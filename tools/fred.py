#!/usr/bin/env python3
"""FRED (Florence RGB-Event Drone dataset, ACM MM 2025, Apache 2.0) as an
addition for the drone gaps (README.md, Procedure, step 10). One zip per
sequence on Hugging Face; inside, event data first and labels and RGB
frames at the end: only these are read, with HTTP range requests.

    python tools/fred.py --step label      # step 0: only the labels of all sequences
    python tools/fred.py --step anteprima  # step 0: one frame per sequence and sheets, to drop the indoor ones
    python tools/fred.py --step frame --seq 107 210 18     # step 1: RGB and labels -> 1_frame
    python tools/fred.py --step annotati --seq 107 210 18  # step 2: drop frames without boxes -> 2_annotati
    python tools/fred.py --step dedup --seq 107 210 18     # step 3: 16-bit hash on whole frames -> 3_dedup
    python tools/fred.py --step ritaglio --seq 107 210 18  # step 4: 640x512 of the kept ones -> 4_ritagli
    python tools/fred.py --step sam --seq 107 210 18       # step 5: SAM 2 with the box as prompt -> 5_box/sam
    python tools/fred.py --step fogli_sam --seq 107 210 18 # step 5: check sheets -> 5_box/fogli_sam
    python tools/fred.py --step yolo --seq 107 210 18      # step 5: detector ($DETECTOR_PESI) boxes -> 5_box/yolo
    python tools/fred.py --step fogli_yolo --seq 107 210 18
    python tools/fred.py --step movimento --seq 18           # step 5, courtyards: median background + SAM with a point
    python tools/fred.py --step fogli_movimento --seq 18
    python tools/fred.py --step sessioni                    # step 6: session proposal and sheets -> 6_sessioni
    python tools/fred.py --step proposte --seq ...          # step 5: YOLO on sky, motion on courtyards -> proposte
    python tools/fred.py --step conta                       # step 7: count per cell -> 7_conteggio
    python tools/fred.py --step cvat                        # step 5: one CVAT task 'fred_<seq>' per sequence, with boxes
    python tools/fred.py --step cvat_import                 # step 5: CVAT YOLO zips in 5_box/cvat_zip -> 5_box/manuale
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import INPUT, BAND_EDGES, BANDS  # noqa: E402
from tools import pipeline  # noqa: E402
from tools.pipeline import (annotated, count, crop, cvat_export, cvat_import, dedup, motion,  # noqa: E402
                          proposals, sam, sam_sheets, yolo)

HF = "https://huggingface.co/datasets/GabrieleMagrini/FRED"
PREFIX = "FRED"  # video name (`<PREFIX>_<seq>`); tools/extra_sources.py changes it for the other sources


class HttpFile(io.RawIOBase):
    """Read-only remote file for `zipfile`: each read is a Range request, with at
    least `chunk` bytes read ahead and a few retries (the proxy comes and goes)."""

    def __init__(self, url: str, chunk: int = 1 << 20):
        self.url, self.chunk, self.pos = url, chunk, 0
        self.size = int(self._get("HEAD").headers["Content-Length"])
        self.buf, self.buf_at = b"", 0

    def _get(self, method: str, rng: str | None = None) -> requests.Response:
        for k in range(8):
            try:
                r = requests.request(method, self.url, headers={"Range": rng} if rng else {},
                                     allow_redirects=True, timeout=60)
                r.raise_for_status()
                return r
            except requests.RequestException:
                time.sleep(5 * (k + 1))
        raise IOError(f"{method} {self.url} {rng}: troppe riprove")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if not (self.buf_at <= self.pos and self.pos + n <= self.buf_at + len(self.buf)):
            end = min(self.size, self.pos + max(n, self.chunk)) - 1
            self.buf, self.buf_at = self._get("GET", f"bytes={self.pos}-{end}").content, self.pos
        out = self.buf[self.pos - self.buf_at:self.pos - self.buf_at + n]
        self.pos += len(out)
        return out


def sequences() -> list[tuple[str, str]]:
    """(split, sequence) of all zips, from the Hugging Face API."""
    api = HF.replace("/datasets/", "/api/datasets/") + "/tree/main/"
    out = []
    for split in ("train", "test"):
        for k in range(8):
            try:
                d = requests.get(api + split, timeout=60).json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(5 * (k + 1))
        out += [(split, Path(x["path"]).stem) for x in d if x["path"].endswith(".zip")]
    return sorted(out, key=lambda x: int(x[1]))


def parse_labels(text: str) -> list[tuple[float, float, float, float, float, str]]:
    """Lines `time: x1, y1, x2, y2, id, model` -> tuples."""
    rows = []
    for ln in text.splitlines():
        if ":" not in ln:
            continue
        t, rest = ln.split(":", 1)
        p = [x.strip() for x in rest.split(",")]
        rows.append((float(t), *map(float, p[:4]), p[5] if len(p) > 5 else ""))
    return rows


def seq_labels(args) -> tuple[list, str]:
    """One sequence: zip index and `coordinates_rgb.txt` -> (row of
    `sequenze.csv`, label text)."""
    split, seq = args
    hf = HttpFile(f"{HF}/resolve/main/{split}/{seq}.zip")
    z = zipfile.ZipFile(hf)
    info = {Path(x.filename).name: x for x in z.infolist()}
    rgb = sorted(x.filename for x in z.infolist() if "/RGB/" in x.filename and x.filename.endswith(".jpg"))
    text = z.read(info["coordinates_rgb.txt"]).decode()
    rows = parse_labels(text)
    side = np.array([np.sqrt(max(x2 - x1, 0) * max(y2 - y1, 0)) for _, x1, y1, x2, y2, *_ in rows])
    bands = Counter(np.digitize(side, BAND_EDGES)) if len(side) else Counter()
    hours = sorted(("_".join(Path(r).stem.split("_")[-3:]) for r in rgb), key=lambda h: _sec(h + ".jpg"))
    dates = Counter(x.date_time[:3] for x in z.infolist() if x.filename.endswith(".jpg"))
    return [seq, split, len(rgb), len(rows), "|".join(sorted({r[-1] for r in rows})),
            "-".join(map(str, dates.most_common(1)[0][0])) if dates else "",
            hours[0] if hours else "", hours[-1] if hours else "",
            info["coordinates_rgb.txt"].header_offset, round(hf.size / 1e6),
            *(bands[i] for i in range(len(BANDS)))], text


def fetch_labels(raw: Path, workers: int = 8) -> None:
    """Step 0: for each sequence the zip index and `coordinates_rgb.txt` ->
    `0_raw/label/<seq>.txt` and `0_raw/sequenze.csv` (frames, boxes, models, dates,
    boxes per band on sqrt(w*h) in original pixels). FRED boxes are not
    tight: the bands here only serve to choose the sequences. `workers`
    sequences in parallel: the time is the proxy latency, not the bytes."""
    from concurrent.futures import ThreadPoolExecutor

    lab = raw / "label"
    lab.mkdir(parents=True, exist_ok=True)
    out = raw / "sequenze.csv"
    done = {r["sequenza"] for r in csv.DictReader(open(out))} if out.exists() else set()
    new = not out.exists()
    todo = [s for s in sequences() if s[1] not in done]
    with open(out, "a", newline="") as f, ThreadPoolExecutor(workers) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["sequenza", "split", "frame", "box", "modelli", "data_zip", "ora_inizio", "ora_fine",
                        "offset_label", "mb_zip", *BANDS])
        for row, text in ex.map(seq_labels, todo):
            (lab / f"{row[0]}.txt").write_text(text)
            w.writerow(row)
            f.flush()
            print(f"fred: {row[1]}/{row[0]} {row[2]} frame, {row[3]} box", flush=True)


def seq_preview(args) -> str:
    """The RGB frame at mid-sequence, with the boxes of the nearest frame ->
    `0_raw/anteprima/<seq>.jpg`."""
    import cv2

    split, seq, out = args
    z = zipfile.ZipFile(HttpFile(f"{HF}/resolve/main/{split}/{seq}.zip"))
    rgb = sorted((x for x in z.infolist() if x.filename.endswith(".jpg")), key=lambda x: x.filename)
    x = rgb[len(rgb) // 2]
    img = cv2.imdecode(np.frombuffer(z.read(x), np.uint8), cv2.IMREAD_COLOR)
    (out / f"{seq}.jpg").write_bytes(cv2.imencode(".jpg", img)[1].tobytes())
    return seq


def previews(raw: Path, workers: int = 8, per: int = 24) -> None:
    """One frame per sequence in `0_raw/anteprima/` and numbered sheets in
    `0_raw/fogli/` (`per` sequences per sheet, number = sequence) to
    pick by eye the indoor ones to drop."""
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    out = raw / "anteprima"
    out.mkdir(exist_ok=True)
    todo = [(sp, sq, out) for sp, sq in sequences() if not (out / f"{sq}.jpg").exists()]
    with ThreadPoolExecutor(workers) as ex:
        for seq in ex.map(seq_preview, todo):
            print(f"fred: anteprima {seq}", flush=True)
    sheets = raw / "fogli"
    sheets.mkdir(exist_ok=True)
    seqs = sorted((p.stem for p in out.glob("*.jpg")), key=int)
    for k in range(0, len(seqs), per):
        tiles = []
        for sq in seqs[k:k + per]:
            t = cv2.resize(cv2.imread(str(out / f"{sq}.jpg")), (320, 180))
            cv2.putText(t, sq, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(t, sq, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(sheets / f"{k // per + 1:02d}.jpg"),
                    np.vstack([np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)]))
    print(f"fred: {len(seqs)} anteprime, fogli in {sheets}")


def _sec(name: str) -> float:
    """Frame time from the name (`..._HH_MM_SS.micro.jpg`) in seconds."""
    h, m, x = Path(name).stem.split("_")[-3:]
    return int(h) * 3600 + int(m) * 60 + float(x)


def _split(seq: str) -> str:
    return next(r["split"] for r in csv.DictReader(open(INPUT / "fred" / "0_raw" / "sequenze.csv"))
                if r["sequenza"] == seq)


def download_frames(src: Path, seq: str, W: int = 1280, H: int = 720) -> None:
    """Step 1: only the `RGB/` frames and the labels of the zip (read in position
    order, in 32 MB blocks: no event data) -> `1_frame/FRED_<seq>/`
    with `images/FRED_<seq>_fNNNNN.jpg` (original bytes, in time order) and
    `labels/*.txt` YOLO (class 2, drone) on the nearest frame in time,
    within half a frame. Boxes clipped to the frame."""
    v = f"{PREFIX}_{seq}"
    dst = src / "1_frame" / v
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(exist_ok=True)
    z = zipfile.ZipFile(HttpFile(f"{HF}/resolve/main/{_split(seq)}/{seq}.zip", chunk=32 << 20))
    rgb = sorted((x for x in z.infolist() if "/RGB/" in x.filename and x.filename.endswith(".jpg")),
                 key=lambda x: _sec(x.filename))
    t = np.array([_sec(x.filename) for x in rgb]) - _sec(rgb[0].filename)
    rows = parse_labels(z.read(f"{seq}/coordinates_rgb.txt").decode())
    lab = [[] for _ in rgb]
    miss = 0
    for tt, x1, y1, x2, y2, _ in rows:
        i = int(np.abs(t - tt).argmin())
        if abs(t[i] - tt) > 1 / 60:
            miss += 1
            continue
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            lab[i].append(f"2 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
    name = {x.filename: i for i, x in enumerate(rgb)}
    for x in sorted(rgb, key=lambda x: x.header_offset):  # in position order: contiguous reads
        i = name[x.filename]
        (dst / "images" / f"{v}_f{i:05d}.jpg").write_bytes(z.read(x))
        (dst / "labels" / f"{v}_f{i:05d}.txt").write_text("".join(lab[i]))
    print(f"fred: {v} {len(rgb)} frame, {sum(map(len, lab))} box, {miss} label senza frame entro mezzo frame",
          flush=True)


def _embed(raw: Path, seqs: list[str]) -> dict[str, np.ndarray]:
    """DINOv2 ViT-S/14 embedding (CLS, norm 1) of the preview of each
    sequence, without the black border, as for the Svanstrom sessions."""
    import cv2
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(dev).eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = {}
    with torch.no_grad():
        for s in seqs:
            im = cv2.imread(str(raw / "anteprima" / f"{s}.jpg"))[36:693, 28:1247]
            x = (cv2.resize(im, (448, 252))[..., ::-1] / 255 - mean) / std  # multiples of 14
            x = torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32, device=dev)
            out[s] = torch.nn.functional.normalize(m(x), dim=1).cpu().numpy()
    return out


def session_sheets(src: Path, gap: float = 3600, thr: float = 0.9) -> None:
    """Step 6, session proposal without indoor and night: same **day**
    (hourly blocks: sequences in number order, new block when the time
    goes back or after a pause of more than `gap` s) **or** same **place**
    (groups by average linkage of the DINOv2 similarity of the previews above
    `thr`), merged by transitivity; then the exceptions of `correzioni.txt`. ->
    `6_sessioni/sessioni.csv` (clip, session), `proposta.csv` (sequence,
    proposal, session, block, visual group, time, drone, scenario), `embedding.npz` and one sheet per
    session in `6_sessioni/fogli/SNN.jpg` (number, time, block = day, visual
    group = place, drone; at the end
    in gray the most similar sequence outside the session)."""
    import cv2

    from tools.common import propose_sessions

    raw = src / "0_raw"
    out = src / "6_sessioni"
    (out / "fogli").mkdir(parents=True, exist_ok=True)
    for p in (out / "fogli").glob("*.jpg"):
        p.unlink()
    sc = {r["sequenza"]: r["scenario"] for r in csv.DictReader(open(raw / "scenari.csv"))}
    R = sorted((r for r in csv.DictReader(open(raw / "sequenze.csv")) if sc[r["sequenza"]] != "interni"),
               key=lambda r: int(r["sequenza"]))

    def t(h):
        try:
            return _sec(h + ".jpg")
        except ValueError:
            return None

    blk, k, prev, bad = {}, 0, None, []
    for r in R:
        a = t(r["ora_inizio"])
        bad += [r["sequenza"]] if a is None or t(r["ora_fine"]) is None else []
        if prev is None or (a is not None and (a < prev or a - prev > gap)):
            k += 1
        blk[r["sequenza"]] = f"B{k:02d}"
        prev = t(r["ora_fine"]) or prev
    f = out / "embedding.npz"
    emb = dict(np.load(f)) if f.exists() else {}
    if set(blk) - set(emb):
        emb = _embed(raw, sorted(blk, key=int))
        np.savez(f, **emb)
    seqs, sim, groups = propose_sessions({s: emb[s] for s in blk}, thr)
    par = {}

    def find(x):
        while par.setdefault(x, x) != x:
            x = par[x]
        return x

    for s, g in zip(seqs, groups):
        par[find(f"g{g}")] = find(blk[s])
    roots = sorted({find(blk[s]) for s in seqs}, key=lambda x: min(int(s) for s in seqs if find(blk[s]) == x))
    ses = {s: f"S{roots.index(find(blk[s])) + 1:02d}" for s in seqs}
    vis = {s: f"g{g:02d}" for s, g in zip(seqs, groups)}
    prop = dict(ses)
    # corrections: "SNN: via a b c-d" -> the sequences leave SNN and form a new session
    cf = out / "correzioni.txt"
    for k2, ln in enumerate(cf.read_text().splitlines() if cf.exists() else []):
        if ":" not in ln or ln.startswith("#"):
            continue
        S, rest = ln.split(":", 1)
        for tok in rest.split()[1:]:
            a, b = (tok.split("-") + [tok])[:2]
            for q in range(int(a), int(b) + 1):
                if prop.get(str(q)) == S.strip():
                    ses[str(q)] = f"{S.strip()}.{k2}"
    # "separa a b c-d": the listed sequences together form a new session (sequence numbers: stable)
    for k2, ln in enumerate(cf.read_text().splitlines() if cf.exists() else []):
        if ln.startswith("separa"):
            for tok in ln.split()[1:]:
                a, b = (tok.split("-") + [tok])[:2]
                for q in range(int(a), int(b) + 1):
                    if str(q) in ses:
                        ses[str(q)] = f"sep{k2}"
    # "unisci a b c": the sessions of sequences a, b, c become one (sequence numbers: stable)
    for ln in cf.read_text().splitlines() if cf.exists() else []:
        if ln.startswith("unisci"):
            qs = [q for q in ln.split()[1:] if q in ses]
            old = {ses[q] for q in qs}
            for q in seqs:
                if ses[q] in old:
                    ses[q] = ses[qs[0]]
    # the night (removed by the user) stays in the grouping, so the proposal and the corrections do not change,
    # and is removed here
    for q in [q for q in seqs if sc[q] == "notte"]:
        del ses[q]
    allseq, seqs = seqs, [q for q in seqs if q in ses]  # allseq: indices of `sim`
    order = sorted(set(ses.values()), key=lambda x: min(int(q) for q in seqs if ses[q] == x))
    ses = {q: f"S{order.index(ses[q]) + 1:02d}" for q in seqs}
    by = {r["sequenza"]: r for r in R}
    with open(out / "sessioni.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "sessione"])
        w.writerows((f"{PREFIX}_{q}/s0", f"{PREFIX}_{ses[q]}") for q in sorted(seqs, key=int))
    with open(out / "proposta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequenza", "proposta", "sessione", "blocco", "gruppo_visivo", "ora_inizio", "drone", "scenario"])
        w.writerows((s, prop[s], ses[s], blk[s], vis[s], by[s]["ora_inizio"][:8], by[s]["modelli"], sc[s])
                    for s in sorted(seqs, key=int))
    short = lambda m: "+".join(x.replace("DarwinFPV ", "").replace("DJI ", "").replace("Betafpv ", "")
                               .replace(" EDU", "") for x in m.split("|"))

    def tile(s, grey=False):
        im = cv2.resize(cv2.imread(str(raw / "anteprima" / f"{s}.jpg")), (256, 144))
        if grey:
            im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        txt = f"{s} {by[s]['ora_inizio'][:5].replace('_', ':')} {blk[s]} {vis[s]} {short(by[s]['modelli'])}"
        cv2.rectangle(im, (0, 0), (256, 20), (0, 0, 0), -1)
        cv2.putText(im, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    idx = {s: i for i, s in enumerate(allseq)}
    for S in sorted(set(ses.values())):
        mem = sorted((s for s in seqs if ses[s] == S), key=int)
        rest = [s for s in seqs if ses[s] != S]
        near = max(rest, key=lambda o: max(sim[idx[m], idx[o]] for m in mem)) if rest else None
        tiles = [tile(s) for s in mem] + ([tile(near, grey=True)] if near else [])
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(out / "fogli" / f"{S}.jpg"), np.vstack([np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)]))
    n = sorted((sum(v == S for v in ses.values()) for S in set(ses.values())), reverse=True)
    print(f"fred: {len(seqs)} sequenze, {k} blocchi orari, {len(set(groups))} gruppi visivi (soglia {thr}) -> "
          f"{len(n)} sessioni {n}; ora non leggibile: {' '.join(bad) or '-'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=INPUT / "fred", help="one folder per step")
    p.add_argument("--step", choices=["label", "anteprima", "frame", "annotati", "dedup", "ritaglio", "sam", "yolo", "movimento", "fogli_sam", "fogli_yolo", "fogli_movimento", "sessioni", "proposte", "conta", "cvat", "cvat_import"], required=True)
    p.add_argument("--seq", nargs="+", default=[], help="sequences (numbers) for frame, dedup, ritaglio")
    a = p.parse_args(argv)
    pipeline.PREFIX = PREFIX
    if a.step == "label":
        fetch_labels(a.src / "0_raw")
    if a.step == "anteprima":
        previews(a.src / "0_raw")
    if a.step == "frame":
        for seq in a.seq:
            download_frames(a.src, seq)
    if a.step == "annotati":
        annotated(a.src, a.seq)
    if a.step == "dedup":
        dedup(a.src, a.seq)
    if a.step == "ritaglio":
        crop(a.src, a.seq)
    if a.step == "sam":
        sam(a.src, a.seq)
    if a.step == "yolo":
        yolo(a.src, a.seq)
    if a.step == "sessioni":
        session_sheets(a.src)
    if a.step == "proposte":
        proposals(a.src, a.seq)
    if a.step == "conta":
        count(a.src)
    if a.step == "cvat":
        cvat_export(a.src)
    if a.step == "cvat_import":
        cvat_import(a.src)
    if a.step == "movimento":
        motion(a.src, a.seq)
    if a.step in ("fogli_sam", "fogli_yolo", "fogli_movimento"):
        sam_sheets(a.src, a.seq, source=a.step.split("_")[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
