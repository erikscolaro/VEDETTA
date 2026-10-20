#!/usr/bin/env python3
"""FRED (Florence RGB-Event Drone dataset, ACM MM 2025, Apache 2.0) come
aggiunta per i buchi dei droni (docs/procedura.md, passo 10). Uno zip per
sequenza su Hugging Face; dentro, prima i dati a eventi e in fondo label e
frame RGB: si leggono solo questi, con richieste HTTP a intervalli.

    python tools/fred.py --step label      # passo 0: solo le label di tutte le sequenze
    python tools/fred.py --step anteprima  # passo 0: un frame per sequenza e fogli, per togliere le indoor
    python tools/fred.py --step frame --seq 107 210 18     # passo 1: RGB e label -> 1_frame
    python tools/fred.py --step annotati --seq 107 210 18  # passo 2: via i frame senza box -> 2_annotati
    python tools/fred.py --step dedup --seq 107 210 18     # passo 3: hash a 16 bit sui frame interi -> 3_dedup
    python tools/fred.py --step ritaglio --seq 107 210 18  # passo 4: 640x512 dei tenuti -> 4_ritagli
    python tools/fred.py --step sam --seq 107 210 18       # passo 5: SAM 2 col box come prompt -> 5_box/sam
    python tools/fred.py --step fogli_sam --seq 107 210 18 # passo 5: fogli di controllo -> 5_box/fogli_sam
    python tools/fred.py --step yolo --seq 107 210 18      # passo 5: box proposti da yolo26m-intern -> 5_box/yolo
    python tools/fred.py --step fogli_yolo --seq 107 210 18
    python tools/fred.py --step movimento --seq 18           # passo 5, cortili: sfondo mediano + SAM a punto
    python tools/fred.py --step fogli_movimento --seq 18
    python tools/fred.py --step sessioni                    # passo 6: proposta di sessioni e fogli -> 6_sessioni
    python tools/fred.py --step proposte --seq ...          # passo 5: YOLO sul cielo, movimento sui cortili -> 5_box/proposte
    python tools/fred.py --step conta                       # passo 7: conteggio per cella -> 7_conteggio
    python tools/fred.py --step cvat                        # passo 5: un task CVAT 'fred_<seq>' per sequenza, con i box proposti
    python tools/fred.py --step cvat_import                 # passo 5: zip YOLO di CVAT in 5_box/cvat_zip -> 5_box/manuale
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
from tools.svanstrom import BAND_EDGES, BANDS  # noqa: E402

HF = "https://huggingface.co/datasets/GabrieleMagrini/FRED"
PREFIX = "FRED"  # nome dei video (`<PREFIX>_<seq>`); tools/aggiunte.py lo cambia per le altre fonti


class HttpFile(io.RawIOBase):
    """File remoto in sola lettura per `zipfile`: ogni lettura e' una richiesta
    Range, con almeno `chunk` byte letti in anticipo e qualche riprova (il
    proxy va a tratti)."""

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
    """(split, sequenza) di tutti gli zip, dall'API di Hugging Face."""
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
    """Righe `tempo: x1, y1, x2, y2, id, modello` -> tuple."""
    rows = []
    for ln in text.splitlines():
        if ":" not in ln:
            continue
        t, rest = ln.split(":", 1)
        p = [x.strip() for x in rest.split(",")]
        rows.append((float(t), *map(float, p[:4]), p[5] if len(p) > 5 else ""))
    return rows


def seq_labels(args) -> tuple[list, str]:
    """Una sequenza: indice dello zip e `coordinates_rgb.txt` -> (riga di
    `sequenze.csv`, testo delle label)."""
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
    """Passo 0: per ogni sequenza l'indice dello zip e `coordinates_rgb.txt` ->
    `0_raw/label/<seq>.txt` e `0_raw/sequenze.csv` (frame, box, modelli, date,
    box per fascia sulla sqrt(w*h) in pixel originali). I box di FRED non sono
    aderenti: le fasce qui servono solo per scegliere le sequenze. `workers`
    sequenze in parallelo: il tempo e' la latenza del proxy, non i byte."""
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
    """Il frame RGB a meta' sequenza, con i box del frame piu' vicino ->
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
    """Un frame per sequenza in `0_raw/anteprima/` e fogli numerati in
    `0_raw/fogli/` (`per` sequenze per foglio, numero = sequenza) per
    scegliere a occhio quelle indoor da togliere."""
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
    """Ora del frame dal nome (`..._HH_MM_SS.micro.jpg`) in secondi."""
    h, m, x = Path(name).stem.split("_")[-3:]
    return int(h) * 3600 + int(m) * 60 + float(x)


def _split(seq: str) -> str:
    return next(r["split"] for r in csv.DictReader(open(ROOT / "data" / "fred" / "0_raw" / "sequenze.csv"))
                if r["sequenza"] == seq)


def download_frames(src: Path, seq: str, W: int = 1280, H: int = 720) -> None:
    """Passo 1: solo i frame di `RGB/` e le label dello zip (letti in ordine di
    posizione, a blocchi da 32 MB: niente dati a eventi) -> `1_frame/FRED_<seq>/`
    con `images/FRED_<seq>_fNNNNN.jpg` (byte originali, in ordine di tempo) e
    `labels/*.txt` YOLO (classe 2, drone) sul frame piu' vicino nel tempo,
    entro mezzo frame. Box tagliati al frame."""
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
    for x in sorted(rgb, key=lambda x: x.header_offset):  # in ordine di posizione: letture contigue
        i = name[x.filename]
        (dst / "images" / f"{v}_f{i:05d}.jpg").write_bytes(z.read(x))
        (dst / "labels" / f"{v}_f{i:05d}.txt").write_text("".join(lab[i]))
    print(f"fred: {v} {len(rgb)} frame, {sum(map(len, lab))} box, {miss} label senza frame entro mezzo frame",
          flush=True)


def annotated(src: Path, seqs: list[str]) -> None:
    """Passo 2: via i frame senza box -> `2_annotati/annotati.csv` (clip,
    frame), aggiornato per sequenza."""
    out = src / "2_annotati"
    out.mkdir(exist_ok=True)
    f = out / "annotati.csv"
    clips = {f"{PREFIX}_{s}/s0" for s in seqs}
    rows = [r for r in csv.DictReader(open(f)) if r["clip"] not in clips] if f.exists() else []
    for s in seqs:
        v = f"{PREFIX}_{s}"
        lab = sorted((src / "1_frame" / v / "labels").glob("*.txt"))
        idx = [int(t.stem.rsplit("_f", 1)[1]) for t in lab if t.stat().st_size]
        rows += [{"clip": f"{v}/s0", "frame": i} for i in idx]
        print(f"fred: {v} {len(idx)} frame con box, {len(lab) - len(idx)} senza (tolti)", flush=True)
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["clip", "frame"])
        w.writeheader()
        w.writerows(rows)


def dedup(src: Path, seqs: list[str], hash_thr: int = 16, geo_thr: float = 0.05) -> None:
    """Passo 3 (prima del ritaglio e di SAM): quasi-duplicati sui frame di
    `2_annotati/annotati.csv`, interi, con l'hash di Svanstrom (`dedup_clip`,
    16 bit) -> `3_dedup/tenuti.csv`
    (clip, frame), aggiornato per sequenza."""
    from tools.svanstrom import dedup_clip

    out = src / "3_dedup"
    out.mkdir(exist_ok=True)
    f = out / "tenuti.csv"
    old = [r for r in csv.DictReader(open(f))] if f.exists() else []
    clips = {f"{PREFIX}_{s}/s0" for s in seqs}
    rows = [r for r in old if r["clip"] not in clips]
    for s in seqs:
        v = f"{PREFIX}_{s}"
        idx = [int(r["frame"]) for r in csv.DictReader(open(src / "2_annotati" / "annotati.csv"))
               if r["clip"] == f"{v}/s0"]
        clip, keep, _ = dedup_clip((src / "1_frame", src / "1_frame" / v / "labels", f"{v}/s0", idx,
                                    hash_thr, geo_thr))
        rows += [{"clip": clip, "frame": i} for i in keep]
        print(f"fred: {v} {len(keep)} frame tenuti su {len(idx)} ({len(keep) / max(len(idx), 1):.0%})", flush=True)
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["clip", "frame"])
        w.writeheader()
        w.writerows(rows)


def _valid(images: list[Path], thr: int = 8) -> tuple[int, int, int, int]:
    """Area senza il bordo nero di padding: righe e colonne della mediana di
    qualche frame sopra `thr`."""
    import cv2

    g = np.median([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in images[::max(1, len(images) // 8)]], axis=0)
    rows, cols = np.where(g.max(1) > thr)[0], np.where(g.max(0) > thr)[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop(src: Path, seqs: list[str], w: int = 640, h: int = 512, seed: int = 0) -> None:
    """Passo 4: i frame tenuti ritagliati a `w`x`h` senza ridimensionare,
    dentro l'area senza padding, con il box piu' grande in posizione casuale
    (seed fisso per frame) -> `4_ritagli/FRED_<seq>/{images,labels}` e
    `4_ritagli/ritagli.csv` (clip, frame, x0, y0). Gli altri box tagliati
    alla finestra (parte visibile), via se restano sotto 1 px. Il box piu'
    grande resta intero solo se sta in `w`x`h` e fuori dal padding: i casi
    contrari si contano (tagliato) e si scrivono in `ritagli.csv`."""
    import random

    import cv2

    kept = {}
    for r in csv.DictReader(open(src / "3_dedup" / "tenuti.csv")):
        kept.setdefault(r["clip"], []).append(int(r["frame"]))
    out = src / "4_ritagli"
    log = []
    for s in seqs:
        v = f"{PREFIX}_{s}"
        fr = src / "1_frame" / v
        X0, Y0, X1, Y1 = _valid(sorted((fr / "images").glob("*.jpg")))
        if X1 - X0 < w or Y1 - Y0 < h:  # video scuri: la soglia del bordo nero taglia troppo, si usa il frame intero
            H, W = cv2.imread(str(next((fr / "images").glob("*.jpg")))).shape[:2]
            X0, Y0, X1, Y1 = 0, 0, W, H
        (out / v / "images").mkdir(parents=True, exist_ok=True)
        (out / v / "labels").mkdir(exist_ok=True)
        for i in kept.get(f"{v}/s0", []):
            img = cv2.imread(str(fr / "images" / f"{v}_f{i:05d}.jpg"))
            H, W = img.shape[:2]
            b = np.array([ln.split() for ln in (fr / "labels" / f"{v}_f{i:05d}.txt").read_text().splitlines()],
                         float).reshape(-1, 5)
            cls, b = b[:, 0].astype(int), b[:, 1:] * [W, H, W, H]
            b = np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2]  # x1 y1 x2 y2
            rng = random.Random(f"{seed}-{v}-{i}")
            if len(b):
                x1, y1, x2, y2 = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
                lo_x, hi_x = max(X0, x2 - w), min(X1 - w, x1)
                lo_y, hi_y = max(Y0, y2 - h), min(Y1 - h, y1)
                x0 = int(rng.uniform(lo_x, hi_x)) if lo_x <= hi_x else int(np.clip((x1 + x2 - w) / 2, X0, X1 - w))
                y0 = int(rng.uniform(lo_y, hi_y)) if lo_y <= hi_y else int(np.clip((y1 + y2 - h) / 2, Y0, Y1 - h))
            else:
                x0, y0 = rng.randint(X0, X1 - w), rng.randint(Y0, Y1 - h)
            lines, cut = [], 0
            for c, (bx1, by1, bx2, by2) in zip(cls, b - [x0, y0, x0, y0]):
                cut += bx1 < 0 or by1 < 0 or bx2 > w or by2 > h
                bx1, by1, bx2, by2 = max(bx1, 0), max(by1, 0), min(bx2, w), min(by2, h)
                if bx2 - bx1 >= 1 and by2 - by1 >= 1:
                    lines.append(f"{c} {(bx1 + bx2) / 2 / w:.6f} {(by1 + by2) / 2 / h:.6f} "
                                 f"{(bx2 - bx1) / w:.6f} {(by2 - by1) / h:.6f}\n")
            stem = f"{v}_f{i:05d}"
            cv2.imwrite(str(out / v / "images" / f"{stem}.jpg"), img[y0:y0 + h, x0:x0 + w],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / v / "labels" / f"{stem}.txt").write_text("".join(lines))
            log.append((f"{v}/s0", i, x0, y0, len(b), cut, len(b) - len(lines)))
        n = [r for r in log if r[0] == f"{v}/s0"]
        print(f"fred: {v} {len(n)} ritagli, area valida {X0},{Y0}-{X1},{Y1}; frame con box tagliati "
              f"{sum(r[5] > 0 for r in n)}, con box fuori {sum(r[6] > 0 for r in n)}", flush=True)
    f = out / "ritagli.csv"
    old = [r for r in csv.DictReader(open(f))] if f.exists() else []
    clips = {f"{PREFIX}_{s}/s0" for s in seqs}
    with open(f, "w", newline="") as fh:
        wr = csv.writer(fh)
        cols = ["clip", "frame", "x0", "y0", "box", "tagliati", "fuori"]
        wr.writerow(cols)
        wr.writerows([[r.get(k, "") for k in cols] for r in old if r["clip"] not in clips] + log)


def sam(src: Path, seqs: list[str]) -> None:
    """Passo 5: box aderenti con SAM 2.1 Large sui ritagli, il box originale
    (largo) come prompt -> `5_box/sam/FRED_<seq>.json` (frame, box x y w h in
    pixel del ritaglio o None). Su Svanstrom il box come prompt ridava il box
    stesso, ma li' gli oggetti erano di 4-6 px; qui sono grandi: da verificare
    sui fogli (`fogli_sam`)."""
    import json

    import cv2
    from ultralytics import SAM

    from tools.svanstrom import _mask_box

    model = SAM(str(ROOT / "data" / "modelli" / "sam2.1_l.pt"))
    out = src / "5_box" / "sam"
    out.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        v = f"{PREFIX}_{s}"
        d = src / "4_ritagli" / v
        rec = {"frame": [], "orig": [], "box": []}
        for t in sorted((d / "labels").glob("*.txt")):
            img = cv2.imread(str(d / "images" / f"{t.stem}.jpg"))
            H, W = img.shape[:2]
            for ln in t.read_text().splitlines():
                cx, cy, bw, bh = (float(x) for x in ln.split()[1:])
                x1, y1, x2, y2 = (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H
                res = model(img, bboxes=[[x1, y1, x2, y2]], verbose=False)[0]
                b = _mask_box(res.masks.data[0].cpu().numpy() > 0.5) if res.masks is not None and len(res.masks.data) else None
                rec["frame"].append(int(t.stem.rsplit("_f", 1)[1]))
                rec["orig"].append([x1, y1, x2 - x1, y2 - y1])
                rec["box"].append(None if b is None else [float(x) for x in b])
        (out / f"{v}.json").write_text(json.dumps(rec))
        n = sum(b is not None for b in rec["box"])
        r = [np.sqrt(b[2] * b[3] / (o[2] * o[3])) for b, o in zip(rec["box"], rec["orig"]) if b is not None and o[2] * o[3]]
        print(f"fred: {v} SAM su {len(rec['box'])} box, {n} con maschera, lato SAM/originale mediana "
              f"{np.median(r):.2f}" if r else f"fred: {v} nessun box", flush=True)


def yolo(src: Path, seqs: list[str], weights: str = "yolo26m-intern.pt", grow: float = 2.0, conf: float = 0.1) -> None:
    """Passo 5, alternativa a SAM: box proposti da una YOLO (`data/modelli/`,
    classi UAV, aereo, elicottero, uccello) sui ritagli; per ogni box di FRED
    la detection UAV (classi 0 e 1) col centro dentro il box allargato `grow`
    volte e la confidenza piu' alta -> `5_box/yolo/FRED_<seq>.json` (frame,
    orig, box, conf; box None se non trova niente). Il box di FRED dice solo
    dove cercare: i frame senza detection restano, per la revisione in CVAT."""
    import json

    from ultralytics import YOLO

    model = YOLO(str(ROOT / "data" / "modelli" / weights))
    out = src / "5_box" / "yolo"
    out.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        v = f"{PREFIX}_{s}"
        d = src / "4_ritagli" / v
        rec = {"frame": [], "orig": [], "box": [], "conf": []}
        for t in sorted((d / "labels").glob("*.txt")):
            res = model(str(d / "images" / f"{t.stem}.jpg"), conf=conf, verbose=False)[0]
            H, W = res.orig_shape
            det = [(b.xyxy[0].tolist(), float(b.conf)) for b in res.boxes if int(b.cls) in (0, 1)]
            for ln in t.read_text().splitlines():
                cx, cy, bw, bh = (float(x) for x in ln.split()[1:])
                cx, cy, bw, bh = cx * W, cy * H, bw * W, bh * H
                ok = [(xy, c) for xy, c in det if abs((xy[0] + xy[2]) / 2 - cx) <= bw * grow / 2
                      and abs((xy[1] + xy[3]) / 2 - cy) <= bh * grow / 2]
                xy, c = max(ok, key=lambda x: x[1]) if ok else (None, 0.0)
                rec["frame"].append(int(t.stem.rsplit("_f", 1)[1]))
                rec["orig"].append([cx - bw / 2, cy - bh / 2, bw, bh])
                rec["box"].append(None if xy is None else [xy[0], xy[1], xy[2] - xy[0], xy[3] - xy[1]])
                rec["conf"].append(c)
        (out / f"{v}.json").write_text(json.dumps(rec))
        r = [np.sqrt(b[2] * b[3] / (o[2] * o[3])) for b, o in zip(rec["box"], rec["orig"]) if b and o[2] * o[3]]
        n = sum(b is not None for b in rec["box"])
        print(f"fred: {v} YOLO su {len(rec['box'])} box, {n} trovati ({n / max(len(rec['box']), 1):.0%})"
              + (f", lato YOLO/originale mediana {np.median(r):.2f}" if r else ""), flush=True)


def motion(src: Path, seqs: list[str], grow: float = 1.5, n_bg: int = 60, use_sam: bool = True) -> None:
    """Passo 5 per i cortili: la camera e' su un treppiede, quindi lo sfondo
    e' la mediana di `n_bg` frame sparsi della sequenza (il drone che si muove
    sparisce). Nel box di FRED allargato `grow` volte, i pixel che si staccano
    dallo sfondo (soglia: mediana + 6 MAD della differenza, almeno 20) e la
    componente piu' grande danno il box; poi SAM 2 con un punto al centro
    della componente, sul ritaglio -> `5_box/movimento/FRED_<seq>.json`
    (frame, orig, mov, box: box = SAM se ha maschera, altrimenti mov)."""
    import json

    import cv2

    from tools.svanstrom import _mask_box

    model = None
    if use_sam:
        from ultralytics import SAM
        model = SAM(str(ROOT / "data" / "modelli" / "sam2.1_l.pt"))
    off = {(r["clip"], int(r["frame"])): (int(r["x0"]), int(r["y0"]))
           for r in csv.DictReader(open(src / "4_ritagli" / "ritagli.csv"))}
    out = src / "5_box" / "movimento"
    out.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        v = f"{PREFIX}_{s}"
        imgs = sorted((src / "1_frame" / v / "images").glob("*.jpg"))
        bg = np.median([cv2.imread(str(p)) for p in imgs[::max(1, len(imgs) // n_bg)]], axis=0).astype(np.int16)
        d = src / "4_ritagli" / v
        rec = {"frame": [], "orig": [], "mov": [], "box": []}
        for t in sorted((d / "labels").glob("*.txt")):
            i = int(t.stem.rsplit("_f", 1)[1])
            x0, y0 = off[(f"{v}/s0", i)]
            crop = cv2.imread(str(d / "images" / f"{t.stem}.jpg"))
            h, w = crop.shape[:2]
            full = cv2.imread(str(src / "1_frame" / v / "images" / f"{t.stem}.jpg")).astype(np.int16)
            diff = np.abs(full - bg).max(2)[y0:y0 + h, x0:x0 + w]
            for ln in t.read_text().splitlines():
                cx, cy, bw, bh = (float(x) for x in ln.split()[1:])
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                ax0, ay0 = int(max(cx - bw * grow / 2, 0)), int(max(cy - bh * grow / 2, 0))
                ax1, ay1 = int(min(cx + bw * grow / 2, w)), int(min(cy + bh * grow / 2, h))
                reg = diff[ay0:ay1, ax0:ax1]
                mov = box = None
                if reg.size:
                    med = np.median(reg)
                    m = (reg > max(20, med + 6 * np.median(np.abs(reg - med)))).astype(np.uint8)
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                    k, lab, st, cen = cv2.connectedComponentsWithStats(m)
                    if k > 1:
                        j = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
                        bx, by, bw2, bh2 = st[j, :4]
                        mov = [float(bx + ax0), float(by + ay0), float(bw2), float(bh2)]
                        box = mov
                        if model is not None:
                            px, py = cen[j] + [ax0, ay0]
                            res = model(crop, points=[[float(px), float(py)]], labels=[1], verbose=False)[0]
                            if res.masks is not None and len(res.masks.data):
                                b = _mask_box(res.masks.data[0].cpu().numpy() > 0.5)
                                # SAM solo se resta vicino al movimento (non l'intero muro)
                                if b is not None and b[2] * b[3] <= 4 * max(bw2 * bh2, 16):
                                    box = [float(x) for x in b]
                rec["frame"].append(i)
                rec["orig"].append([cx - bw / 2, cy - bh / 2, bw, bh])
                rec["mov"].append(mov)
                rec["box"].append(box)
        (out / f"{v}.json").write_text(json.dumps(rec))
        n = sum(b is not None for b in rec["box"])
        r = [np.sqrt(b[2] * b[3] / (o[2] * o[3])) for b, o in zip(rec["box"], rec["orig"]) if b and o[2] * o[3]]
        print(f"fred: {v} movimento su {len(rec['box'])} box, {n} trovati"
              + (f", lato/originale mediana {np.median(r):.2f}" if r else ""), flush=True)


def proposals(src: Path, seqs: list[str], W: int = 640, H: int = 512) -> None:
    """Passo 5, box proposti per CVAT e per il conteggio: YOLO sulle sequenze di
    cielo, sfondo fermo (`movimento`) sulle altre, come deciso il 16/10/2026 ->
    `5_box/proposte/FRED_<seq>/<stem>.txt` YOLO 640x512 (classe 2). I box non
    trovati restano fuori: si aggiungono in CVAT."""
    import json

    sc = {r["sequenza"]: r["scenario"] for r in csv.DictReader(open(src / "0_raw" / "scenari.csv"))}
    for s in seqs:
        v = f"{PREFIX}_{s}"
        fonte = "yolo" if sc[s].startswith("cielo") else "movimento"
        rec = json.loads((src / "5_box" / fonte / f"{v}.json").read_text())
        out = src / "5_box" / "proposte" / v
        out.mkdir(parents=True, exist_ok=True)
        lines = {f: [] for f in rec["frame"]}
        for f, b in zip(rec["frame"], rec["box"]):
            if b:
                x, y, w, h = b
                lines[f].append(f"2 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        for f, ls in lines.items():
            (out / f"{v}_f{f:05d}.txt").write_text("".join(ls))
        n = sum(map(len, lines.values()))
        print(f"fred: {v} ({sc[s]}, {fonte}) {n} box proposti su {len(rec['box'])}", flush=True)


def boxes(src: Path) -> Path:
    """Box da contare e da mandare in CVAT: `5_box/preselezione/` (tools/preselezione.py)
    se c'e', altrimenti le proposte."""
    pre = src / "5_box" / "preselezione"
    return pre if pre.exists() else src / "5_box" / "proposte"


def excluded(src: Path) -> set[str]:
    """Ritagli tolti dalla preselezione (`5_box/preselezione/esclusi.txt`)."""
    f = src / "5_box" / "preselezione" / "esclusi.txt"
    return {ln.split()[0] for ln in open(f) if ln.strip()} if f.exists() else set()


def count(src: Path) -> None:
    """Passo 7: istanze, frame, clip, video e sessioni per cella classe x
    fascia sui box di `5_box/manuale/` (da CVAT) per le sequenze riviste, di
    `5_box/proposte/` per le altre, per le sequenze in `6_sessioni/sessioni.csv` ->
    `7_conteggio/conteggio.csv`, stesso formato di Svanstrom (per tools/buchi.py)."""
    from collections import defaultdict

    from tools.svanstrom import NAMES

    ses = {r["clip"]: r["sessione"] for r in csv.DictReader(open(src / "6_sessioni" / "sessioni.csv"))}
    out_ = excluded(src)
    inst, sets = Counter(), defaultdict(lambda: (set(), set(), set(), set()))
    for d in sorted((src / "5_box" / "proposte").iterdir()):
        clip = f"{d.name}/s0"
        if clip not in ses:
            continue
        man = src / "5_box" / "manuale" / d.name
        for t in sorted(d.glob("*.txt")):
            if t.stem in out_:
                continue
            t = boxes(src) / d.name / t.name
            # sequenza rivista in CVAT: solo le sue label (frame senza txt = nessun box)
            if man.exists():
                t = man / t.name
                if not t.exists():
                    continue
            for ln in t.read_text().splitlines():
                c, _, _, w, h = ln.split()
                cell = (NAMES[int(c)], BANDS[np.digitize(np.sqrt(float(w) * 640 * float(h) * 512), BAND_EDGES)])
                inst[cell] += 1
                for st, x in zip(sets[cell], (t.stem, clip, d.name, ses[clip])):
                    st.add(x)
    out = src / "7_conteggio"
    out.mkdir(exist_ok=True)
    with open(out / "conteggio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "fascia", "istanze", "frame", "clip", "video", "sessioni"])
        w.writerows([*cell, inst[cell], *map(len, sets[cell])] for cell in
                    ((c, b) for c in NAMES for b in BANDS) if inst[cell])
    for c in sorted({c for c, _ in inst}):
        print(f"{PREFIX}: conteggio {c} " + ", ".join(f"{b} {inst[c, b]}" for b in BANDS))
    print(f"-> {out / 'conteggio.csv'}")


def cvat_export(src: Path, prefix: str | None = None, per_patch: int = 2000, per_task: int = 0) -> None:
    """Passo 5, revisione a mano: un task CVAT `fred_<seq>` per sequenza (un
    task unico da 12.000 file caricati dal disco dava errore 500), con i ritagli
    tenuti dalla deduplica (CVAT monta solo data/svanstrom: si caricano dal
    disco) e i box proposti (`5_box/proposte/`) come rettangoli. Le sequenze
    con il task gia' fatto si saltano. Si corregge in CVAT, si esporta
    "Ultralytics YOLO Detection 1.0" senza immagini in `5_box/cvat_zip/`, poi
    `--step cvat_import`. Sequenze corte (FBD, DVIDS) insieme, in ordine di
    sessione, fino a `per_task` ritagli per task: `<prefix>_<prima>-<ultima>`."""
    from cvat_sdk.api_client import models
    from cvat_sdk.core.proxies.tasks import ResourceType

    from tools.svanstrom import NAMES, _cvat_client

    prefix = prefix or PREFIX.lower()
    ses = {r["clip"]: r["sessione"] for r in csv.DictReader(open(src / "6_sessioni" / "sessioni.csv"))}
    seqs = sorted((d.name for d in (src / "5_box" / "proposte").iterdir() if f"{d.name}/s0" in ses),
                  key=lambda v: (ses[f"{v}/s0"], len(v), v))
    out_ = excluded(src)
    with _cvat_client() as client:
        tasks = client.tasks.list()
        have = {t.name for t in tasks}
        # sequenze gia' in un task: si saltano, cosi' le sequenze nuove non spostano i gruppi fatti
        done = {f.name.rsplit("_f", 1)[0] for t in tasks if t.name.startswith(f"{prefix}_") and t.size for f in t.get_frames_info()}
        seqs = [v for v in seqs if v not in done]
        groups, cur, n = [], [], 0
        for v in seqs:
            k = sum(p.stem not in out_ for p in (src / "4_ritagli" / v / "images").glob("*.jpg"))
            if not k:  # tutta esclusa dalla preselezione
                continue
            if cur and n + k > per_task:
                groups.append(cur)
                cur, n = [], 0
            cur, n = cur + [v], n + k
        groups += [cur] if cur else []
        for g in groups:
            ids = [v.split("_", 1)[1] for v in g]  # AUAV_<data>_<ora>_<cam>_<pezzo>: tutto dopo il prefisso
            name = f"{prefix}_{ids[0]}" if len(g) == 1 else f"{prefix}_{ids[0]}-{ids[-1]}"
            if name in have:
                continue
            imgs = sorted((p for v in g for p in (src / "4_ritagli" / v / "images").glob("*.jpg") if p.stem not in out_),
                          key=lambda p: p.name)  # come CVAT
            v = g[0]
            # un solo zip per sequenza: oltre ~1000 file per richiesta il server da' errore 500
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                zp = Path(tmp) / f"{v}.zip"
                with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
                    for p in imgs:
                        zf.write(p, p.name)
                task = client.tasks.create_from_data(
                    spec={"name": name, "labels": [{"name": n} for n in NAMES]},
                    resource_type=ResourceType.LOCAL, resources=[str(zp)],
                    data_params={"image_quality": 95, "sorting_method": "lexicographical"})
            assert task.size == len(imgs), f"{name}: {task.size} frame invece di {len(imgs)}"
            label = {lb.name: lb.id for lb in task.get_labels()}
            shapes = []
            for frame, p in enumerate(imgs):
                t = boxes(src) / p.stem.rsplit("_f", 1)[0] / f"{p.stem}.txt"
                for ln in (t.read_text().splitlines() if t.exists() else []):
                    c, cx, cy, w, h = ln.split()
                    cx, cy, w, h = float(cx) * 640, float(cy) * 512, float(w) * 640, float(h) * 512
                    shapes.append(models.LabeledShapeRequest(
                        type="rectangle", frame=frame, label_id=label[NAMES[int(c)]],
                        points=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
            for k in range(0, len(shapes), per_patch):  # a pezzi: uno solo va in timeout
                task.update_annotations(models.PatchedLabeledDataRequest(shapes=shapes[k:k + per_patch]))
            print(f"cvat: task {name} ({task.id}), {len(imgs)} frame, {len(shapes)} box proposti, "
                  f"sessioni {' '.join(sorted({ses[f'{x}/s0'] for x in g}))}", flush=True)


def cvat_import(src: Path) -> None:
    """Zip YOLO di CVAT in `5_box/cvat_zip/` -> `5_box/manuale/FRED_<seq>/<stem>.txt`;
    uno zip piu' recente sovrascrive i frame di uno vecchio."""
    box = src / "5_box"
    for z in sorted((box / "cvat_zip").glob("*.zip"), key=lambda p: p.stat().st_mtime):
        n = Counter()
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.startswith("labels/") and name.endswith(".txt"):
                    stem = Path(name).stem
                    d = box / "manuale" / stem.rsplit("_f", 1)[0]
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{stem}.txt").write_bytes(zf.read(name))
                    n[d.name] += 1
        print(f"cvat: {z.name} -> " + ", ".join(f"{k} ({v} frame)" for k, v in n.items()))


def sam_sheets(src: Path, seqs: list[str], n: int = 24, seed: int = 0, fonte: str = "sam") -> None:
    """Fogli di controllo in `5_box/fogli_<fonte>/FRED_<seq>.jpg` (fonte sam o
    yolo): `n` ritagli a caso attorno al box, rosso il box originale, verde
    quello proposto."""
    import json
    import random

    import cv2

    out = src / "5_box" / f"fogli_{fonte}"
    out.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        v = f"{PREFIX}_{s}"
        rec = json.loads((src / "5_box" / fonte / f"{v}.json").read_text())
        idx = random.Random(seed).sample(range(len(rec["frame"])), min(n, len(rec["frame"])))
        tiles = []
        for k in sorted(idx):
            img = cv2.imread(str(src / "4_ritagli" / v / "images" / f"{v}_f{rec['frame'][k]:05d}.jpg"))
            o, b = rec["orig"][k], rec["box"][k]
            cv2.rectangle(img, (int(o[0]), int(o[1])), (int(o[0] + o[2]), int(o[1] + o[3])), (0, 0, 255), 1)
            if b:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), (0, 255, 0), 1)
            side = int(max(o[2], o[3]) * 2 + 20)
            cx, cy = int(o[0] + o[2] / 2), int(o[1] + o[3] / 2)
            x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
            t = cv2.resize(img[y0:y0 + side, x0:x0 + side], (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, str(rec["frame"][k]), (4, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
        cv2.imwrite(str(out / f"{v}.jpg"), np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)]))
    print(f"fred: fogli {fonte} in {out}")


def _embed(raw: Path, seqs: list[str]) -> dict[str, np.ndarray]:
    """Embedding DINOv2 ViT-S/14 (CLS, norma 1) dell'anteprima di ogni
    sequenza, senza il bordo nero, come per le sessioni di Svanstrom."""
    import cv2
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(dev).eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = {}
    with torch.no_grad():
        for s in seqs:
            im = cv2.imread(str(raw / "anteprima" / f"{s}.jpg"))[36:693, 28:1247]
            x = (cv2.resize(im, (448, 252))[..., ::-1] / 255 - mean) / std  # multipli di 14
            x = torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32, device=dev)
            out[s] = torch.nn.functional.normalize(m(x), dim=1).cpu().numpy()
    return out


def session_sheets(src: Path, gap: float = 3600, thr: float = 0.9) -> None:
    """Passo 6, proposta di sessioni senza interne e notte: stesso **giorno**
    (blocchi orari: sequenze in ordine di numero, blocco nuovo quando l'ora
    torna indietro o dopo una pausa di oltre `gap` s) **o** stesso **posto**
    (gruppi per legame medio della somiglianza DINOv2 delle anteprime sopra
    `thr`), uniti per transitivita'; poi le eccezioni di `correzioni.txt`. ->
    `6_sessioni/sessioni.csv` (clip, sessione), `proposta.csv` (sequenza,
    proposta, sessione, blocco, gruppo visivo, ora, drone, scenario), `embedding.npz` e un foglio per
    sessione in `6_sessioni/fogli/SNN.jpg` (numero, ora, blocco = giorno, gruppo
    visivo = posto, drone; in fondo
    in grigio la sequenza piu' simile fuori dalla sessione)."""
    import cv2

    from tools.svanstrom import propose_sessions

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
    # correzioni: "SNN: via a b c-d" -> le sequenze escono da SNN e formano una sessione nuova
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
    # "separa a b c-d": le sequenze elencate formano insieme una sessione nuova (numeri di sequenza: stabili)
    for k2, ln in enumerate(cf.read_text().splitlines() if cf.exists() else []):
        if ln.startswith("separa"):
            for tok in ln.split()[1:]:
                a, b = (tok.split("-") + [tok])[:2]
                for q in range(int(a), int(b) + 1):
                    if str(q) in ses:
                        ses[str(q)] = f"sep{k2}"
    # "unisci a b c": le sessioni delle sequenze a, b, c diventano una (numeri di sequenza: stabili)
    for ln in cf.read_text().splitlines() if cf.exists() else []:
        if ln.startswith("unisci"):
            qs = [q for q in ln.split()[1:] if q in ses]
            old = {ses[q] for q in qs}
            for q in seqs:
                if ses[q] in old:
                    ses[q] = ses[qs[0]]
    # la notte (tolta dall'utente) resta nel raggruppamento, cosi' la proposta e le correzioni non cambiano,
    # e si toglie qui
    for q in [q for q in seqs if sc[q] == "notte"]:
        del ses[q]
    allseq, seqs = seqs, [q for q in seqs if q in ses]  # allseq: indici di `sim`
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
    p.add_argument("--src", type=Path, default=ROOT / "data" / "fred", help="una cartella per passo")
    p.add_argument("--step", choices=["label", "anteprima", "frame", "annotati", "dedup", "ritaglio", "sam", "yolo", "movimento", "fogli_sam", "fogli_yolo", "fogli_movimento", "sessioni", "proposte", "conta", "cvat", "cvat_import"], required=True)
    p.add_argument("--seq", nargs="+", default=[], help="sequenze (numeri) per frame, dedup, ritaglio")
    a = p.parse_args(argv)
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
        sam_sheets(a.src, a.seq, fonte=a.step.split("_")[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
