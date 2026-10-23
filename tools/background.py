#!/usr/bin/env python3
"""Step 13, background from independent sources (Clouds-1500, Skyfinder; licenses in README, Sources and licenses).
One folder per step in `data/input/<source>/`, then the choice per split in `data/intermediate/14_background/`.

    python tools/background.py --fonte clouds1500 --step ritagli   # 1_ritagli: 640x512 on the sky-horizon band
    python tools/background.py --fonte clouds1500 --step detector  # 2_detector: drop the crops with an object
    python tools/background.py --fonte clouds1500 --step dedup     # 3_dedup: drop the near-duplicates per camera
    python tools/background.py --fonte clouds1500 --step fogli     # anteprima/: sample of kept and excluded
    python tools/background.py --step split                        # intermediate/14_background: ~10% per split
    python tools/background.py --step raccolta                     # 0_raw: the choices as hardlinks
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import DATA, INPUT, INTERMEDIATE  # noqa: E402
W, H = 640, 512
DARK = 35  # mean luminance below which the crop is night (README, Rules, as Anti-UAV and DVIDS)


def sources(source: str) -> list[tuple[Path, str]]:
    """(image, camera) of the source. Clouds-1500: camera = folder (two cameras in a single place, folders per
    period). Skyfinder: camera = AMOS folder, without `GSV` (Street View) and without the cameras under 640x512."""
    import cv2

    raw = INPUT / source / "0_raw" / "estratto"
    if source == "clouds1500":
        return [(p, p.parent.parent.name) for p in sorted(raw.glob("*/img/*")) if p.suffix.lower() in (".jpg", ".png")]
    out = []
    for d in sorted((raw / "OriginalImages").iterdir()):
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        h, w = cv2.imread(str(fs[0])).shape[:2]
        if d.name != "GSV" and w >= W and h >= H:
            out += [(p, d.name) for p in fs]
    return out


def _crop(args) -> list[tuple]:
    """640x512 crops without resizing, top-left corner random in the upper half (sky and horizon, away from the
    captions at the bottom); dark crops are dropped."""
    import cv2

    p, cam, k, seed, dst = args
    im = cv2.imread(str(p))
    if im is None:
        return []
    h, w = im.shape[:2]
    rng = np.random.default_rng(seed)
    out = []
    for j in range(k):
        x0 = int(rng.integers(0, w - W + 1))
        y0 = int(rng.integers(0, max(0, min(h - H, h // 2 - H // 2)) + 1))
        c = im[y0:y0 + H, x0:x0 + W]
        if cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean() < DARK:
            continue
        name = f"{cam}_{p.stem}_{j}".replace(" ", "")
        cv2.imwrite(str(dst / f"{name}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append((name, str(p.relative_to(DATA)), cam, x0, y0))
    return out


def crops(source: str, seed: int = 0) -> None:
    """1_ritagli: `ritagli.csv` (ritaglio, sorgente, camera, x0, y0) and `images/`. Clouds-1500 two crops per
    image (1,500 images), Skyfinder one."""
    from concurrent.futures import ProcessPoolExecutor

    dst = INPUT / source / "1_ritagli" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.iterdir():
        f.unlink()
    src = sources(source)
    k = 2 if source == "clouds1500" else 1
    with ProcessPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(_crop, [(p, cam, k, seed + i, dst) for i, (p, cam) in enumerate(src)], chunksize=32)
                for r in rs]
    with open(dst.parent / "ritagli.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ritaglio", "sorgente", "camera", "x0", "y0"])
        w.writerows(rows)
    print(f"{source} ritagli: {len(src)} immagini, {len(rows)} ritagli ({len(src) * k - len(rows)} scuri via), "
          f"{len({r[2] for r in rows})} camere -> {dst.parent}")


def filter_detector(source: str, conf: float = 0.1) -> None:
    """2_detector: proposals detector (`common.detector`) on every crop, low confidence:
    a crop with at least one detection is dropped (`esclusi.csv`: ritaglio, classe, conf, box), the others go in
    `tenuti.csv`. Better to throw away a good sky than to keep a bird without a box."""
    from tools.common import detector

    model = detector()
    src = INPUT / source / "1_ritagli"
    rows = list(csv.DictReader(open(src / "ritagli.csv")))
    out = INPUT / source / "2_detector"
    out.mkdir(exist_ok=True)
    excl, kept = [], []
    paths = [str(src / "images" / f"{r['ritaglio']}.jpg") for r in rows]
    for k in range(0, len(paths), 64):
        for r, res in zip(rows[k:k + 64], model(paths[k:k + 64], conf=conf, verbose=False)):
            if len(res.boxes):
                b = max(res.boxes, key=lambda x: float(x.conf))
                excl.append((r["ritaglio"], model.names[int(b.cls)], round(float(b.conf), 3),
                            " ".join(f"{v:.0f}" for v in b.xyxy[0].tolist())))
            else:
                kept.append(r)
    with open(out / "esclusi.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ritaglio", "classe", "conf", "box"])
        w.writerows(excl)
    with open(out / "tenuti.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(kept)
    print(f"{source} detector (conf {conf}): {len(excl)} via ({', '.join(f'{c} {n}' for c, n in Counter(e[1] for e in excl).most_common())}), "
          f"{len(kept)} tenuti -> {out}")


def dedup(source: str, thr: float = 0.95) -> None:
    """3_dedup: DINOv2 ViT-S embedding (`split.embed`) of the kept crops; per camera, in source order, a
    crop stays if its similarity with all those already kept is below `thr` (0.95, the duplicates threshold) ->
    `tenuti.csv`, `embedding.npy` (of the kept, same order)."""
    from tools.split import embed

    rows = list(csv.DictReader(open(INPUT / source / "2_detector" / "tenuti.csv")))
    out = INPUT / source / "3_dedup"
    out.mkdir(exist_ok=True)
    e = embed([str(INPUT / source / "1_ritagli" / "images" / f"{r['ritaglio']}.jpg") for r in rows]).astype(np.float32)
    keep = []
    by_cam = defaultdict(list)
    for i, r in enumerate(rows):
        by_cam[r["camera"]].append(i)
    for cam, idx in by_cam.items():
        kept = []
        for i in idx:
            if not kept or (e[kept] @ e[i]).max() < thr:
                kept.append(i)
        keep += kept
    keep.sort()
    with open(out / "tenuti.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows[i] for i in keep)
    np.save(out / "embedding.npy", e[keep].astype(np.float16))
    n = Counter(rows[i]["camera"] for i in keep)
    print(f"{source} dedup (< {thr}): {len(keep)} su {len(rows)}; per camera min {min(n.values())}, max {max(n.values())} -> {out}")


def sheets(source: str, n: int = 48, seed: int = 0) -> None:
    """anteprima/: `tenuti.jpg` (n random kept crops) and `esclusi.jpg` (n excluded by the detector, with box and
    class), for visual checking."""
    import cv2

    rng = np.random.default_rng(seed)
    dst = INPUT / source / "anteprima"
    dst.mkdir(exist_ok=True)
    img = INPUT / source / "1_ritagli" / "images"
    tw, th = 256, 205

    def sheet(items, name):
        tiles = []
        for stem, text, box in items:
            im = cv2.imread(str(img / f"{stem}.jpg"))
            if box:
                x0, y0, x1, y1 = map(int, map(float, box.split()))
                cv2.rectangle(im, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 255), 3)
            im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.rectangle(im, (0, 0), (tw, 14), (255, 255, 255), -1)
            cv2.putText(im, text[:40], (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.pad(im, ((0, 4), (0, 4), (0, 0)), constant_values=255))
        tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 8)
        cv2.imwrite(str(dst / name), np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    kept = list(csv.DictReader(open(INPUT / source / "3_dedup" / "tenuti.csv")))
    sheet([(r["ritaglio"], r["ritaglio"], "") for r in rng.choice(kept, min(n, len(kept)), replace=False)], "tenuti.jpg")
    excl = list(csv.DictReader(open(INPUT / source / "2_detector" / "esclusi.csv")))
    if excl:
        sheet([(r["ritaglio"], f"{r['classe']} {r['conf']}", r["box"]) for r in rng.choice(excl, min(n, len(excl)), replace=False)],
              "esclusi.jpg")
    print(f"{source} fogli -> {dst}")


def split(srcs=("clouds1500", "skyfinder"), share: float = 0.1, seed: int = 0) -> None:
    """intermediate/14_background: ~10% of background in every split (bg = images / 9, so they are 10% of the total).
    Whole cameras in one split (the same camera never in two splits): in turn, each camera goes to the split with the
    greatest need relative to the crops already assigned; then within each split pick in turn among the cameras. ->
    `background.csv` (immagine, fonte, camera, split)."""
    agg = INTERMEDIATE
    need = Counter(r["split"] for r in csv.DictReader(open(agg / "12_split" / "frame_split.csv")))
    need["train"] = sum(1 for _ in csv.DictReader(open(agg / "13_train_compensation" / "train.csv")))
    need = {s: round(need[s] * share / (1 - share)) for s in ("train", "val", "test")}
    cams = defaultdict(list)
    for f in srcs:
        for r in csv.DictReader(open(INPUT / f / "3_dedup" / "tenuti.csv")):
            cams[f, r["camera"]].append(f"{f}/1_ritagli/images/{r['ritaglio']}.jpg")
    rng = np.random.default_rng(seed)
    got, lab = Counter(), {}
    for c in sorted(cams, key=lambda c: -len(cams[c])):  # big cameras first, then the small ones fill in
        s = min(need, key=lambda s: got[s] / need[s])
        lab[c] = s
        got[s] += len(cams[c])
    out = agg / "14_background"
    out.mkdir(exist_ok=True)
    rows = []
    for s in need:
        q = [list(rng.permutation(cams[c])) for c in cams if lab[c] == s]
        pick = []
        while len(pick) < need[s] and any(q):
            for v in q:
                if v and len(pick) < need[s]:
                    pick.append(v.pop())
        rows += [(p, p.split("/")[0], next(c[1] for c in cams if c[0] == p.split("/")[0] and p in cams[c]), s)
                 for p in pick]
    with open(out / "background.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "fonte", "camera", "split"])
        w.writerows(rows)
    for s in need:
        n = Counter(r[1] for r in rows if r[3] == s)
        print(f"  {s}: {sum(n.values())} su {need[s]} ({', '.join(f'{f} {k}' for f, k in n.items())}), "
              f"{sum(1 for c in lab if lab[c] == s)} camere")
    print(f"-> {out / 'background.csv'}")


def collect() -> None:
    """data/input/background/0_raw: the images chosen in `intermediate/14_background/background.csv` as hardlinks (same
    name as the crop), plus `background.csv` with the new path. A source of its own, for the later steps (review).
    Skips the names in `rimossi.txt` (operator choices: backgrounds removed by hand because they had objects)."""
    import os

    dst = INPUT / "background" / "0_raw"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    for f in (dst / "images").iterdir():
        f.unlink()
    f = dst / "rimossi.txt"
    drop = {ln.strip() for ln in open(f) if ln.strip() and not ln.startswith("#")} if f.exists() else set()
    rows = [r for r in csv.DictReader(open(INTERMEDIATE / "14_background" / "background.csv"))
            if Path(r["immagine"]).name not in drop]
    names = [Path(r["immagine"]).name for r in rows]
    assert len(set(names)) == len(names), "nomi doppi"
    for r, n in zip(rows, names):
        os.link(DATA / r["immagine"], dst / "images" / n)
    with open(dst / "background.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "sorgente", "fonte", "camera", "split"])
        w.writerows((f"background/0_raw/images/{n}", r["immagine"], r["fonte"], r["camera"], r["split"])
                    for r, n in zip(rows, names))
    print(f"background: {len(rows)} immagini in hardlink ({', '.join(f'{s} {k}' for s, k in Counter(r['split'] for r in rows).items())}) -> {dst}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", choices=["clouds1500", "skyfinder"])
    p.add_argument("--step", choices=["ritagli", "detector", "dedup", "fogli", "split", "raccolta"], required=True)
    a = p.parse_args(argv)
    if a.step == "split":
        split()
    elif a.step == "raccolta":
        collect()
    else:
        {"ritagli": crops, "detector": filter_detector, "dedup": dedup, "fogli": sheets}[a.step](a.fonte)
    return 0


if __name__ == "__main__":
    sys.exit(main())
