"""Scenario of each clip (sky, horizon, ground, wall) for the domain of the turret: ground camera looking at the sky.

    python tools/scenario.py --step fogli     # contact sheets, 4 clips x 5 frames -> intermediate/12_scenario/fogli/
    python tools/scenario.py --step metrica   # background texture around each box -> intermediate/12_scenario/metrica.csv

The proposals are written by agents (as a human operator would) in `12_scenario/proposte/*.csv`
(id, scenario, ala_fissa, nota) and read by `split.py`; the user reviews them (TODO.md)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from tools.common import DATA, INTERMEDIATE

OUT = INTERMEDIATE / "12_scenario"
NAMES = ["airplane", "bird", "drone", "helicopter"]  # order of the intermediate labels
COLORS = [(0, 140, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # BGR: orange, red, cyan, magenta


def _clips() -> dict[tuple[str, str], list[dict]]:
    by = defaultdict(list)
    for r in csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")):
        by[r["fonte"], r["clip"]].append(r)
    return dict(sorted(by.items()))


def _boxes(r: dict) -> list[tuple[int, float, float, float, float]]:
    return [(int(c), float(x), float(y), float(w), float(h)) for c, x, y, w, h in
            (ln.split() for ln in (DATA / r["label"]).read_text().splitlines() if ln.strip())]


def sheets(per_sheet: int = 4, frames: int = 5, tw: int = 300, th: int = 240) -> None:
    """One row per clip: `frames` frames spread over the clip, boxes with a thick ring (small targets stay visible),
    clip id on the left. Writes `fogli/F<nnn>.jpg` and `indice.csv` (id, foglio, fonte, clip, sessione, frame)."""
    import cv2

    (OUT / "fogli").mkdir(parents=True, exist_ok=True)
    clips = list(_clips().items())
    idx = []
    for s in range(0, len(clips), per_sheet):
        rows_img = []
        for k, ((src, clip), rows) in enumerate(clips[s:s + per_sheet], start=s):
            cid = f"C{k:04d}"
            idx.append((cid, f"F{s // per_sheet:03d}", src, clip, rows[0]["sessione"], len(rows)))
            tiles = []
            for r in [rows[i] for i in np.linspace(0, len(rows) - 1, min(frames, len(rows))).astype(int)]:
                im = cv2.imread(str(DATA / r["immagine"]))
                H, W = im.shape[:2]
                for c, x, y, w, h in _boxes(r):
                    x0, y0, x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)
                    m = max(6, int(0.15 * max(x1 - x0, y1 - y0)))  # margin: the ring must not hide the background
                    cv2.rectangle(im, (x0 - m, y0 - m), (x1 + m, y1 + m), COLORS[c], 3)
                    cv2.putText(im, NAMES[c], (x0 - m, max(14, y0 - m - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[c], 2)
                tiles.append(cv2.resize(im, (tw, th)))
            tiles += [np.zeros((th, tw, 3), np.uint8)] * (frames - len(tiles))
            lab = np.zeros((th, 110, 3), np.uint8)
            cv2.putText(lab, cid, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(lab, src, (6, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            rows_img.append(np.vstack([np.hstack([lab] + tiles), np.full((6, 110 + frames * tw, 3), 255, np.uint8)]))
        cv2.imwrite(str(OUT / "fogli" / f"F{s // per_sheet:03d}.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(OUT / "indice.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "foglio", "fonte", "clip", "sessione", "frame"])
        w.writerows(idx)
    print(f"scenario: {len(idx)} clip in {len(range(0, len(clips), per_sheet))} fogli -> {OUT / 'fogli'}")


def metric() -> None:
    """Background around each box: mean |Laplacian| of the gray image in a ring as wide as the box (at least 24 px),
    the box excluded. Low = smooth (sky), high = textured (vegetation, walls). Writes `metrica.csv`."""
    import cv2

    with open(OUT / "metrica.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["immagine", "k", "classe", "lato", "tessitura"])
        for (src, clip), rows in _clips().items():
            for r in rows:
                g = cv2.imread(str(DATA / r["immagine"]), cv2.IMREAD_GRAYSCALE)
                H, W = g.shape
                lap = np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F))
                for k, (c, x, y, bw, bh) in enumerate(_boxes(r)):
                    b = [(x - bw / 2) * W, (y - bh / 2) * H, (x + bw / 2) * W, (y + bh / 2) * H]
                    m = max(b[2] - b[0], b[3] - b[1], 24)
                    x0, y0, x1, y1 = (int(v) for v in (max(0, b[0] - m), max(0, b[1] - m), min(W, b[2] + m), min(H, b[3] + m)))
                    mask = np.ones((y1 - y0, x1 - x0), bool)
                    mask[max(0, int(b[1]) - y0):int(b[3]) - y0, max(0, int(b[0]) - x0):int(b[2]) - x0] = False
                    w.writerow([r["immagine"], k, NAMES[c], round(float(np.sqrt(bw * W * bh * H)), 1),
                                round(float(lap[y0:y1, x0:x1][mask].mean()), 2) if mask.any() else ""])
    print(f"scenario: metrica -> {OUT / 'metrica.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["fogli", "metrica"], required=True)
    a = p.parse_args()
    {"fogli": sheets, "metrica": metric}[a.step]()
