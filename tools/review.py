"""Label review with two independent detectors (user, 25/10/2026: TODO.md, night plan, point 5).

    python tools/review.py --step predizioni   # both models on every annotated frame -> 13_revisione/pred.pkl
    python tools/review.py --step candidati    # candidates -> 13_revisione/candidati.csv
    python tools/review.py --step fogli        # sheets of 24 candidates -> 13_revisione/fogli/R<nnn>.jpg
    python tools/review.py --step applica      # decisions (decisioni/*.csv) -> 13_revisione/label/ (copy of the labels)

Models: `DETECTOR_PESI` (proposal detector, 5 classes) and `REVIEW_PESI` (yolo26p trained on the frozen dataset).
Candidate kinds:
- `senza_box`: a confident prediction (>= 0.5) that overlaps no label (IoU < 0.1): object without a box?
- `vuota`: a label that no prediction touches (any class, conf >= 0.05, IoU >= 0.1) on a flat patch: empty box?
- `classe`: both models predict, confidently and on the box (IoU >= 0.5), the same class, different from the label.
- `bird_dvids`: every bird of DVIDS (jets labelled bird were seen in val).
Agents decide on the sheets (as a human operator would); the originals in `input/` are never touched."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from tools.common import DATA, DETECTOR_CLASSES, INTERMEDIATE, ROOT, detector

OUT = INTERMEDIATE / "13_revisione"
NAMES = ["airplane", "bird", "drone", "helicopter"]  # order of the intermediate labels
PICO = {0: 2, 1: 0, 2: 3, 3: 1}  # yolo26p of the frozen dataset (drone, airplane, helicopter, bird) -> ours


def _frames() -> list[dict]:
    return list(csv.DictReader(open(INTERMEDIATE / "9_split" / "frame.csv")))


def _labels(r: dict, W: int = 640, H: int = 512) -> tuple[np.ndarray, np.ndarray]:
    b = [ln.split() for ln in (DATA / r["label"]).read_text().splitlines() if ln.strip()]
    c = np.array([int(x[0]) for x in b], int)
    xywh = np.array([[float(v) for v in x[1:]] for x in b]).reshape(-1, 4)
    return c, np.c_[(xywh[:, 0] - xywh[:, 2] / 2) * W, (xywh[:, 1] - xywh[:, 3] / 2) * H,
                    (xywh[:, 0] + xywh[:, 2] / 2) * W, (xywh[:, 1] + xywh[:, 3] / 2) * H]


def _side(b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None))


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (n, 4) and b (m, 4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    i = np.prod(np.clip(rb - lt, 0, None), 2)
    return i / (_side(a)[:, None] ** 2 + _side(b)[None] ** 2 - i + 1e-9)


def predictions(batch: int = 64) -> None:
    """Both models, conf >= 0.05, one-to-one head (nms=False): {image: [(model, boxes, classes, confs)]}."""
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    rows = _frames()
    models = {"detector": (detector(), DETECTOR_CLASSES), "pico": (YOLO(str(ROOT / os.environ["REVIEW_PESI"])), PICO)}
    pred = defaultdict(dict)
    for name, (m, cmap) in models.items():
        for k in range(0, len(rows), batch):
            paths = [str(DATA / r["immagine"]) for r in rows[k:k + batch]]
            for r, res in zip(rows[k:k + batch], m.predict(paths, conf=0.05, nms=False, verbose=False, batch=batch)):
                bx = res.boxes
                raw = bx.cls.cpu().numpy().astype(int)  # kept: the detector tells rotary (0) from fixed wing (1)
                pred[r["immagine"]][name] = (bx.xyxy.cpu().numpy(), np.array([cmap[c] for c in raw], int), bx.conf.cpu().numpy(), raw)
            print(f"\r{name}: {min(k + batch, len(rows))}/{len(rows)}", end="", flush=True)
        print()
    pickle.dump(dict(pred), open(OUT / "pred.pkl", "wb"))


def candidates(conf_new: float = 0.5, flat: float = 1.5) -> None:
    """Candidates from `pred.pkl` -> `candidati.csv` (id, tipo, fonte, clip, immagine, label, k, classe, proposta,
    x1, y1, x2, y2, conf). `flat`: maximum contrast (|mean inside - mean ring| / std ring) of an empty box."""
    import cv2

    pred = pickle.load(open(OUT / "pred.pkl", "rb"))
    out, n = [], Counter()
    for r in _frames():
        gc, gb = _labels(r)
        p = pred[r["immagine"]]
        base = dict(fonte=r["fonte"], clip=r["clip"], immagine=r["immagine"], label=r["label"])
        # boxes without a label: confident predictions of either model, merged across models
        new = []
        for name, (pb, pc, ps, _) in p.items():
            for b, c, s in zip(pb, pc, ps):
                if s >= conf_new and _side(b) >= 8 and (not len(gb) or _iou(b[None], gb).max() < 0.1):
                    new.append((b, c, s, name))
        kept = []
        for b, c, s, name in sorted(new, key=lambda x: -x[2]):
            if all(_iou(b[None], k_[0][None])[0, 0] < 0.3 for k_ in kept):
                kept.append((b, c, s, name))
        for b, c, s, name in kept:
            out.append(base | dict(tipo="senza_box", k=-1, classe="", proposta=NAMES[c], box=b, conf=s))
        # labels nobody sees, on a flat patch; labels both models call another class; DVIDS birds
        g = None
        for k, (c, b) in enumerate(zip(gc, gb)):
            touched = any(len(pb) and _iou(b[None], pb).max() >= 0.1 for pb, *_ in p.values())
            if not touched:
                if g is None:
                    g = cv2.imread(str(DATA / r["immagine"]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                x0, y0, x1, y1 = (int(v) for v in b)
                m = max(x1 - x0, y1 - y0, 12)
                inner = g[max(0, y0):y1, max(0, x0):x1]
                ring = g[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
                if inner.size and ring.size and abs(inner.mean() - ring.mean()) / (ring.std() + 1) < flat and inner.std() < ring.std() * 1.2:
                    out.append(base | dict(tipo="vuota", k=k, classe=NAMES[c], proposta="togli", box=b, conf=0.0))
            votes = []
            for pb, pc, ps, _ in p.values():
                if len(pb):
                    io = _iou(b[None], pb)[0]
                    j = int(io.argmax())
                    votes.append(pc[j] if io[j] >= 0.5 and ps[j] >= 0.5 else -1)
            if len(votes) == 2 and votes[0] == votes[1] >= 0 and votes[0] != c:
                out.append(base | dict(tipo="classe", k=k, classe=NAMES[c], proposta=NAMES[votes[0]], box=b, conf=0.0))
            elif r["fonte"] == "dvids" and NAMES[c] == "bird":
                out.append(base | dict(tipo="bird_dvids", k=k, classe="bird", proposta="", box=b, conf=0.0))
    with open(OUT / "candidati.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "tipo", "fonte", "clip", "immagine", "label", "k", "classe", "proposta", "x1", "y1", "x2", "y2", "conf"])
        for i, o in enumerate(out):
            n[o["tipo"], o["fonte"]] += 1
            w.writerow([f"K{i:05d}", o["tipo"], o["fonte"], o["clip"], o["immagine"], o["label"], o["k"], o["classe"],
                        o["proposta"], *(round(float(v), 1) for v in o["box"]), round(float(o["conf"]), 3)])
    print(f"revisione: {len(out)} candidati -> {OUT / 'candidati.csv'}")
    for (t, f), v in sorted(n.items()):
        print(f"  {t:<11}{f:<11}{v}")


def sheets(per: int = 24, cols: int = 6, size: int = 260) -> None:
    """Sheets of `per` candidates: zoomed crop (context 5x the box, at least 120 px), the candidate box in yellow,
    the other labels of the image in green, id and kind on top."""
    import cv2

    (OUT / "fogli").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(OUT / "candidati.csv")))
    labels = {r["immagine"]: r for r in _frames()}
    for s in range(0, len(rows), per):
        tiles = []
        for c in rows[s:s + per]:
            im = cv2.imread(str(DATA / c["immagine"]))
            H, W = im.shape[:2]
            b = np.array([float(c[k]) for k in ("x1", "y1", "x2", "y2")])
            gc, gb = _labels(labels[c["immagine"]], W, H)
            for k, (cl, q) in enumerate(zip(gc, gb)):
                if str(k) != c["k"]:
                    cv2.rectangle(im, (int(q[0]) - 2, int(q[1]) - 2), (int(q[2]) + 2, int(q[3]) + 2), (0, 200, 0), 1)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            sz = max(5 * max(b[2] - b[0], b[3] - b[1]), 120)
            x0, y0 = int(np.clip(cx - sz / 2, 0, max(0, W - sz))), int(np.clip(cy - sz / 2, 0, max(0, H - sz)))
            x1, y1 = int(min(W, x0 + sz)), int(min(H, y0 + sz))
            z = size / max(x1 - x0, y1 - y0)
            t = cv2.resize(im[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST if z > 2 else cv2.INTER_AREA)
            p0 = (int((b[0] - x0) * z) - 4, int((b[1] - y0) * z) - 4)
            p1 = (int((b[2] - x0) * z) + 4, int((b[3] - y0) * z) + 4)
            cv2.rectangle(t, p0, p1, (0, 255, 255), 2)
            t = cv2.copyMakeBorder(t, 24, size - t.shape[0], 0, size - t.shape[1], cv2.BORDER_CONSTANT, value=0)
            what = f"{c['classe']}?" if c["tipo"] != "senza_box" else f"+{c['proposta']} {float(c['conf']):.2f}"
            if c["tipo"] == "classe":
                what = f"{c['classe']}->{c['proposta']}?"
            cv2.putText(t, f"{c['id']} {c['tipo']} {what}", (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        cv2.imwrite(str(OUT / "fogli" / f"R{s // per:03d}.jpg"),
                    np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"revisione: {len(rows)} candidati in {-(-len(rows) // per)} fogli -> {OUT / 'fogli'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["predizioni", "candidati", "fogli"], required=True)
    a = p.parse_args()
    {"predizioni": predictions, "candidati": candidates, "fogli": sheets}[a.step]()
