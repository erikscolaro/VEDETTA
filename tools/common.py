"""Costanti e funzioni comuni a tutte le fonti: classi, fasce di grandezza, deduplica, sessioni, CVAT."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "input"  # fonti, una cartella per passo
INTERMEDIATE = DATA / "intermediate"  # passi su tutte le fonti
OUTPUT = DATA / "output"  # dataset finale, congelato
# classi del detector delle proposte (5 classi, due di droni) -> le nostre (0 airplane, 1 bird, 2 drone, 3 helicopter)
DETECTOR_CLASSES = {0: 2, 1: 2, 2: 0, 3: 3, 4: 1}


def detector():
    """YOLO delle proposte e del filtro del background: pesi privati, fuori dalla repo, dal percorso in
    `$DETECTOR_PESI`. Le classi sono mappate con `DETECTOR_CLASSI`."""
    from ultralytics import YOLO

    pesi = os.environ.get("DETECTOR_PESI")
    if not pesi:
        raise SystemExit("serve DETECTOR_PESI=<percorso dei pesi del detector delle proposte>")
    return YOLO(pesi)


NAMES = ["airplane", "bird", "drone", "helicopter"]  # stesso ordine di AOD4


BANDS = ["<8", "8-16", "16-32", "32-64", "64-96", ">96"]  # sqrt(w*h) in px, README.md (Regole)


BAND_EDGES = [8, 16, 32, 64, 96]


def dhash(gray: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).ravel()


def signature(gray: np.ndarray, boxes: np.ndarray):
    """Hash del ritaglio attorno all'unione dei box + geometria normalizzata."""
    h, w = gray.shape
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
    pad = max(x1 - x0, y1 - y0, 8)  # contesto: lo sfondo vicino conta
    crop = gray[int(max(0, y0 - pad)):int(min(h, y1 + pad)),
                int(max(0, x0 - pad)):int(min(w, x1 + pad))]
    geo = np.array([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                    np.log(max(x1 - x0, 1)), np.log(max(y1 - y0, 1))])
    return dhash(crop), geo, len(boxes)


def is_new(sig, kept, hash_thr: int, geo_thr: float) -> bool:
    # ponytail: confronto con tutti i frame tenuti, O(n*k) per video; basta
    # finche' i video restano da ~300 frame
    hsh, geo, k = sig
    for h2, g2, k2 in kept:
        if (k == k2 and np.count_nonzero(hsh != h2) <= hash_thr
                and (geo is None or (np.hypot(*(geo[:2] - g2[:2])) <= geo_thr
                                     and np.abs(geo[2:] - g2[2:]).max() <= 0.3))):
            return False
    return True


def dedup_clip(args) -> tuple[str, list[int], int]:
    """Frame tenuti di una clip, in ordine: un frame resta se nessun frame gia'
    tenuto della clip ha lo stesso numero di box, dHash del ritaglio sui box
    entro `hash_thr` bit e posizione/grandezza vicine (`is_new`). Frame senza
    box: dHash del frame intero."""
    import cv2

    frames, man, clip, idx, hash_thr, geo_thr = args
    v = clip.split("/")[0]
    kept, keep = [], []
    for i in idx:
        gray = cv2.imread(str(frames / v / "images" / f"{v}_f{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        t = man / f"{v}_f{i:05d}.txt"
        rows = np.array([ln.split()[1:] for ln in t.read_text().splitlines() if ln.strip()] if t.exists() else [],
                        float).reshape(-1, 4) * ([gray.shape[1], gray.shape[0]] * 2)
        boxes = np.c_[rows[:, :2] - rows[:, 2:] / 2, rows[:, 2:]]  # (x, y, w, h) in pixel
        sig = signature(gray, boxes) if len(boxes) else (dhash(gray), None, 0)
        if is_new(sig, kept, hash_thr, geo_thr):
            kept.append(sig)
            keep.append(i)
    return clip, keep, len(idx)


def mask_box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return None if not len(xs) else (xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())


def propose_sessions(emb: dict[str, np.ndarray], thr: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Somiglianza fra clip = massimo fra le coppie dei loro frame; gruppi per
    legame medio sopra `thr`. Il legame singolo
    incatena tutto: su cielo uniforme quasi ogni coppia supera 0,9."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    vids = sorted(emb)
    sim = np.eye(len(vids))
    for a in range(len(vids)):
        for b in range(a + 1, len(vids)):
            sim[a, b] = sim[b, a] = (emb[vids[a]] @ emb[vids[b]].T).max()
    groups = fcluster(linkage(squareform(1 - sim, checks=False), "average"), 1 - thr, "distance")
    return vids, sim, groups


def cvat_client():
    """Client CVAT su localhost:8080, credenziali in ~/.config/cvat.env
    (CVAT_USER, CVAT_PASS)."""
    from cvat_sdk import make_client

    env = dict(ln.split("=", 1) for ln in (Path.home() / ".config" / "cvat.env").read_text().split())
    return make_client("http://localhost:8080", credentials=(env["CVAT_USER"], env["CVAT_PASS"]))
