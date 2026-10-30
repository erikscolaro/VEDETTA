"""Constants and functions shared by all sources: classes, size bands, dedup, sessions, CVAT."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "input"  # sources, one folder per step
INTERMEDIATE = DATA / "intermediate"  # steps over all sources
OUTPUT = DATA / "output"  # final dataset, frozen
# proposal detector classes (5 classes, two of drones) -> ours (0 airplane, 1 bird, 2 drone, 3 helicopter)
DETECTOR_CLASSES = {0: 2, 1: 2, 2: 0, 3: 3, 4: 1}


def _load_env(f: Path = ROOT / ".env") -> None:
    """Local settings from `.env` (template in `.env.example`): `KEY=value` lines, `#` for comments. Variables
    already in the environment win."""
    if f.exists():
        for ln in f.read_text().splitlines():
            k, sep, v = ln.split("#", 1)[0].partition("=")
            if sep and k.strip() and v.strip():  # empty = not set
                os.environ.setdefault(k.strip(), v.strip())


_load_env()


def _weights(var: str) -> str:
    """Path of the weights from `$<var>`, relative to the repo root if not absolute."""
    p = os.environ.get(var)
    if not p:
        raise SystemExit(f"serve {var} in .env (vedi .env.example)")
    return str(ROOT / p)  # ROOT / absolute = absolute


def detector():
    """YOLO for the proposals and the background filter: private weights, outside the repo, from `DETECTOR_PESI`.
    Classes are mapped with `DETECTOR_CLASSES`."""
    from ultralytics import YOLO

    return YOLO(_weights("DETECTOR_PESI"))


def sam():
    """SAM 2 for the boxes (Svanström, FRED), weights from `SAM_PESI`; Ultralytics downloads them if missing."""
    from ultralytics import SAM

    return SAM(_weights("SAM_PESI"))


NAMES = ["airplane", "bird", "drone", "helicopter"]  # same order as AOD4


BANDS = ["<8", "8-16", "16-32", "32-64", "64-128", "128-256", ">256"]  # sqrt(w*h) in px; powers of 2 (user, 30/10/2026: was 64-96, >96)


BAND_EDGES = [8, 16, 32, 64, 128, 256]


def dhash(gray: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).ravel()


def signature(gray: np.ndarray, boxes: np.ndarray):
    """Hash of the crop around the union of the boxes + normalized geometry."""
    h, w = gray.shape
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
    pad = max(x1 - x0, y1 - y0, 8)  # context: the nearby background matters
    crop = gray[int(max(0, y0 - pad)):int(min(h, y1 + pad)),
                int(max(0, x0 - pad)):int(min(w, x1 + pad))]
    geo = np.array([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                    np.log(max(x1 - x0, 1)), np.log(max(y1 - y0, 1))])
    return dhash(crop), geo, len(boxes)


def is_new(sig, kept, hash_thr: int, geo_thr: float) -> bool:
    # ponytail: compared against all kept frames, O(n*k) per video; enough
    # as long as videos stay around ~300 frames
    hsh, geo, k = sig
    for h2, g2, k2 in kept:
        if (k == k2 and np.count_nonzero(hsh != h2) <= hash_thr
                and (geo is None or (np.hypot(*(geo[:2] - g2[:2])) <= geo_thr
                                     and np.abs(geo[2:] - g2[2:]).max() <= 0.3))):
            return False
    return True


def dedup_clip(args) -> tuple[str, list[int], int]:
    """Kept frames of a clip, in order: a frame stays if no frame already kept
    from the clip has the same number of boxes, a crop dHash on the boxes
    within `hash_thr` bits and a close position/size (`is_new`). Frames
    without boxes: dHash of the whole frame."""
    import cv2

    frames, man, clip, idx, hash_thr, geo_thr = args
    v = clip.split("/")[0]
    kept, keep = [], []
    for i in idx:
        gray = cv2.imread(str(frames / v / "images" / f"{v}_f{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        t = man / f"{v}_f{i:05d}.txt"
        rows = np.array([ln.split()[1:] for ln in t.read_text().splitlines() if ln.strip()] if t.exists() else [],
                        float).reshape(-1, 4) * ([gray.shape[1], gray.shape[0]] * 2)
        boxes = np.c_[rows[:, :2] - rows[:, 2:] / 2, rows[:, 2:]]  # (x, y, w, h) in pixels
        sig = signature(gray, boxes) if len(boxes) else (dhash(gray), None, 0)
        if is_new(sig, kept, hash_thr, geo_thr):
            kept.append(sig)
            keep.append(i)
    return clip, keep, len(idx)


def mask_box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return None if not len(xs) else (xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())


def propose_sessions(emb: dict[str, np.ndarray], thr: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Similarity between clips = maximum over the pairs of their frames; groups by
    average linkage above `thr`. Single linkage
    chains everything: on uniform sky almost every pair exceeds 0.9."""
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
    """CVAT client from `.env` (`CVAT_URL`, `CVAT_USER`, `CVAT_PASS`); else credentials from ~/.config/cvat.env."""
    from cvat_sdk import make_client

    env = os.environ
    if "CVAT_USER" not in env:
        env = dict(ln.split("=", 1) for ln in (Path.home() / ".config" / "cvat.env").read_text().split())
    return make_client(os.environ.get("CVAT_URL", "http://localhost:8080"), credentials=(env["CVAT_USER"], env["CVAT_PASS"]))