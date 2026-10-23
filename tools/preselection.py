"""Preselection before CVAT (user decision, 19/10/2026), on the proposed boxes:

1. DVIDS, video class: every non-bird box takes the video class
   (`0_raw/video.csv`).
2. DVIDS, close-ups: drop the crops with a box over 25% of the crop or
   cut on 2+ sides (before: over half).
3. Night (user, 19/10/2026: "no night domain"): drop the Anti-UAV
   and DVIDS crops with mean brightness below 35 (lights in the dark, night-vision goggles). Not
   FRED: its dark crops are underexposed daytime yards. For Anti-UAV also
   the shots from 19:00 on (hour in the name): strong city lights raise the mean
   above the threshold (user, 20/10/2026); the 18:00 dusk stays. Applied
   after the cap (6), which thus does not change.
4. Airshow smoke (user, 19/10/2026): drop the crops of the DVIDS segments in
   `0_raw/fumo.txt` (id, start in frames), marked on the `fogli_fumo/`.
4b. Grayscale and archive footage (user, 20/10/2026): drop the crops of the
   DVIDS segments in `0_raw/grigi.txt`, marked on the `fogli_grigi/` (G; N for the night that slipped past the
   threshold); after the cap (6).
4d. Yellow star of the "U.S. ARMY" logo in the DVIDS crops (`stars`, user,
   20/10/2026: it breaks the model); after the cap (6).
4e. Duplicate DVIDS boxes (IoU >= 0.8) or inside another (>= 90%), after the cap
   (`clean_boxes`, user, 20/10/2026).
4c. FRED sequences in `fred/0_raw/esclusi.txt` (sequence, reason), written
   by the operator (user, 20/10/2026: 209); after the cap (6).
5. Boxes under 4 px^2 (sqrt(w*h) < 2 px, user, 19/10/2026): removed from the labels
   of all sources.
6. Cap with margin: (4,000 train + 857 val/test) * 1.1 instances per cell over
   all sources (Svanstrom, already reviewed, counts but is not touched); drop the
   crops with all boxes in cells above the cap, taking turns among the videos.

Reads `data/input/<source>/5_box/proposte/` (the YOLO or source labels, left
untouched) and writes `data/input/<source>/5_box/preselezione/`: corrected labels (1, 5) and
`esclusi.txt` (stem, reason) with the dropped crops (2, 3, 4, 6), which are not
deleted. Count and CVAT read from there (`fred.boxes`, `fred.excluded`).
The smoke (4) comes from `data/dvids/0_raw/fumo.txt`, written by the operator with
`extra_sources.py --step fogli_fumo` and `--step fumo`. Rerun after every `catena`.

    python tools/preselection.py
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import INPUT, BAND_EDGES, BANDS, NAMES  # noqa: E402

SOURCES = ["dvids", "fred", "fbd_sv", "anti_uav"]
CAP = round((4000 + 857) * 1.1)
DARK = 35
MIN_AREA = 4  # px^2


def fix_labels() -> None:
    """`5_box/proposte/` (left untouched) -> `5_box/preselezione/`: video class
    on DVIDS (1) and drop the boxes under `MIN_AREA` (5)."""
    import shutil

    vc = {r["id"]: NAMES.index(r["classe"]) for r in csv.DictReader(open(INPUT / "dvids" / "0_raw" / "video.csv"))}
    for s in SOURCES:
        out = INPUT / s / "5_box" / "preselezione"
        shutil.rmtree(out, ignore_errors=True)
        n = Counter()
        for d in sorted((INPUT / s / "5_box" / "proposte").iterdir()):
            (out / d.name).mkdir(parents=True)
            k = vc[d.name.split("_")[1]] if s == "dvids" else None
            for t in d.glob("*.txt"):
                new = []
                for ln in t.read_text().splitlines():
                    c, cx, cy, w, h = ln.split()
                    n["box"] += 1
                    if float(w) * 640 * float(h) * 512 < MIN_AREA:
                        n["piccole"] += 1
                        continue
                    if k is not None and c not in ("1", str(k)):
                        c, n["classe"] = str(k), n["classe"] + 1
                    new.append(f"{c} {cx} {cy} {w} {h}\n")
                (out / d.name / t.name).write_text("".join(new))
        print(f"{s}: {n['box']} box, classe del video cambiata {n['classe']}, sotto i {MIN_AREA} px^2 tolte {n['piccole']}")


def smoke(name: str = "fumo") -> dict[str, str]:
    """Crops of the segments in `0_raw/<name>.txt` (fumo or grigi; id, segment start in frames; tratti.csv, +-1 frame)
    -> operator letter (fourth column, if present)."""
    f = INPUT / "dvids" / "0_raw" / f"{name}.txt"
    if not f.exists():
        return {}
    want = {(x[0], float(x[1])): (x[3] if len(x) > 3 else "") for x in (ln.split() for ln in open(f) if ln.strip() and not ln.startswith("#"))}
    T = defaultdict(list)
    for r in csv.DictReader(open(INPUT / "dvids" / "0_raw" / "tratti.csv")):
        T[r["id"]].append((float(r["inizio"]) * 5, float(r["fine"]) * 5))
    out = {}
    for d in (INPUT / "dvids" / "4_ritagli").glob("DVIDS_*"):
        i = d.name.split("_")[1]
        for p in (d / "images").glob("*.jpg"):
            k = int(p.stem.rsplit("_f", 1)[1])
            a = next((a for a, b in T[i] if a - 1 <= k <= b + 1), -1)
            if (i, a) in want:
                out[p.stem] = want[i, a]
    return out


STAR_REF = INPUT / "dvids" / "4_ritagli" / "DVIDS_1002305" / "images" / "DVIDS_1002305_f00283.jpg"  # star at (5-100, 190-280)


def _yellow(im: np.ndarray) -> np.ndarray:
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return (((h[..., 0] >= 21) & (h[..., 0] <= 30) & (h[..., 1] >= 170) & (h[..., 2] >= 170)) * 255).astype(np.uint8)


def stars() -> set[str]:
    """DVIDS crops with the yellow star of the "U.S. ARMY" logo (user, 20/10/2026: it breaks
    the model): a yellow blob (H 21-30, S and V >= 170), as wide as tall, with the
    shape of the reference star (matchShapes <= 0.2; stars are below 0.13,
    lights and flowers above 0.26)."""
    from concurrent.futures import ThreadPoolExecutor

    cs, _ = cv2.findContours(_yellow(cv2.imread(str(STAR_REF)))[190:280, 5:100], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ref = max(cs, key=cv2.contourArea)

    def one(p: Path) -> bool:
        cs, _ = cv2.findContours(_yellow(cv2.imread(str(p))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return any(cv2.contourArea(c) >= 60 and 0.75 <= cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3] <= 1.35
                   and cv2.matchShapes(c, ref, cv2.CONTOURS_MATCH_I1, 0) <= 0.2 for c in cs)

    ps = sorted((INPUT / "dvids" / "4_ritagli").glob("DVIDS_*/images/*.jpg"))
    with ThreadPoolExecutor(16) as ex:
        return {p.stem for p, y in zip(ps, ex.map(one, ps)) if y}


def big(cx: float, cy: float, w: float, h: float) -> bool:
    """Close-up: box over 25% of the crop or cut on 2+ sides (user, 19/10/2026)."""
    cut = (cx - w / 2 < 0.002) + (cy - h / 2 < 0.002) + (cx + w / 2 > 0.998) + (cy + h / 2 > 0.998)
    return w * h > 0.25 or cut >= 2


def cells(t: Path) -> list[tuple[str, str]]:
    out = []
    for ln in t.read_text().splitlines():
        c, _, _, w, h = ln.split()
        out.append((NAMES[int(c)], BANDS[np.digitize(np.sqrt(float(w) * 640 * float(h) * 512), BAND_EDGES)]))
    return out


def select() -> None:
    drop = {s: {} for s in SOURCES}
    smoked = smoke()
    crops = []  # (source, video, stem, cells)
    for s in SOURCES:
        for d in sorted((INPUT / s / "5_box" / "preselezione").iterdir()):
            for t in sorted(d.glob("*.txt")):
                if s in ("anti_uav", "dvids") and cv2.imread(str(INPUT / s / "4_ritagli" / d.name / "images" / f"{t.stem}.jpg"), 0).mean() < DARK:
                    drop[s][t.stem] = "notte"
                    continue
                if t.stem in smoked:
                    drop[s][t.stem] = "fumo"
                    continue
                if s == "dvids" and any(big(*map(float, ln.split()[1:])) for ln in t.read_text().splitlines()):
                    drop[s][t.stem] = "primo_piano"
                    continue
                crops.append((s, d.name, t.stem, [c for c in cells(t) if c[1] != "<8"]))
    cnt = Counter(c for *_, cs in crops for c in cs)
    for r in csv.DictReader(open(next((INPUT / "svanstrom").glob("*_conteggio/conteggio.csv")))):
        cnt[r["classe"], r["fascia"]] += int(r["istanze"])
    print("sopra il tetto", CAP, ":", ", ".join(f"{c} {b} {cnt[c, b]}" for c, b in sorted(cnt) if cnt[c, b] > CAP))
    random.Random(0).shuffle(crops)
    byv = defaultdict(list)
    for x in crops:
        byv[x[1]].append(x)
    changed = True
    while changed:  # round-robin: one crop per video, videos with more crops first
        changed = False
        for v in sorted(byv, key=lambda v: -len(byv[v])):
            L = byv[v]
            i = next((i for i, x in enumerate(L) if x[3] and all(cnt[c] > CAP for c in x[3])), None)
            if i is not None:
                s, _, stem, cs = L.pop(i)
                for c in cs:
                    cnt[c] -= 1
                drop[s][stem] = "tetto"
                changed = True
    # Anti-UAV from 19:00 on: after the cap, which stays the one of the CVAT tasks already uploaded
    for s, _, stem, _ in crops:
        if s == "anti_uav" and int(stem.split("_")[2][:2]) >= 19:
            drop[s][stem] = "notte"
    # grayscale or archive DVIDS (user, 20/10/2026), also after the cap
    grays = smoke("grigi") | smoke("verdi")  # verdi: night-vision goggles (N)
    for s, _, stem, _ in crops:
        if stem in grays:
            drop[s][stem] = "notte" if grays[stem] == "N" else "grigio"  # N: night lights marked on the sheets
    # yellow star of the "U.S. ARMY" logo (DVIDS, user 20/10/2026), after the cap
    st = stars()
    for s, _, stem, _ in crops:
        if s == "dvids" and stem in st:
            drop[s][stem] = "stella"
    # FRED sequences dropped by the operator (`fred/0_raw/esclusi.txt`, user 20/10/2026), also after the cap
    f = INPUT / "fred" / "0_raw" / "esclusi.txt"
    excl = {f"FRED_{x[0]}": x[1] for x in (ln.split() for ln in open(f) if ln.strip() and not ln.startswith("#"))} if f.exists() else {}
    for s, v, stem, _ in crops:
        if v in excl:
            drop[s][stem] = excl[v]
    for s in SOURCES:
        f = INPUT / s / "5_box" / "preselezione" / "esclusi.txt"
        f.write_text("".join(f"{k} {m}\n" for k, m in sorted(drop[s].items())))
        print(f"{s}: esclusi {len(drop[s])} ({dict(Counter(drop[s].values()))}) -> {f}")


def clean_boxes(dup: float = 0.8, inside: float = 0.9, edge: float = 4) -> None:
    """DVIDS, after the cap (which does not change): drop the duplicate boxes (IoU >= `dup`, the
    video class is kept first, then the largest) and those inside another for at least
    `inside` of their area (the outermost is kept). User, 20/10/2026: the most
    common errors in CVAT. Also drop the crop leftovers: boxes touching the border with the
    short side <= `edge` px (user, 22/10/2026)."""
    vc = {r["id"]: NAMES.index(r["classe"]) for r in csv.DictReader(open(INPUT / "dvids" / "0_raw" / "video.csv"))}

    def xyxy(b):
        cx, cy, w, h = map(float, b[1:])
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def inter(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    area = lambda a: (a[2] - a[0]) * (a[3] - a[1])  # noqa: E731
    n = Counter()
    W, H = 640, 512
    for t in (INPUT / "dvids" / "5_box" / "preselezione").glob("DVIDS_*/*.txt"):
        bs = [ln.split() for ln in t.read_text().splitlines()]
        n0 = len(bs)
        px = [(x[0] * W, x[1] * H, x[2] * W, x[3] * H) for x in map(xyxy, bs)]
        bs = [b for b, (x0, y0, x1, y1) in zip(bs, px)
              if not ((x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1) and min(x1 - x0, y1 - y0) <= edge)]
        n["bordo"] += n0 - len(bs)
        k = str(vc[t.parent.name.split("_")[1]])
        keep = []
        for b in sorted(bs, key=lambda b: (b[0] != k, -area(xyxy(b)))):
            x = xyxy(b)
            if any(inter(x, xyxy(o)) / (area(x) + area(xyxy(o)) - inter(x, xyxy(o))) >= dup for o in keep):
                n["doppie"] += 1
            else:
                keep.append(b)
        out = []
        for b in sorted(keep, key=lambda b: -area(xyxy(b))):
            x = xyxy(b)
            if any(inter(x, xyxy(o)) >= inside * area(x) for o in out):
                n["dentro"] += 1
            else:
                out.append(b)
        if len(out) < n0:
            t.write_text("".join(" ".join(b) + "\n" for b in out))
    print(f"dvids: box doppie tolte {n['doppie']}, dentro un'altra {n['dentro']}, residui sul bordo {n['bordo']}")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fix_labels()
    select()
    clean_boxes()
