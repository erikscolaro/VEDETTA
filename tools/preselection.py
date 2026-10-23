"""Preselezione prima di CVAT (decisione dell'utente, 19/10/2026), sui box proposti:

1. DVIDS, classe del video: ogni box non-uccello prende la classe del video
   (`0_raw/video.csv`).
2. DVIDS, primi piani: via i ritagli con una box oltre il 25% del ritaglio o
   tagliata su 2+ lati (prima: oltre meta').
3. Notte (utente, 19/10/2026: "no dominio notturno"): via i ritagli di Anti-UAV
   e DVIDS con luminosita' media sotto 35 (luci nel buio, visori notturni). Non
   FRED: i suoi ritagli bui sono cortili di giorno sottoesposti. Per Anti-UAV anche
   le riprese dalle 19 in poi (ora nel nome): luci della citta' forti alzano la media
   sopra la soglia (utente, 20/10/2026); il crepuscolo delle 18 resta. Si
   applica dopo il tetto (6), che cosi' non cambia.
4. Fumo degli airshow (utente, 19/10/2026): via i ritagli dei tratti DVIDS in
   `0_raw/fumo.txt` (id, inizio in frame), segnati sui `fogli_fumo/`.
4b. Scala di grigi e filmati d'archivio (utente, 20/10/2026): via i ritagli dei
   tratti DVIDS in `0_raw/grigi.txt`, segnati sui `fogli_grigi/` (G; N per la notte sfuggita alla
   soglia); dopo il tetto (6).
4d. Stella gialla del logo "U.S. ARMY" nei ritagli DVIDS (`stars`, utente,
   20/10/2026: rompe il modello); dopo il tetto (6).
4e. Box DVIDS doppie (IoU >= 0,8) o dentro un'altra (>= 90%), dopo il tetto
   (`clean_boxes`, utente, 20/10/2026).
4c. Sequenze FRED in `fred/0_raw/esclusi.txt` (sequenza, motivo), scritto
   dall'operatore (utente, 20/10/2026: 209); dopo il tetto (6).
5. Box sotto i 4 px^2 (sqrt(w*h) < 2 px, utente, 19/10/2026): tolte dalle label
   di tutte le fonti.
6. Tetto con margine: (4.000 train + 857 val/test) * 1,1 istanze per cella su
   tutte le fonti (Svanstrom, gia' rivisto, conta ma non si tocca); via i
   ritagli con tutte le box in celle sopra il tetto, a turno fra i video.

Legge `data/input/<fonte>/5_box/proposte/` (le label della YOLO o delle fonti, non si
toccano) e scrive `data/input/<fonte>/5_box/preselezione/`: label corrette (1, 5) e
`esclusi.txt` (stem, motivo) con i ritagli tolti (2, 3, 4, 6), che non si
cancellano. Conteggio e CVAT leggono da li' (`fred.boxes`, `fred.excluded`).
Il fumo (4) viene da `data/dvids/0_raw/fumo.txt`, scritto dall'operatore con
`extra_sources.py --step fogli_fumo` e `--step fumo`. Si rilancia dopo ogni `catena`.

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
    """`5_box/proposte/` (non si tocca) -> `5_box/preselezione/`: classe del video
    su DVIDS (1) e via le box sotto `MIN_AREA` (5)."""
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


def smoke(nome: str = "fumo") -> dict[str, str]:
    """Ritagli dei tratti in `0_raw/<nome>.txt` (fumo o grigi; id, inizio del tratto in frame; tratti.csv, +-1 frame)
    -> lettera dell'operatore (quarta colonna, se c'e')."""
    f = INPUT / "dvids" / "0_raw" / f"{nome}.txt"
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


STAR_REF = INPUT / "dvids" / "4_ritagli" / "DVIDS_1002305" / "images" / "DVIDS_1002305_f00283.jpg"  # stella a (5-100, 190-280)


def _yellow(im: np.ndarray) -> np.ndarray:
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return (((h[..., 0] >= 21) & (h[..., 0] <= 30) & (h[..., 1] >= 170) & (h[..., 2] >= 170)) * 255).astype(np.uint8)


def stars() -> set[str]:
    """Ritagli DVIDS con la stella gialla del logo "U.S. ARMY" (utente, 20/10/2026: rompe
    il modello): una macchia gialla (H 21-30, S e V >= 170), larga quanto alta, con la
    forma della stella di riferimento (matchShapes <= 0,2; le stelle stanno sotto 0,13,
    luci e fiori sopra 0,26)."""
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
    """Primo piano: box oltre il 25% del ritaglio o tagliata su 2+ lati (utente, 19/10/2026)."""
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
    fumo = smoke()
    crops = []  # (fonte, video, stem, celle)
    for s in SOURCES:
        for d in sorted((INPUT / s / "5_box" / "preselezione").iterdir()):
            for t in sorted(d.glob("*.txt")):
                if s in ("anti_uav", "dvids") and cv2.imread(str(INPUT / s / "4_ritagli" / d.name / "images" / f"{t.stem}.jpg"), 0).mean() < DARK:
                    drop[s][t.stem] = "notte"
                    continue
                if t.stem in fumo:
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
    while changed:  # a turno: un ritaglio per video, prima i video con piu' ritagli
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
    # Anti-UAV dalle 19 in poi: dopo il tetto, che resta quello dei task CVAT gia' caricati
    for s, _, stem, _ in crops:
        if s == "anti_uav" and int(stem.split("_")[2][:2]) >= 19:
            drop[s][stem] = "notte"
    # DVIDS in scala di grigi o d'archivio (utente, 20/10/2026), anch'essi dopo il tetto
    grigi = smoke("grigi") | smoke("verdi")  # verdi: visore notturno (N)
    for s, _, stem, _ in crops:
        if stem in grigi:
            drop[s][stem] = "notte" if grigi[stem] == "N" else "grigio"  # N: luci di notte segnate sui fogli
    # stella gialla del logo "U.S. ARMY" (DVIDS, utente 20/10/2026), dopo il tetto
    st = stars()
    for s, _, stem, _ in crops:
        if s == "dvids" and stem in st:
            drop[s][stem] = "stella"
    # sequenze FRED tolte dall'operatore (`fred/0_raw/esclusi.txt`, utente 20/10/2026), anch'esse dopo il tetto
    f = INPUT / "fred" / "0_raw" / "esclusi.txt"
    via = {f"FRED_{x[0]}": x[1] for x in (ln.split() for ln in open(f) if ln.strip() and not ln.startswith("#"))} if f.exists() else {}
    for s, v, stem, _ in crops:
        if v in via:
            drop[s][stem] = via[v]
    for s in SOURCES:
        f = INPUT / s / "5_box" / "preselezione" / "esclusi.txt"
        f.write_text("".join(f"{k} {m}\n" for k, m in sorted(drop[s].items())))
        print(f"{s}: esclusi {len(drop[s])} ({dict(Counter(drop[s].values()))}) -> {f}")


def clean_boxes(dup: float = 0.8, inside: float = 0.9, bordo: float = 4) -> None:
    """DVIDS, dopo il tetto (che non cambia): via le box doppie (IoU >= `dup`, si tiene
    prima la classe del video, poi la piu' grande) e quelle dentro un'altra per almeno
    `inside` della loro area (si tiene la piu' esterna). Utente, 20/10/2026: gli errori
    piu' comuni in CVAT. Via anche i residui del crop: box che toccano il bordo con il
    lato corto <= `bordo` px (utente, 22/10/2026)."""
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
              if not ((x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1) and min(x1 - x0, y1 - y0) <= bordo)]
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
