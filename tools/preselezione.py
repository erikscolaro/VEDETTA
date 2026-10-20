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
5. Box sotto i 4 px^2 (sqrt(w*h) < 2 px, utente, 19/10/2026): tolte dalle label
   di tutte le fonti.
6. Tetto con margine: (4.000 train + 857 val/test) * 1,1 istanze per cella su
   tutte le fonti (Svanstrom, gia' rivisto, conta ma non si tocca); via i
   ritagli con tutte le box in celle sopra il tetto, a turno fra i video.

Legge `data/<fonte>/5_box/proposte/` (le label della YOLO o delle fonti, non si
toccano) e scrive `data/<fonte>/5_box/preselezione/`: label corrette (1, 5) e
`esclusi.txt` (stem, motivo) con i ritagli tolti (2, 3, 4, 6), che non si
cancellano. Conteggio e CVAT leggono da li' (`fred.boxes`, `fred.excluded`).
Il fumo (4) viene da `data/dvids/0_raw/fumo.txt`, scritto dall'operatore con
`aggiunte.py --step fogli_fumo` e `--step fumo`. Si rilancia dopo ogni `catena`.

    python tools/preselezione.py
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
from tools.svanstrom import BAND_EDGES, BANDS, NAMES  # noqa: E402

DATA = ROOT / "data"
FONTI = ["dvids", "fred", "fbd_sv", "anti_uav"]
CAP = round((4000 + 857) * 1.1)
DARK = 35
MIN_AREA = 4  # px^2


def fix_labels() -> None:
    """`5_box/proposte/` (non si tocca) -> `5_box/preselezione/`: classe del video
    su DVIDS (1) e via le box sotto `MIN_AREA` (5)."""
    import shutil

    vc = {r["id"]: NAMES.index(r["classe"]) for r in csv.DictReader(open(DATA / "dvids" / "0_raw" / "video.csv"))}
    for s in FONTI:
        out = DATA / s / "5_box" / "preselezione"
        shutil.rmtree(out, ignore_errors=True)
        n = Counter()
        for d in sorted((DATA / s / "5_box" / "proposte").iterdir()):
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


def smoke() -> set[str]:
    """Ritagli dei tratti in `0_raw/fumo.txt` (id, inizio del tratto in frame; tratti.csv, +-1 frame)."""
    f = DATA / "dvids" / "0_raw" / "fumo.txt"
    if not f.exists():
        return set()
    want = {(i, float(k)) for i, k in (ln.split()[:2] for ln in open(f) if ln.strip() and not ln.startswith("#"))}
    T = defaultdict(list)
    for r in csv.DictReader(open(DATA / "dvids" / "0_raw" / "tratti.csv")):
        T[r["id"]].append((float(r["inizio"]) * 5, float(r["fine"]) * 5))
    out = set()
    for d in (DATA / "dvids" / "4_ritagli").glob("DVIDS_*"):
        i = d.name.split("_")[1]
        for p in (d / "images").glob("*.jpg"):
            k = int(p.stem.rsplit("_f", 1)[1])
            a = next((a for a, b in T[i] if a - 1 <= k <= b + 1), -1)
            if (i, a) in want:
                out.add(p.stem)
    return out


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
    drop = {s: {} for s in FONTI}
    fumo = smoke()
    crops = []  # (fonte, video, stem, celle)
    for s in FONTI:
        for d in sorted((DATA / s / "5_box" / "preselezione").iterdir()):
            for t in sorted(d.glob("*.txt")):
                if s in ("anti_uav", "dvids") and cv2.imread(str(DATA / s / "4_ritagli" / d.name / "images" / f"{t.stem}.jpg"), 0).mean() < DARK:
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
    for r in csv.DictReader(open(next((DATA / "svanstrom").glob("*_conteggio/conteggio.csv")))):
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
    for s in FONTI:
        f = DATA / s / "5_box" / "preselezione" / "esclusi.txt"
        f.write_text("".join(f"{k} {m}\n" for k, m in sorted(drop[s].items())))
        print(f"{s}: esclusi {len(drop[s])} ({dict(Counter(drop[s].values()))}) -> {f}")


if __name__ == "__main__":
    fix_labels()
    select()
