#!/usr/bin/env python3
"""Passo 9 della procedura: celle classe x fascia (>= 8 px) sotto i minimi, su
tutte le fonti insieme. Somma i `data/<fonte>/*_conteggio/conteggio.csv` (le
sessioni e i video di fonti diverse sono distinti) -> `data/aggregati/8_buchi/buchi.csv`
con quanto manca.

    python tools/buchi.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.svanstrom import BANDS, NAMES  # noqa: E402

# minimi per cella (regole.md, 19/10/2026): 2.000 istanze nel train (10.000 per classe / 5 fasce), split
# 70/20/10 -> 2.857 in tutto; train >= 10 video; split per sessione, quindi almeno una sessione e un video per split
MIN_INST = round(2000 / 0.7)
MARGIN = 0.10  # finche' i box non sono rivisti in CVAT si scarica fino a +10% (regole.md)
MIN_SESSIONS = 3
MIN_VIDEOS = 10 + 1 + 1
MIN_CLASS = {"frame": 1500, "istanze": 10_000}  # train, per classe, tutte le fasce
KEYS = ("istanze", "frame", "sessioni", "video")


def main() -> int:
    data = ROOT / "data"
    srcs = sorted(p.parents[1].name for p in data.glob("*/*_conteggio/conteggio.csv"))
    tot, per_src = Counter(), Counter()
    for s in srcs:
        for r in csv.DictReader(open(next((data / s).glob("*_conteggio/conteggio.csv")))):
            r = {k.strip(): v.strip() for k, v in r.items()}  # anche col csv allineato a spazi
            for k in KEYS:
                tot[r["classe"], r["fascia"], k] += int(r[k])
            per_src[r["classe"], r["fascia"], s] += int(r["istanze"])
    out = data / "aggregati" / "8_buchi"
    out.mkdir(exist_ok=True)
    recs = []
    for c in NAMES:
        for b in BANDS[1:]:  # dalla 8-16
            n, ses, vid = (tot[c, b, k] for k in ("istanze", "sessioni", "video"))
            miss = (max(0, MIN_INST - n), max(0, MIN_SESSIONS - ses), max(0, MIN_VIDEOS - vid))
            recs.append((c, b, n, ses, vid, *miss, any(miss), max(0, round(MIN_INST * (1 + MARGIN)) - n),
                         *(per_src[c, b, s] for s in srcs)))
    with open(out / "buchi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "fascia", "istanze", "sessioni", "video", "mancano_istanze", "mancano_sessioni",
                    "mancano_video", "buco", "mancano_col_margine", *(f"istanze_{s}" for s in srcs)])
        w.writerows(recs)
    print(f"buchi: fonti {', '.join(srcs)}; minimi per cella {MIN_INST} istanze, {MIN_SESSIONS} sessioni, "
          f"{MIN_VIDEOS} video")
    for c, b, n, ses, vid, mi, ms, mv, gap, mm, *_ in recs:
        if gap or mm:
            print(f"  {c:<11}{b:>6}  {n:>5} ist {ses:>3} ses {vid:>3} vid   mancano "
                  + ", ".join(f"{m} {k}" for m, k in zip((mi, ms, mv), ("istanze", "sessioni", "video")) if m)
                  + f" (col margine {mm})")
    for c in NAMES:
        got = {k: sum(tot[c, b, k] for b in BANDS) for k in MIN_CLASS}
        if any(got[k] < MIN_CLASS[k] for k in MIN_CLASS):
            print(f"  {c}: per classe {got} sotto {MIN_CLASS}")
    print(f"{sum(r[8] for r in recs)} buchi su {len(recs)} celle -> {out / 'buchi.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
