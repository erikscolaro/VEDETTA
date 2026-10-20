"""Campioni di ritagli per stimare la qualita' dei box prima di CVAT.

    python tools/campione.py --step fogli --nome revisione --fonti dvids:70 fred:20 fbd_sv:18 anti_uav:12 --seed 0
    python tools/campione.py --step fogli --nome qualita --fonti fbd_sv:48 anti_uav:48 --seed 1 --zoom --senza-esclusi
    python tools/campione.py --step stima --nome revisione

`fogli`: ritagli a caso (seed fisso) di `data/<fonte>/4_ritagli/` con le box di
`5_box/preselezione/` (o `proposte/`) disegnate, numerati, in
`data/aggregati/campioni/<nome>/` con `indice.csv` (n, fonte, frame, box). Con
`--zoom` accanto a ogni ritaglio uno zoom 4x sulla box piu' grande; con
`--senza-esclusi` solo i ritagli non tolti dalla preselezione.

L'operatore guarda i fogli e scrive `giudizi.csv` nella stessa cartella: righe
`n,giudizio,tipo` solo per i ritagli da correggere (giudizio `fix` o `dubbio`;
tipo per esempio classe, mancato, box_unite, primo_piano, falso_positivo).

`stima`: quota da correggere per fonte e per tipo, e proiezione sui ritagli
che oggi vanno in CVAT.
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.fred as fred  # noqa: E402

DATA = ROOT / "data"
COL = {0: (0, 165, 255), 1: (255, 0, 255), 2: (0, 0, 255), 3: (255, 255, 0)}
LET = "AUDE"  # aereo, uccello, drone, elicottero


def sheets(nome: str, fonti: dict[str, int], seed: int, zoom: bool, senza_esclusi: bool) -> None:
    import cv2

    out = DATA / "aggregati" / "campioni" / nome
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows, tiles = [], []
    for src, n in fonti.items():
        ex = fred.excluded(DATA / src) if senza_esclusi else set()
        imgs = [p for p in sorted((DATA / src / "4_ritagli").glob("*/images/*.jpg")) if p.stem not in ex]
        for p in rng.sample(imgs, n):
            lab = fred.boxes(DATA / src) / p.parent.parent.name / f"{p.stem}.txt"
            im = cv2.imread(str(p))
            raw = im.copy()
            bs = [(int(c), *map(float, r)) for c, *r in (ln.split() for ln in lab.read_text().splitlines())] if lab.exists() else []
            bs = [(c, cx * 640 - w * 320, cy * 512 - h * 256, cx * 640 + w * 320, cy * 512 + h * 256) for c, cx, cy, w, h in bs]
            for c, x1, y1, x2, y2 in bs:
                cv2.rectangle(im, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), COL[c], 1)
                cv2.putText(im, LET[c], (int(x2) + 5, max(int(y1) + 7, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[c], 1)
            k = len(rows) + 1
            if zoom:
                z = np.zeros((384, 480, 3), np.uint8)
                if bs:
                    _, x1, y1, x2, y2 = max(bs, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
                    s = max(40, 1.6 * max(x2 - x1, (y2 - y1) * 1.25))
                    X0, Y0 = int(np.clip((x1 + x2 - s) / 2, 0, 640 - s)), int(np.clip((y1 + y2) / 2 - s / 2.5, 0, 512 - s / 1.25))
                    z = cv2.resize(raw[Y0:Y0 + int(s / 1.25), X0:X0 + int(s)], (480, 384), interpolation=cv2.INTER_NEAREST)
                    f = 480 / int(s)
                    cv2.rectangle(z, (int((x1 - X0) * f), int((y1 - Y0) * f)), (int((x2 - X0) * f), int((y2 - Y0) * f)), (0, 255, 0), 1)
                im = np.hstack([cv2.resize(im, (480, 384), interpolation=cv2.INTER_AREA), z])
            cv2.rectangle(im, (0, 0), (50, 24), (0, 0, 0), -1)
            cv2.putText(im, str(k), (3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            tiles.append(cv2.copyMakeBorder(im, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
            rows.append((k, src, p.stem, len(bs)))
    cols = 2 if zoom else 3
    for s in range(0, len(tiles), 6):
        g = tiles[s:s + 6] + [np.zeros_like(tiles[0])] * (-len(tiles[s:s + 6]) % cols)
        cv2.imwrite(str(out / f"{s // 6 + 1:02d}.jpg"), np.vstack([np.hstack(g[j:j + cols]) for j in range(0, len(g), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out / "indice.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "fonte", "frame", "box"])
        w.writerows(rows)
    print(f"campione {nome}: {len(rows)} ritagli, {sum(r[3] for r in rows)} box -> {out}")


def estimate(nome: str) -> None:
    d = DATA / "aggregati" / "campioni" / nome
    idx = {r["n"]: r["fonte"] for r in csv.DictReader(open(d / "indice.csv"))}
    g = {r["n"]: r for r in csv.DictReader(open(d / "giudizi.csv"))}
    for src in dict.fromkeys(idx.values()):
        n = sum(v == src for v in idx.values())
        fix = [g[k] for k, v in idx.items() if v == src and k in g and g[k]["giudizio"] == "fix"]
        dub = sum(1 for k, v in idx.items() if v == src and k in g and g[k]["giudizio"] == "dubbio")
        ex = fred.excluded(DATA / src)
        tot = sum(p.stem not in ex for p in (DATA / src / "4_ritagli").glob("*/images/*.jpg"))
        print(f"{src}: {len(fix)}/{n} da correggere ({len(fix) / n:.0%}), {dub} dubbi; "
              f"per tipo {dict(Counter(r['tipo'] for r in fix))}; in CVAT {tot} ritagli -> ~{round(tot * len(fix) / n)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["fogli", "stima"], required=True)
    p.add_argument("--nome", required=True)
    p.add_argument("--fonti", nargs="+", default=[], help="fonte:numero, per esempio dvids:70")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zoom", action="store_true")
    p.add_argument("--senza-esclusi", action="store_true")
    a = p.parse_args(argv)
    if a.step == "fogli":
        sheets(a.nome, {f.split(":")[0]: int(f.split(":")[1]) for f in a.fonti}, a.seed, a.zoom, a.senza_esclusi)
    else:
        estimate(a.nome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
