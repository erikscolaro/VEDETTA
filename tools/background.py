#!/usr/bin/env python3
"""Passo 13, background da fonti indipendenti (Clouds-1500, Skyfinder; licenze in docs/licenze.md). Una cartella per
passo in `data/<fonte>/`, poi la scelta per split in `data/aggregati/14_background/`.

    python tools/background.py --fonte clouds1500 --step ritagli   # 1_ritagli: 640x512 sulla fascia cielo-orizzonte
    python tools/background.py --fonte clouds1500 --step detector  # 2_detector: via i ritagli con un oggetto
    python tools/background.py --fonte clouds1500 --step dedup     # 3_dedup: via i quasi-duplicati per camera
    python tools/background.py --fonte clouds1500 --step fogli     # anteprima/: tenuti ed esclusi a campione
    python tools/background.py --step split                        # aggregati/14_background: ~10% per split
    python tools/background.py --step raccolta                     # data/background/0_raw: le scelte in hardlink
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
DATA = ROOT / "data"
W, H = 640, 512
DARK = 35  # luminosita' media sotto cui il ritaglio e' notte (regole.md, come Anti-UAV e DVIDS)


def sources(fonte: str) -> list[tuple[Path, str]]:
    """(immagine, camera) della fonte. Clouds-1500: camera = cartella (due camere in un solo posto, cartelle per
    periodo). Skyfinder: camera = cartella AMOS, senza `GSV` (Street View) e senza le camere sotto 640x512."""
    import cv2

    raw = DATA / fonte / "0_raw" / "estratto"
    if fonte == "clouds1500":
        return [(p, p.parent.parent.name) for p in sorted(raw.glob("*/img/*")) if p.suffix.lower() in (".jpg", ".png")]
    out = []
    for d in sorted((raw / "OriginalImages").iterdir()):
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        h, w = cv2.imread(str(fs[0])).shape[:2]
        if d.name != "GSV" and w >= W and h >= H:
            out += [(p, d.name) for p in fs]
    return out


def _crop(args) -> list[tuple]:
    """Ritagli 640x512 senza ridimensionare, con l'angolo in alto a sinistra a caso nella meta' alta (cielo e
    orizzonte, lontano dalle scritte in basso); via i ritagli scuri."""
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


def crops(fonte: str, seed: int = 0) -> None:
    """1_ritagli: `ritagli.csv` (ritaglio, sorgente, camera, x0, y0) e `images/`. Clouds-1500 due ritagli per
    immagine (1.500 immagini), Skyfinder uno."""
    from concurrent.futures import ProcessPoolExecutor

    dst = DATA / fonte / "1_ritagli" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.iterdir():
        f.unlink()
    src = sources(fonte)
    k = 2 if fonte == "clouds1500" else 1
    with ProcessPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(_crop, [(p, cam, k, seed + i, dst) for i, (p, cam) in enumerate(src)], chunksize=32)
                for r in rs]
    with open(dst.parent / "ritagli.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ritaglio", "sorgente", "camera", "x0", "y0"])
        w.writerows(rows)
    print(f"{fonte} ritagli: {len(src)} immagini, {len(rows)} ritagli ({len(src) * k - len(rows)} scuri via), "
          f"{len({r[2] for r in rows})} camere -> {dst.parent}")


def detector(fonte: str, conf: float = 0.1) -> None:
    """2_detector: YOLO delle proposte (`data/modelli/yolo26m-intern.pt`, le 4 classi) su ogni ritaglio, confidenza
    bassa: un ritaglio con almeno una detection va via (`esclusi.csv`: ritaglio, classe, conf, box), gli altri in
    `tenuti.csv`. Meglio buttare un cielo buono che tenere un uccello senza box."""
    from ultralytics import YOLO

    model = YOLO(str(DATA / "modelli" / "yolo26m-intern.pt"))
    src = DATA / fonte / "1_ritagli"
    rows = list(csv.DictReader(open(src / "ritagli.csv")))
    out = DATA / fonte / "2_detector"
    out.mkdir(exist_ok=True)
    esc, ten = [], []
    paths = [str(src / "images" / f"{r['ritaglio']}.jpg") for r in rows]
    for k in range(0, len(paths), 64):
        for r, res in zip(rows[k:k + 64], model(paths[k:k + 64], conf=conf, verbose=False)):
            if len(res.boxes):
                b = max(res.boxes, key=lambda x: float(x.conf))
                esc.append((r["ritaglio"], model.names[int(b.cls)], round(float(b.conf), 3),
                            " ".join(f"{v:.0f}" for v in b.xyxy[0].tolist())))
            else:
                ten.append(r)
    with open(out / "esclusi.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ritaglio", "classe", "conf", "box"])
        w.writerows(esc)
    with open(out / "tenuti.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(ten)
    print(f"{fonte} detector (conf {conf}): {len(esc)} via ({', '.join(f'{c} {n}' for c, n in Counter(e[1] for e in esc).most_common())}), "
          f"{len(ten)} tenuti -> {out}")


def dedup(fonte: str, thr: float = 0.95) -> None:
    """3_dedup: embedding DINOv2 ViT-S (`split.embed`) dei ritagli tenuti; per camera, in ordine di sorgente, un
    ritaglio resta se la somiglianza con tutti quelli gia' tenuti e' sotto `thr` (0,95, la soglia dei doppioni) ->
    `tenuti.csv`, `embedding.npy` (dei tenuti, stesso ordine)."""
    from tools.split import embed

    rows = list(csv.DictReader(open(DATA / fonte / "2_detector" / "tenuti.csv")))
    out = DATA / fonte / "3_dedup"
    out.mkdir(exist_ok=True)
    e = embed([str(DATA / fonte / "1_ritagli" / "images" / f"{r['ritaglio']}.jpg") for r in rows]).astype(np.float32)
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
    print(f"{fonte} dedup (< {thr}): {len(keep)} su {len(rows)}; per camera min {min(n.values())}, max {max(n.values())} -> {out}")


def sheets(fonte: str, n: int = 48, seed: int = 0) -> None:
    """anteprima/: `tenuti.jpg` (n ritagli tenuti a caso) ed `esclusi.jpg` (n esclusi dal detector, con il box e
    la classe), per il controllo a occhio."""
    import cv2

    rng = np.random.default_rng(seed)
    dst = DATA / fonte / "anteprima"
    dst.mkdir(exist_ok=True)
    img = DATA / fonte / "1_ritagli" / "images"
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

    ten = list(csv.DictReader(open(DATA / fonte / "3_dedup" / "tenuti.csv")))
    sheet([(r["ritaglio"], r["ritaglio"], "") for r in rng.choice(ten, min(n, len(ten)), replace=False)], "tenuti.jpg")
    esc = list(csv.DictReader(open(DATA / fonte / "2_detector" / "esclusi.csv")))
    if esc:
        sheet([(r["ritaglio"], f"{r['classe']} {r['conf']}", r["box"]) for r in rng.choice(esc, min(n, len(esc)), replace=False)],
              "esclusi.jpg")
    print(f"{fonte} fogli -> {dst}")


def split(fonti=("clouds1500", "skyfinder"), share: float = 0.1, seed: int = 0) -> None:
    """aggregati/14_background: ~10% di background in ogni split (bg = immagini / 9, cosi' sono il 10% del totale).
    Camere intere in uno split (la stessa camera mai in due split): a turno, ogni camera va nello split che ha piu'
    bisogno rispetto ai ritagli gia' assegnati; poi in ogni split si pesca a turno fra le camere. ->
    `background.csv` (immagine, fonte, camera, split)."""
    agg = DATA / "aggregati"
    need = Counter(r["split"] for r in csv.DictReader(open(agg / "12_split" / "frame_split.csv")))
    need["train"] = sum(1 for _ in csv.DictReader(open(agg / "13_train_compensation" / "train.csv")))
    need = {s: round(need[s] * share / (1 - share)) for s in ("train", "val", "test")}
    cams = defaultdict(list)
    for f in fonti:
        for r in csv.DictReader(open(DATA / f / "3_dedup" / "tenuti.csv")):
            cams[f, r["camera"]].append(f"{f}/1_ritagli/images/{r['ritaglio']}.jpg")
    rng = np.random.default_rng(seed)
    got, lab = Counter(), {}
    for c in sorted(cams, key=lambda c: -len(cams[c])):  # le camere grandi prima, poi riempiono le piccole
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
    """data/background/0_raw: le immagini scelte in `aggregati/14_background/background.csv` in hardlink (stesso
    nome del ritaglio), piu' `background.csv` con il nuovo percorso. Una fonte a se', per i passi dopo (revisione).
    Salta i nomi in `rimossi.txt` (scelte dell'operatore: sfondi tolti a mano perche' avevano oggetti)."""
    import os

    dst = DATA / "background" / "0_raw"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    for f in (dst / "images").iterdir():
        f.unlink()
    f = dst / "rimossi.txt"
    via = {ln.strip() for ln in open(f) if ln.strip() and not ln.startswith("#")} if f.exists() else set()
    rows = [r for r in csv.DictReader(open(DATA / "aggregati" / "14_background" / "background.csv"))
            if Path(r["immagine"]).name not in via]
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
        {"ritagli": crops, "detector": detector, "dedup": dedup, "fogli": sheets}[a.step](a.fonte)
    return 0


if __name__ == "__main__":
    sys.exit(main())
