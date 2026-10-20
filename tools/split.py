#!/usr/bin/env python3
"""Passi 11-14 della procedura, su tutte le fonti insieme (`data/aggregati/`).

Passo 11, gruppi di clip (16/10/2026, idea dell'utente): un'impronta per clip
(embedding DINOv2 ViT-S/14 dei frame di una striscia, uno al secondo, a 640x512
come per le sessioni di Svanstrom), legame medio su tutte le clip di tutte le
fonti con **una sola soglia**; lo split si fa per gruppi, quindi nessuna coppia
di clip simili sta in split diversi.

    python tools/split.py --step gruppi --soglia 0.95
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


def scartate() -> set[str]:
    """Clip tolte dal dataset in ogni fonte: scie e scartate di Svanstrom,
    scartate di FRED (`*/*_box/scie.txt`, `*/*_box/scartate.txt`)."""
    out = set()
    for f in list(DATA.glob("*/*_box/scie.txt")) + list(DATA.glob("*/*_box/scartate.txt")):
        out |= {ln.split("#")[0].strip() for ln in open(f) if ln.split("#")[0].strip()}
    return out


def fred_strips() -> dict[str, list[Path]]:
    """FRED: i ritagli 640x512 dei frame tenuti dall'hash, uno al secondo
    (30 frame) per sequenza, solo le sequenze in `6_sessioni/sessioni.csv`
    con i ritagli fatti."""
    ses = {r["clip"] for r in csv.DictReader(open(DATA / "fred" / "6_sessioni" / "sessioni.csv"))}
    out = {}
    for d in sorted((DATA / "fred" / "4_ritagli").glob("FRED_*")):
        if f"{d.name}/s0" not in ses or f"{d.name}/s0" in scartate():
            continue
        imgs = sorted((d / "images").glob("*.jpg"))
        sec = defaultdict(list)
        for p in imgs:
            sec[int(p.stem.rsplit("_f", 1)[1]) // 30].append(p)
        out[f"{d.name}/s0"] = [v[0] for _, v in sorted(sec.items())]
    return out


def embed(paths: dict[str, list[Path]]) -> dict[str, np.ndarray]:
    """Come `svanstrom.embed_strips`: 640x512 -> 280x224, CLS, norma 1."""
    import cv2
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(dev).eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = {}
    with torch.no_grad():
        for v, ps in paths.items():
            x = [(cv2.resize(cv2.imread(str(p)), (280, 224))[..., ::-1] / 255 - mean) / std for p in ps]
            x = torch.tensor(np.stack(x).transpose(0, 3, 1, 2), dtype=torch.float32, device=dev)
            out[v] = torch.nn.functional.normalize(model(x), dim=1).cpu().numpy()
    return out


def frames_per_clip() -> Counter:
    """Frame per clip dopo la deduplica (`*_dedup/tenuti.csv` di ogni fonte)."""
    n = Counter()
    for f in DATA.glob("*/*_dedup/tenuti.csv"):
        n.update(r["clip"] for r in csv.DictReader(open(f)))
    return n


def groups(thr: float) -> None:
    """Passo 11a: gruppi di clip -> `aggregati/9_split/gruppi.csv` (clip, fonte,
    gruppo, sessione manuale) e il riepilogo: grandezza dei gruppi in frame e
    accordo con le sessioni fatte a mano (coppie nello stesso gruppo)."""
    from tools.svanstrom import propose_sessions

    out = DATA / "aggregati" / "9_split"
    out.mkdir(parents=True, exist_ok=True)
    emb = dict(np.load(DATA / "svanstrom" / "3_sessioni" / "embedding.npz"))
    f = out / "embedding_fred.npz"
    ef = dict(np.load(f)) if f.exists() else {}
    strips = fred_strips()
    if set(strips) - set(ef):
        ef = embed(strips)
        np.savez(f, **ef)
    emb.update(ef)
    n = frames_per_clip()
    out_ = scartate()
    clips = [c for c in emb if n[c] and c not in out_]  # solo clip con frame tenuti, senza scie e scartate
    vids, sim, g = propose_sessions({c: emb[c] for c in clips}, thr)
    man = {}
    for src, p in (("svanstrom", DATA / "svanstrom" / "3_sessioni" / "sessioni.csv"),
                   ("fred", DATA / "fred" / "6_sessioni" / "sessioni.csv")):
        man.update({r["clip"]: f"{src}:{r['sessione']}" for r in csv.DictReader(open(p))})
    with open(out / "gruppi.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "fonte", "gruppo", "sessione_manuale", "frame"])
        w.writerows((c, "fred" if c.startswith("FRED_") else "svanstrom", f"G{k:03d}", man.get(c, ""), n[c])
                    for c, k in zip(vids, g))
    tot = sum(n[c] for c in vids)
    size = Counter()
    for c, k in zip(vids, g):
        size[k] += n[c]
    big = size.most_common(5)
    gi = dict(zip(vids, g))
    same_g = {(a, b) for i, a in enumerate(vids) for b in vids[i + 1:] if gi[a] == gi[b]}
    same_m = {(a, b) for i, a in enumerate(vids) for b in vids[i + 1:] if man.get(a) and man.get(a) == man.get(b)}
    mixed = sum(len({c.startswith("FRED_") for c, kk in zip(vids, g) if kk == k}) > 1 for k in set(g))
    print(f"gruppi: soglia {thr}, {len(vids)} clip, {tot} frame -> {len(set(g))} gruppi; i piu' grandi "
          + ", ".join(f"{s / tot:.1%}" for _, s in big)
          + f"; coppie nello stesso gruppo {len(same_g)}, nella stessa sessione manuale {len(same_m)}, "
            f"in comune {len(same_g & same_m)}; gruppi con piu' fonti {mixed}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["gruppi"], required=True)
    p.add_argument("--soglia", type=float, default=0.95)
    a = p.parse_args(argv)
    if a.step == "gruppi":
        groups(a.soglia)
    return 0


if __name__ == "__main__":
    sys.exit(main())
