#!/usr/bin/env python3
"""Distribuzione continua della grandezza sqrt(w*h) per classe sulle label finali (stessi frame di
tools/buchi.py: Svanström sui frame tenuti, aggiunte da 5_box/manuale), box >= 8 px in origine.

Due figure in docs/figure/ (considerazioni in docs/distribuzioni.md), con la curva ideale (TOTALE alzato per classe con SCALA)
come linea continua e lo stesso asse verticale nei 4 grafici:
- distribuzioni.png: istogramma per classe con le istanze per fascia (rosso: sotto il minimo per cella);
- distribuzioni_scale.png: la stessa con lo scale di Ultralytics (s = 0,25 e 0,5) calcolato in modo esatto.

    python tools/distribuzioni.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plots import _P, STYLE, WIDTH, save  # noqa: E402
from tools.svanstrom import NAMES  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "docs" / "figure"
W, H = 640, 512
MIN = 8  # box sotto gli 8 px in origine esclusi (utente, 22/10/2026)
TARGET = round(2000 / 0.7)  # minimo per cella (regole.md)
EDGES = [8, 16, 32, 64, 96]
BANDS = [(8, 16), (16, 32), (32, 64), (64, 96), (96, 1e9)]
CLASSI = ["airplane", "bird", "drone", "helicopter"]
# distribuzione ideale: densita' in log s proporzionale a s^-alpha su [SMIN, SMAX] (docs/distribuzioni.md)
ALPHA = dict.fromkeys(["airplane", "bird", "drone", "helicopter"], 0.3)  # stessa curva per tutte (utente, 23/10/2026)
SMIN, SMAX = 8, 320
TOTALE = round(10_000 / 0.7)  # box per classe della curva ideale: 10.000 nel train, split 70/20/10 (regole.md)
# curva alzata (utente, 23/10/2026): almeno 10.000 istanze per classe nel train, 2.000 in val, 1.000 in test; la stessa
# curva traslata in alto fa da tetto (sopra) e da obiettivo dei mosaici (sotto), cosi' la forma resta s^-alpha
SCALA = dict.fromkeys(["airplane", "bird", "drone", "helicopter"], 1.2)
TOT = {c: TOTALE * k for c, k in SCALA.items()}
# fine della curva per classe (utente, 23/10/2026): gli uccelli sopra i 96 px sono troppo pochi, stanno solo nel train
# e la loro curva si ferma a 96
CUT = {"airplane": SMAX, "bird": 96, "drone": SMAX, "helicopter": SMAX}
BINS = np.geomspace(2, 640, 70)


def sizes() -> dict[str, np.ndarray]:
    """sqrt(w*h) in px per classe, box >= MIN."""
    out = defaultdict(list)

    def add(t: Path) -> None:
        for ln in t.read_text().splitlines():
            if ln.strip():
                c, _, _, w, h = ln.split()
                out[NAMES[int(c)]].append(np.sqrt(float(w) * W * float(h) * H))

    sv = DATA / "svanstrom"  # frame tenuti dalla deduplica, label di 4_box/manuale (come svanstrom.count_cells)
    for r in csv.DictReader(open(sv / "6_dedup" / "tenuti.csv")):
        clip, v = r["clip"], r["clip"].split("/")[0]
        t = sv / "4_box" / "manuale" / clip.replace("/", "_") / f"{v}_f{int(r['frame']):05d}.txt"
        if t.exists():
            add(t)
    for s in ["fred", "dvids", "fbd_sv", "anti_uav"]:  # come fred.count
        src = DATA / s
        ses = {r["clip"] for r in csv.DictReader(open(src / "6_sessioni" / "sessioni.csv"))}
        f = src / "5_box" / "preselezione" / "esclusi.txt"
        esclusi = {ln.split()[0] for ln in open(f) if ln.strip()} if f.exists() else set()
        for d in sorted((src / "5_box" / "proposte").iterdir()):
            if f"{d.name}/s0" not in ses:
                continue
            man = src / "5_box" / "manuale" / d.name
            for t in d.glob("*.txt"):
                if t.stem in esclusi:
                    continue
                t = man / t.name if man.exists() else src / "5_box" / "preselezione" / d.name / t.name
                if t.exists():
                    add(t)
    return {c: np.array([v for v in out[c] if v >= MIN]) for c in CLASSI}


def attese(x: np.ndarray, s: float, edges) -> np.ndarray:
    """Istanze attese per intervallo con lo scale di Ultralytics: un box di grandezza x diventa uniforme
    in [x(1-s), x(1+s)], quindi P(intervallo) = sovrapposizione / (2 s x). Esatto, senza campionamento."""
    edges = np.asarray(edges, float)
    if s == 0:
        return np.histogram(x, edges)[0].astype(float)
    lo, hi = (x * (1 - s))[:, None], (x * (1 + s))[:, None]
    ov = np.clip(np.minimum(hi, edges[None, 1:]) - np.maximum(lo, edges[None, :-1]), 0, None)
    return (ov / (hi - lo)).sum(0)


def ideale(n: int, alpha: float, edges, smax: float = SMAX) -> np.ndarray:
    """Istanze per intervallo con densita' in log s proporzionale a s^-alpha su [SMIN, smax], totale n."""
    e = np.clip(np.asarray(edges, float), SMIN, smax)
    F = lambda v: -v ** -alpha / alpha  # noqa: E731  primitiva di s^-alpha d(ln s)
    return n * (F(e[1:]) - F(e[:-1])) / (F(smax) - F(SMIN))


def uniforme(n: float, edges, smax: float = SMAX) -> np.ndarray:
    """Istanze per intervallo con densita' costante in log s su [SMIN, smax], totale n (obiettivo del test)."""
    e = np.log(np.clip(np.asarray(edges, float), SMIN, smax))
    return n * np.diff(e) / np.log(smax / SMIN)


def curva(n: int, alpha: float, smax: float = SMAX) -> tuple[np.ndarray, np.ndarray]:
    """La distribuzione ideale come linea continua (s, istanze attese per intervallo di BINS), totale n."""
    s = np.geomspace(SMIN, smax, 400)
    dlog = np.log(BINS[1] / BINS[0])  # BINS a passo costante in log
    return s, n * s ** -alpha * dlog * alpha / (SMIN ** -alpha - smax ** -alpha)


def _assi(ax, c: str, n: int) -> None:
    from matplotlib.ticker import NullLocator

    ax.axvspan(2, 8, color="0.92", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(2, 512)
    ticks = [4, 8, 16, 32, 64, 128, 256, 512]  # potenze di 2
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())  # le tacche secondarie in base 10 non cadono sulle potenze di 2
    ax.set_title(f"{c} ({n:,} box)".replace(",", "."), fontsize=8)


def fig_attuale(size: dict, plt) -> Path:
    f, axs = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.72), sharex=True, sharey=True)
    hs = {c: attese(size[c], 0, BINS) for c in CLASSI}
    top = max(max(hs[c].max(), curva(TOT[c], ALPHA[c], CUT[c])[1].max()) for c in CLASSI) * 1.3  # stesso asse per le 4
    for ax, c in zip(axs.flat, CLASSI):
        x = size[c]
        ax.bar(BINS[:-1], hs[c], width=np.diff(BINS), align="edge", color=_P[0], alpha=.85, lw=0)
        for e in EDGES[:-1]:  # senza la linea a 96 px: l'asse va in potenze di 2 (utente)
            ax.axvline(e, color="0.55", lw=.6, ls="--")
        ax.plot(*curva(TOT[c], ALPHA[c], CUT[c]), color="k", lw=.8)
        for j, (a, b) in enumerate(BANDS):
            n = int(((x >= a) & (x < b)).sum())
            ax.text(np.sqrt(a * min(b, 300)), top * (.97 if j % 2 == 0 else .87), f"{n:,}".replace(",", "."),
                    ha="center", va="top", fontsize=6.5, color=_P[7] if n < TARGET else "0.2",
                    weight="bold" if n < TARGET else "normal")
        ax.set_ylim(0, top)
        _assi(ax, c, len(x))
    for ax in axs[1]:
        ax.set_xlabel(r"grandezza $\sqrt{w\,h}$ (px)")
    for ax in axs[:, 0]:
        ax.set_ylabel("istanze per intervallo")
    f.suptitle(f"Box >= 8 px dopo la revisione CVAT — in rosso le fasce sotto {TARGET:,}".replace(",", ".")
               + "; in nero l'ideale (curva alzata per classe)", fontsize=8.5)
    f.tight_layout()
    return save(f, OUT / "distribuzioni", pdf=False)


def fig_scale(size: dict, plt) -> Path:
    f, axs = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.78), sharex=True, sharey=True)
    curve = [(0, _P[0], "senza scale"), (.25, _P[1], "scale 0,25 (×0,75–1,25)"), (.5, _P[2], "scale 0,5 (×0,5–1,5)")]
    ymax = 0
    for ax, c in zip(axs.flat, CLASSI):
        x = size[c]
        for s, col, lab in curve:
            h = attese(x, s, BINS)
            ax.stairs(h, BINS, color=col, lw=1.2, label=lab, fill=s == 0, alpha=.35 if s == 0 else 1)
            ymax = max(ymax, h.max())
        sc, hi = curva(TOT[c], ALPHA[c], CUT[c])
        ax.plot(sc, hi, color="k", lw=.8, label="ideale (alzata per classe)")
        ymax = max(ymax, hi.max())
        ax.text(.98, .95, f"ideale ∝ $s^{{-{ALPHA[c]:g}}}$, ×{SCALA[c]:g}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        for e in EDGES[:-1]:
            ax.axvline(e, color="0.8", lw=.5)
        _assi(ax, c, len(x))
    axs[0, 0].set_ylim(0, ymax * 1.1)  # sharey: stesso asse per le 4
    for ax in axs[1]:
        ax.set_xlabel(r"grandezza $\sqrt{w\,h}$ (px)")
    for ax in axs[:, 0]:
        ax.set_ylabel("istanze attese per epoca")
    hs, ls = axs[0, 0].get_legend_handles_labels()
    f.legend(hs, ls, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(.5, -.01))
    f.suptitle("Box >= 8 px: attuale, con scale (uniforme in ±s·px) e distribuzione ideale (alzata per classe)",
               fontsize=8.5)
    f.tight_layout(rect=(0, .04, 1, 1))
    return save(f, OUT / "distribuzioni_scale", pdf=False)


def main() -> int:
    import argparse

    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.parse_args()
    plt.style.use(STYLE)
    size = sizes()
    print(fig_attuale(size, plt))
    print(fig_scale(size, plt))
    print("ideale droni per fascia:" + "".join(f"{round(v):>8}" for v in ideale(TOT["drone"], ALPHA["drone"], [8, 16, 32, 64, 96, 1e6])))
    print(f"{'':<11}{'s':>5}" + "".join(f"{a:>8}" for a in ["<8", "8-16", "16-32", "32-64", "64-96", ">96"]))
    for c in CLASSI:
        for s in (0, .25, .5):
            n = attese(size[c], s, [0, 8, 16, 32, 64, 96, 1e6])
            print(f"{c:<11}{s:>5}" + "".join(f"{round(v):>8}" for v in n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
