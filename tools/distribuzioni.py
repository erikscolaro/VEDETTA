#!/usr/bin/env python3
"""Distribuzione continua della grandezza sqrt(w*h) per classe sulle label finali (stessi frame di
tools/buchi.py: Svanström sui frame tenuti, aggiunte da 5_box/manuale), box >= 8 px in origine.

Due figure in docs/figure/ (considerazioni in docs/distribuzioni.md):
- distribuzioni.png: istogramma per classe con le istanze per fascia (rosso: sotto il minimo per cella);
- distribuzioni_scale.png: la stessa con lo scale di Ultralytics (s = 0,5 e 0,75) calcolato in modo
  esatto e la distribuzione ideale a parita' di totale.

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
ALPHA = {"drone": 0.3, "airplane": 0.45, "bird": 0.45, "helicopter": 0.45}
SMIN, SMAX = 8, 320
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


def ideale(n: int, alpha: float, edges) -> np.ndarray:
    """Istanze per intervallo con densita' in log s proporzionale a s^-alpha su [SMIN, SMAX], totale n."""
    e = np.clip(np.asarray(edges, float), SMIN, SMAX)
    F = lambda v: -v ** -alpha / alpha  # noqa: E731  primitiva di s^-alpha d(ln s)
    return n * (F(e[1:]) - F(e[:-1])) / (F(SMAX) - F(SMIN))


def _assi(ax, c: str, n: int) -> None:
    ax.axvspan(2, 8, color="0.92", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(2, 640)
    ax.set_xticks([4, 8, 16, 32, 64, 96, 256, 640])
    ax.set_xticklabels(["4", "8", "16", "32", "64", "96", "256", "640"])
    ax.set_title(f"{c} ({n:,} box)".replace(",", "."), fontsize=8)


def fig_attuale(size: dict, plt) -> Path:
    f, axs = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.72), sharex=True)
    for ax, c in zip(axs.flat, CLASSI):
        x = size[c]
        h = attese(x, 0, BINS)
        ax.bar(BINS[:-1], h, width=np.diff(BINS), align="edge", color=_P[0], alpha=.85, lw=0)
        for e in EDGES:
            ax.axvline(e, color="0.55", lw=.6, ls="--")
        top = h.max() * 1.38
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
    f.suptitle(f"Box >= 8 px dopo la revisione CVAT — in rosso le fasce sotto {TARGET:,}".replace(",", "."),
               fontsize=8.5)
    f.tight_layout()
    return save(f, OUT / "distribuzioni", pdf=False)


def fig_scale(size: dict, plt) -> Path:
    f, axs = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.78), sharex=True)
    curve = [(0, _P[0], "senza scale"), (.5, _P[1], "scale 0,5 (×0,5–1,5)"), (.75, _P[2], "scale 0,75 (×0,25–1,75)")]
    for ax, c in zip(axs.flat, CLASSI):
        x = size[c]
        ymax = 0
        for s, col, lab in curve:
            h = attese(x, s, BINS)
            ax.stairs(h, BINS, color=col, lw=1.2, label=lab, fill=s == 0, alpha=.35 if s == 0 else 1)
            ymax = max(ymax, h.max())
        hi = ideale(len(x), ALPHA[c], BINS)
        ax.stairs(hi, BINS, color="k", lw=1.4, ls="--", label="ideale, stesso totale")
        ax.text(.98, .95, f"ideale ∝ $s^{{-{ALPHA[c]:g}}}$", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        for e in EDGES:
            ax.axvline(e, color="0.8", lw=.5)
        ax.set_ylim(0, max(ymax, hi.max()) * 1.1)
        _assi(ax, c, len(x))
    for ax in axs[1]:
        ax.set_xlabel(r"grandezza $\sqrt{w\,h}$ (px)")
    for ax in axs[:, 0]:
        ax.set_ylabel("istanze attese per epoca")
    hs, ls = axs[0, 0].get_legend_handles_labels()
    f.legend(hs, ls, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(.5, -.01))
    f.suptitle("Box >= 8 px: attuale, con scale (uniforme in ±s·px) e distribuzione ideale a parità di totale",
               fontsize=8.5)
    f.tight_layout(rect=(0, .04, 1, 1))
    return save(f, OUT / "distribuzioni_scale", pdf=False)


def main() -> int:
    import matplotlib.pyplot as plt

    plt.style.use(STYLE)
    size = sizes()
    print(fig_attuale(size, plt))
    print(fig_scale(size, plt))
    print(f"{'':<11}{'s':>5}" + "".join(f"{a:>8}" for a in ["<8", "8-16", "16-32", "32-64", "64-96", ">96"]))
    for c in CLASSI:
        for s in (0, .5, .75):
            n = attese(size[c], s, [0, 8, 16, 32, 64, 96, 1e6])
            print(f"{c:<11}{s:>5}" + "".join(f"{round(v):>8}" for v in n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
