#!/usr/bin/env python3
"""Continuous distribution of the size sqrt(w*h) per class on the final labels
(Svanström on the kept frames, the other sources from 5_box/manuale), boxes >= 8 px at origin.

Two figures in docs/figure/ (rationale in README, Rules), with the ideal curve (TOTAL raised per class with SCALE)
as a continuous line and the same vertical axis in the 4 plots:
- it_size_review.png: histogram per class with the instances per band (red: below the per-cell minimum);
- it_size_scale.png: the same with the Ultralytics scale (s = 0.25 and 0.5) computed exactly.

    python tools/distributions.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import INPUT, NAMES  # noqa: E402
from tools.plots import _P, WIDTH, save, use  # noqa: E402

OUT = ROOT / "docs" / "figure"
W, H = 640, 512
MIN = 8  # boxes under 8 px at origin are excluded (user, 22/10/2026)
TARGET = round(2000 / 0.7)  # per-cell minimum (README, Rules)
EDGES = [8, 16, 32, 64, 96]
BANDS = [(8, 16), (16, 32), (32, 64), (64, 96), (96, 1e9)]
CLASSES = ["airplane", "bird", "drone", "helicopter"]
# ideal distribution: density in log s proportional to s^-alpha on [SMIN, SMAX] (README, Rules)
ALPHA = dict.fromkeys(["airplane", "bird", "drone", "helicopter"], 0.3)  # same curve for all (user, 23/10/2026)
SMIN, SMAX = 8, 320
TOTAL = round(10_000 / 0.7)  # boxes per class of the ideal curve: 10,000 in train, split 70/20/10 (README, Rules)
# raised curve (user, 23/10/2026): at least 10,000 instances per class in train, 2,000 in val, 1,000 in test; the same
# curve shifted up acts as the cap (above) and as the mosaics target (below), so the shape stays s^-alpha
SCALE = dict.fromkeys(["airplane", "bird", "drone", "helicopter"], 1.2)
TOT = {c: TOTAL * k for c, k in SCALE.items()}
# end of the curve per class: on 23/10/2026 the birds stopped at 96 px (too few above); from 28/10/2026 the Open
# Images birds in flight reach 256 px and the bird curve is the same as the others (user)
CUT = {"airplane": SMAX, "bird": SMAX, "drone": SMAX, "helicopter": SMAX}
BINS = np.geomspace(2, 640, 70)


def sizes() -> dict[str, np.ndarray]:
    """sqrt(w*h) in px per class, boxes >= MIN."""
    out = defaultdict(list)

    def add(t: Path) -> None:
        for ln in t.read_text().splitlines():
            if ln.strip():
                c, _, _, w, h = ln.split()
                out[NAMES[int(c)]].append(np.sqrt(float(w) * W * float(h) * H))

    sv = INPUT / "svanstrom"  # frames kept by the dedup, labels from 4_box/manuale (as svanstrom.count_cells)
    for r in csv.DictReader(open(sv / "6_dedup" / "tenuti.csv")):
        clip, v = r["clip"], r["clip"].split("/")[0]
        t = sv / "4_box" / "manuale" / clip.replace("/", "_") / f"{v}_f{int(r['frame']):05d}.txt"
        if t.exists():
            add(t)
    for s in ["fred", "dvids", "fbd_sv", "anti_uav"]:  # as fred.count
        src = INPUT / s
        sessions = {r["clip"] for r in csv.DictReader(open(src / "6_sessioni" / "sessioni.csv"))}
        f = src / "5_box" / "preselezione" / "esclusi.txt"
        excluded = {ln.split()[0] for ln in open(f) if ln.strip()} if f.exists() else set()
        for d in sorted((src / "5_box" / "proposte").iterdir()):
            if f"{d.name}/s0" not in sessions:
                continue
            man = src / "5_box" / "manuale" / d.name
            for t in d.glob("*.txt"):
                if t.stem in excluded:
                    continue
                t = man / t.name if man.exists() else src / "5_box" / "preselezione" / d.name / t.name
                if t.exists():
                    add(t)
    return {c: np.array([v for v in out[c] if v >= MIN]) for c in CLASSES}


def expected(x: np.ndarray, s: float, edges) -> np.ndarray:
    """Expected instances per interval with the Ultralytics scale: a box of size x becomes uniform
    in [x(1-s), x(1+s)], so P(interval) = overlap / (2 s x). Exact, no sampling."""
    edges = np.asarray(edges, float)
    if s == 0:
        return np.histogram(x, edges)[0].astype(float)
    lo, hi = (x * (1 - s))[:, None], (x * (1 + s))[:, None]
    ov = np.clip(np.minimum(hi, edges[None, 1:]) - np.maximum(lo, edges[None, :-1]), 0, None)
    return (ov / (hi - lo)).sum(0)


def ideal(n: int, alpha: float, edges, smax: float = SMAX) -> np.ndarray:
    """Instances per interval with density in log s proportional to s^-alpha on [SMIN, smax], total n."""
    e = np.clip(np.asarray(edges, float), SMIN, smax)
    F = lambda v: -v ** -alpha / alpha  # noqa: E731  antiderivative of s^-alpha d(ln s)
    return n * (F(e[1:]) - F(e[:-1])) / (F(smax) - F(SMIN))


def uniform(n: float, edges, smax: float = SMAX) -> np.ndarray:
    """Instances per interval with constant density in log s on [SMIN, smax], total n (test target)."""
    e = np.log(np.clip(np.asarray(edges, float), SMIN, smax))
    return n * np.diff(e) / np.log(smax / SMIN)


def curve(n: int, alpha: float, smax: float = SMAX) -> tuple[np.ndarray, np.ndarray]:
    """The ideal distribution as a continuous line (s, expected instances per interval of BINS), total n."""
    s = np.geomspace(SMIN, smax, 400)
    dlog = np.log(BINS[1] / BINS[0])  # BINS with constant step in log
    return s, n * s ** -alpha * dlog * alpha / (SMIN ** -alpha - smax ** -alpha)


ORDER = ["drone", "airplane", "helicopter", "bird"]  # class order of the dataset and of the other figures
CLASS_IT = {"drone": "Drone", "airplane": "Aereo", "helicopter": "Elicottero", "bird": "Uccello"}


def _axes(ax, k: int, c: str, n: int) -> None:
    from matplotlib.ticker import NullLocator

    ax.axvspan(2, 8, color="0.93", lw=0, zorder=0, label="Escluse (< 8 px)")
    ax.set_xscale("log")
    ax.set_xlim(2, 512)
    ticks = [4, 8, 16, 32, 64, 128, 256, 512]  # powers of 2
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())  # the base-10 minor ticks do not fall on the powers of 2
    ax.set_title(f"({'abcd'[k]}) {CLASS_IT[c]}, " + f"{n:,}".replace(",", ".") + " box")


def _labels(f, axs, y: str) -> None:
    for ax in axs[1]:
        ax.set_xlabel("Lato della box √(w·h) [px]")
    for ax in axs[:, 0]:
        ax.set_ylabel(y)
    hs, ls = axs[0, 0].get_legend_handles_labels()
    f.legend(hs, ls, loc="outside lower center", ncol=len(hs))


def fig_current(size: dict, plt) -> Path:
    """Sizes after the CVAT review, with the boxes per band (red: under the per-cell minimum) and the ideal curve."""
    f, axs = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
    hs = {c: expected(size[c], 0, BINS) for c in CLASSES}
    top = max(max(hs[c].max(), curve(TOT[c], ALPHA[c], CUT[c])[1].max()) for c in CLASSES) * 1.3  # same axis for all 4
    for k, (ax, c) in enumerate(zip(axs.flat, ORDER)):
        x = size[c]
        _axes(ax, k, c, len(x))
        ax.bar(BINS[:-1], hs[c], width=np.diff(BINS), align="edge", color=_P[0], alpha=.85, lw=0, label="Box")
        for e in EDGES[:-1]:  # without the line at 96 px: the axis is in powers of 2 (user)
            ax.axvline(e, color="0.55", lw=.5, ls="--")
        ax.plot(*curve(TOT[c], ALPHA[c], CUT[c]), color="0.1", lw=1, label="Ideale (alzata per classe)")
        for j, (a, b) in enumerate(BANDS):
            n = int(((x >= a) & (x < b)).sum())
            ax.text(np.sqrt(a * min(b, 300)), top * (.97 if j % 2 == 0 else .87), f"{n:,}".replace(",", "."),
                    ha="center", va="top", fontsize=7, color=_P[7] if n < TARGET else "0.2",
                    weight="bold" if n < TARGET else "normal")
        ax.set_ylim(0, top)
    axs[0, 0].plot([], [], ls="none", marker="$n$", color=_P[7], label="Fascia sotto il minimo per cella")
    _labels(f, axs, "Istanze per intervallo")
    return save(f, OUT / "size_review", pdf=False)


def fig_scale(size: dict, plt) -> Path:
    """The same sizes with the Ultralytics scale augmentation (s = 0.25 and 0.5, uniform in ±s), computed exactly."""
    f, axs = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
    scales = [(0, _P[0], "Senza scale"), (.25, _P[1], "Scale 0,25 (×0,75–1,25)"), (.5, _P[2], "Scale 0,5 (×0,5–1,5)")]
    ymax = 0
    for k, (ax, c) in enumerate(zip(axs.flat, ORDER)):
        x = size[c]
        _axes(ax, k, c, len(x))
        for s, col, lab in scales:
            h = expected(x, s, BINS)
            ax.stairs(h, BINS, color=col, lw=0 if s == 0 else 1.2, label=lab, fill=s == 0, alpha=.35 if s == 0 else 1)
            ymax = max(ymax, h.max())
        sc, hi = curve(TOT[c], ALPHA[c], CUT[c])
        ax.plot(sc, hi, color="0.1", lw=1, label="Ideale (alzata per classe)")
        ymax = max(ymax, hi.max())
    axs[0, 0].set_ylim(0, ymax * 1.1)  # sharey: same axis for all 4
    _labels(f, axs, "Istanze attese per epoca")
    return save(f, OUT / "size_scale", pdf=False)


def main() -> int:
    import argparse

    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()
    use("it")
    size = sizes()
    print(fig_current(size, plt))
    print(fig_scale(size, plt))
    print("ideale droni per fascia:" + "".join(f"{round(v):>8}" for v in ideal(TOT["drone"], ALPHA["drone"], [8, 16, 32, 64, 96, 1e6])))
    print(f"{'':<11}{'s':>5}" + "".join(f"{a:>8}" for a in ["<8", "8-16", "16-32", "32-64", "64-96", ">96"]))
    for c in CLASSES:
        for s in (0, .25, .5):
            n = expected(size[c], s, [0, 8, 16, 32, 64, 96, 1e6])
            print(f"{c:<11}{s:>5}" + "".join(f"{round(v):>8}" for v in n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
