"""Style and saving shared by all plots (report, experiments, notebooks).

    import matplotlib.pyplot as plt
    from tools.plots import STYLE, WIDTH, color, save

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    ax.bar(qs, vals, color=[color(q) for q in qs])
    save(fig, out / "latency")          # latency.png + latency.pdf

The color follows the entity, not the position: int8 is orange in every figure
of the thesis, even when a plot lacks fp16 or fp32.
"""

from __future__ import annotations

from pathlib import Path

STYLE = Path(__file__).with_name("plots.mplstyle")

#: \textwidth of an A4 thesis (15 cm); half width for side-by-side figures
WIDTH = 15 / 2.54
# palette slots in plots.mplstyle
_P = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLORS = {
    # split
    "train": _P[0], "val": _P[1], "test": _P[2], "calib": _P[3],
    # precision
    "fp32": _P[0], "fp16": _P[2], "int8": _P[1],
    # backend
    "onnxruntime": _P[0], "onnxruntime_py": _P[4], "openvino": _P[1],
    "openvino_qdq": _P[7], "tensorrt": _P[2],
    "executorch": _P[3], "axelera": _P[6],
    # board
    "wks4_rtx6000": _P[0], "jetson_orin": _P[2], "rpi5": _P[1], "rpi5_axelera": _P[6],
}


def color(key, i: int = 0) -> str:
    """Color of the entity `key`; for new keys the palette slot `i`."""
    return COLORS.get(str(key), _P[i % len(_P)])


def save(fig, path: Path, pdf: bool = True) -> Path:
    """PNG (preview, report) and PDF (LaTeX) with the same name; closes the figure."""
    import matplotlib.pyplot as plt

    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf") if pdf else ("png",):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return path.with_suffix(".png")
