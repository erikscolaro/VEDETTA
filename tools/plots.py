"""Style and saving shared by all plots (report, experiments, notebooks).

    import matplotlib.pyplot as plt
    from tools.plots import WIDTH, color, save, use

    use("it")                           # style of the thesis, decimal comma
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    ax.bar(qs, vals, color=[color(q) for q in qs])
    save(fig, out / "latency")          # latency.png + latency.pdf

The color follows the entity, not the position: int8 is orange in every figure
of the thesis, even when a plot lacks fp16 or fp32.
"""

from __future__ import annotations

from pathlib import Path

STYLE = Path(__file__).with_name("plots.mplstyle")
#: every figure of the scripts also ends up here (user, 26/10/2026), whatever folder it is saved to, as
#: `<lang>_<name>` (it_distribution, en_distribution): the language of the last use()
FIG = Path(__file__).resolve().parents[1] / "docs" / "figure"
_LOCALE = "C"

#: full width of a two-column page (figure*, 7.16 in as IEEE): the thesis and the papers are on two columns (user,
#: 26/10/2026); COLUMN for a figure inside one column
WIDTH = 7.16
COLUMN = 3.5
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


def use(lang: str = "en") -> None:
    """The style of `plots.mplstyle`; with `lang` "it" the numeric axes use the decimal comma (it_IT locale).
    Call it before creating the figure: the tick formatters read the setting when they are made."""
    import locale

    import matplotlib.pyplot as plt

    global _LOCALE
    plt.style.use(STYLE)
    _LOCALE = "it_IT.UTF-8" if lang == "it" else "C"
    locale.setlocale(locale.LC_NUMERIC, _LOCALE)  # again in save(): an interactive backend (Tk) resets it
    plt.rcParams["axes.formatter.use_locale"] = lang == "it"


def color(key, i: int = 0) -> str:
    """Color of the entity `key`; for new keys the palette slot `i`."""
    return COLORS.get(str(key), _P[i % len(_P)])


def save(fig, path: Path, pdf: bool = True, name: str | None = None) -> Path:
    """PNG (preview, report) and PDF (LaTeX) with the same name; closes the figure. In `docs/figure` as
    `<lang>_<name>` (`name`: default the stem of `path`): a `path` there is only saved with the prefix, a `path`
    elsewhere is also copied there."""
    import locale
    import shutil

    import matplotlib.pyplot as plt

    locale.setlocale(locale.LC_NUMERIC, _LOCALE)  # the tick labels are formatted while saving
    path = Path(path).with_suffix("")
    doc = FIG / f"{'it' if _LOCALE.startswith('it') else 'en'}_{name or path.name}"
    if path.parent.resolve() == FIG:
        path = doc
    path.parent.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf") if pdf else ("png",):
        fig.savefig(path.with_suffix(f".{ext}"))
        if path != doc:
            shutil.copy(path.with_suffix(f".{ext}"), doc.with_suffix(f".{ext}"))
    plt.close(fig)
    return path.with_suffix(".png")
