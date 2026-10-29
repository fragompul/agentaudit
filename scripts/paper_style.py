# ruff: noqa: E501
"""One visual identity for every figure in the paper.

Rules: figures are drawn at the text width (6.5 in) so that point sizes are final sizes; body text 8 pt,
ticks and legends 7 pt, nothing below 7 pt; one colour per *meaning* across all figures (a method or an
agent keeps its colour everywhere); horizontal hairline grid only; no top/right spines; direct panel tags.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

WIDTH = 6.5  # inches, equals \textwidth
INK, MUTED, GRID, PAPER = "#1c2233", "#6b7280", "#e6e8ee", "#ffffff"

# semantic palette: warm = the shortcut that fails, cool = the task-aware procedure
CORAL, INDIGO, TEAL, AMBER, VIOLET, SLATE = (
    "#ef5a3c",
    "#3547d4",
    "#0e9f8e",
    "#e3a21a",
    "#8b4fd1",
    "#7a8394",
)
BLUE, ORANGE, AQUA, GREY = INDIGO, CORAL, TEAL, SLATE  # legacy names used by the asset scripts

METHOD = {  # interval / test procedures
    "wilson_pooled": CORAL,
    "clopper_pearson_pooled": AMBER,
    "bootstrap_tasks": INDIGO,
    "bootstrap_tasks_bca": "#7f8df0",
    "beta_binomial_profile": TEAL,
    "bootstrap_two_stage": AMBER,
    "bootstrap_runs_only": SLATE,
}


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "sans",
            "mathtext.it": "sans:italic",
            "mathtext.bf": "sans:bold",
            "font.size": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 5,
            "axes.labelsize": 8,
            "axes.labelpad": 3,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.8,
            "lines.solid_capstyle": "round",
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.columnspacing": 1.4,
            "legend.borderaxespad": 0.2,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.w_pad": 0.06,
            "figure.constrained_layout.h_pad": 0.06,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def tag(axes) -> None:  # noqa: ANN001
    """Panel tags are intentionally not drawn (captions say left/right)."""


def refline(ax, y: float, label: str | None = None) -> None:  # noqa: ANN001
    """Reference line (nominal level) in ink, dashed, drawn under the data."""
    ax.axhline(y, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate(
            label,
            xy=(1, y),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=INK,
        )
