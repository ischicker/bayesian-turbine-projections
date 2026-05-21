"""Paper-ready plotting utilities and Applied Energy figure settings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


MM_PER_INCH = 25.4
APPLIED_ENERGY_FULL_WIDTH_IN = 190 / MM_PER_INCH


def set_paper_style() -> None:
    """Apply shared matplotlib settings for Applied Energy figures."""

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    """Save a figure as PDF and 300 dpi PNG."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
