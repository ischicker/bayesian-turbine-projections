"""Run prior sensitivity analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.sensitivity import (
    PRIOR_SET_LABELS,
    convergence_table,
    run_prior_sensitivity,
    sensitivity_summary,
)
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

PRIOR_COLORS = {
    "A_informative": "#1D3557",
    "B_weakly_informative": "#E76F51",
    "C_diffuse_uniform": "#2A9D8F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-run fits even if sensitivity posteriors exist.")
    return parser.parse_args()


def plot_prior_sensitivity(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    x_base = np.array([0, 1], dtype=float)
    offsets = {"A_informative": -0.18, "B_weakly_informative": 0.0, "C_diffuse_uniform": 0.18}
    for row, region in enumerate(config.REGIONS):
        for col, metric in enumerate(config.METRICS):
            ax = axes[row, col]
            subset = predictions.loc[(predictions["region"] == region) & (predictions["metric"] == metric)]
            for prior_set in config.PRIOR_SETS:
                values = subset.loc[subset["prior_set"] == prior_set].sort_values("year")
                x = x_base + offsets[prior_set]
                y = values["median"].to_numpy(dtype=float)
                yerr = np.vstack(
                    [
                        y - values["q2_5"].to_numpy(dtype=float),
                        values["q97_5"].to_numpy(dtype=float) - y,
                    ]
                )
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="o",
                    capsize=3,
                    markersize=4,
                    color=PRIOR_COLORS[prior_set],
                    label=PRIOR_SET_LABELS[prior_set] if row == 0 and col == 0 else None,
                )
            ax.set_xticks(x_base)
            ax.set_xticklabels(["2030", "2055"])
            if row == 0:
                ax.set_title(metric.replace("_", " ").title())
            if col == 0:
                ax.set_ylabel(region)
            ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"prior_sensitivity_projection.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(summary: pd.DataFrame) -> None:
    sensitive = summary.loc[summary["prior_sensitive"]].copy()
    lines = ["Prior sensitivity summary", ""]
    if sensitive.empty:
        lines.append("No region-metric combination exceeded the sensitivity threshold.")
    else:
        lines.append("Prior-sensitive combinations using >10% max relative posterior-median deviation:")
        for _, row in sensitive.iterrows():
            lines.append(
                f"- {row['region']} {row['metric']}: "
                f"{row['max_relative_deviation_pct']:.1f}% max relative deviation "
                f"({row['max_abs_deviation']:.2f} absolute)"
            )
    robust = summary.loc[~summary["prior_sensitive"]]
    if not robust.empty:
        lines.extend(["", "Robust combinations:"])
        for _, row in robust.iterrows():
            lines.append(f"- {row['region']} {row['metric']}: {row['max_relative_deviation_pct']:.1f}%")
    (config.TABLES_DIR / "prior_sensitivity_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_logging()
    ensure_directories([config.TABLES_DIR, config.FIGURES_DIR, config.POSTERIORS_DIR])
    args = parse_args()
    predictions = run_prior_sensitivity(force=args.force)
    convergence = convergence_table(predictions)
    summary = sensitivity_summary(convergence)
    plot_prior_sensitivity(predictions)
    write_summary_text(summary)
    LOGGER.info("=== Prior Sensitivity Summary ===\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
