"""Run benchmark extrapolation models."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.hindcast import MODEL_ORDER, load_region_data, run_hindcast_benchmarks
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

MODEL_LABELS = {
    "linear": "Linear",
    "quadratic": "Quadratic",
    "mle_logistic": "MLE logistic",
    "bayesian_logistic": "Bayesian logistic",
}

MODEL_STYLES = {
    "linear": {"color": "#6C757D", "linestyle": "--"},
    "quadratic": {"color": "#F4A261", "linestyle": "-."},
    "mle_logistic": {"color": "#7B2CBF", "linestyle": ":"},
    "bayesian_logistic": {"color": "#111111", "linestyle": "-"},
}


def write_latex_table(results: pd.DataFrame) -> None:
    table = results.copy()
    metric_names = {
        "hub_height": "Hub height",
        "rotor_diameter": "Rotor diameter",
        "specific_power": "Specific power",
    }
    table["metric"] = table["metric"].map(metric_names)
    table["model"] = table["model"].map(MODEL_LABELS)
    columns = [
        "region",
        "metric",
        "split_year",
        "model",
        "rmse",
        "mae",
        "crps",
        "coverage_95",
        "interval_width_95",
        "waic",
        "loo",
    ]
    formatted = table[columns].sort_values(["split_year", "region", "metric", "model"]).round(3)
    header = " & ".join(formatted.columns).replace("_", "\\_")
    body_rows = []
    for _, row in formatted.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("--")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        body_rows.append(" & ".join(values) + r" \\")
    latex = "\n".join(
        [
            r"\begin{tabular}{lllllllllll}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *body_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (config.TABLES_DIR / "benchmark_comparison_table.tex").write_text(latex, encoding="utf-8")


def plot_hindcast(predictions: pd.DataFrame) -> None:
    split = config.PRIMARY_HINDCAST_TRAIN_END
    pred = predictions.loc[predictions["split_year"] == split].copy()
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False)
    for row, region in enumerate(config.REGIONS):
        data = load_region_data(region)
        for col, metric in enumerate(config.METRICS):
            ax = axes[row, col]
            column = config.METRIC_COLUMNS[metric]
            annual = (
                data.groupby("year", as_index=False)[column]
                .median()
                .rename(columns={column: "observed_median"})
            )
            ax.scatter(
                annual["year"],
                annual["observed_median"],
                s=16,
                color=config.REGION_COLORS[region],
                alpha=0.7,
                label="Annual median" if row == 0 and col == 0 else None,
            )
            subset = pred.loc[(pred["region"] == region) & (pred["metric"] == metric)]
            for model in MODEL_ORDER:
                model_data = subset.loc[subset["model"] == model].sort_values("year")
                if model_data.empty:
                    continue
                style = MODEL_STYLES[model]
                ax.plot(
                    model_data["year"],
                    model_data["prediction"],
                    linewidth=1.8,
                    label=MODEL_LABELS[model] if row == 0 and col == 0 else None,
                    **style,
                )
                if model == "bayesian_logistic" and {"q2_5", "q97_5"}.issubset(model_data.columns):
                    ax.fill_between(
                        model_data["year"].to_numpy(dtype=float),
                        model_data["q2_5"].to_numpy(dtype=float),
                        model_data["q97_5"].to_numpy(dtype=float),
                        color="black",
                        alpha=0.12,
                        linewidth=0,
                    )
            ax.axvline(split + 0.5, color="#333333", linewidth=0.8, alpha=0.6)
            if row == 0:
                ax.set_title(metric.replace("_", " ").title())
            if col == 0:
                ax.set_ylabel(region)
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"hindcast_benchmark.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(calibration: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], color="#333333", linewidth=1.0, linestyle="--")
    markers = {"AT": "o", "DE": "s", "US": "^"}
    for region in config.REGIONS:
        subset = calibration.loc[calibration["region"] == region]
        grouped = subset.groupby("nominal_coverage", as_index=False)["empirical_coverage"].mean()
        ax.plot(
            grouped["nominal_coverage"],
            grouped["empirical_coverage"],
            marker=markers[region],
            color=config.REGION_COLORS[region],
            label=region,
            linewidth=1.8,
        )
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"bayesian_calibration.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    ensure_directories([config.TABLES_DIR, config.FIGURES_DIR])
    results, predictions, calibration = run_hindcast_benchmarks(config.HINDCAST_SPLITS)
    results = results.sort_values(["split_year", "region", "metric", "model"]).reset_index(drop=True)
    results.to_csv(config.TABLES_DIR / "benchmark_comparison_table.csv", index=False)
    predictions.to_csv(config.TABLES_DIR / "benchmark_hindcast_predictions.csv", index=False)
    calibration.to_csv(config.TABLES_DIR / "benchmark_bayesian_calibration.csv", index=False)
    write_latex_table(results)
    plot_hindcast(predictions)
    plot_calibration(calibration)
    LOGGER.info("=== Benchmark Summary ===\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()
