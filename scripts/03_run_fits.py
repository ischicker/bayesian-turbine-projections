"""Run Bayesian logistic fits for all configured regions and metrics."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.bayesian_model import (
    build_logistic_decay_model,
    build_logistic_growth_model,
    compute_derived_capacity,
    convergence_summary,
    fit_model,
    posterior_predictive_samples,
    predict_future,
    warn_on_convergence,
)
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=config.REGIONS + ["all"], default="all")
    parser.add_argument("--metric", choices=config.METRICS, default="hub_height")
    parser.add_argument("--quick", action="store_true", help="Use MCMC_QUICK settings.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing posterior files.")
    parser.add_argument(
        "--subsample-sensitivity",
        action="store_true",
        help="Run subsampling sensitivity for one region and metric.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Optional deterministic row subset per region for debugging.",
    )
    return parser.parse_args()


def load_region_data(region: str, subset: int | None = None) -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / f"{region.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    data = pd.read_csv(path)
    if subset is not None and len(data) > subset:
        data = data.sample(n=subset, random_state=config.RANDOM_SEED).sort_values("year")
    return data


def stratified_year_subsample(data: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Draw a deterministic stratified sample by year for model fitting."""

    if len(data) <= max_samples:
        return data.copy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    years = np.array(sorted(data["year"].dropna().unique()))
    base = max(1, max_samples // len(years))
    remainder = max_samples - base * len(years)
    sampled_parts = []
    for index, year in enumerate(years):
        group = data.loc[data["year"] == year]
        n_year = base + (1 if index < remainder else 0)
        replace = len(group) < n_year
        chosen = rng.choice(group.index.to_numpy(), size=n_year, replace=replace)
        sampled_parts.append(data.loc[chosen])
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)


def build_model_for_metric(region: str, metric: str, data: pd.DataFrame):
    priors = config.PRIOR_CONFIG[region][metric]
    if priors["model_type"] == "growth":
        return build_logistic_growth_model(data, priors, metric)
    return build_logistic_decay_model(data, priors, metric)


def max_fit_samples_for_region(region: str) -> int | None:
    return getattr(config, "REGION_MAX_FIT_SAMPLES", {}).get(region, config.MAX_FIT_SAMPLES)


def save_trace(trace: az.InferenceData, region: str, metric: str) -> Path:
    output = config.POSTERIORS_DIR / f"{region.lower()}_{metric}.nc"
    trace.to_netcdf(output)
    return output


def fit_region(
    region: str,
    mcmc_config: dict[str, int],
    subset: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    full_data = load_region_data(region)
    fit_data = load_region_data(region, subset=subset)
    max_fit_samples = max_fit_samples_for_region(region)
    if subset is None and max_fit_samples is not None and len(fit_data) > max_fit_samples:
        original_n = len(fit_data)
        fit_data = stratified_year_subsample(fit_data, max_fit_samples)
        LOGGER.info(
            "Subsampled %s from %d to %d (stratified by year)",
            region,
            original_n,
            len(fit_data),
        )
    traces = {}
    summaries = []
    predictions = []

    for metric in config.METRICS:
        posterior_path = config.POSTERIORS_DIR / f"{region.lower()}_{metric}.nc"
        start = time.perf_counter()
        if posterior_path.exists() and not force:
            LOGGER.info("Skipping %s %s - posterior already exists", region, metric)
            trace = az.from_netcdf(posterior_path)
            diagnostics = convergence_summary(trace)
            elapsed = 0.0
        else:
            LOGGER.info("Starting fit: %s %s (%d rows)", region, metric, len(fit_data))
            model = build_model_for_metric(region, metric, fit_data)
            trace = fit_model(model, mcmc_config)
            elapsed = time.perf_counter() - start
            diagnostics = warn_on_convergence(trace, region, metric)
            save_trace(trace, region, metric)
        traces[metric] = trace

        summary = az.summary(trace, var_names=["L", "k", "t0", "sigma", "nu"])
        if metric == "specific_power":
            summary = pd.concat([summary, az.summary(trace, var_names=["y_min"])])
        else:
            summary = pd.concat([summary, az.summary(trace, var_names=["y0"])])
        summary.insert(0, "parameter", summary.index)
        summary.insert(0, "metric", metric)
        summary.insert(0, "region", region)
        summaries.append(summary.reset_index(drop=True))

        model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
        pred = predict_future(trace, model_type, config.TARGET_YEARS)
        pred.insert(0, "metric", metric)
        pred.insert(0, "region", region)
        predictions.append(pred)

        LOGGER.info(
            "Finished %s %s in %.1fs | R-hat %.3f | ESS %.0f | divergences %d",
            region,
            metric,
            elapsed,
            diagnostics["max_rhat"],
            diagnostics["min_ess"],
            diagnostics["divergences"],
        )

    capacity = compute_derived_capacity(traces, config.TARGET_YEARS)
    capacity.insert(0, "region", region)
    predictions.append(capacity)

    pd.concat(summaries, ignore_index=True).to_csv(
        config.TABLES_DIR / f"{region.lower()}_posterior_summary.csv", index=False
    )
    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(config.TABLES_DIR / f"{region.lower()}_projections.csv", index=False)

    plot_region_fit(region, full_data, traces)
    plot_region_diagnostics(region, traces)
    return prediction_table


def summarize_parameter_intervals(trace: az.InferenceData, region: str, metric: str, label: str, n_rows: int) -> pd.DataFrame:
    """Summarize posterior medians and 95% intervals for model parameters."""

    var_names = ["L", "k", "t0", "sigma", "nu"]
    var_names.append("y_min" if metric == "specific_power" else "y0")
    rows = []
    for var_name in var_names:
        samples = np.asarray(trace.posterior[var_name].values).reshape(-1)
        rows.append(
            {
                "region": region,
                "metric": metric,
                "sample_label": label,
                "n_fit_samples": n_rows,
                "quantity": var_name,
                "median": np.median(samples),
                "q2_5": np.quantile(samples, 0.025),
                "q97_5": np.quantile(samples, 0.975),
            }
        )
    model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
    predictions = predict_future(trace, model_type, config.TARGET_YEARS)
    for _, row in predictions.iterrows():
        rows.append(
            {
                "region": region,
                "metric": metric,
                "sample_label": label,
                "n_fit_samples": n_rows,
                "quantity": f"prediction_{int(row['year'])}",
                "median": row["median"],
                "q2_5": row["q2_5"],
                "q97_5": row["q97_5"],
            }
        )
    diagnostics = convergence_summary(trace)
    for diagnostic_name, value in diagnostics.items():
        rows.append(
            {
                "region": region,
                "metric": metric,
                "sample_label": label,
                "n_fit_samples": n_rows,
                "quantity": diagnostic_name,
                "median": value,
                "q2_5": np.nan,
                "q97_5": np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_subsample_sensitivity(region: str, metric: str, mcmc_config: dict[str, int]) -> pd.DataFrame:
    """Run fits for increasing stratified subsamples and the full dataset."""

    full_data = load_region_data(region)
    sample_plan: list[tuple[str, pd.DataFrame]] = []
    for n_samples in [2000, 5000, 10000]:
        if len(full_data) > n_samples:
            sample_plan.append((str(n_samples), stratified_year_subsample(full_data, n_samples)))
        else:
            sample_plan.append((str(n_samples), full_data.copy()))
    sample_plan.append(("full", full_data.copy()))

    previous_results = []
    output_path = config.TABLES_DIR / "subsampling_sensitivity.csv"
    if output_path.exists():
        previous = pd.read_csv(output_path)
        previous = previous.loc[
            ~((previous["region"] == region) & (previous["metric"] == metric))
        ]
        if not previous.empty:
            previous_results.append(previous)

    outputs = []
    for label, fit_data in sample_plan:
        LOGGER.info(
            "Subsampling sensitivity fit: %s %s sample=%s n=%d",
            region,
            metric,
            label,
            len(fit_data),
        )
        start = time.perf_counter()
        model = build_model_for_metric(region, metric, fit_data)
        trace = fit_model(model, mcmc_config)
        diagnostics = warn_on_convergence(trace, region, f"{metric}_{label}")
        LOGGER.info(
            "Finished sensitivity fit %s %s sample=%s in %.1fs | R-hat %.3f | ESS %.0f | divergences %d",
            region,
            metric,
            label,
            time.perf_counter() - start,
            diagnostics["max_rhat"],
            diagnostics["min_ess"],
            diagnostics["divergences"],
        )
        outputs.append(summarize_parameter_intervals(trace, region, metric, label, len(fit_data)))
        pd.concat(previous_results + outputs, ignore_index=True).to_csv(output_path, index=False)
        LOGGER.info("Updated subsampling sensitivity results: %s", output_path)

    result = pd.concat(previous_results + outputs, ignore_index=True)
    result.to_csv(output_path, index=False)
    LOGGER.info("Wrote subsampling sensitivity results: %s", output_path)
    return result


def plot_region_fit(region: str, data: pd.DataFrame, traces: dict[str, az.InferenceData]) -> None:
    years_plot = np.arange(int(data["year"].min()), 2056)
    color = config.REGION_COLORS[region]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False)
    for ax, metric in zip(axes, config.METRICS):
        column = config.METRIC_COLUMNS[metric]
        model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
        samples = posterior_predictive_samples(traces[metric], model_type, years_plot, max_samples=1000)
        ax.scatter(data["year"], data[column], s=7, color=color, alpha=0.15, edgecolors="none")
        ax.plot(years_plot, np.median(samples, axis=0), color="black", linewidth=2)
        ax.fill_between(
            years_plot,
            np.quantile(samples, 0.025, axis=0),
            np.quantile(samples, 0.975, axis=0),
            color=color,
            alpha=0.25,
        )
        ax.set_xlabel("Year")
        ax.set_ylabel(config.METRIC_LABELS[column])
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"bayesian_fit_{region.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(regions: list[str]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=False)
    for row, region in enumerate(regions):
        data = load_region_data(region)
        for col, metric in enumerate(config.METRICS):
            trace_path = config.POSTERIORS_DIR / f"{region.lower()}_{metric}.nc"
            if not trace_path.exists():
                continue
            trace = az.from_netcdf(trace_path)
            years_plot = np.arange(int(data["year"].min()), 2056)
            column = config.METRIC_COLUMNS[metric]
            model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
            samples = posterior_predictive_samples(trace, model_type, years_plot, max_samples=700)
            ax = axes[row, col]
            ax.scatter(
                data["year"],
                data[column],
                s=4,
                color=config.REGION_COLORS[region],
                alpha=0.08,
                edgecolors="none",
            )
            ax.plot(years_plot, np.median(samples, axis=0), color="black", linewidth=1.8)
            ax.fill_between(
                years_plot,
                np.quantile(samples, 0.025, axis=0),
                np.quantile(samples, 0.975, axis=0),
                color=config.REGION_COLORS[region],
                alpha=0.22,
            )
            if row == 0:
                ax.set_title(metric)
            if col == 0:
                ax.set_ylabel(region)
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "bayesian_fit_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_diagnostics(region: str, traces: dict[str, az.InferenceData]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for row, metric in enumerate(config.METRICS):
        trace = traces[metric]
        l_values = trace.posterior["L"].values
        for chain in range(l_values.shape[0]):
            axes[row, 0].plot(l_values[chain], alpha=0.65, linewidth=0.8)
        axes[row, 1].hist(l_values.reshape(-1), bins=40, color=config.REGION_COLORS[region], alpha=0.75)
        axes[row, 0].set_title(f"{region} {metric}: L trace")
        axes[row, 1].set_title(f"{region} {metric}: L posterior")
        axes[row, 0].set_xlabel("Draw")
        axes[row, 1].set_xlabel("L")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        config.FIGURES_DIR / f"posterior_diagnostics_{region.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_projection_summary(predictions: pd.DataFrame) -> None:
    plot_data = predictions.loc[predictions["metric"].isin(config.METRICS + ["capacity_mw"])].copy()
    labels = {
        "hub_height": "Hub Height [m]",
        "rotor_diameter": "Rotor Diameter [m]",
        "specific_power": "Specific Power [W/m2]",
        "capacity_mw": "Capacity [MW]",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.ravel(), labels):
        subset = plot_data.loc[plot_data["metric"] == metric]
        x = np.arange(len(subset))
        colors = [config.REGION_COLORS[region] for region in subset["region"]]
        yerr = np.vstack([subset["median"] - subset["q2_5"], subset["q97_5"] - subset["median"]])
        ax.bar(x, subset["median"], color=colors, alpha=0.8)
        ax.errorbar(x, subset["median"], yerr=yerr, fmt="none", ecolor="black", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n{y}" for r, y in zip(subset["region"], subset["year"])])
        ax.set_ylabel(labels[metric])
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "projection_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    ensure_directories([config.FIGURES_DIR, config.TABLES_DIR, config.POSTERIORS_DIR])
    args = parse_args()
    if args.subsample_sensitivity and args.region == "all":
        raise ValueError("--subsample-sensitivity requires a single --region, not 'all'.")
    regions = config.REGIONS if args.region == "all" else [args.region]
    mcmc_config = dict(config.MCMC_QUICK if args.quick else config.MCMC_FULL)
    mcmc_config["target_accept"] = config.TARGET_ACCEPT
    subset = args.subset
    if args.quick and subset is None:
        subset = 500

    if args.subsample_sensitivity:
        run_subsample_sensitivity(args.region, args.metric, mcmc_config)
        return

    all_predictions = []
    for region in regions:
        all_predictions.append(fit_region(region, mcmc_config, subset=subset, force=args.force))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(config.TABLES_DIR / "projection_summary.csv", index=False)
    plot_projection_summary(predictions)
    plot_comparison(regions)

    LOGGER.info("=== Projection Summary ===\n%s", predictions.to_string(index=False))


if __name__ == "__main__":
    main()
