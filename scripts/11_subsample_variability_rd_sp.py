"""Assess subsample variability for rotor diameter and specific power."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.bayesian_model import (
    build_logistic_decay_model,
    build_logistic_growth_model,
    fit_model,
    predict_future,
    warn_on_convergence,
)
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

USE_REPO_MODEL = True
REGIONS = ["DE", "US"]
METRICS = ["rotor_diameter", "specific_power"]
SUBSAMPLE_SIZE = 5000
SUBSAMPLE_SEEDS = config.SUBSAMPLE_SEEDS
TARGET_YEARS = [2030, 2055]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing full-run rows.")
    parser.add_argument("--quick", action="store_true", help="Run one smoke-test fit with MCMC_SMOKE.")
    return parser.parse_args()


def load_region_data(region: str) -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / f"{region.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    return pd.read_csv(path)


def stratified_year_subsample(data: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Draw a deterministic stratified sample by year for an independent seed."""

    if len(data) <= max_samples:
        return data.copy()
    rng = np.random.default_rng(seed)
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
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_repo_model(region: str, metric: str, fit_data: pd.DataFrame):
    priors = config.PRIOR_CONFIG[region][metric]
    if priors["model_type"] == "growth":
        return build_logistic_growth_model(fit_data, priors, metric)
    if priors["model_type"] == "decay":
        return build_logistic_decay_model(fit_data, priors, metric)
    raise ValueError(f"Unsupported model type for {region} {metric}: {priors['model_type']}")


def fit_via_repo(region: str, metric: str, fit_data: pd.DataFrame, mcmc_config: dict) -> tuple[pd.DataFrame, dict]:
    """Fit with the same repository fitter/projection extraction used by script 09."""

    if not USE_REPO_MODEL:
        raise RuntimeError("This revision analysis must use the repository Bayesian model.")
    model = build_repo_model(region, metric, fit_data)
    trace = fit_model(model, {**mcmc_config, "compute_log_likelihood": False})
    diagnostics = warn_on_convergence(trace, region, f"{metric}_subsample")
    model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
    predictions = predict_future(trace, model_type, TARGET_YEARS)
    del trace
    gc.collect()
    return predictions, diagnostics


def result_paths(quick: bool = False) -> tuple[Path, Path, Path]:
    suffix = "_quick" if quick else ""
    return (
        config.TABLES_DIR / f"subsample_variability_rd_sp{suffix}.csv",
        config.TABLES_DIR / f"subsample_variability_rd_sp_aggregate{suffix}.csv",
        config.TABLES_DIR / f"subsample_variability_rd_sp_summary{suffix}.tex",
    )


def load_existing(force: bool, quick: bool) -> pd.DataFrame:
    row_path, _, _ = result_paths(quick)
    if force or quick or not row_path.exists():
        return pd.DataFrame()
    return pd.read_csv(row_path)


def fit_one(region: str, metric: str, seed: int, full_data: pd.DataFrame, mcmc_config: dict) -> dict:
    fit_data = stratified_year_subsample(full_data, SUBSAMPLE_SIZE, seed)
    LOGGER.info("Fitting %s %s seed=%d n=%d", region, metric, seed, len(fit_data))
    start = time.perf_counter()
    predictions, diagnostics = fit_via_repo(region, metric, fit_data, mcmc_config)
    elapsed = time.perf_counter() - start
    row = {
        "region": region,
        "metric": metric,
        "model_type": config.PRIOR_CONFIG[region][metric]["model_type"],
        "subsample_seed": seed,
        "n_fit_samples": len(fit_data),
        "elapsed_seconds": elapsed,
        "max_rhat": diagnostics["max_rhat"],
        "min_ess": diagnostics["min_ess"],
        "divergences": diagnostics["divergences"],
    }
    for _, prediction in predictions.iterrows():
        year = int(prediction["year"])
        row[f"median_{year}"] = float(prediction["median"])
        row[f"q2_5_{year}"] = float(prediction["q2_5"])
        row[f"q97_5_{year}"] = float(prediction["q97_5"])
        row[f"ci_width_{year}"] = float(prediction["q97_5"] - prediction["q2_5"])
    return row


def aggregate_results(rows: pd.DataFrame) -> pd.DataFrame:
    aggregate_rows = []
    for (region, metric), group in rows.groupby(["region", "metric"], sort=True):
        for year in TARGET_YEARS:
            medians = group[f"median_{year}"].to_numpy(dtype=float)
            widths = group[f"ci_width_{year}"].to_numpy(dtype=float)
            mean_ci_width = float(np.mean(widths))
            sd_median = float(np.std(medians, ddof=1)) if len(medians) > 1 else np.nan
            aggregate_rows.append(
                {
                    "region": region,
                    "metric": metric,
                    "year": year,
                    "n_subsamples": int(len(group)),
                    "median_min": float(np.min(medians)),
                    "median_max": float(np.max(medians)),
                    "mean_median": float(np.mean(medians)),
                    "sd_median": sd_median,
                    "mean_ci_width": mean_ci_width,
                    "sd_to_ci": sd_median / mean_ci_width if mean_ci_width > 0 else np.nan,
                    "passes_5pct": bool(sd_median / mean_ci_width < 0.05) if mean_ci_width > 0 else False,
                    "max_rhat": float(group["max_rhat"].max()),
                    "min_ess": float(group["min_ess"].min()),
                    "divergences": int(group["divergences"].sum()),
                }
            )
    return pd.DataFrame(aggregate_rows).sort_values(["region", "metric", "year"]).reset_index(drop=True)


def metric_label(metric: str) -> str:
    return {"rotor_diameter": "Rotor diameter", "specific_power": "Specific power"}[metric]


def write_summary_tex(aggregate: pd.DataFrame, path: Path) -> None:
    lines = []
    for _, row in aggregate.iterrows():
        lines.append(
            f"{row['region']} & {metric_label(row['metric'])} & {int(row['year'])} & "
            f"{row['median_min']:.1f}--{row['median_max']:.1f} & "
            f"{row['mean_median']:.1f} & {row['sd_median']:.2f} & {row['sd_to_ci']:.2f} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: pd.DataFrame, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_path, aggregate_path, tex_path = result_paths(quick)
    rows = rows.sort_values(["region", "metric", "subsample_seed"]).reset_index(drop=True)
    aggregate = aggregate_results(rows)
    rows.to_csv(row_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    write_summary_tex(aggregate, tex_path)
    return rows, aggregate


def print_summary(aggregate: pd.DataFrame) -> None:
    LOGGER.info("=== RD/SP Subsample Variability Summary ===\n%s", aggregate.to_string(index=False))


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_directories([config.TABLES_DIR])
    mcmc_config = config.MCMC_SMOKE if args.quick else config.MCMC_FULL
    existing = load_existing(args.force, args.quick)
    rows = [] if existing.empty else existing.to_dict("records")
    completed = set()
    if not existing.empty and not args.force:
        completed = set(zip(existing["region"], existing["metric"], existing["subsample_seed"]))

    jobs = [(region, metric, seed) for region in REGIONS for metric in METRICS for seed in SUBSAMPLE_SEEDS]
    if args.quick:
        jobs = [("DE", "rotor_diameter", SUBSAMPLE_SEEDS[0])]

    data_cache = {region: load_region_data(region) for region in REGIONS}
    for region, metric, seed in jobs:
        if (region, metric, seed) in completed:
            LOGGER.info("Skipping %s %s seed=%d - already complete", region, metric, seed)
            continue
        rows.append(fit_one(region, metric, seed, data_cache[region], mcmc_config))
        current_rows, _ = write_outputs(pd.DataFrame(rows), quick=args.quick)
        LOGGER.info("Wrote %d rows to %s", len(current_rows), result_paths(args.quick)[0])

    _, aggregate = write_outputs(pd.DataFrame(rows), quick=args.quick)
    print_summary(aggregate)


if __name__ == "__main__":
    main()
