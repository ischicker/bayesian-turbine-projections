"""Assess projection variability across independent stratified subsamples."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

from turbine_projections import config
from turbine_projections.bayesian_model import (
    build_logistic_growth_model,
    convergence_summary,
    fit_model,
    predict_future,
    warn_on_convergence,
)
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

REGIONS = ["DE", "US"]
METRIC = "hub_height"
SUBSAMPLE_SIZE = 5000
SUBSAMPLE_SEEDS = config.SUBSAMPLE_SEEDS
TARGET_YEARS = [2030, 2055]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing rows in the output CSV.")
    return parser.parse_args()


def load_region_data(region: str) -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / f"{region.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    return pd.read_csv(path)


def stratified_year_subsample(data: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Draw a stratified sample by year using an independent seed."""

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


def output_path() -> Path:
    return config.TABLES_DIR / "subsample_variability.csv"


def load_existing(force: bool) -> pd.DataFrame:
    path = output_path()
    if force or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fit_one(region: str, seed: int, full_data: pd.DataFrame) -> dict:
    fit_data = stratified_year_subsample(full_data, SUBSAMPLE_SIZE, seed)
    LOGGER.info(
        "Fitting %s %s subsample seed=%d n=%d",
        region,
        METRIC,
        seed,
        len(fit_data),
    )
    model = build_logistic_growth_model(fit_data, config.PRIOR_CONFIG[region][METRIC], METRIC)
    mcmc_config = {**config.MCMC_FULL, "compute_log_likelihood": False}
    start = time.perf_counter()
    trace = fit_model(model, mcmc_config)
    elapsed = time.perf_counter() - start
    diagnostics = warn_on_convergence(trace, region, f"{METRIC}_subsample_{seed}")
    predictions = predict_future(trace, "growth", TARGET_YEARS)

    row = {
        "region": region,
        "metric": METRIC,
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

    del trace
    gc.collect()
    return row


def add_summary_columns(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    for region in REGIONS:
        region_mask = output["region"] == region
        if int(region_mask.sum()) < len(SUBSAMPLE_SEEDS):
            continue
        for year in TARGET_YEARS:
            medians = output.loc[region_mask, f"median_{year}"].to_numpy(dtype=float)
            ci_widths = output.loc[region_mask, f"ci_width_{year}"].to_numpy(dtype=float)
            mean_median = float(np.mean(medians))
            sd_median = float(np.std(medians, ddof=1))
            mean_ci_width = float(np.mean(ci_widths))
            ratio = sd_median / mean_ci_width if mean_ci_width > 0 else np.nan
            output.loc[region_mask, f"mean_median_{year}"] = mean_median
            output.loc[region_mask, f"sd_median_{year}"] = sd_median
            output.loc[region_mask, f"mean_ci_width_{year}"] = mean_ci_width
            output.loc[region_mask, f"sd_to_ci_width_ratio_{year}"] = ratio
            output.loc[region_mask, f"passes_5pct_rule_{year}"] = bool(ratio < 0.05)
    return output.sort_values(["region", "subsample_seed"]).reset_index(drop=True)


def write_results(results: pd.DataFrame) -> pd.DataFrame:
    summarized = add_summary_columns(results)
    summarized.to_csv(output_path(), index=False)
    return summarized


def print_summary(results: pd.DataFrame) -> None:
    rows = []
    for region in REGIONS:
        subset = results.loc[results["region"] == region]
        if len(subset) < len(SUBSAMPLE_SEEDS):
            LOGGER.info("%s incomplete: %d/%d subsamples finished", region, len(subset), len(SUBSAMPLE_SEEDS))
            continue
        for year in TARGET_YEARS:
            rows.append(
                {
                    "region": region,
                    "year": year,
                    "mean_median": float(subset[f"mean_median_{year}"].iloc[0]),
                    "sd_median": float(subset[f"sd_median_{year}"].iloc[0]),
                    "mean_ci_width": float(subset[f"mean_ci_width_{year}"].iloc[0]),
                    "sd_to_ci_width_ratio": float(subset[f"sd_to_ci_width_ratio_{year}"].iloc[0]),
                    "passes_5pct_rule": bool(subset[f"passes_5pct_rule_{year}"].iloc[0]),
                    "max_rhat": float(subset["max_rhat"].max()),
                    "min_ess": float(subset["min_ess"].min()),
                    "divergences": int(subset["divergences"].sum()),
                }
            )
    if rows:
        summary = pd.DataFrame(rows)
        LOGGER.info("=== Subsample Variability Summary ===\n%s", summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_directories([config.TABLES_DIR])
    existing = load_existing(args.force)
    rows = [] if existing.empty else existing.to_dict("records")

    completed = set()
    if not existing.empty and not args.force:
        completed = set(zip(existing["region"], existing["subsample_seed"]))

    for region in REGIONS:
        full_data = load_region_data(region)
        for seed in SUBSAMPLE_SEEDS:
            if (region, seed) in completed:
                LOGGER.info("Skipping %s seed=%d - already in %s", region, seed, output_path())
                continue
            row = fit_one(region, seed, full_data)
            rows.append(row)
            current = pd.DataFrame(rows)
            write_results(current)
            LOGGER.info("Updated %s after %s seed=%d", output_path(), region, seed)

    final = write_results(pd.DataFrame(rows))
    print_summary(final)


if __name__ == "__main__":
    main()
