"""Run climate-aware energy-yield calculations."""

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
from turbine_projections.climate_scenarios import extract_gowires_wind_climate, weibull_pdf
from turbine_projections.energy_yield import (
    DEFAULT_WIND_SPEEDS,
    propagate_energy_yield,
    joint_technology_samples,
    normalized_power_curve,
    run_energy_yield_propagation,
    uncertainty_decomposition,
)
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

SCENARIO_COLORS = {
    "historical": "#333333",
    "SSP2-4.5": "#457B9D",
    "SSP5-8.5": "#E76F51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gowires-path",
        type=Path,
        default=None,
        help="Optional path to GOWIRES_V1.csv. Defaults to data/raw/GOWIRES/GOWIRES_V1.csv.",
    )
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=1000)
    return parser.parse_args()


def plot_weibull_pdfs(summary) -> None:
    wind_speed = np.linspace(0.01, 30.0, 500)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, region in zip(axes, config.REGIONS):
        subset = summary.loc[summary["region"] == region]
        for _, row in subset.iterrows():
            scenario = row["scenario"]
            ax.plot(
                wind_speed,
                weibull_pdf(wind_speed, row["weibull_k"], row["weibull_A"]),
                color=SCENARIO_COLORS[scenario],
                linewidth=2.0,
                label=scenario if region == config.REGIONS[0] else None,
            )
            if scenario != "historical":
                lower = weibull_pdf(wind_speed, row["weibull_k_min"], row["weibull_A_min"])
                upper = weibull_pdf(wind_speed, row["weibull_k_max"], row["weibull_A_max"])
                ax.fill_between(
                    wind_speed,
                    np.minimum(lower, upper),
                    np.maximum(lower, upper),
                    color=SCENARIO_COLORS[scenario],
                    alpha=0.12,
                    linewidth=0,
                )
        ax.set_title(f"{region}: {config.REFERENCE_SITES[region]['name']}")
        ax.set_xlabel("Wind speed [m/s]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Weibull PDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"gowires_weibull_pdfs.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_latex(summary: pd.DataFrame) -> None:
    columns = [
        "region",
        "year",
        "scenario",
        "cf_median",
        "cf_p5",
        "cf_p95",
        "aep_mwh_median",
        "aep_mwh_p5",
        "aep_mwh_p95",
    ]
    table = summary[columns].copy()
    header = " & ".join(columns).replace("_", "\\_")
    body = []
    for _, row in table.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        body.append(" & ".join(values) + r" \\")
    latex = "\n".join(
        [
            r"\begin{tabular}{lllrrrrrr}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (config.TABLES_DIR / "energy_yield_summary.tex").write_text(latex, encoding="utf-8")


def make_shear_comparison(ple_summary: pd.DataFrame, old_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare site-specific PLE AEP with the former fixed 1/7 exponent."""

    key = ["region", "year", "scenario"]
    baseline = old_summary[key + ["aep_mwh_median"]].rename(columns={"aep_mwh_median": "aep_median_baseline"})
    frames = []
    for alpha_source, frame in [("PLE", ple_summary), ("fixed_0.143", old_summary)]:
        out = frame.copy()
        out["alpha_source"] = alpha_source
        out = out.merge(baseline, on=key, how="left")
        out["delta_aep_pct"] = 100.0 * (out["aep_mwh_median"] - out["aep_median_baseline"]) / out[
            "aep_median_baseline"
        ]
        frames.append(out)
    comparison = pd.concat(frames, ignore_index=True)
    comparison = comparison.rename(
        columns={
            "aep_mwh_median": "aep_median",
            "aep_mwh_p5": "aep_p5",
            "aep_mwh_p95": "aep_p95",
        }
    )
    columns = [
        "region",
        "year",
        "scenario",
        "alpha_source",
        "alpha",
        "cf_median",
        "aep_median",
        "aep_p5",
        "aep_p95",
        "delta_aep_pct",
    ]
    return comparison[columns].sort_values(["region", "year", "scenario", "alpha_source"]).reset_index(drop=True)


def plot_aep_violins(propagation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.5), sharey=False)
    scenarios = ["historical", "SSP2-4.5", "SSP5-8.5"]
    y_limits_by_year = {}
    for year in config.TARGET_YEARS:
        year_values = propagation.loc[propagation["year"] == year, "aep_mwh"].to_numpy(dtype=float)
        y_min = float(np.nanmin(year_values))
        y_max = float(np.nanmax(year_values))
        padding = 0.1 * (y_max - y_min)
        y_limits_by_year[year] = (max(0.0, y_min - padding), y_max + padding)
    for row, region in enumerate(config.REGIONS):
        for col, year in enumerate(config.TARGET_YEARS):
            ax = axes[row, col]
            subset = propagation.loc[(propagation["region"] == region) & (propagation["year"] == year)]
            values = [subset.loc[subset["scenario"] == scenario, "aep_mwh"].to_numpy() for scenario in scenarios]
            positions = np.arange(len(scenarios))
            parts = ax.violinplot(values, positions=positions, widths=0.75, showextrema=False)
            for body, scenario in zip(parts["bodies"], scenarios):
                body.set_facecolor(SCENARIO_COLORS[scenario])
                body.set_edgecolor("black")
                body.set_alpha(0.7)
            medians = [np.median(v) for v in values]
            p5 = [np.quantile(v, 0.05) for v in values]
            p95 = [np.quantile(v, 0.95) for v in values]
            ax.scatter(positions, medians, color="white", edgecolor="black", zorder=3, s=28)
            ax.vlines(positions, p5, p95, color="black", linewidth=1.2)
            ax.set_xticks(positions)
            ax.set_xticklabels(scenarios, rotation=20)
            ax.set_title(f"{region} {year}")
            ax.set_ylabel("AEP [MWh/a]")
            ax.set_ylim(y_limits_by_year[year])
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"energy_yield_aep_violins.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_decomposition(decomp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(decomp))
    tech = decomp["technology_share"].to_numpy(dtype=float)
    climate = decomp["climate_share"].to_numpy(dtype=float)
    ax.bar(x, tech, color="#264653", label="Technology")
    ax.bar(x, climate, bottom=tech, color="#E76F51", label="Climate")
    ax.set_xticks(x)
    ax.set_xticklabels(decomp["region"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Variance share")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(facecolor="white", edgecolor="gray", framealpha=0.9)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"energy_yield_uncertainty_decomposition.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_power_law_sensitivity(gowires_climate: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    rows = []
    historical = gowires_climate.loc[gowires_climate["scenario"] == "historical"].set_index("region")
    for region in config.REGIONS:
        tech = joint_technology_samples(region, 2055, n_samples=n_samples)
        for alpha in [0.10, 0.143, 0.20]:
            propagated = propagate_energy_yield(
                region=region,
                year=2055,
                scenario_row=historical.loc[region],
                technology_samples=tech,
                alpha=alpha,
            )
            rows.append(
                {
                    "region": region,
                    "alpha": alpha,
                    "cf_median": float(propagated["capacity_factor"].median()),
                    "aep_median": float(propagated["aep_mwh"].median()),
                    "aep_p5": float(propagated["aep_mwh"].quantile(0.05)),
                    "aep_p95": float(propagated["aep_mwh"].quantile(0.95)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(config.TABLES_DIR / "power_law_sensitivity.csv", index=False)
    return result


def run_correlation_sensitivity(gowires_climate: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    historical = gowires_climate.loc[
        (gowires_climate["region"] == "DE") & (gowires_climate["scenario"] == "historical")
    ].iloc[0]
    correlated_tech = joint_technology_samples("DE", 2055, n_samples=n_samples)
    correlated = propagate_energy_yield("DE", 2055, historical, correlated_tech)

    rng = np.random.default_rng(config.RANDOM_SEED)
    uncorrelated_tech = correlated_tech.copy()
    uncorrelated_tech["rotor_diameter_m"] = rng.permutation(
        uncorrelated_tech["rotor_diameter_m"].to_numpy(dtype=float)
    )
    uncorrelated_tech["specific_power_wm2"] = rng.permutation(
        uncorrelated_tech["specific_power_wm2"].to_numpy(dtype=float)
    )
    uncorrelated_tech["rated_power_mw"] = (
        uncorrelated_tech["specific_power_wm2"]
        * np.pi
        * (uncorrelated_tech["rotor_diameter_m"] / 2.0) ** 2
        / 1_000_000.0
    )
    uncorrelated = propagate_energy_yield("DE", 2055, historical, uncorrelated_tech)

    result = pd.DataFrame(
        [
            {
                "region": "DE",
                "year": 2055,
                "scenario": "historical",
                "sampling": "correlated_index_matched",
                "aep_variance": float(correlated["aep_mwh"].var(ddof=1)),
                "aep_median": float(correlated["aep_mwh"].median()),
                "n_samples": int(len(correlated)),
            },
            {
                "region": "DE",
                "year": 2055,
                "scenario": "historical",
                "sampling": "uncorrelated_shuffled_rd_sp",
                "aep_variance": float(uncorrelated["aep_mwh"].var(ddof=1)),
                "aep_median": float(uncorrelated["aep_mwh"].median()),
                "n_samples": int(len(uncorrelated)),
            },
        ]
    )
    result.to_csv(config.TABLES_DIR / "correlation_sensitivity.csv", index=False)
    return result


def _gcm_models(neighbors: pd.DataFrame, scenario_code: str) -> list[str]:
    prefix = f"weibull_c_{scenario_code}_"
    return sorted(column.replace(prefix, "") for column in neighbors.columns if column.startswith(prefix))


def _fixed_technology_for_region(region: str) -> pd.Series:
    tech = joint_technology_samples(region, 2055, n_samples=1000)
    return pd.Series(
        {
            "hub_height_m": float(tech["hub_height_m"].median()),
            "rotor_diameter_m": float(tech["rotor_diameter_m"].median()),
            "specific_power_wm2": float(tech["specific_power_wm2"].median()),
            "rated_power_mw": float(tech["rated_power_mw"].median()),
            "sample_id": 0,
        }
    )


def run_gcm_ensemble_variance(neighbors: pd.DataFrame, previous_decomp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenario_map = {"SSP245": "SSP2-4.5", "SSP585": "SSP5-8.5"}
    previous = previous_decomp.set_index("region")
    for region in config.REGIONS:
        region_neighbors = neighbors.loc[neighbors["region"] == region]
        fixed_tech = _fixed_technology_for_region(region)
        tech_df = pd.DataFrame([fixed_tech])
        ple_alpha = float(region_neighbors["PLE"].median()) if "PLE" in region_neighbors else np.nan
        for scenario_code, scenario_name in scenario_map.items():
            model_rows = []
            for model in _gcm_models(region_neighbors, scenario_code):
                c_col = f"weibull_c_{scenario_code}_{model}"
                k_col = f"weibull_k_{scenario_code}_{model}"
                scenario_row = pd.Series(
                    {
                        "scenario": scenario_name,
                        "weibull_A": float(region_neighbors[c_col].median()),
                        "weibull_k": float(region_neighbors[k_col].median()),
                    }
                )
                fixed_alpha = propagate_energy_yield(region, 2055, scenario_row, tech_df, alpha=0.143).iloc[0]
                ple_result = (
                    propagate_energy_yield(region, 2055, scenario_row, tech_df, alpha=ple_alpha).iloc[0]
                    if np.isfinite(ple_alpha)
                    else None
                )
                model_rows.append(
                    {
                        "region": region,
                        "scenario": scenario_name,
                        "gcm": model,
                        "alpha": 0.143,
                        "ple_alpha": ple_alpha,
                        "weibull_k": scenario_row["weibull_k"],
                        "weibull_A": scenario_row["weibull_A"],
                        "aep_mwh": float(fixed_alpha["aep_mwh"]),
                        "cf": float(fixed_alpha["capacity_factor"]),
                        "aep_mwh_ple": float(ple_result["aep_mwh"]) if ple_result is not None else np.nan,
                        "cf_ple": float(ple_result["capacity_factor"]) if ple_result is not None else np.nan,
                    }
                )
            model_df = pd.DataFrame(model_rows)
            var_gcm = float(model_df["aep_mwh"].var(ddof=1))
            var_gcm_ple = float(model_df["aep_mwh_ple"].var(ddof=1))
            var_tech = float(previous.loc[region, "var_technology"])
            var_previous_climate = float(previous.loc[region, "var_climate"])
            for row in model_rows:
                row.update(
                    {
                        "var_climate_gcm": var_gcm,
                        "var_climate_gcm_ple": var_gcm_ple,
                        "var_climate_previous_3scenario": var_previous_climate,
                        "var_technology": var_tech,
                        "technology_share_gcm": var_tech / (var_tech + var_gcm) if var_tech + var_gcm > 0 else np.nan,
                        "climate_share_gcm": var_gcm / (var_tech + var_gcm) if var_tech + var_gcm > 0 else np.nan,
                        "technology_share_gcm_ple": var_tech / (var_tech + var_gcm_ple)
                        if var_tech + var_gcm_ple > 0
                        else np.nan,
                        "climate_share_gcm_ple": var_gcm_ple / (var_tech + var_gcm_ple)
                        if var_tech + var_gcm_ple > 0
                        else np.nan,
                    }
                )
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(config.TABLES_DIR / "gcm_ensemble_variance.csv", index=False)
    return result


def plot_gcm_ensemble_variance(gcm: pd.DataFrame) -> None:
    summary = (
        gcm.loc[gcm["scenario"] == "SSP2-4.5"]
        .groupby("region", as_index=False)
        .agg(
            var_climate_gcm=("var_climate_gcm_ple", "first"),
            var_climate_previous_3scenario=("var_climate_previous_3scenario", "first"),
            climate_share_gcm=("climate_share_gcm_ple", "first"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    x = np.arange(len(summary))
    width = 0.34
    axes[0].bar(x - width / 2, summary["var_climate_previous_3scenario"], width=width, color="#6C757D", label="3 scenarios\n(PLE)")
    axes[0].bar(x + width / 2, summary["var_climate_gcm"], width=width, color="#457B9D", label="SSP2-4.5 GCM\nensemble (PLE)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["region"])
    axes[0].set_ylabel("Climate variance [MWh²/a²]")
    axes[0].set_ylim(0, 1.4e7)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(summary["region"], summary["climate_share_gcm"], color="#E76F51", alpha=0.85)
    axes[1].set_ylim(0, max(0.2, float(summary["climate_share_gcm"].max()) * 1.25))
    axes[1].set_ylabel("Climate share with GCM variance")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"gcm_ensemble_variance.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_power_law_sensitivity(sensitivity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    alphas = sorted(sensitivity["alpha"].unique())
    x = np.arange(len(config.REGIONS))
    width = 0.22
    offsets = np.linspace(-width, width, len(alphas))
    colors = ["#6C757D", "#457B9D", "#E76F51"]
    for offset, alpha, color in zip(offsets, alphas, colors):
        subset = sensitivity.loc[sensitivity["alpha"] == alpha].set_index("region").loc[config.REGIONS]
        y = subset["aep_median"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - subset["aep_p5"].to_numpy(dtype=float),
                subset["aep_p95"].to_numpy(dtype=float) - y,
            ]
        )
        ax.bar(x + offset, y, width=width, color=color, label=f"alpha={alpha:g}", alpha=0.85)
        ax.errorbar(x + offset, y, yerr=yerr, fmt="none", ecolor="black", capsize=2, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(config.REGIONS)
    ax.set_ylabel("AEP 2055 historical [MWh/a]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"power_law_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_propagation_schematic(propagation: pd.DataFrame, region: str = "DE", year: int = 2055) -> None:
    subset = propagation.loc[
        (propagation["region"] == region)
        & (propagation["year"] == year)
        & (propagation["scenario"] == "historical")
    ].copy()
    if subset.empty:
        return
    sample = subset.sample(n=min(120, len(subset)), random_state=config.RANDOM_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].scatter(
        sample["rotor_diameter_m"],
        sample["specific_power_wm2"],
        s=12,
        color=config.REGION_COLORS[region],
        alpha=0.35,
        edgecolors="none",
    )
    axes[0].set_xlabel("Rotor diameter [m]")
    axes[0].set_ylabel("Specific power [W/m2]")
    axes[0].set_title("Technology posterior")
    axes[0].grid(alpha=0.25)

    for _, row in sample.head(80).iterrows():
        _, cf_curve, _, _ = normalized_power_curve(row["specific_power_wm2"], DEFAULT_WIND_SPEEDS)
        axes[1].plot(DEFAULT_WIND_SPEEDS, cf_curve, color="#111111", alpha=0.08, linewidth=0.8)
    median_row = subset.loc[(subset["specific_power_wm2"] - subset["specific_power_wm2"].median()).abs().idxmin()]
    _, median_curve, _, _ = normalized_power_curve(median_row["specific_power_wm2"], DEFAULT_WIND_SPEEDS)
    axes[1].plot(DEFAULT_WIND_SPEEDS, median_curve, color="#E76F51", linewidth=2.0)
    axes[1].set_xlabel("Wind speed [m/s]")
    axes[1].set_ylabel("Normalized power")
    axes[1].set_title("Synthetic power curves")
    axes[1].grid(alpha=0.25)

    axes[2].hist(subset["aep_mwh"], bins=35, color=config.REGION_COLORS[region], alpha=0.75)
    axes[2].axvline(subset["aep_mwh"].median(), color="black", linewidth=2.0)
    axes[2].set_xlabel("AEP [MWh/a]")
    axes[2].set_ylabel("Samples")
    axes[2].set_title("AEP distribution")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"energy_yield_propagation_schematic_{region.lower()}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    ensure_directories([config.PROCESSED_DATA_DIR, config.TABLES_DIR, config.FIGURES_DIR])
    args = parse_args()
    summary, neighbors = extract_gowires_wind_climate(args.gowires_path, n_neighbors=args.n_neighbors)
    output_columns = [
        "region",
        "scenario",
        "weibull_k",
        "weibull_A",
        "weibull_k_min",
        "weibull_k_max",
        "weibull_A_min",
        "weibull_A_max",
    ]
    output = summary[output_columns]
    output.to_csv(config.PROCESSED_DATA_DIR / "gowires_wind_climate.csv", index=False)
    output.to_csv(config.TABLES_DIR / "gowires_wind_climate.csv", index=False)
    neighbors.to_csv(config.PROCESSED_DATA_DIR / "gowires_reference_neighbors.csv", index=False)
    plot_weibull_pdfs(summary)

    propagation, energy_summary = run_energy_yield_propagation(output, n_samples=args.n_samples)
    fixed_alpha = {region: 0.143 for region in config.REGIONS}
    propagation_fixed, energy_summary_fixed = run_energy_yield_propagation(
        output,
        n_samples=args.n_samples,
        alpha_by_region=fixed_alpha,
    )
    shear_comparison = make_shear_comparison(energy_summary, energy_summary_fixed)
    decomp = uncertainty_decomposition(propagation, year=2055)
    propagation.to_csv(config.PROCESSED_DATA_DIR / "energy_yield_propagation_samples.csv", index=False)
    energy_summary.to_csv(config.TABLES_DIR / "energy_yield_summary.csv", index=False)
    energy_summary.to_csv(config.TABLES_DIR / "energy_yield_summary_ple.csv", index=False)
    energy_summary_fixed.to_csv(config.TABLES_DIR / "energy_yield_summary_alpha0143.csv", index=False)
    shear_comparison.to_csv(config.TABLES_DIR / "energy_yield_shear_comparison.csv", index=False)
    decomp.to_csv(config.TABLES_DIR / "energy_yield_uncertainty_decomposition.csv", index=False)
    write_summary_latex(energy_summary)
    plot_aep_violins(propagation)
    plot_uncertainty_decomposition(decomp)
    plot_propagation_schematic(propagation, region="DE", year=2055)
    power_law = run_power_law_sensitivity(output, n_samples=args.n_samples)
    plot_power_law_sensitivity(power_law)
    correlation = run_correlation_sensitivity(output, n_samples=args.n_samples)
    gcm_variance = run_gcm_ensemble_variance(neighbors, decomp)
    plot_gcm_ensemble_variance(gcm_variance)

    LOGGER.info("=== GOWIRES Wind Climate ===\n%s", output.to_string(index=False))
    LOGGER.info("=== Energy Yield Summary ===\n%s", energy_summary.to_string(index=False))
    LOGGER.info("=== Energy Yield Shear Comparison ===\n%s", shear_comparison.to_string(index=False))
    LOGGER.info("=== Energy Yield Variance Decomposition ===\n%s", decomp.to_string(index=False))
    LOGGER.info("=== Power Law Sensitivity ===\n%s", power_law.to_string(index=False))
    LOGGER.info("=== Correlation Sensitivity ===\n%s", correlation.to_string(index=False))
    LOGGER.info(
        "=== GCM Ensemble Climate Variance (SSP2-4.5) ===\n%s",
        gcm_variance.loc[gcm_variance["scenario"] == "SSP2-4.5"]
        .groupby("region", as_index=False)
        .agg(
            var_climate_gcm=("var_climate_gcm", "first"),
            var_climate_previous_3scenario=("var_climate_previous_3scenario", "first"),
            climate_share_gcm=("climate_share_gcm", "first"),
            ple_alpha=("ple_alpha", "first"),
            var_climate_gcm_ple=("var_climate_gcm_ple", "first"),
            climate_share_gcm_ple=("climate_share_gcm_ple", "first"),
        )
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
