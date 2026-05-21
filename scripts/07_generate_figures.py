"""Generate paper-ready descriptive data-section figures and tables."""

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
from turbine_projections.data_preparation import (
    _extract_year,
    _find_column,
    _to_numeric,
    compute_derived_metrics,
    find_at_raw_file,
    find_de_raw_path,
    find_us_raw_file,
    load_at_raw_excel,
)
from turbine_projections.plotting import APPLIED_ENERGY_FULL_WIDTH_IN, save_figure, set_paper_style
from turbine_projections.utils import configure_logging

LOGGER = logging.getLogger(__name__)

METRICS = [
    ("hub_height_m", "Hub height [m]", "HH"),
    ("rotor_diameter_m", "Rotor diameter [m]", "RD"),
    ("specific_power_wm2", "Specific power [W/m2]", "SP"),
]

PERIODS = [
    ("1986-1999", 1986, 1999),
    ("2000-2009", 2000, 2009),
    ("2010-2019", 2010, 2019),
    ("2020-2025", 2020, 2025),
]


def load_clean_data() -> dict[str, pd.DataFrame]:
    """Load cleaned regional turbine-level data."""

    data = {}
    for region in config.REGIONS:
        path = config.PROCESSED_DATA_DIR / f"{region.lower()}_clean.csv"
        data[region] = pd.read_csv(path)
    return data


def annual_quantiles(data: pd.DataFrame) -> pd.DataFrame:
    """Compute annual medians and IQRs from turbine-level data."""

    grouped = data.groupby("year")
    out = grouped.size().rename("n_turbines").to_frame()
    for metric, _, _ in METRICS:
        quantiles = grouped[metric].quantile([0.25, 0.5, 0.75]).unstack()
        out[f"{metric}_p25"] = quantiles[0.25]
        out[f"{metric}_median"] = quantiles[0.5]
        out[f"{metric}_p75"] = quantiles[0.75]
    return out.reset_index()


def all_annual_quantiles(clean: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {region: annual_quantiles(frame) for region, frame in clean.items()}


def period_mask(frame: pd.DataFrame, start: int, end: int) -> pd.Series:
    return frame["year"].between(start, end)


def figure_violins(clean: dict[str, pd.DataFrame]) -> None:
    """Figure 1: metric distributions by period and region."""

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(APPLIED_ENERGY_FULL_WIDTH_IN, 7.4),
        sharey="row",
    )
    offsets = {"AT": -0.24, "DE": 0.0, "US": 0.24}

    for row, (metric, label, _) in enumerate(METRICS):
        max_n = max(
            len(frame.loc[period_mask(frame, start, end), metric].dropna())
            for frame in clean.values()
            for _, start, end in PERIODS
        )
        for col, (period_label, start, end) in enumerate(PERIODS):
            ax = axes[row, col]
            regions = ["US"] if col == 0 else config.REGIONS
            for region in regions:
                values = clean[region].loc[period_mask(clean[region], start, end), metric].dropna()
                if values.empty:
                    continue
                n = len(values)
                width = 0.10 + 0.28 * np.sqrt(n / max_n)
                position = 1.0 + offsets[region]
                parts = ax.violinplot(
                    values.to_numpy(),
                    positions=[position],
                    widths=width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(config.REGION_COLORS[region])
                    body.set_edgecolor(config.REGION_COLORS[region])
                    body.set_alpha(0.55)
                q25, median, q75 = values.quantile([0.25, 0.5, 0.75])
                ax.plot([position, position], [q25, q75], color="black", linewidth=1.3, zorder=4)
                ax.scatter(position, median, s=18, color="white", edgecolor="black", linewidth=0.5, zorder=5)
            if row == 0:
                ax.set_title(period_label)
            if col == 0:
                ax.set_ylabel(label)
            ax.set_xlim(0.45, 1.55)
            ax.set_xticks([1 + offsets[r] for r in config.REGIONS])
            ax.set_xticklabels(config.REGIONS)
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, config.FIGURES_DIR / "data_descriptive_violins")
    plt.close(fig)


def figure_exploration_comparison(clean: dict[str, pd.DataFrame], annual: dict[str, pd.DataFrame]) -> None:
    """Figure 2: anonymized AT exploration and open-data DE/US scatter."""

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(APPLIED_ENERGY_FULL_WIDTH_IN, 7.3),
        sharex=False,
        sharey="row",
    )
    for row, (metric, label, _) in enumerate(METRICS):
        y_min = min(frame[metric].quantile(0.01) for frame in clean.values())
        y_max = max(frame[metric].quantile(0.99) for frame in clean.values())
        pad = 0.05 * (y_max - y_min)
        for col, region in enumerate(config.REGIONS):
            ax = axes[row, col]
            color = config.REGION_COLORS[region]
            yearly = annual[region]
            if region == "AT":
                ax.fill_between(
                    yearly["year"],
                    yearly[f"{metric}_p25"],
                    yearly[f"{metric}_p75"],
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                )
                ax.plot(yearly["year"], yearly[f"{metric}_median"], color=color, linewidth=2)
            else:
                ax.scatter(clean[region]["year"], clean[region][metric], s=4, alpha=0.12, color=color, edgecolors="none")
                ax.plot(yearly["year"], yearly[f"{metric}_median"], color="black", linewidth=1.8)
            if row == 0:
                ax.set_title(region)
            if col == 0:
                ax.set_ylabel(label)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlabel("Year")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, config.FIGURES_DIR / "data_exploration_comparison")
    plt.close(fig)


def figure_sample_sizes(annual: dict[str, pd.DataFrame]) -> None:
    """Figure 3: annual and cumulative sample sizes."""

    years = np.arange(
        min(frame["year"].min() for frame in annual.values()),
        max(frame["year"].max() for frame in annual.values()) + 1,
    )
    counts = pd.DataFrame(index=years)
    for region, frame in annual.items():
        counts[region] = frame.set_index("year")["n_turbines"].reindex(years, fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(APPLIED_ENERGY_FULL_WIDTH_IN, 5.7), sharex=True)
    bottom = np.zeros(len(years))
    for region in config.REGIONS:
        axes[0].bar(years, counts[region], bottom=bottom, color=config.REGION_COLORS[region], label=region, width=0.9)
        bottom += counts[region].to_numpy()
    axes[0].set_ylabel("Annual turbines")
    axes[0].legend(ncol=3, frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for region in config.REGIONS:
        axes[1].plot(years, counts[region].cumsum(), color=config.REGION_COLORS[region], linewidth=2, label=region)
    axes[1].set_ylabel("Cumulative turbines")
    axes[1].set_xlabel("Year")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, config.FIGURES_DIR / "data_sample_sizes")
    plt.close(fig)


def figure_normalized_trends(annual: dict[str, pd.DataFrame], reference_year: int = 2010) -> None:
    """Figure 4: normalized median and IQR trends with 2010 = 100."""

    fig, axes = plt.subplots(1, 3, figsize=(APPLIED_ENERGY_FULL_WIDTH_IN, 3.0), sharey=False)
    for ax, (metric, label, _) in zip(axes, METRICS):
        for region in config.REGIONS:
            yearly = annual[region].copy()
            reference = yearly.loc[yearly["year"] == reference_year, f"{metric}_median"]
            if reference.empty or not np.isfinite(reference.iloc[0]) or reference.iloc[0] == 0:
                continue
            ref = reference.iloc[0]
            color = config.REGION_COLORS[region]
            x = yearly["year"]
            median = 100 * yearly[f"{metric}_median"] / ref
            p25 = 100 * yearly[f"{metric}_p25"] / ref
            p75 = 100 * yearly[f"{metric}_p75"] / ref
            ax.fill_between(x, p25, p75, color=color, alpha=0.16, linewidth=0)
            ax.plot(x, median, color=color, linewidth=2, label=region)
        ax.axhline(100, color="0.4", linewidth=0.8, linestyle=":")
        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Index (2010 = 100)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, config.FIGURES_DIR / "data_normalized_trends")
    plt.close(fig)


def latex_escape(text: object) -> str:
    return str(text).replace("_", "\\_")


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(df.columns)
    alignment = "l" * len(columns)
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(latex_escape(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[column]) for column in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    latex = "\n".join(lines)
    path.write_text(latex, encoding="utf-8")


def raw_counts() -> dict[str, int]:
    """Compute raw source counts for Table 1."""

    counts: dict[str, int] = {}
    at_file = find_at_raw_file()
    counts["AT"] = len(pd.read_excel(at_file, sheet_name="Gegenwart", header=2)) + len(
        pd.read_excel(at_file, sheet_name="Zukunft I", header=4)
    )
    counts["US"] = sum(1 for _ in open(find_us_raw_file(), encoding="utf-8")) - 1
    de_path = find_de_raw_path()
    files = sorted(de_path.glob("*.csv")) if de_path.is_dir() else [de_path]
    counts["DE"] = sum(sum(1 for _ in open(file, encoding="utf-8")) - 1 for file in files)
    return counts


def table_summary(clean: dict[str, pd.DataFrame]) -> None:
    raw_n = raw_counts()
    sources = {"AT": "IGW/UVP", "DE": "MaStR", "US": "USWTDB"}
    rows = []
    for region in config.REGIONS:
        frame = clean[region]
        row = {
            "Region": region,
            "Source": sources[region],
            "Period": f"{int(frame['year'].min())}-{int(frame['year'].max())}",
            "N (raw)": f"{raw_n[region]:,}",
            "N (QC)": f"{len(frame):,}",
        }
        frame_2024 = frame.loc[frame["year"] == 2024]
        for metric, _, short in METRICS:
            q25, median, q75 = frame_2024[metric].quantile([0.25, 0.5, 0.75])
            unit = "[W/m2]" if metric == "specific_power_wm2" else "[m]"
            row[f"Median {short} 2024 {unit}"] = f"{median:.1f}"
            row[f"IQR {short} 2024 {unit}"] = f"{q25:.1f}-{q75:.1f}"
        rows.append(row)
    write_latex_table(
        pd.DataFrame(rows),
        config.TABLES_DIR / "data_summary_table.tex",
        "Summary of regional wind turbine datasets after quality control.",
        "tab:data-summary",
    )


def _count_common_exclusions(frame: pd.DataFrame) -> tuple[int, int, int]:
    with_derived = compute_derived_metrics(frame)
    before_missing = len(with_derived)
    required = ["year", "hub_height_m", "rotor_diameter_m", "capacity_kw", "specific_power_wm2"]
    valid = with_derived.dropna(subset=required)
    missing = before_missing - len(valid)
    limits = config.PLAUSIBILITY_LIMITS
    plausible_mask = (
        valid["year"].between(1900, 2060)
        & valid["hub_height_m"].between(*limits["hub_height_m"], inclusive="neither")
        & valid["rotor_diameter_m"].between(*limits["rotor_diameter_m"], inclusive="neither")
        & valid["specific_power_wm2"].between(*limits["specific_power_wm2"], inclusive="neither")
        & (valid["capacity_kw"] > limits["capacity_kw"][0])
    )
    plausible = valid.loc[plausible_mask].copy()
    plausibility = len(valid) - len(plausible)
    duplicates = len(plausible) - len(plausible.drop_duplicates())
    return missing, plausibility, duplicates


def exclusion_counts() -> dict[str, dict[str, int | str]]:
    """Recompute exclusion-stage counts from raw files."""

    counts: dict[str, dict[str, int | str]] = {}

    at_raw = load_at_raw_excel(find_at_raw_file())
    cap_col = _find_column(at_raw.columns.tolist(), ["Total Power", "Nennleistung", "Leistung"])
    capacity = _to_numeric(at_raw[cap_col])
    if capacity.dropna().median() < 100:
        capacity = capacity * 1000.0
    at_frame = pd.DataFrame(
        {
            "region": "AT",
            "year": _extract_year(at_raw[_find_column(at_raw.columns.tolist(), ["Year", "Baujahr"])]),
            "hub_height_m": _to_numeric(
                at_raw[_find_column(at_raw.columns.tolist(), ["Nabenhoehe", "Nabenh\u00f6he"])]
            ),
            "rotor_diameter_m": _to_numeric(
                at_raw[_find_column(at_raw.columns.tolist(), ["Rotordurchmesser"])]
            ),
            "capacity_kw": capacity,
        }
    )
    missing, plausibility, duplicates = _count_common_exclusions(at_frame)
    counts["AT"] = {
        "raw": len(at_raw),
        "onshore": "n/a",
        "missing": missing,
        "plausibility": plausibility,
        "duplicates": duplicates,
        "after_filters": len(at_raw) - missing - plausibility,
    }

    us_raw = pd.read_csv(find_us_raw_file(), low_memory=False)
    us_frame = pd.DataFrame(
        {
            "region": "US",
            "year": _extract_year(us_raw["p_year"]),
            "hub_height_m": _to_numeric(us_raw["t_hh"]),
            "rotor_diameter_m": _to_numeric(us_raw["t_rd"]),
            "capacity_kw": _to_numeric(us_raw["t_cap"]),
            "longitude": _to_numeric(us_raw["xlong"]),
            "latitude": _to_numeric(us_raw["ylat"]),
            "state": us_raw.get("t_state", pd.Series(index=us_raw.index, dtype="string")).astype("string"),
            "offshore": us_raw.get("t_offshore", pd.Series(index=us_raw.index, dtype="string")).astype("string"),
        }
    )
    us_geo = us_frame.loc[
        us_frame["latitude"].between(24, 50)
        & us_frame["longitude"].between(-125, -66)
        & ~us_frame["state"].isin(["AK", "HI", "PR", "GU", "VI"])
        & ~us_frame["offshore"].str.lower().isin(["t", "true", "1", "yes", "y"])
    ].copy()
    missing, plausibility, duplicates = _count_common_exclusions(us_geo)
    counts["US"] = {
        "raw": len(us_raw),
        "onshore": len(us_raw) - len(us_geo),
        "missing": missing,
        "plausibility": plausibility,
        "duplicates": duplicates,
        "after_filters": len(us_raw) - (len(us_raw) - len(us_geo)) - missing - plausibility,
    }

    de_path = find_de_raw_path()
    files = sorted(de_path.glob("*.csv")) if de_path.is_dir() else [de_path]
    de_raw = pd.concat(
        [pd.read_csv(file, sep=";", decimal=",", dtype="string", low_memory=False) for file in files],
        ignore_index=True,
    )
    columns = de_raw.columns.tolist()
    onshore_col = _find_column(columns, ["Wind an Land oder auf See", "Lage"], required=False)
    energy_col = _find_column(columns, ["Energietr\u00e4ger", "Energietraeger"], required=False)
    de_filtered = de_raw.copy()
    before_onshore = len(de_filtered)
    if onshore_col:
        onshore = de_filtered[onshore_col].astype("string").str.lower()
        de_filtered = de_filtered.loc[
            onshore.str.contains("land", na=False) & ~onshore.str.contains("see|offshore", na=False)
        ].copy()
    if energy_col:
        energy = de_filtered[energy_col].astype("string").str.lower()
        de_filtered = de_filtered.loc[energy.str.contains("wind", na=False)].copy()
    de_frame = pd.DataFrame(
        {
            "region": "DE",
            "year": _extract_year(
                de_filtered[_find_column(columns, ["Inbetriebnahmedatum der Einheit", "Inbetriebnahmedatum"])]
            ),
            "hub_height_m": _to_numeric(
                de_filtered[
                    _find_column(
                        columns,
                        ["Nabenh\u00f6he der Windenergieanlage", "Nabenhoehe der Windenergieanlage", "Nabenhoehe"],
                    )
                ]
            ),
            "rotor_diameter_m": _to_numeric(
                de_filtered[_find_column(columns, ["Rotordurchmesser der Windenergieanlage", "Rotordurchmesser"])]
            ),
            "capacity_kw": _to_numeric(
                de_filtered[
                    _find_column(
                        columns,
                        ["Nettonennleistung der Einheit", "Nettonennleistung", "Installierte Leistung der EEG-Anlage"],
                    )
                ]
            ),
        }
    )
    missing, plausibility, duplicates = _count_common_exclusions(de_frame)
    counts["DE"] = {
        "raw": len(de_raw),
        "onshore": before_onshore - len(de_filtered),
        "missing": missing,
        "plausibility": plausibility,
        "duplicates": duplicates,
        "after_filters": len(de_raw) - (before_onshore - len(de_filtered)) - missing - plausibility,
    }
    return counts


def table_exclusions(clean: dict[str, pd.DataFrame]) -> None:
    """Write a compact exclusions table using recomputed stage counts."""

    counts = exclusion_counts()
    rows = []
    for region in config.REGIONS:
        final_n = len(clean[region])
        region_counts = counts[region]
        rows.append(
            {
                "Region": region,
                "Onshore filter": region_counts["onshore"],
                "Missing fields": f"{region_counts['missing']:,}",
                "Plausibility": f"{region_counts['plausibility']:,}",
                "Duplicates": f"{max(0, int(region_counts['after_filters']) - final_n):,}",
                "Final N": f"{final_n:,}",
                "Total excluded": f"{int(region_counts['raw']) - final_n:,}",
            }
        )
    write_latex_table(
        pd.DataFrame(rows),
        config.TABLES_DIR / "data_exclusions_table.tex",
        "Data exclusions during quality control.",
        "tab:data-exclusions",
    )


def main() -> None:
    configure_logging()
    set_paper_style()
    clean = load_clean_data()
    annual = all_annual_quantiles(clean)
    figure_violins(clean)
    figure_exploration_comparison(clean, annual)
    figure_sample_sizes(annual)
    figure_normalized_trends(annual)
    table_summary(clean)
    table_exclusions(clean)
    LOGGER.info("Generated data-section figures and tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
