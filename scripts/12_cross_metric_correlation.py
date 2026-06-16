"""Reserve M4 analysis: empirical cross-metric posterior correlations.

This script reuses the same index-matched technology projection draws used by
the AEP propagation and the correlation-shuffle sensitivity. No MCMC is run.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.energy_yield import joint_technology_samples


REGIONS = ["AT", "DE", "US"]
TARGET_YEARS = [2030, 2055]
METRIC_COLUMNS = {
    "hub_height": "hub_height_m",
    "rotor_diameter": "rotor_diameter_m",
    "specific_power": "specific_power_wm2",
}
PAIR_LABELS = {
    ("hub_height", "rotor_diameter"): "HH-RD",
    ("hub_height", "specific_power"): "HH-SP",
    ("rotor_diameter", "specific_power"): "RD-SP",
}


def compute_correlations(n_samples: int = 1000) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        for year in TARGET_YEARS:
            tech = joint_technology_samples(region, year, n_samples=n_samples)
            matrix = np.vstack([tech[column].to_numpy(float) for column in METRIC_COLUMNS.values()])
            corr = np.corrcoef(matrix)
            metric_names = list(METRIC_COLUMNS)
            print(
                f"{region} {year}: "
                f"HH-RD={corr[0, 1]:+.3f} "
                f"HH-SP={corr[0, 2]:+.3f} "
                f"RD-SP={corr[1, 2]:+.3f}"
            )
            for a, b in combinations(range(len(metric_names)), 2):
                metric_pair = (metric_names[a], metric_names[b])
                rows.append(
                    {
                        "region": region,
                        "year": year,
                        "pair": PAIR_LABELS[metric_pair],
                        "pearson_r": float(corr[a, b]),
                        "n_samples": int(len(tech)),
                    }
                )
    return pd.DataFrame(rows)


def write_summary_tex(correlations: pd.DataFrame) -> None:
    pivot = correlations.pivot_table(
        index=["region", "year"],
        columns="pair",
        values="pearson_r",
        aggfunc="first",
    )
    ordered_pairs = ["HH-RD", "HH-SP", "RD-SP"]
    lines = []
    for (region, year), row in pivot.iterrows():
        values = " & ".join(f"{row[pair]:+.2f}" for pair in ordered_pairs)
        lines.append(f"{region} & {year} & {values} \\\\")
    (config.TABLES_DIR / "cross_metric_correlation_summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    correlations = compute_correlations(n_samples=1000)
    correlations.to_csv(config.TABLES_DIR / "cross_metric_correlation.csv", index=False)
    write_summary_tex(correlations)
    max_abs = correlations["pearson_r"].abs().max()
    print(f"\nMax |r| = {max_abs:.3f}")
    print(
        "Wrote "
        f"{config.TABLES_DIR / 'cross_metric_correlation.csv'} and "
        f"{config.TABLES_DIR / 'cross_metric_correlation_summary.tex'}"
    )


if __name__ == "__main__":
    main()
