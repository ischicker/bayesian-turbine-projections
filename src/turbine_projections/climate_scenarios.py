"""Climate scenario and Weibull parameter handling."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from turbine_projections import config

LOGGER = logging.getLogger(__name__)

GOWIRES_COUNTRIES = {
    "AT": "Austria",
    "DE": "Germany",
    "US": "United States",
}

SCENARIO_COLUMN_PREFIX = {
    "historical": None,
    "SSP2-4.5": "SSP245",
    "SSP5-8.5": "SSP585",
}


def gowires_csv_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    candidate = config.RAW_DATA_DIR / "GOWIRES" / "GOWIRES_V1.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing GOWIRES CSV: {candidate}")
    return candidate


def load_gowires(path: str | Path | None = None) -> pd.DataFrame:
    """Load GOWIRES CSV with the dataset's semicolon delimiter."""

    csv_path = gowires_csv_path(path)
    columns = pd.read_csv(csv_path, sep=";", nrows=0, encoding="latin1").columns.tolist()
    usecols = [
        col
        for col in columns
        if col
        in {
            "full_id",
            "longitude",
            "latitude",
            "country",
            "wb_c_hist",
            "wb_k_hist",
            "weibull_c_hist",
            "weibull_k_hist",
            "PLE",
        }
        or col.startswith("weibull_c_SSP")
        or col.startswith("weibull_k_SSP")
    ]
    data = pd.read_csv(csv_path, sep=";", usecols=usecols, low_memory=False, encoding="latin1")
    for column in data.columns:
        if column not in {"full_id", "country"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in kilometres."""

    radius_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def nearest_turbines(gowires: pd.DataFrame, region: str, n_neighbors: int = 50) -> pd.DataFrame:
    """Return nearest GOWIRES turbine rows for one configured reference region."""

    site = config.REFERENCE_SITES[region]
    country = GOWIRES_COUNTRIES.get(region)
    subset = gowires.copy()
    if country and "country" in subset:
        country_subset = subset.loc[subset["country"] == country].copy()
        if len(country_subset) >= n_neighbors:
            subset = country_subset
        else:
            LOGGER.warning("Only %d GOWIRES rows for %s; using all countries.", len(country_subset), country)
    subset = subset.dropna(subset=["latitude", "longitude"]).copy()
    subset["distance_km"] = haversine_km(
        site["lat"],
        site["lon"],
        subset["latitude"].to_numpy(dtype=float),
        subset["longitude"].to_numpy(dtype=float),
    )
    subset["region"] = region
    subset["reference_name"] = site["name"]
    return subset.nsmallest(n_neighbors, "distance_km").reset_index(drop=True)


def _column_pair_models(columns: pd.Index, scenario_prefix: str) -> list[str]:
    c_prefix = f"weibull_c_{scenario_prefix}_"
    return sorted(col.replace(c_prefix, "") for col in columns if col.startswith(c_prefix))


def summarize_weibull(neighbors: pd.DataFrame, region: str) -> list[dict[str, float | str]]:
    """Summarize historical and scenario Weibull parameters for 50-neighbor means."""

    rows: list[dict[str, float | str]] = []
    historical_c = "wb_c_hist" if "wb_c_hist" in neighbors.columns else "weibull_c_hist"
    historical_k = "wb_k_hist" if "wb_k_hist" in neighbors.columns else "weibull_k_hist"
    rows.append(
        {
            "region": region,
            "scenario": "historical",
            "weibull_k": float(neighbors[historical_k].median()),
            "weibull_A": float(neighbors[historical_c].median()),
            "weibull_k_min": float(neighbors[historical_k].quantile(0.05)),
            "weibull_k_max": float(neighbors[historical_k].quantile(0.95)),
            "weibull_A_min": float(neighbors[historical_c].quantile(0.05)),
            "weibull_A_max": float(neighbors[historical_c].quantile(0.95)),
            "n_turbines": int(len(neighbors)),
            "max_distance_km": float(neighbors["distance_km"].max()),
        }
    )
    for scenario, prefix in SCENARIO_COLUMN_PREFIX.items():
        if scenario == "historical" or prefix is None:
            continue
        model_values = []
        for model in _column_pair_models(neighbors.columns, prefix):
            c_col = f"weibull_c_{prefix}_{model}"
            k_col = f"weibull_k_{prefix}_{model}"
            if c_col not in neighbors or k_col not in neighbors:
                continue
            model_values.append(
                {
                    "model": model,
                    "weibull_A": float(neighbors[c_col].median()),
                    "weibull_k": float(neighbors[k_col].median()),
                }
            )
        model_df = pd.DataFrame(model_values).dropna(subset=["weibull_A", "weibull_k"])
        rows.append(
            {
                "region": region,
                "scenario": scenario,
                "weibull_k": float(model_df["weibull_k"].median()),
                "weibull_A": float(model_df["weibull_A"].median()),
                "weibull_k_min": float(model_df["weibull_k"].min()),
                "weibull_k_max": float(model_df["weibull_k"].max()),
                "weibull_A_min": float(model_df["weibull_A"].min()),
                "weibull_A_max": float(model_df["weibull_A"].max()),
                "n_turbines": int(len(neighbors)),
                "n_gcms": int(len(model_df)),
                "max_distance_km": float(neighbors["distance_km"].max()),
            }
        )
    return rows


def extract_gowires_wind_climate(
    path: str | Path | None = None,
    n_neighbors: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract 100 m Weibull climate summaries for configured reference sites."""

    gowires = load_gowires(path)
    summary_rows = []
    neighbor_frames = []
    for region in config.REGIONS:
        neighbors = nearest_turbines(gowires, region, n_neighbors=n_neighbors)
        neighbor_frames.append(neighbors)
        summary_rows.extend(summarize_weibull(neighbors, region))
        LOGGER.info(
            "GOWIRES %s: selected %d turbines, max distance %.1f km",
            region,
            len(neighbors),
            neighbors["distance_km"].max(),
        )
    summary = pd.DataFrame(summary_rows)
    neighbors = pd.concat(neighbor_frames, ignore_index=True)
    return summary, neighbors


def weibull_pdf(wind_speed_ms: np.ndarray, weibull_k: float, weibull_A: float) -> np.ndarray:
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=float)
    return (weibull_k / weibull_A) * (wind_speed_ms / weibull_A) ** (weibull_k - 1.0) * np.exp(
        -((wind_speed_ms / weibull_A) ** weibull_k)
    )
