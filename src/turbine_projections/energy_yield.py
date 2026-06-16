"""Synthetic power curves and annual energy production calculations.

The power-curve generator implemented here follows the idea used by Ryberg et
al. (2019), Energy 182, Appendix A: when no manufacturer curve is available, a
synthetic normalized turbine power curve is inferred from the turbine's specific
power and then scaled to rated power. This keeps the project independent from
RESKit/windtools while preserving the same modelling assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.bayesian_model import posterior_predictive_samples
from turbine_projections.climate_scenarios import weibull_pdf


DEFAULT_WIND_SPEEDS = np.arange(0.0, 30.0 + 0.5, 0.5)
DEFAULT_CUT_OUT_SPEED = 25.0


@dataclass(frozen=True)
class SyntheticPowerCurve:
    """Synthetic wind turbine power curve.

    Attributes
    ----------
    wind_speed_ms:
        Wind-speed grid in metres per second.
    power_kw:
        Electrical power output in kW at each wind speed.
    capacity_factor:
        Normalized power output in the interval [0, 1].
    specific_power_wm2:
        Turbine specific power in W/m2 used to generate the curve.
    rated_power_kw:
        Rated turbine power in kW.
    rotor_diameter_m:
        Rotor diameter in metres.
    cut_in_speed_ms:
        Estimated cut-in wind speed.
    rated_speed_ms:
        Estimated rated wind speed where the curve reaches rated power.
    cut_out_speed_ms:
        Cut-out wind speed.
    """

    wind_speed_ms: np.ndarray
    power_kw: np.ndarray
    capacity_factor: np.ndarray
    specific_power_wm2: float
    rated_power_kw: float
    rotor_diameter_m: float
    cut_in_speed_ms: float
    rated_speed_ms: float
    cut_out_speed_ms: float


def _validate_positive(name: str, value: float) -> float:
    value_float = float(value)
    if not np.isfinite(value_float) or value_float <= 0:
        raise ValueError(f"{name} must be a positive finite value, got {value!r}.")
    return value_float


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Cubic Hermite transition from 0 to 1 with zero end slopes."""

    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def estimate_cut_in_speed(specific_power_wm2: float) -> float:
    """Estimate cut-in speed from turbine specific power.

    Lower-specific-power turbines generally begin producing at slightly lower
    wind speeds. The range is constrained to the 3-4 m/s interval used in the
    Ryberg-style synthetic curve assumption.
    """

    sp = _validate_positive("specific_power_wm2", specific_power_wm2)
    return float(np.interp(np.clip(sp, 150.0, 500.0), [150.0, 500.0], [3.0, 4.0]))


def estimate_rated_speed(specific_power_wm2: float) -> float:
    """Estimate rated wind speed from turbine specific power.

    The estimate is anchored on the cubic wind-power relation. A reference
    turbine with 300 W/m2 reaches rated power near 12 m/s; lower specific power
    shifts the rated point down and higher specific power shifts it up.
    """

    sp = _validate_positive("specific_power_wm2", specific_power_wm2)
    rated = 12.0 * (sp / 300.0) ** (1.0 / 3.0)
    return float(np.clip(rated, 9.0, 16.0))


def normalized_power_curve(
    specific_power_wm2: float,
    wind_speed_ms: np.ndarray | None = None,
    cut_out_speed_ms: float = DEFAULT_CUT_OUT_SPEED,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate a normalized synthetic power curve from specific power.

    Parameters
    ----------
    specific_power_wm2:
        Turbine specific power in W/m2.
    wind_speed_ms:
        Optional wind-speed grid in m/s. Defaults to 0-30 m/s in 0.5 m/s steps.
    cut_out_speed_ms:
        Cut-out wind speed in m/s. Defaults to 25 m/s.

    Returns
    -------
    tuple
        Wind-speed grid, normalized capacity factor, cut-in speed, rated speed.
    """

    sp = _validate_positive("specific_power_wm2", specific_power_wm2)
    speeds = DEFAULT_WIND_SPEEDS.copy() if wind_speed_ms is None else np.asarray(wind_speed_ms, dtype=float)
    if speeds.ndim != 1 or not np.all(np.isfinite(speeds)):
        raise ValueError("wind_speed_ms must be a one-dimensional finite array.")

    cut_in = estimate_cut_in_speed(sp)
    rated = max(estimate_rated_speed(sp), cut_in + 1.0)
    cut_out = _validate_positive("cut_out_speed_ms", cut_out_speed_ms)

    capacity_factor = np.zeros_like(speeds, dtype=float)
    ramp_mask = (speeds >= cut_in) & (speeds < rated)
    plateau_mask = (speeds >= rated) & (speeds <= cut_out)

    ramp_position = (speeds[ramp_mask] - cut_in) / (rated - cut_in)
    capacity_factor[ramp_mask] = _smoothstep(ramp_position)
    capacity_factor[plateau_mask] = 1.0
    capacity_factor[speeds > cut_out] = 0.0

    return speeds, np.clip(capacity_factor, 0.0, 1.0), cut_in, rated


def synthetic_power_curve(
    specific_power_wm2: float,
    rated_power_kw: float,
    rotor_diameter_m: float,
    wind_speed_ms: np.ndarray | None = None,
    cut_out_speed_ms: float = DEFAULT_CUT_OUT_SPEED,
) -> SyntheticPowerCurve:
    """Create a Ryberg-style synthetic wind turbine power curve.

    Parameters
    ----------
    specific_power_wm2:
        Turbine specific power in W/m2.
    rated_power_kw:
        Rated turbine power in kW.
    rotor_diameter_m:
        Rotor diameter in m. The value is validated and stored for traceability;
        specific power is supplied explicitly because this project samples it
        directly from the Bayesian posterior.
    wind_speed_ms:
        Optional wind-speed grid in m/s. Defaults to 0-30 m/s in 0.5 m/s steps.
    cut_out_speed_ms:
        Cut-out wind speed in m/s. Defaults to 25 m/s.

    Returns
    -------
    SyntheticPowerCurve
        Dataclass containing wind speeds, capacity factors and power in kW.
    """

    sp = _validate_positive("specific_power_wm2", specific_power_wm2)
    rated_power = _validate_positive("rated_power_kw", rated_power_kw)
    rotor_diameter = _validate_positive("rotor_diameter_m", rotor_diameter_m)
    speeds, capacity_factor, cut_in, rated = normalized_power_curve(
        sp,
        wind_speed_ms=wind_speed_ms,
        cut_out_speed_ms=cut_out_speed_ms,
    )
    power_kw = np.minimum(capacity_factor * rated_power, rated_power)
    power_kw = np.where(speeds > cut_out_speed_ms, 0.0, power_kw)

    return SyntheticPowerCurve(
        wind_speed_ms=speeds,
        power_kw=power_kw,
        capacity_factor=capacity_factor,
        specific_power_wm2=sp,
        rated_power_kw=rated_power,
        rotor_diameter_m=rotor_diameter,
        cut_in_speed_ms=cut_in,
        rated_speed_ms=rated,
        cut_out_speed_ms=float(cut_out_speed_ms),
    )


def power_output_kw(
    wind_speed_ms: np.ndarray,
    specific_power_wm2: float,
    rated_power_kw: float,
    rotor_diameter_m: float,
    cut_out_speed_ms: float = DEFAULT_CUT_OUT_SPEED,
) -> np.ndarray:
    """Evaluate synthetic turbine power output at arbitrary wind speeds."""

    wind_speed = np.asarray(wind_speed_ms, dtype=float)
    curve = synthetic_power_curve(
        specific_power_wm2=specific_power_wm2,
        rated_power_kw=rated_power_kw,
        rotor_diameter_m=rotor_diameter_m,
        wind_speed_ms=DEFAULT_WIND_SPEEDS,
        cut_out_speed_ms=cut_out_speed_ms,
    )
    return np.interp(wind_speed, curve.wind_speed_ms, curve.power_kw, left=0.0, right=0.0)


def rated_power_mw(specific_power_wm2: np.ndarray, rotor_diameter_m: np.ndarray) -> np.ndarray:
    """Compute rated turbine power from specific power and rotor diameter."""

    area_m2 = np.pi * (np.asarray(rotor_diameter_m, dtype=float) / 2.0) ** 2
    return np.asarray(specific_power_wm2, dtype=float) * area_m2 / 1_000_000.0


def capacity_factor_from_weibull(
    specific_power_wm2: float,
    rated_power_mw_value: float,
    rotor_diameter_m: float,
    weibull_k: float,
    weibull_A: float,
    wind_speed_ms: np.ndarray | None = None,
) -> float:
    """Integrate normalized synthetic power over a Weibull wind distribution."""

    speeds = DEFAULT_WIND_SPEEDS if wind_speed_ms is None else np.asarray(wind_speed_ms, dtype=float)
    curve = synthetic_power_curve(
        specific_power_wm2=specific_power_wm2,
        rated_power_kw=rated_power_mw_value * 1000.0,
        rotor_diameter_m=rotor_diameter_m,
        wind_speed_ms=speeds,
    )
    pdf = weibull_pdf(curve.wind_speed_ms, weibull_k=weibull_k, weibull_A=weibull_A)
    return float(np.trapezoid(curve.capacity_factor * pdf, curve.wind_speed_ms))


def _load_trace(region: str, metric: str) -> az.InferenceData:
    path = config.POSTERIORS_DIR / f"{region.lower()}_{metric}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing posterior: {path}")
    return az.from_netcdf(path)


def _common_sample_indices(traces: dict[str, az.InferenceData], n_samples: int) -> np.ndarray:
    n_available = min(int(np.prod(trace.posterior["L"].shape)) for trace in traces.values())
    if n_available < n_samples:
        n_samples = n_available
    rng = np.random.default_rng(config.RANDOM_SEED)
    return np.sort(rng.choice(n_available, size=n_samples, replace=False))


def joint_technology_samples(region: str, year: int, n_samples: int = 1000) -> pd.DataFrame:
    """Draw joint technology samples using the same flattened index per metric."""

    traces = {metric: _load_trace(region, metric) for metric in config.METRICS}
    indices = _common_sample_indices(traces, n_samples)
    samples = {}
    for metric in config.METRICS:
        model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
        predicted = posterior_predictive_samples(traces[metric], model_type, np.asarray([year], dtype=float))
        samples[metric] = predicted.reshape(-1)[indices]
    result = pd.DataFrame(
        {
            "sample_id": np.arange(len(indices)),
            "posterior_index": indices,
            "year": year,
            "hub_height_m": samples["hub_height"],
            "rotor_diameter_m": samples["rotor_diameter"],
            "specific_power_wm2": samples["specific_power"],
        }
    )
    result["rated_power_mw"] = rated_power_mw(result["specific_power_wm2"], result["rotor_diameter_m"])
    return result


def propagate_energy_yield(
    region: str,
    year: int,
    scenario_row: pd.Series,
    technology_samples: pd.DataFrame,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Propagate technology and Weibull climate uncertainty to CF and AEP."""

    alpha_value = config.SITE_PLE_ALPHA[region] if alpha is None else float(alpha)
    rows = []
    for _, sample in technology_samples.iterrows():
        weibull_A_hub = float(scenario_row["weibull_A"]) * (float(sample["hub_height_m"]) / 100.0) ** alpha_value
        cf = capacity_factor_from_weibull(
            specific_power_wm2=float(sample["specific_power_wm2"]),
            rated_power_mw_value=float(sample["rated_power_mw"]),
            rotor_diameter_m=float(sample["rotor_diameter_m"]),
            weibull_k=float(scenario_row["weibull_k"]),
            weibull_A=weibull_A_hub,
        )
        aep_mwh = cf * float(sample["rated_power_mw"]) * 8760.0
        rows.append(
            {
                "region": region,
                "year": year,
                "scenario": scenario_row["scenario"],
                "sample_id": int(sample["sample_id"]),
                "hub_height_m": float(sample["hub_height_m"]),
                "rotor_diameter_m": float(sample["rotor_diameter_m"]),
                "specific_power_wm2": float(sample["specific_power_wm2"]),
                "rated_power_mw": float(sample["rated_power_mw"]),
                "weibull_k": float(scenario_row["weibull_k"]),
                "weibull_A_100m": float(scenario_row["weibull_A"]),
                "weibull_A_hub": weibull_A_hub,
                "alpha": alpha_value,
                "capacity_factor": cf,
                "aep_mwh": aep_mwh,
            }
        )
    return pd.DataFrame(rows)


def run_energy_yield_propagation(
    gowires_climate: pd.DataFrame,
    years: list[int] | tuple[int, ...] = tuple(config.TARGET_YEARS),
    n_samples: int = 1000,
    alpha_by_region: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run end-to-end probabilistic AEP propagation for all regions."""

    alpha_by_region = config.SITE_PLE_ALPHA if alpha_by_region is None else alpha_by_region
    propagation_frames = []
    technology_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for region in config.REGIONS:
        climate_region = gowires_climate.loc[gowires_climate["region"] == region]
        for year in years:
            tech = joint_technology_samples(region, year, n_samples=n_samples)
            technology_cache[(region, year)] = tech
            for _, scenario_row in climate_region.iterrows():
                propagation_frames.append(
                    propagate_energy_yield(region, year, scenario_row, tech, alpha=alpha_by_region[region])
                )
    propagation = pd.concat(propagation_frames, ignore_index=True)
    summary = (
        propagation.groupby(["region", "year", "scenario"], as_index=False)
        .agg(
            cf_median=("capacity_factor", "median"),
            cf_p5=("capacity_factor", lambda s: float(np.quantile(s, 0.05))),
            cf_p95=("capacity_factor", lambda s: float(np.quantile(s, 0.95))),
            aep_mwh_median=("aep_mwh", "median"),
            aep_mwh_p5=("aep_mwh", lambda s: float(np.quantile(s, 0.05))),
            aep_mwh_p95=("aep_mwh", lambda s: float(np.quantile(s, 0.95))),
            rated_power_mw_median=("rated_power_mw", "median"),
            alpha=("alpha", "first"),
            n_samples=("sample_id", "count"),
        )
        .sort_values(["region", "year", "scenario"])
    )
    return propagation, summary


def uncertainty_decomposition(propagation: pd.DataFrame, year: int = 2055) -> pd.DataFrame:
    """Decompose 2055 variance into technology and climate components."""

    rows = []
    for region in config.REGIONS:
        subset = propagation.loc[(propagation["region"] == region) & (propagation["year"] == year)]
        historical = subset.loc[subset["scenario"] == "historical"]
        var_tech = float(np.var(historical["aep_mwh"], ddof=1))

        scenario_medians = subset.groupby("scenario")["aep_mwh"].median()
        var_climate = float(np.var(scenario_medians.to_numpy(dtype=float), ddof=1))
        total = var_tech + var_climate
        rows.append(
            {
                "region": region,
                "year": year,
                "var_technology": var_tech,
                "var_climate": var_climate,
                "total_variance": total,
                "technology_share": var_tech / total if total > 0 else np.nan,
                "climate_share": var_climate / total if total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
