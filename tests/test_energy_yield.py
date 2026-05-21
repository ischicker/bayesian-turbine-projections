import numpy as np
import pytest

from turbine_projections.energy_yield import (
    DEFAULT_WIND_SPEEDS,
    estimate_rated_speed,
    power_output_kw,
    synthetic_power_curve,
)


def test_synthetic_power_curve_shape_and_bounds() -> None:
    curve = synthetic_power_curve(
        specific_power_wm2=300.0,
        rated_power_kw=3_000.0,
        rotor_diameter_m=113.0,
    )

    assert np.array_equal(curve.wind_speed_ms, DEFAULT_WIND_SPEEDS)
    assert curve.power_kw.shape == DEFAULT_WIND_SPEEDS.shape
    assert curve.capacity_factor.min() >= 0.0
    assert curve.capacity_factor.max() <= 1.0
    assert curve.power_kw.max() <= 3_000.0
    assert curve.power_kw[DEFAULT_WIND_SPEEDS < curve.cut_in_speed_ms].max() == 0.0
    assert np.all(curve.power_kw[DEFAULT_WIND_SPEEDS > 25.0] == 0.0)


def test_specific_power_shifts_rated_speed() -> None:
    low_sp_rated = estimate_rated_speed(200.0)
    high_sp_rated = estimate_rated_speed(500.0)

    assert low_sp_rated < high_sp_rated


def test_power_output_interpolates_arbitrary_wind_speeds() -> None:
    speeds = np.array([0.0, 3.5, 8.0, 12.0, 26.0])
    output = power_output_kw(
        speeds,
        specific_power_wm2=300.0,
        rated_power_kw=3_000.0,
        rotor_diameter_m=113.0,
    )

    assert output.shape == speeds.shape
    assert output[0] == 0.0
    assert output[-1] == 0.0
    assert output[2] < output[3] <= 3_000.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        synthetic_power_curve(0.0, 3_000.0, 113.0)
