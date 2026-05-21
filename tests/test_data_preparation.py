from pathlib import Path

import pandas as pd

from turbine_projections.data_preparation import aggregate_annual_metrics, clean_at_data


def test_clean_at_data_keeps_turbine_level_observations() -> None:
    raw = pd.DataFrame(
        {
            "Location": ["A", "B", "C"],
            "Year": [2020, 2020, 2021],
            "Nabenhoehe": [120, 130, 140],
            "Rotordurchmesser": [110, 120, 130],
            "Total Power": ["3 MW", "4", "5 MW"],
            "source_sheet": ["gegenwart", "gegenwart", "zukunft1"],
        }
    )

    clean = clean_at_data(raw)

    assert len(clean) == 3
    assert clean["region"].unique().tolist() == ["AT"]
    assert {"hub_height_m", "rotor_diameter_m", "specific_power_wm2", "capacity_kw"}.issubset(
        clean.columns
    )
    assert clean["specific_power_wm2"].notna().all()


def test_aggregate_annual_metrics_uses_descriptive_statistics_only() -> None:
    clean = pd.DataFrame(
        {
            "region": ["AT", "AT", "AT"],
            "year": [2020, 2020, 2021],
            "hub_height_m": [100.0, 120.0, 140.0],
            "rotor_diameter_m": [100.0, 120.0, 140.0],
            "specific_power_wm2": [300.0, 250.0, 220.0],
            "capacity_kw": [2000.0, 3000.0, 4000.0],
        }
    )

    annual = aggregate_annual_metrics(clean)

    row_2020 = annual.loc[annual["year"] == 2020].iloc[0]
    assert row_2020["n_turbines"] == 2
    assert row_2020["hub_height_m_median"] == 110.0
    assert row_2020["specific_power_wm2_mean"] == 275.0


def test_at_raw_file_exists_in_workspace() -> None:
    expected = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "AT"
        / "Windparks_Gegenwart_und_Nahe_Zukunft_FINAL.xlsx"
    )
    assert expected.exists()
