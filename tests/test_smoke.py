from pathlib import Path

import pandas as pd

from turbine_projections import config


def test_core_configuration_matches_project_decisions() -> None:
    assert config.PUBLISH_RAW_DATA is False
    assert config.HINDCAST_SPLITS == [2015, 2018]
    assert config.PRIMARY_HINDCAST_TRAIN_END == 2015
    assert config.METRICS == ["hub_height", "rotor_diameter", "specific_power"]
    assert "informative" in config.PRIOR_SETS
    assert config.PRIOR_SETS["informative"]["specific_power"]["model_type"] == "decay"


def test_cleaned_regional_csvs_pass_plausibility_checks() -> None:
    required = ["year", "hub_height_m", "rotor_diameter_m", "capacity_kw", "specific_power_wm2"]
    processed = Path(config.PROCESSED_DATA_DIR)
    summaries = []

    for region in config.REGIONS:
        path = processed / f"{region.lower()}_clean.csv"
        assert path.exists(), f"Missing processed file: {path}"
        df = pd.read_csv(path)
        assert not df[required].isna().any().any(), f"{region} has NaN in required fields"

        limits = config.PLAUSIBILITY_LIMITS
        assert df["hub_height_m"].between(*limits["hub_height_m"], inclusive="neither").all()
        assert df["rotor_diameter_m"].between(
            *limits["rotor_diameter_m"], inclusive="neither"
        ).all()
        assert df["specific_power_wm2"].between(
            *limits["specific_power_wm2"], inclusive="neither"
        ).all()
        assert (df["capacity_kw"] > limits["capacity_kw"][0]).all()

        expected_min_year, expected_max_year = config.EXPECTED_YEAR_RANGES[region]
        assert df["year"].min() >= expected_min_year
        assert df["year"].max() <= expected_max_year

        expected_min_count, expected_max_count = config.EXPECTED_TURBINE_COUNTS[region]
        assert expected_min_count <= len(df) <= expected_max_count

        year_2024 = df.loc[df["year"] == 2024, "specific_power_wm2"]
        median_2024 = year_2024.median() if not year_2024.empty else float("nan")
        summaries.append(
            f"{region}: {len(df):,} turbines, {int(df['year'].min())}-{int(df['year'].max())}, "
            f"median SP 2024: {median_2024:.0f} W/m2"
        )

    print("\n=== Data Preparation Summary ===")
    for summary in summaries:
        print(summary)
    print("All plausibility checks passed.")
