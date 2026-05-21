"""Data loading, cleaning, aggregation, and exploration plots."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbine_projections import config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDataPaths:
    """Output paths produced by a region-specific preparation step."""

    raw_clean: Path
    annual: Path


CANONICAL_COLUMNS = [
    "region",
    "year",
    "hub_height_m",
    "rotor_diameter_m",
    "capacity_kw",
    "rotor_area_m2",
    "specific_power_wm2",
]

METRIC_COLUMNS = ["hub_height_m", "rotor_diameter_m", "specific_power_wm2"]
REQUIRED_COLUMNS = ["year", "hub_height_m", "rotor_diameter_m", "capacity_kw"]


def _normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    translation = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
    text = text.translate(translation)
    text = (
        text.replace("\u00e4", "ae")
        .replace("\u00f6", "oe")
        .replace("\u00fc", "ue")
        .replace("\u00df", "ss")
    )
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _find_column(columns: list[str], aliases: list[str], required: bool = True) -> str | None:
    normalized = {_normalize_column_name(column): column for column in columns}
    for alias in aliases:
        key = _normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Could not find any of these columns: {aliases}")
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    values = (
        series.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace("MW", "", case=False, regex=False)
        .str.replace("kW", "", case=False, regex=False)
        .str.strip()
    )
    has_comma = values.str.contains(",", regex=False, na=False)
    german_decimal = values[has_comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    standard_decimal = values[~has_comma].str.replace(",", "", regex=False)
    normalized = pd.Series(index=values.index, dtype="string")
    normalized.loc[has_comma] = german_decimal
    normalized.loc[~has_comma] = standard_decimal
    return pd.to_numeric(normalized, errors="coerce")


def _extract_year(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    year = parsed.dt.year
    numeric = pd.to_numeric(text, errors="coerce")
    return year.fillna(numeric).astype("Int64")


def _log_exclusions(region: str, stage_counts: dict[str, int]) -> None:
    previous = stage_counts["input"]
    for stage, count in stage_counts.items():
        if stage == "input":
            continue
        LOGGER.info("%s excluded at %s: %d", region, stage, previous - count)
        previous = count


def _apply_common_cleaning(df: pd.DataFrame, region: str) -> pd.DataFrame:
    stage_counts = {"input": len(df)}
    data = compute_derived_metrics(df)

    data = data.dropna(subset=REQUIRED_COLUMNS + ["specific_power_wm2"])
    stage_counts["valid_required_fields"] = len(data)

    limits = config.PLAUSIBILITY_LIMITS
    mask = (
        data["year"].between(1900, 2060)
        & data["hub_height_m"].between(*limits["hub_height_m"], inclusive="neither")
        & data["rotor_diameter_m"].between(*limits["rotor_diameter_m"], inclusive="neither")
        & data["specific_power_wm2"].between(*limits["specific_power_wm2"], inclusive="neither")
        & (data["capacity_kw"] > limits["capacity_kw"][0])
    )
    data = data.loc[mask].copy()
    stage_counts["plausibility"] = len(data)

    data["year"] = data["year"].astype(int)
    data = data.drop_duplicates()
    stage_counts["duplicates"] = len(data)

    _log_exclusions(region, stage_counts)
    LOGGER.info(
        "%s cleaned: %d turbines, years %s-%s",
        region,
        len(data),
        int(data["year"].min()) if not data.empty else "NA",
        int(data["year"].max()) if not data.empty else "NA",
    )
    ordered = CANONICAL_COLUMNS + [
        column for column in data.columns if column not in CANONICAL_COLUMNS
    ]
    return data.loc[:, ordered].sort_values(["year"]).reset_index(drop=True)


def find_at_raw_file(raw_data_dir: Path = config.RAW_DATA_DIR) -> Path:
    """Return the Austrian wind-park Excel file path.

    Parameters
    ----------
    raw_data_dir:
        Root folder containing raw regional data.

    Returns
    -------
    pathlib.Path
        Path to ``Windparks_Gegenwart_und_Nahe_Zukunft_FINAL.xlsx``.

    Raises
    ------
    FileNotFoundError
        If the expected Excel file cannot be found.
    """

    candidates = [
        raw_data_dir / "AT" / "Windparks_Gegenwart_und_Nahe_Zukunft_FINAL.xlsx",
        config.WORKSPACE_ROOT
        / "Turbine metrics extrapolation"
        / "Windparks_Gegenwart_und_Nahe_Zukunft_FINAL.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Austrian raw data. Expected "
        "data/raw/AT/Windparks_Gegenwart_und_Nahe_Zukunft_FINAL.xlsx."
    )


def load_at_raw_excel(path: Path | None = None) -> pd.DataFrame:
    """Load and combine the Austrian current and near-future Excel sheets.

    Parameters
    ----------
    path:
        Optional explicit path to the Austrian Excel workbook.

    Returns
    -------
    pandas.DataFrame
        Combined raw observations with original source columns.
    """

    workbook = path or find_at_raw_file()
    LOGGER.info("Loading AT workbook: %s", workbook)

    sheets = {
        "gegenwart": pd.read_excel(workbook, sheet_name="Gegenwart", header=2),
        "zukunft1": pd.read_excel(workbook, sheet_name="Zukunft I", header=4),
    }

    frames = []
    required = ["Location", "Year", "Nabenhoehe", "Rotordurchmesser", "Total Power"]
    for sheet_name, frame in sheets.items():
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"AT sheet {sheet_name!r} is missing columns: {missing}")
        selected = frame[required].copy()
        selected["source_sheet"] = sheet_name
        frames.append(selected)
        LOGGER.info("Loaded AT sheet %s with %d rows", sheet_name, len(selected))

    return pd.concat(frames, ignore_index=True)


def load_and_clean_at(path: str) -> pd.DataFrame:
    """Load and clean Austrian wind park data from the IGW/UVP Excel workbook.

    Parameters
    ----------
    path:
        Path to the Austrian IGW/UVP Excel workbook.

    Returns
    -------
    pandas.DataFrame
        Clean turbine-level observations.
    """

    raw = load_at_raw_excel(Path(path))
    return clean_at_data(raw)


def clean_at_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean Austrian turbine-level observations.

    Parameters
    ----------
    raw:
        Combined raw Austrian observations.

    Returns
    -------
    pandas.DataFrame
        Clean turbine-level observations with canonical column names and units.
    """

    year_col = _find_column(raw.columns.tolist(), ["Year", "Baujahr"])
    location_col = _find_column(raw.columns.tolist(), ["Location", "Standort"], required=False)
    hub_col = _find_column(raw.columns.tolist(), ["Nabenhoehe", "Nabenhöhe", "Nabenhöhe [m]"])
    rotor_col = _find_column(raw.columns.tolist(), ["Rotordurchmesser", "Rotordurchmesser [m]"])
    capacity_col = _find_column(raw.columns.tolist(), ["Total Power", "Nennleistung", "Leistung"])
    source_col = _find_column(raw.columns.tolist(), ["Datenquelle", "Quelle", "source"], required=False)

    capacity = _to_numeric(raw[capacity_col])
    if capacity.dropna().median() < 100:
        capacity = capacity * 1000.0

    data = pd.DataFrame(
        {
            "region": "AT",
            "year": _extract_year(raw[year_col]),
            "hub_height_m": _to_numeric(raw[hub_col]),
            "rotor_diameter_m": _to_numeric(raw[rotor_col]),
            "capacity_kw": capacity,
            "source_sheet": raw.get("source_sheet", pd.Series(index=raw.index, dtype="string")),
        }
    )
    if location_col:
        data["location"] = raw[location_col].astype("string")
    if source_col:
        data["source"] = raw[source_col].astype("string")
    else:
        data["source"] = data["source_sheet"].astype("string")

    return _apply_common_cleaning(data, "AT")


def load_and_clean_us(path: str) -> pd.DataFrame:
    """USWTDB CSV laden und bereinigen.

    Parameters
    ----------
    path:
        Path to a USWTDB CSV file.

    Returns
    -------
    pandas.DataFrame
        Clean turbine-level observations for the continental US.
    """

    raw = pd.read_csv(path, low_memory=False)
    data = pd.DataFrame(
        {
            "region": "US",
            "year": _extract_year(raw["p_year"]),
            "hub_height_m": _to_numeric(raw["t_hh"]),
            "rotor_diameter_m": _to_numeric(raw["t_rd"]),
            "capacity_kw": _to_numeric(raw["t_cap"]),
            "longitude": _to_numeric(raw["xlong"]),
            "latitude": _to_numeric(raw["ylat"]),
            "state": raw.get("t_state", pd.Series(index=raw.index, dtype="string")).astype("string"),
            "offshore": raw.get("t_offshore", pd.Series(index=raw.index, dtype="string")).astype("string"),
            "source": "USWTDB",
        }
    )
    before_geo = len(data)
    data = data.loc[
        data["latitude"].between(24, 50)
        & data["longitude"].between(-125, -66)
        & ~data["state"].isin(["AK", "HI", "PR", "GU", "VI"])
        & ~data["offshore"].str.lower().isin(["t", "true", "1", "yes", "y"])
    ].copy()
    LOGGER.info("US excluded by continental/onshore filter: %d", before_geo - len(data))
    return _apply_common_cleaning(data, "US")


def load_and_clean_de(path: str) -> pd.DataFrame:
    """Marktstammdatenregister CSV laden und bereinigen.

    Parameters
    ----------
    path:
        Path to one MaStR CSV file or a directory containing MaStR CSV files.

    Returns
    -------
    pandas.DataFrame
        Clean onshore wind turbine observations for Germany.
    """

    input_path = Path(path)
    files = sorted(input_path.glob("*.csv")) if input_path.is_dir() else [input_path]
    if not files:
        raise FileNotFoundError(f"No MaStR CSV files found at {path}")

    frames = [pd.read_csv(file, sep=";", decimal=",", dtype="string", low_memory=False) for file in files]
    raw = pd.concat(frames, ignore_index=True)
    columns = raw.columns.tolist()

    year_col = _find_column(columns, ["Inbetriebnahmedatum der Einheit", "Inbetriebnahmedatum"])
    capacity_col = _find_column(
        columns,
        ["Nettonennleistung der Einheit", "Nettonennleistung", "Installierte Leistung der EEG-Anlage"],
    )
    hub_col = _find_column(
        columns,
        ["Nabenhöhe der Windenergieanlage", "Nabenhoehe der Windenergieanlage", "Nabenhoehe"],
    )
    rotor_col = _find_column(
        columns,
        ["Rotordurchmesser der Windenergieanlage", "Rotordurchmesser"],
    )
    onshore_col = _find_column(columns, ["Wind an Land oder auf See", "Lage"], required=False)
    status_col = _find_column(columns, ["Betriebs-Status", "Betriebsstatus"], required=False)
    energy_col = _find_column(columns, ["Energietr\u00e4ger", "Energietraeger"], required=False)
    id_col = _find_column(columns, ["MaStR-Nr. der Einheit"], required=False)
    name_col = _find_column(columns, ["Anzeige-Name der Einheit"], required=False)
    state_col = _find_column(columns, ["Bundesland"], required=False)

    data = pd.DataFrame(
        {
            "region": "DE",
            "year": _extract_year(raw[year_col]),
            "hub_height_m": _to_numeric(raw[hub_col]),
            "rotor_diameter_m": _to_numeric(raw[rotor_col]),
            "capacity_kw": _to_numeric(raw[capacity_col]),
            "source": "MaStR",
        }
    )
    if id_col:
        data["unit_id"] = raw[id_col].astype("string")
    if name_col:
        data["unit_name"] = raw[name_col].astype("string")
    if state_col:
        data["state"] = raw[state_col].astype("string")
    if status_col:
        data["status"] = raw[status_col].astype("string")
    if onshore_col:
        onshore = raw[onshore_col].astype("string").str.lower()
        before_onshore = len(data)
        data = data.loc[
            onshore.str.contains("land", na=False) & ~onshore.str.contains("see|offshore", na=False)
        ].copy()
        LOGGER.info("DE excluded by onshore filter: %d", before_onshore - len(data))
    if energy_col:
        energy = raw.loc[data.index, energy_col].astype("string").str.lower()
        before_energy = len(data)
        data = data.loc[energy.str.contains("wind", na=False)].copy()
        LOGGER.info("DE excluded by energy-carrier filter: %d", before_energy - len(data))

    return _apply_common_cleaning(data, "DE")


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Specific Power, Rotorflaeche, Kapazitaet berechnen.

    Parameters
    ----------
    df:
        DataFrame containing ``rotor_diameter_m`` and ``capacity_kw``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``rotor_area_m2`` and ``specific_power_wm2``.
    """

    data = df.copy()
    if "region" not in data.columns:
        data["region"] = "UNKNOWN"
    if "region" in df.columns:
        data["region"] = df["region"]
    data["rotor_area_m2"] = np.pi * (data["rotor_diameter_m"] / 2.0) ** 2
    data["specific_power_wm2"] = data["capacity_kw"] * 1000.0 / data["rotor_area_m2"]
    return data


def compute_annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Jaehrliche Mediane, Mittelwerte, Standardabweichungen und n berechnen.

    Parameters
    ----------
    clean:
        Canonical turbine-level observations.

    Returns
    -------
    pandas.DataFrame
        Annual medians, means, standard deviations, and turbine counts.
    """

    annual = df.groupby("year", as_index=False).agg(
        n_turbines=("year", "size"),
        **{
            f"{column}_{stat}": (column, stat)
            for column in METRIC_COLUMNS
            for stat in ["median", "mean", "std"]
        },
    )
    return annual.sort_values("year").reset_index(drop=True)


def aggregate_annual_metrics(clean: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for annual descriptive statistics."""

    return compute_annual_summary(clean)


def plot_region_exploration(clean: pd.DataFrame, annual: pd.DataFrame, region: str, output: Path) -> None:
    """Create a three-panel regional exploration plot."""

    output.parent.mkdir(parents=True, exist_ok=True)
    color = config.REGION_COLORS[region]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, metric in zip(axes, METRIC_COLUMNS):
        ax.scatter(clean["year"], clean[metric], s=8, alpha=0.18, color=color, edgecolors="none")
        ax.plot(
            annual["year"],
            annual[f"{metric}_median"],
            color="black",
            linewidth=1.8,
            label="Annual median",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel(config.METRIC_LABELS[metric])
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(annual_by_region: dict[str, pd.DataFrame], output: Path) -> None:
    """Create a 3x3 comparison plot of annual regional median trends."""

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=False)
    for row, metric in enumerate(METRIC_COLUMNS):
        for col, region in enumerate(config.REGIONS):
            ax = axes[row, col]
            annual = annual_by_region.get(region)
            if annual is not None and not annual.empty:
                ax.plot(
                    annual["year"],
                    annual[f"{metric}_median"],
                    color=config.REGION_COLORS[region],
                    linewidth=2,
                )
            if row == 0:
                ax.set_title(region)
            if col == 0:
                ax.set_ylabel(config.METRIC_LABELS[metric])
            ax.set_xlabel("Year")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_at_data(
    output_dir: Path = config.PROCESSED_DATA_DIR,
    raw_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, PreparedDataPaths]:
    """Prepare Austrian raw and annual turbine metric datasets.

    Parameters
    ----------
    output_dir:
        Directory for processed CSV outputs.
    raw_file:
        Optional explicit path to the Austrian Excel workbook.

    Returns
    -------
    tuple
        Clean raw data, annual aggregates, and output paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    clean = load_and_clean_at(str(raw_file or find_at_raw_file()))
    annual = compute_annual_summary(clean)

    raw_output = output_dir / "at_clean.csv"
    annual_output = output_dir / "at_annual.csv"
    clean.to_csv(raw_output, index=False)
    annual.to_csv(annual_output, index=False)

    LOGGER.info("Wrote AT turbine-level data: %s", raw_output)
    LOGGER.info("Wrote AT annual aggregates: %s", annual_output)
    return clean, annual, PreparedDataPaths(raw_clean=raw_output, annual=annual_output)


def find_us_raw_file(raw_data_dir: Path = config.RAW_DATA_DIR) -> Path:
    """Return the newest-looking USWTDB CSV path."""

    candidates = sorted((raw_data_dir / "uswtdbCSV").glob("uswtdb*.csv"))
    candidates += sorted((raw_data_dir / "US").glob("*.csv")) if (raw_data_dir / "US").exists() else []
    if not candidates:
        raise FileNotFoundError("Could not find USWTDB CSV under data/raw/uswtdbCSV or data/raw/US.")
    return candidates[-1]


def find_de_raw_path(raw_data_dir: Path = config.RAW_DATA_DIR) -> Path:
    """Return the MaStR raw folder or CSV path."""

    for name in ["DE", "MARKTSTAMMDATENREGISTER"]:
        candidate = raw_data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find MaStR data under data/raw/DE or data/raw/MARKTSTAMMDATENREGISTER.")


def prepare_region(region: str, output_dir: Path = config.PROCESSED_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one region and write clean and annual CSV outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    region_upper = region.upper()
    if region_upper == "AT":
        clean = load_and_clean_at(str(find_at_raw_file()))
    elif region_upper == "US":
        clean = load_and_clean_us(str(find_us_raw_file()))
    elif region_upper == "DE":
        clean = load_and_clean_de(str(find_de_raw_path()))
    else:
        raise ValueError(f"Unsupported region: {region}")

    annual = compute_annual_summary(clean)
    clean.to_csv(output_dir / f"{region_upper.lower()}_clean.csv", index=False)
    annual.to_csv(output_dir / f"{region_upper.lower()}_annual.csv", index=False)
    plot_region_exploration(
        clean,
        annual,
        region_upper,
        config.FIGURES_DIR / f"data_exploration_{region_upper.lower()}.png",
    )
    return clean, annual


def prepare_all_regions(output_dir: Path = config.PROCESSED_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Prepare all configured regions and create the comparison plot."""

    annual_by_region = {}
    for region in config.REGIONS:
        _, annual = prepare_region(region, output_dir=output_dir)
        annual_by_region[region] = annual
    plot_comparison(annual_by_region, config.FIGURES_DIR / "data_exploration_comparison.png")
    return annual_by_region
