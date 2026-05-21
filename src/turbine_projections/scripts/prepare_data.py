"""Prepare raw regional wind turbine datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from turbine_projections import config
from turbine_projections.data_preparation import prepare_all_regions, prepare_region
from turbine_projections.utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        choices=config.REGIONS + ["all"],
        default="all",
        help="Region to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROCESSED_DATA_DIR,
        help="Directory for processed CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run data preparation."""

    configure_logging()
    args = parse_args()

    if args.region == "all":
        prepare_all_regions(output_dir=args.output_dir)
    else:
        prepare_region(args.region, output_dir=args.output_dir)

    LOGGER.info("Data preparation complete.")


if __name__ == "__main__":
    main()
