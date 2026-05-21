"""Wrapper for preparing raw regional datasets."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from turbine_projections.scripts.prepare_data import main


if __name__ == "__main__":
    main()
