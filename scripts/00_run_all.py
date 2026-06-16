"""Run the full reproducibility pipeline in script order.

This entry point assumes raw data have already been placed under ``data/raw``.
It deliberately keeps the heavyweight Bayesian production settings; cached
posteriors are reused by scripts that implement resume logic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


PIPELINE = [
    ["scripts/01_download_data.py"],
    ["scripts/02_prepare_data.py", "--region", "all"],
    ["scripts/03_run_fits.py", "--region", "all"],
    ["scripts/04_run_benchmarks.py"],
    ["scripts/05_run_sensitivity.py"],
    ["scripts/06_run_energy_yield.py"],
    ["scripts/07_generate_figures.py"],
    ["scripts/08_functional_form_comparison.py"],
    ["scripts/08_prior_predictive.py"],
    ["scripts/09_subsample_variability.py"],
    ["scripts/10_prior_posterior_overlap.py"],
    ["scripts/11_subsample_variability_rd_sp.py"],
    ["scripts/12_cross_metric_correlation.py"],
]

LIGHTWEIGHT_ONLY = {
    "scripts/01_download_data.py",
    "scripts/02_prepare_data.py",
    "scripts/04_run_benchmarks.py",
    "scripts/06_run_energy_yield.py",
    "scripts/07_generate_figures.py",
    "scripts/08_prior_predictive.py",
    "scripts/10_prior_posterior_overlap.py",
    "scripts/12_cross_metric_correlation.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Run only deterministic/lightweight steps and skip MCMC-heavy fits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = [
        command
        for command in PIPELINE
        if not args.skip_heavy or command[0] in LIGHTWEIGHT_ONLY
    ]
    for command in commands:
        print(f"\n=== Running {' '.join(command)} ===", flush=True)
        subprocess.run([sys.executable, *command], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
