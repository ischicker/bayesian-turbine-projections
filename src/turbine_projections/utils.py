"""Utility helpers shared across scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: int = logging.INFO) -> None:
    """Configure project logging.

    Parameters
    ----------
    level:
        Root logging level. Defaults to ``logging.INFO``.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories(paths: list[Path]) -> None:
    """Create required output directories if they do not exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
