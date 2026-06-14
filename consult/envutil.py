"""Tolerant numeric environment-variable parsing.

Several runtime knobs (cost cap, heartbeat interval, tail-dropout window,
trim budgets, retry policy) are numbers read from the environment at call
time. Some call sites already handled garbage gracefully —
CONSULT_MAX_PANEL_SIZE warned and fell back — while others called
`float()`/`int()` bare, so a typo like `CONSULT_MAX_RUN_USD=5,00` crashed
mid-run as INTERNAL_ERROR. One helper, one behaviour: log a warning and
fall back to the default. Imports nothing from the package so any module
can use it without cycles.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def env_float(name: str, default: float) -> float:
    """Read a float env var; unset/blank/garbage falls back to `default`."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def env_int(name: str, default: int) -> int:
    """Read an int env var; unset/blank/garbage falls back to `default`."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean env var. True for 1/true/yes/on, False for
    0/false/no/off (case-insensitive); unset/blank/garbage falls back to
    `default`."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    logger.warning("%s=%r is not a boolean; using %s", name, raw, default)
    return default
