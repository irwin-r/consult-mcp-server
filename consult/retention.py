"""`consult-gc` — prune old run artifacts to bound disk growth.

Run directories under the runs root (see `consult.artifacts.runs_root`) are
kept forever otherwise; a long-lived desktop install accumulates every
prompt and raw provider response indefinitely. This command removes runs
past an age and/or count bound. Defaults come from the
`CONSULT_RUNS_RETENTION_DAYS` and `CONSULT_RUNS_MAX` environment variables;
CLI flags override them.
"""

from __future__ import annotations

import argparse
import os

from . import __version__, artifacts


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="consult-gc",
        description="Prune old consult run artifacts under the runs directory.",
    )
    parser.add_argument("--version", action="version", version=f"consult-gc {__version__}")
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=None,
        help="Delete runs older than this many days. Default: CONSULT_RUNS_RETENTION_DAYS.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Keep only the newest N runs. Default: CONSULT_RUNS_MAX.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the runs that would be deleted without removing anything.",
    )
    args = parser.parse_args()

    max_age = (
        args.max_age_days if args.max_age_days is not None else _env_float("CONSULT_RUNS_RETENTION_DAYS")
    )
    max_count = args.max_count if args.max_count is not None else _env_int("CONSULT_RUNS_MAX")
    root = artifacts.runs_root()

    if max_age is None and max_count is None:
        print(
            "Nothing to prune. Pass --max-age-days / --max-count, or set "
            "CONSULT_RUNS_RETENTION_DAYS / CONSULT_RUNS_MAX."
        )
        return

    deleted = artifacts.prune_runs(max_age_days=max_age, max_count=max_count, dry_run=args.dry_run)
    verb = "Would prune" if args.dry_run else "Pruned"
    print(f"{verb} {len(deleted)} run(s) from {root}.")
    for rid in deleted:
        print(f"  {rid}")
