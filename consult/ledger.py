"""Daily cost ledger.

Scans ~/.consult/runs/ (or `$CONSULT_RUNS_DIR`) for runs whose IDs share a
given YYYYMMDD prefix, reads each `manifest.json`, and aggregates `cost_usd`
+ `cost_known` into a single `DailyLedger`.

Useful for spot-checking spend without spinning up the MCP server. Exposed
as a console entry point: `consult-ledger [YYYY-MM-DD]` (defaults to today,
local clock).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from pydantic import Field

from . import artifacts
from .types import StrictModel

logger = logging.getLogger(__name__)


class LedgerRunEntry(StrictModel):
    """One row in the daily ledger — a single run's summary."""

    run_id: str
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool
    status_counts: dict[str, int] = Field(default_factory=dict)
    panel_size: int = Field(..., ge=0)


class DailyLedger(StrictModel):
    """All runs whose IDs start with the given date, aggregated.

    `total_known=False` when at least one run had unknown pricing for some
    panellist — the displayed total is then a lower bound, not the truth.
    """

    date: date
    total_usd: float = Field(..., ge=0.0)
    total_known: bool
    runs: list[LedgerRunEntry] = Field(default_factory=list)


def _read_manifest(run_dir: Path) -> dict | None:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("ledger: skipping %s — manifest unreadable: %s", run_dir.name, e)
        return None


def daily_ledger(d: date | None = None) -> DailyLedger:
    """Aggregate per-run costs for a given date.

    `d` defaults to today (local clock). Run IDs are `YYYYMMDD-HHMMSS-<rand>`
    so the date filter is a string-prefix match — no parsing of the random
    suffix. Missing/malformed manifests are logged and skipped, never raised
    (so a single corrupt run can't hide the rest of the day's spend).
    """
    if d is None:
        d = datetime.now().date()
    prefix = d.strftime("%Y%m%d")
    root = artifacts.runs_root()

    runs: list[LedgerRunEntry] = []
    total = 0.0
    total_known = True
    for run_dir in sorted(root.glob(f"{prefix}-*")):
        if not run_dir.is_dir():
            continue
        data = _read_manifest(run_dir)
        if data is None:
            continue
        cost = float(data.get("cost_usd") or 0.0)
        known = bool(data.get("cost_known", False))
        manifest = data.get("manifest") or []
        status_counts: dict[str, int] = {}
        for m in manifest:
            s = m.get("status") or "UNKNOWN"
            status_counts[s] = status_counts.get(s, 0) + 1
        runs.append(
            LedgerRunEntry(
                run_id=data.get("run_id") or run_dir.name,
                cost_usd=cost,
                cost_known=known,
                status_counts=status_counts,
                panel_size=len(manifest),
            )
        )
        total += cost
        if not known:
            total_known = False

    return DailyLedger(date=d, total_usd=total, total_known=total_known, runs=runs)


def _parse_arg(arg: str | None) -> date:
    if arg is None or arg == "today":
        return datetime.now().date()
    try:
        return datetime.strptime(arg, "%Y-%m-%d").date()
    except ValueError as e:
        raise SystemExit(
            f"consult-ledger: bad date {arg!r}; expected YYYY-MM-DD or 'today'"
        ) from e


def cli() -> None:
    """Console entry point. Usage: `consult-ledger [YYYY-MM-DD]`."""
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    d = _parse_arg(arg)
    ledger = daily_ledger(d)
    print(ledger.model_dump_json(indent=2))
