"""After-fix probe driver for issue #55.

Runs the two fixed probe prompts through consult.mcp.handlers.consult —
the exact function the MCP server dispatches to — so the working tree's
code (the fix branch) is exercised end to end, server restart not
required. Prints the run economics the PR table needs.

Usage: .venv/bin/python scripts/probe_55.py [--dry-run] p1|p2
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from consult.mcp import handlers  # noqa: E402

P1 = "Polars vs DuckDB for a 10GB Parquet timeseries?"
P2 = (
    "What is the current state of WebGPU adoption across browsers, "
    "and what are the main blockers for production use in 2026?"
)


async def main() -> None:
    dry = "--dry-run" in sys.argv
    which = next((a for a in sys.argv[1:] if not a.startswith("--")), "p1")
    args: dict = {
        "prompt": P1 if which == "p1" else P2,
        "tier": "standard",
        "max_run_usd": 3,
        "dry_run": dry,
    }
    if which == "p2":
        args["capsule_kind"] = "research"
    result = await handlers.consult(args)
    out = {
        "run_id": result.get("run_id"),
        "partial_reason": result.get("partial_reason"),
        "cost_usd": result.get("cost_usd"),
        "cost_known": result.get("cost_known"),
        "wall_ms": result.get("wall_ms"),
        "run_summary": result.get("run_summary"),
        "per_panellist": [
            {
                "slug": m.get("slug"),
                "status": m.get("status"),
                "finish_reason": m.get("finish_reason"),
                "tokens_out": m.get("tokens_out"),
                "cost_usd": m.get("cost_usd"),
            }
            for m in (result.get("manifest") or [])
        ],
    }
    print(json.dumps(out, indent=1))
    synth = result.get("synthesis") or ""
    if synth:
        p = Path(f"/tmp/probe55_{which}_synthesis.md")
        p.write_text(synth)
        print(f"synthesis -> {p}", file=sys.stderr)


asyncio.run(main())
