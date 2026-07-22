#!/usr/bin/env python
"""Live smoke for the issue-92 research engine: director contracts + loop.

Runs a real, bounded research loop (quick tier, 2 rounds, $3 cap) against
live providers. Validates that the brief/plan/judge JSON contracts survive
contact with the default director and that the loop's artifacts land.
Spends real money (typically under a dollar); run manually:

    .venv/bin/python scripts/probe_92_research.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv(Path(__file__).parents[1] / ".env")

from consult import research  # noqa: E402
from consult.progress import event_message  # noqa: E402

GOAL = (
    "Recommend a brand name, a one-line positioning statement, and a launch "
    "checklist for a hypothetical espresso-bean subscription aimed at remote "
    "workers in Australia. Keep every deliverable concrete enough to act on."
)


async def main() -> int:
    async def show(event) -> None:
        print(f":: {event_message(event)}", flush=True)

    result = await research.research(
        GOAL,
        tier="quick",
        max_rounds=2,
        max_run_usd=3.0,
        on_progress=show,
    )
    last_verdict = result.verdicts[-1] if result.verdicts else None
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "stop_reason": result.stop_reason,
                "converged": result.converged,
                "rounds_completed": result.rounds_completed,
                "brief_sections": [s.id for s in result.brief.sections] if result.brief else None,
                "assumptions": result.brief.assumptions if result.brief else None,
                "section_status": last_verdict.section_status if last_verdict else None,
                "open_gaps": [f"{g.id}: {g.text[:60]}" for g in result.open_gaps],
                "cost_usd": round(result.cost_usd, 4),
                "cost_known": result.cost_known,
                "partial_reason": result.partial_reason,
                "dossier_chars": len(result.dossier),
                "wall_s": result.wall_ms // 1000,
            },
            indent=2,
        )
    )
    # Smoke passes when the contracts held: a brief exists, at least one
    # round planned and executed, and the loop stopped for a legitimate
    # reason rather than a director failure.
    ok = result.brief is not None and result.rounds_completed >= 1 and result.stop_reason != "director_error"
    print(f"SMOKE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
