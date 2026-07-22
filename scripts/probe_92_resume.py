#!/usr/bin/env python
"""Live crash-resume validation for the issue-92 research loop.

Two modes, driven by the harness that runs them:

    .venv/bin/python scripts/probe_92_resume.py start
    .venv/bin/python scripts/probe_92_resume.py resume <run_id>

`start` launches a bounded research run whose goal demands current, cited
facts (so the director should plan an evidence item — the live validation
of the web-search path). The harness SIGKILLs the process mid-run, then
`resume` recovers it from the journal and runs it to completion. Spends
real money (a few dollars across both halves); run manually.
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
    "Recommend a defensible niche and a launch plan for a new specialty "
    "coffee subscription in Australia. The competitor section must cite "
    "current facts about at least two existing Australian coffee "
    "subscription services, including their present pricing, gathered "
    "from the live web. Keep every deliverable concrete enough to act on."
)


def _summary(result) -> dict:
    last = next((v for v in reversed(result.verdicts) if v.parsed_ok), None)
    return {
        "run_id": result.run_id,
        "stop_reason": result.stop_reason,
        "converged": result.converged,
        "rounds_completed": result.rounds_completed,
        "section_status": last.section_status if last else None,
        "evidence_items": sum(1 for r in result.rounds for i in r.work_items if i.kind == "evidence"),
        "cost_usd": round(result.cost_usd, 4),
        "cost_known": result.cost_known,
        "partial_reason": result.partial_reason,
        "dossier_chars": len(result.dossier),
        "wall_s": result.wall_ms // 1000,
    }


async def main() -> int:
    async def show(event) -> None:
        print(f":: {event_message(event)}", flush=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    kwargs = dict(tier="quick", max_rounds=3, max_run_usd=8.0, on_progress=show)
    if mode == "resume":
        result = await research.research("", continuation_id=sys.argv[2], **kwargs)
        print(json.dumps({"mode": "resume", **_summary(result)}, indent=2))
        ok = result.rounds_completed >= 1 and result.stop_reason not in ("director_error", "")
        print(f"RESUME {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    result = await research.research(GOAL, **kwargs)
    print(json.dumps({"mode": "start", **_summary(result)}, indent=2))
    print("START COMPLETED WITHOUT BEING KILLED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
