"""Research — director-loop deep consultation (issue #92, engine core).

A director model freezes a *brief* in round 0: explicit assumptions plus a
section skeleton with stable ids and per-section acceptance bars. Each round
then runs plan → execute → assemble → judge:

- **plan**: the director emits 1-4 work items, each owning the brief
  sections its output will replace. Fail-closed: a plan that doesn't parse
  or validate is never partially executed (a salvaged plan spends real
  money on corrupted work — panel verdict on issue #92).
- **execute**: each work item runs as an ordinary `orchestrate.consult`
  sub-run (its own run_id, ledger entry, and viewer page). Failures become
  `status="error"` results the judge re-plans around; they never crash the
  round.
- **assemble**: deterministic Python replacement by section id — the work
  item's synthesis *is* the section body. There is no LLM integrator.
- **judge**: the director scores the dossier against the frozen brief
  (per-section missing/draft/accepted plus blocking gaps with stable ids).
  Salvage-tolerant: a failed judge call is a round without signal, not a
  zero score. Two consecutive dead judge calls abort the loop.

Stop conditions: acceptance (every section accepted, zero blocking gaps),
stall (two consecutive rounds with no new accepted section and no drop in
blocking-gap count), budget (accrued spend plus the round's projection
over the cap, checked BEFORE the round executes), or `max_rounds`.
Resolved gap ids cannot reopen: a gap the judge stopped reporting is
recorded as resolved and filtered from later verdicts.

Budget contract: the cap cannot be a pre-flight guarantee — rounds are
planned dynamically — so it is enforced per round. A strict cap combined
with unknown pricing refuses the round rather than spending blind.
`max_run_usd=None` opts into uncapped; stall detection never turns off.

Durable state: `journal.jsonl` in the parent run dir records each phase
boundary (brief, plan, item results, assembly, verdict) so a crashed run
can be replayed to its last committed round (resume lands with issue #92
PR 5). `brief.json`, `dossier.md`, and `verdicts.json` are written as they
change; sub-runs are ordinary sibling runs, and the parent's manifest
carries only director-side spend so daily ledger totals don't double-count.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, cast

import litellm

from . import artifacts, evidence, orchestrate, pricing, provider_caps, registry, runner
from .cost import CostMeter
from .jsonparse import extract_json
from .progress import ProgressCallback, ProgressEvent, ResearchPhase, append_progress_log
from .redact import redact_exc, scrub_exception_attrs
from .types import (
    Brief,
    ModelSpec,
    ResearchGap,
    ResearchResult,
    ResearchRound,
    ResearchVerdict,
    WorkItem,
    WorkItemResult,
)

logger = logging.getLogger(__name__)

# Tool-level default cap. Deliberately higher than the $5 panel default:
# a research run is a multi-round loop, and the per-round gate (not this
# number) is the real enforcement point. Explicit None = uncapped.
DEFAULT_MAX_RUN_USD = 25.0

# Per-section character cap for the dossier view shown to the director.
# Keeps a long dossier from blowing the judge's context; the on-disk
# dossier.md is never clipped.
_SECTION_VIEW_MAX_CHARS = 8000

_BRIEF_PROMPT = """\
You are the director of a multi-round research consultation. Turn the goal \
below into a frozen brief that later rounds will be judged against.

Goal:
{goal}

Rules:
- State every assumption you are forced to make (region, budget, audience, \
scope) as an explicit entry in `assumptions` — do not silently assume.
- Decompose the goal into 3-8 deliverable sections. Each section gets a \
stable kebab-case `id`, a `title`, a one-sentence `goal`, and an \
`acceptance` bar: the concrete test a judge applies to call the section done.
- Acceptance bars must be checkable from the section text alone \
("names at least three candidate niches with evidence", not "is good").

Return EXACTLY this JSON object (no commentary, no markdown fences):

{{
  "assumptions": ["<explicit assumption>", "..."],
  "sections": [
    {{"id": "<kebab-case-id>", "title": "<short title>", "goal": "<one sentence>", "acceptance": "<concrete bar>"}}
  ]
}}
"""

_PLAN_PROMPT = """\
You are the director of a multi-round research consultation, planning round \
{round_num} of at most {max_rounds}.

The frozen brief:
{brief}

Current dossier state (per section):
{dossier_state}

Open blocking gaps from the judge (address these first):
{gaps}

Plan 1-4 work items for this round. Each item is one question sent to a \
fresh model panel; its answer will REPLACE the listed sections wholesale.

Rules:
- Target only sections that are missing, draft, or named in a gap — never \
re-do an accepted section without a gap that demands it.
- `kind` is "panel" for contested questions that benefit from diverse \
opinions, "consult" for questions needing one consolidated answer.
- `kind` "evidence" gathers live web facts with citations. An evidence \
item does NOT write its sections; its findings attach automatically to \
every work item in LATER rounds. Plan evidence early (usually round 1) \
when sections need current facts, prices, or competitor specifics; its \
`section_ids` name the sections the facts will support.
- Every `section_ids` entry must be an id from the brief. Give each item a \
short stable `id` like "r{round_num}-1".
- Write each `question` to be fully self-contained: the panel answering it \
sees NOTHING except that question, so restate the goal context, the \
relevant assumptions, and the acceptance bar it must satisfy.

Return EXACTLY this JSON object (no commentary, no markdown fences):

{{
  "items": [
    {{"id": "<id>", "kind": "panel|consult", "question": "<self-contained question>", "section_ids": ["<brief section id>"], "rationale": "<one sentence>"}}
  ]
}}
"""

_JUDGE_PROMPT = """\
You are the judge of a multi-round research consultation, scoring round \
{round_num}. Score the dossier against the frozen brief — the brief's \
acceptance bars are the ONLY standard; do not invent new requirements.

The frozen brief:
{brief}

The dossier:
{dossier}

Previously resolved gap ids (closed permanently — do NOT re-raise them):
{resolved}

Open gap ids from the prior round (carry each forward VERBATIM if still \
unresolved; omit it if resolved):
{prior_gaps}

For each brief section, status is:
- "missing"  — no usable content yet
- "draft"    — content exists but fails its acceptance bar
- "accepted" — meets its acceptance bar

A blocking gap is a specific, actionable defect that keeps a section from \
acceptance. New gaps get a new short id like "g{round_num}-1". A section \
can only be "accepted" when no blocking gap points at it.

Return EXACTLY this JSON object (no commentary, no markdown fences):

{{
  "section_status": {{"<section id>": "missing|draft|accepted"}},
  "blocking_gaps": [
    {{"id": "<stable gap id>", "text": "<specific defect and what would fix it>", "section_id": "<section id or null>"}}
  ],
  "next_focus": "<one sentence for the next round>",
  "reasoning": "<1-3 sentences>"
}}
"""

_RETRY_NUDGE = (
    "\n\nIMPORTANT: your previous reply could not be parsed as JSON. "
    "Return ONLY the JSON object specified above — no prose, no markdown "
    "fences, no commentary before or after it."
)


async def _director_json(
    prompt: str, alias: str, *, label: str
) -> tuple[dict[str, Any] | None, float | None, bool, str | None]:
    """One director-role JSON call: primary alias, then the configured
    synthesiser fallbacks, with one parse-failure retry per candidate.

    Returns `(data, cost_usd, cost_known, error)`. Cost accrues across
    every billed attempt (mirrors refine's arbiter accounting — the
    provider bills unparseable responses too). `data=None` means every
    candidate failed; the caller decides whether that is fail-closed
    (brief, plan) or salvage-tolerant (judge).
    """
    cost: float | None = 0.0
    cost_known = True

    async def _attempt(
        attempt_prompt: str, litellm_id: str, budget: int, timeout: float
    ) -> tuple[dict[str, Any] | None, str | None]:
        nonlocal cost, cost_known
        call_kwargs: dict[str, Any] = {
            "model": litellm_id,
            "messages": [{"role": "user", "content": attempt_prompt}],
            "max_completion_tokens": budget,
            "response_format": {"type": "json_object"},
        }
        provider_caps.apply_temperature(call_kwargs, litellm_id, 0.0)
        try:
            resp = cast(Any, await asyncio.wait_for(litellm.acompletion(**call_kwargs), timeout=timeout))
        except Exception as e:  # noqa: BLE001 — LiteLLM raises many concrete types
            scrub_exception_attrs(e)
            logger.warning("director %s call failed: %s", label, redact_exc(e))
            cost_known = False
            return None, redact_exc(e, limit=150)
        try:
            c = litellm.completion_cost(completion_response=resp)
        except Exception as e:  # noqa: BLE001
            logger.warning("director %s cost lookup failed: %s", label, redact_exc(e))
            c = None
        if c is None:
            cost_known = False
        else:
            cost = (cost or 0.0) + c
        text = resp.choices[0].message.content or ""
        data = extract_json(text)
        if data is None:
            logger.warning("director %s returned non-JSON; sample=%r", label, text[:120].replace("\n", " "))
            return None, "json_parse_failed"
        return data, None

    candidates = [alias]
    for fb in registry.synthesiser_fallbacks():
        if fb not in candidates:
            candidates.append(fb)

    last_err: str | None = None
    for idx, candidate in enumerate(candidates):
        try:
            entry = registry.resolve_model(candidate)
        except KeyError:
            if idx == 0:
                raise
            logger.warning("director fallback alias %r not in registry; skipping", candidate)
            continue
        litellm_id = entry.get("litellm_id")
        if not litellm_id:
            continue
        budget = max(2000, entry.get("default_budget_tokens", 0))
        timeout = entry.get("default_timeout_s", 180)
        if idx > 0:
            logger.warning("director %s falling back to %s after: %s", label, litellm_id, last_err)
        data, err = await _attempt(prompt, litellm_id, budget, timeout)
        if err is not None and err != "json_parse_failed":
            last_err = err
            continue
        if data is None:
            data, _retry_err = await _attempt(prompt + _RETRY_NUDGE, litellm_id, budget, timeout)
        if data is not None:
            return data, cost, cost_known, None
        last_err = "json_parse_failed"
    return None, cost, cost_known, last_err or "json_parse_failed"


# ---- Parsers -----------------------------------------------------------------


def _norm_id(raw: Any) -> str:
    """Normalise a director-emitted id to the kebab-case grammar the types
    enforce. LLMs freely emit `Market_Research` or `niche selection`; the
    id is a join key across brief, plan, and verdict, so all three parsers
    must normalise identically."""
    return re.sub(r"[^a-z0-9]+", "-", str(raw or "").lower()).strip("-")[:64]


def _parse_brief(data: dict[str, Any]) -> Brief:
    """Build the frozen Brief. Raises ValueError on structural junk — a run
    cannot proceed without a valid contract to judge against."""
    raw_sections = data.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("brief has no sections")
    sections = []
    for s in raw_sections:
        if not isinstance(s, dict):
            raise ValueError(f"brief section is not an object: {s!r}")
        sections.append(
            {
                "id": _norm_id(s.get("id") or s.get("title")),
                "title": str(s.get("title") or "").strip() or str(s.get("id") or ""),
                "goal": str(s.get("goal") or "").strip(),
                "acceptance": str(s.get("acceptance") or "").strip(),
            }
        )
    assumptions = [str(a) for a in data.get("assumptions") or [] if str(a).strip()]
    return Brief(assumptions=assumptions, sections=sections)  # type: ignore[arg-type]


def _parse_plan(data: dict[str, Any], brief: Brief, round_num: int) -> list[WorkItem] | None:
    """Fail-closed plan parser. Any structurally invalid item rejects the
    WHOLE plan (returns None) — executing the valid half of a half-parsed
    plan is how corrupted rounds spend real money."""
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        logger.warning("round %d plan has no items", round_num)
        return None
    if len(raw_items) > 4:
        raw_items = raw_items[:4]
    known_ids = brief.section_ids()
    items: list[WorkItem] = []
    for i, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            logger.warning("round %d plan item %d is not an object; rejecting plan", round_num, i)
            return None
        try:
            item = WorkItem(
                id=str(raw.get("id") or f"r{round_num}-{i}"),
                kind=str(raw.get("kind") or ""),
                question=str(raw.get("question") or ""),
                section_ids=[_norm_id(s) for s in raw.get("section_ids") or []],
                tier=raw.get("tier") if isinstance(raw.get("tier"), str) else None,
                rationale=str(raw.get("rationale") or ""),
            )
        except ValueError as e:
            logger.warning("round %d plan item %d invalid (%s); rejecting plan", round_num, i, e)
            return None
        unknown = set(item.section_ids) - known_ids
        if unknown:
            logger.warning(
                "round %d plan item %s targets unknown sections %s; rejecting plan",
                round_num,
                item.id,
                sorted(unknown),
            )
            return None
        items.append(item)
    return items


def _parse_verdict(
    data: dict[str, Any],
    brief: Brief,
    round_num: int,
    resolved_gap_ids: set[str],
    *,
    cost_usd: float | None,
    cost_known: bool,
) -> ResearchVerdict:
    """Salvage-tolerant verdict parser. Unknown sections are ignored,
    missing sections default to "draft" (never silently "accepted"), and
    the reopen guard drops any gap whose id was previously resolved."""
    known_ids = brief.section_ids()
    raw_status = data.get("section_status")
    section_status: dict[str, str] = {}
    if isinstance(raw_status, dict):
        for sid, status in raw_status.items():
            key = _norm_id(sid)
            if key in known_ids and status in ("missing", "draft", "accepted"):
                section_status[key] = status
    for sid in known_ids:
        section_status.setdefault(sid, "draft")

    gaps: list[ResearchGap] = []
    raw_gaps = data.get("blocking_gaps")
    if isinstance(raw_gaps, list):
        for i, raw in enumerate(raw_gaps, start=1):
            if not isinstance(raw, dict):
                continue
            gap_id = str(raw.get("id") or f"g{round_num}-{i}")
            if gap_id in resolved_gap_ids:
                logger.info("reopen guard: dropping resolved gap %s re-raised in round %d", gap_id, round_num)
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            section_id = _norm_id(raw.get("section_id")) if isinstance(raw.get("section_id"), str) else None
            if section_id not in known_ids:
                section_id = None
            gaps.append(ResearchGap(id=gap_id, text=text, section_id=section_id))

    return ResearchVerdict(
        round=round_num,
        section_status=section_status,
        blocking_gaps=gaps,
        next_focus=str(data.get("next_focus") or ""),
        reasoning=str(data.get("reasoning") or ""),
        cost_usd=cost_usd,
        cost_known=cost_known,
    )


# ---- Dossier -----------------------------------------------------------------


def _clip(text: str, limit: int = _SECTION_VIEW_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... clipped {len(text) - limit:,} chars for the director's view ...]"


def _render_brief(brief: Brief) -> str:
    lines = []
    if brief.assumptions:
        lines.append("Assumptions: " + "; ".join(brief.assumptions))
    for s in brief.sections:
        lines.append(f"- [{s.id}] {s.title} — goal: {s.goal} — acceptance: {s.acceptance}")
    return "\n".join(lines)


def _render_dossier_state(brief: Brief, sections: dict[str, dict[str, Any]], statuses: dict[str, str]) -> str:
    lines = []
    for s in brief.sections:
        status = statuses.get(s.id, "missing" if s.id not in sections else "draft")
        chars = len(sections.get(s.id, {}).get("body", ""))
        lines.append(f"- [{s.id}] {status}, {chars:,} chars")
    return "\n".join(lines)


def _render_dossier(
    brief: Brief, sections: dict[str, dict[str, Any]], *, provenance: bool, clip: bool
) -> str:
    """Deterministic assembly: brief order, one heading per section, the
    owning work item's synthesis as the body. The judge's view sets
    `provenance=False` so no model or run identity can bias the verdict."""
    parts = []
    for s in brief.sections:
        state = sections.get(s.id)
        body = state["body"] if state else "_pending — no content yet._"
        if clip:
            body = _clip(body)
        parts.append(f"## {s.title} <!-- section:{s.id} -->\n\n{body.strip()}")
        if provenance and state:
            parts.append(f"<sub>source: run {state['run_id']} (round {state['round']})</sub>")
    return "\n\n".join(parts) + "\n"


def _render_gaps(gaps: list[ResearchGap]) -> str:
    if not gaps:
        return "(none)"
    return "\n".join(f"- [{g.id}] ({g.section_id or 'general'}) {g.text}" for g in gaps)


# ---- Journal -----------------------------------------------------------------


def _journal_write(paths: artifacts.RunPaths, record: dict[str, Any]) -> None:
    with (paths.root / "journal.jsonl").open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


async def _journal(paths: artifacts.RunPaths, **record: Any) -> None:
    await asyncio.to_thread(_journal_write, paths, record)


# ---- Execution ---------------------------------------------------------------


async def _execute_item(
    item: WorkItem,
    *,
    tier: str,
    max_run_usd: float | None,
    max_output_tokens: int | None,
    sem: asyncio.Semaphore,
    evidence_context: str = "",
) -> tuple[WorkItemResult, str, evidence.EvidencePack | None]:
    """Run one work item. Returns (result record, section body, evidence
    pack). Panel/consult items produce a body and no pack; evidence items
    produce a pack and no body. Never raises — a failed item is journal
    data the judge re-plans around.

    `evidence_context`, when non-empty, is the rendered pack from PRIOR
    rounds' evidence items; it rides along with every panel/consult
    question so grounded facts reach the deliberating panels."""
    if item.kind == "evidence":
        try:
            async with sem:
                pack = await evidence.gather_evidence(
                    item.question,
                    max_run_usd=max_run_usd,
                    max_output_tokens=max_output_tokens,
                )
        except Exception as e:  # noqa: BLE001 — sub-run failure must not crash the round
            scrub_exception_attrs(e)
            logger.warning("evidence item %s failed: %s", item.id, redact_exc(e))
            return (
                WorkItemResult(
                    item_id=item.id,
                    status="error",
                    cost_usd=None,
                    cost_known=False,
                    error=redact_exc(e, limit=200),
                ),
                "",
                None,
            )
        if pack.partial or not pack.records:
            return (
                WorkItemResult(
                    item_id=item.id,
                    run_id=pack.run_id,
                    status="error",
                    cost_usd=pack.cost_usd,
                    cost_known=pack.cost_known,
                    error=pack.partial_reason or "evidence pass harvested zero sources",
                ),
                "",
                None,
            )
        return (
            WorkItemResult(
                item_id=item.id,
                run_id=pack.run_id,
                status="ok",
                cost_usd=pack.cost_usd,
                cost_known=pack.cost_known,
            ),
            "",
            pack,
        )

    question = item.question if not evidence_context else f"{item.question}\n\n{evidence_context}"
    try:
        async with sem:
            res = await orchestrate.consult(
                question,
                tier=item.tier or tier,
                capsule_kind="decision",
                max_run_usd=max_run_usd,
                max_output_tokens=max_output_tokens,
            )
    except Exception as e:  # noqa: BLE001 — sub-run failure must not crash the round
        scrub_exception_attrs(e)
        logger.warning("work item %s failed: %s", item.id, redact_exc(e))
        return (
            WorkItemResult(
                item_id=item.id,
                status="error",
                cost_usd=None,
                cost_known=False,
                error=redact_exc(e, limit=200),
            ),
            "",
            None,
        )
    if res.partial or not (res.synthesis or "").strip():
        return (
            WorkItemResult(
                item_id=item.id,
                run_id=res.run_id,
                status="error",
                cost_usd=res.cost_usd,
                cost_known=res.cost_known,
                error=res.partial_reason or "sub-run returned no synthesis",
            ),
            "",
            None,
        )
    return (
        WorkItemResult(
            item_id=item.id,
            run_id=res.run_id,
            status="ok",
            cost_usd=res.cost_usd,
            cost_known=res.cost_known,
        ),
        res.synthesis,
        None,
    )


async def _project_round_cost(items: list[WorkItem], tier: str) -> tuple[float, bool]:
    """Floor estimate for a planned round: per-item panel cost at the item's
    tier (evidence items price their web workers instead). Excludes per-item
    synthesis, director calls, and provider search fees, so it is a floor —
    the same contract as every other estimate in the engine."""
    total = 0.0
    all_known = True
    for item in items:
        if item.kind == "evidence":
            aliases = list(evidence.DEFAULT_MODELS)
        else:
            aliases = registry.resolve_tier(item.tier or tier)
        specs = [ModelSpec(model=a) for a in aliases]
        est, known = await runner.aestimate_cost(specs, item.question)
        total += est
        all_known = all_known and known
    return total, all_known


# ---- The loop ----------------------------------------------------------------


async def research(
    prompt: str,
    *,
    tier: str = "standard",
    director: str | None = None,
    max_rounds: int = 6,
    max_run_usd: float | None = DEFAULT_MAX_RUN_USD,
    max_parallel_items: int = 3,
    max_output_tokens: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> ResearchResult:
    """Run the director loop until the dossier passes the frozen brief.

    `max_run_usd=None` is the explicit uncapped opt-in; the default cap is
    `DEFAULT_MAX_RUN_USD`. Stall detection is always on. `tier` is the
    default worker tier; the director may override per work item.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    pricing.ensure_registered()
    director_alias = director or registry.default_synthesiser()
    registry.resolve_model(director_alias)
    registry.resolve_tier(tier)

    paths = artifacts.create_run()
    await asyncio.to_thread(paths.prompt_txt.write_text, prompt)
    await asyncio.to_thread(
        paths.registry_snapshot.write_text, json.dumps(registry.models_config(), indent=2)
    )

    start = time.time()
    meter = CostMeter()

    async def _emit(event: ProgressEvent) -> None:
        append_progress_log(paths.root, event)
        if on_progress is None:
            return
        try:
            await on_progress(event)
        except Exception as e:  # noqa: BLE001 — progress is best-effort
            logger.debug("research on_progress failed: %s", e)

    def _result(**kwargs: Any) -> ResearchResult:
        return ResearchResult(
            run_id=paths.run_id,
            cost_usd=meter.total,
            cost_known=meter.known,
            wall_ms=int((time.time() - start) * 1000),
            **kwargs,
        )

    # ---- Round 0: the frozen brief ------------------------------------------
    await _emit(ResearchPhase(done=0, total=max_rounds, phase="brief", round=0))
    data, cost, cost_known, err = await _director_json(
        _BRIEF_PROMPT.format(goal=prompt), director_alias, label="brief"
    )
    meter.add(cost or 0.0, cost_known)
    brief: Brief | None = None
    if data is not None:
        try:
            brief = _parse_brief(data)
        except ValueError as e:
            err = f"brief invalid: {e}"
    if brief is None:
        reason = f"director failed to produce a brief: {err}"
        await _journal(paths, phase="brief_failed", error=reason)
        return _result(
            brief=None,
            rounds_completed=0,
            partial=True,
            partial_reason=reason,
            stop_reason="director_error",
        )
    await asyncio.to_thread((paths.root / "brief.json").write_text, json.dumps(brief.model_dump(), indent=2))
    await _journal(paths, phase="brief", brief=brief.model_dump())

    # ---- Rounds --------------------------------------------------------------
    sections: dict[str, dict[str, Any]] = {}
    statuses: dict[str, str] = {}
    resolved_gap_ids: set[str] = set()
    open_gaps: list[ResearchGap] = []
    verdicts: list[ResearchVerdict] = []
    rounds: list[ResearchRound] = []
    packs: list[evidence.EvidencePack] = []
    stall = 0
    judge_failures = 0
    progress_prev: tuple[int, int] | None = None
    stop_reason = "max_rounds"
    partial_reason: str | None = None
    item_sem = asyncio.Semaphore(max(1, max_parallel_items))

    for round_num in range(1, max_rounds + 1):
        # Plan — fail-closed. The dossier state carries an evidence tally so
        # the director knows whether grounding already happened.
        await _emit(ResearchPhase(done=round_num - 1, total=max_rounds, phase="plan", round=round_num))
        dossier_state = _render_dossier_state(brief, sections, statuses)
        gathered = evidence.merged_records(packs)
        if gathered:
            dossier_state += f"\nEvidence gathered: {len(gathered)} web source(s) from {len(packs)} pass(es)"
        plan_prompt = _PLAN_PROMPT.format(
            round_num=round_num,
            max_rounds=max_rounds,
            brief=_render_brief(brief),
            dossier_state=dossier_state,
            gaps=_render_gaps(open_gaps),
        )
        data, cost, cost_known, err = await _director_json(plan_prompt, director_alias, label="plan")
        meter.add(cost or 0.0, cost_known)
        items = _parse_plan(data, brief, round_num) if data is not None else None
        if items is None:
            stop_reason = "director_error"
            partial_reason = f"round {round_num} plan unusable: {err or 'failed validation'}"
            await _journal(paths, phase="plan_failed", round=round_num, error=partial_reason)
            break
        await _journal(paths, phase="plan", round=round_num, items=[i.model_dump() for i in items])

        # Per-round budget gate: accrued + projection, BEFORE execution.
        if max_run_usd is not None:
            projection, proj_known = await _project_round_cost(items, tier)
            if not proj_known:
                stop_reason = "budget"
                partial_reason = (
                    f"round {round_num} projection includes unknown pricing; refusing to spend "
                    f"blind under a strict cap (spent ${meter.total:.2f}, cap ${max_run_usd:.2f}). "
                    "Pass max_run_usd=None to run uncapped."
                )
                await _journal(paths, phase="budget_refused", round=round_num, reason=partial_reason)
                break
            if meter.total + projection > max_run_usd:
                stop_reason = "budget"
                partial_reason = (
                    f"round {round_num} would exceed cap: spent ${meter.total:.2f}, "
                    f"projection ${projection:.2f}, cap ${max_run_usd:.2f}"
                )
                await _journal(paths, phase="budget_refused", round=round_num, reason=partial_reason)
                break

        # Execute — child cap slices; failures trapped per item. Prior
        # rounds' evidence rides along with every panel/consult question.
        await _emit(ResearchPhase(done=round_num - 1, total=max_rounds, phase="execute", round=round_num))
        per_item_cap = None if max_run_usd is None else max(0.0, max_run_usd - meter.total) / len(items)
        evidence_context = evidence.render_evidence_pack(gathered) if gathered else ""
        outcomes = await asyncio.gather(
            *(
                _execute_item(
                    item,
                    tier=tier,
                    max_run_usd=per_item_cap,
                    max_output_tokens=max_output_tokens,
                    sem=item_sem,
                    evidence_context=evidence_context,
                )
                for item in items
            )
        )
        results: list[WorkItemResult] = []
        for item, (res, body, pack) in zip(items, outcomes, strict=True):
            results.append(res)
            meter.add(res.cost_usd or 0.0, res.cost_known)
            await _journal(paths, phase="item_result", round=round_num, result=res.model_dump())
            if res.status != "ok":
                continue
            if pack is not None:
                # Evidence items feed later rounds, never the dossier —
                # writing their empty body would clobber a real section.
                packs.append(pack)
                await _journal(
                    paths, phase="evidence", round=round_num, item_id=item.id, sources=len(pack.records)
                )
                continue
            for sid in item.section_ids:
                sections[sid] = {"body": body, "run_id": res.run_id, "round": round_num}
        await asyncio.to_thread(
            (paths.root / "dossier.md").write_text,
            _render_dossier(brief, sections, provenance=True, clip=False),
        )
        await _journal(paths, phase="assembled", round=round_num, sections=sorted(sections))

        # Judge — salvage-tolerant.
        await _emit(ResearchPhase(done=round_num - 1, total=max_rounds, phase="judge", round=round_num))
        judge_prompt = _JUDGE_PROMPT.format(
            round_num=round_num,
            brief=_render_brief(brief),
            dossier=_render_dossier(brief, sections, provenance=False, clip=True),
            resolved=", ".join(sorted(resolved_gap_ids)) or "(none)",
            prior_gaps=_render_gaps(open_gaps),
        )
        data, cost, cost_known, err = await _director_json(judge_prompt, director_alias, label="judge")
        meter.add(cost or 0.0, cost_known)
        if data is None:
            verdict = ResearchVerdict(
                round=round_num,
                cost_usd=cost,
                cost_known=cost_known,
                parsed_ok=False,
                error=err or "judge_failed",
            )
        else:
            verdict = _parse_verdict(
                data, brief, round_num, resolved_gap_ids, cost_usd=cost, cost_known=cost_known
            )
        verdicts.append(verdict)
        rounds.append(ResearchRound(round=round_num, work_items=items, results=results, verdict=verdict))
        await _journal(paths, phase="verdict", round=round_num, verdict=verdict.model_dump())

        if not verdict.parsed_ok:
            judge_failures += 1
            if judge_failures >= 2:
                stop_reason = "director_error"
                partial_reason = f"judge failed twice in a row (last: {verdict.error})"
                break
            continue
        judge_failures = 0
        statuses = dict(verdict.section_status)

        # A gap the judge stopped reporting is resolved — permanently.
        reported = {g.id for g in verdict.blocking_gaps}
        resolved_gap_ids |= {g.id for g in open_gaps if g.id not in reported}
        open_gaps = list(verdict.blocking_gaps)

        if verdict.accepted(brief):
            stop_reason = "accepted"
            break

        # Stall: no newly accepted section AND no drop in blocking gaps,
        # two rounds running.
        accepted_n = sum(1 for s in statuses.values() if s == "accepted")
        progress = (accepted_n, -len(open_gaps))
        improved = progress_prev is None or progress[0] > progress_prev[0] or progress[1] > progress_prev[1]
        stall = 0 if improved else stall + 1
        progress_prev = progress
        if stall >= 2:
            stop_reason = "stalled"
            break

    # ---- Finalise ------------------------------------------------------------
    dossier = _render_dossier(brief, sections, provenance=True, clip=False)
    await asyncio.to_thread((paths.root / "dossier.md").write_text, dossier)
    await asyncio.to_thread(
        (paths.root / "verdicts.json").write_text,
        json.dumps([v.model_dump() for v in verdicts], indent=2),
    )

    result = _result(
        brief=brief,
        rounds_completed=len(rounds),
        rounds=rounds,
        verdicts=verdicts,
        dossier=dossier,
        converged=stop_reason == "accepted",
        stop_reason=stop_reason,
        open_gaps=open_gaps,
        partial=partial_reason is not None,
        partial_reason=partial_reason,
    )
    # The parent manifest carries only director-side spend context; sub-runs
    # wrote their own manifests, so the daily ledger never double-counts.
    await asyncio.to_thread(
        artifacts.write_manifest,
        paths,
        {"kind": "research", **result.model_dump(exclude={"dossier"})},
    )
    return result
