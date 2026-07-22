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
boundary (brief, plan, item results, assembly, verdict — director costs
included) and is the substrate for `continuation_id` resume: committed
rounds replay through the same `_apply_verdict` reducer the live loop
uses, and the in-flight round restarts fresh, losing at most one round.
`brief.json`, `dossier.md`, and `verdicts.json` are written as they
change; sub-runs are ordinary sibling runs, and the parent's manifest
carries only director-side spend so daily ledger totals don't double-count.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
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

# Patience floor for every model call in a research run (sub-run panellists
# and director calls alike). Registry timeouts are sized for interactive
# panels; a research run would rather wait hours than lose a deep model's
# answer. None = keep registry defaults.
DEFAULT_MODEL_TIMEOUT_FLOOR_S = 7200.0

# Headroom reserved out of the cap before slicing per-item budgets, so the
# round's judge call and the next plan can always run (review finding, #92:
# director spend was unreserved and capped runs could overshoot).
_DIRECTOR_RESERVE_USD = 0.50

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
    {{"id": "<id>", "kind": "panel|consult|evidence", "question": "<self-contained question>", "section_ids": ["<brief section id>"], "rationale": "<one sentence>"}}
  ]
}}

One more rule: give each panel/consult item exactly ONE section id — its \
whole answer replaces each listed section verbatim, so multi-section items \
duplicate one blob everywhere. Evidence items may list every section their \
facts support. No two items may share a section id in the same round.
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
    prompt: str, alias: str, *, label: str, timeout_floor_s: float | None = None
) -> tuple[dict[str, Any] | None, float | None, bool, str | None]:
    """One director-role JSON call: primary alias, then the configured
    synthesiser fallbacks, with one parse-failure retry per candidate.

    Returns `(data, cost_usd, cost_known, error)`. Cost accrues across
    every billed attempt (mirrors refine's arbiter accounting — the
    provider bills unparseable responses too). `data=None` means every
    candidate failed; the caller decides whether that is fail-closed
    (brief, plan) or salvage-tolerant (judge).

    Calls go through the transport retry (transient errors recover on the
    same candidate instead of burning a fallback) and the shared provider
    semaphores, so director spend obeys the same rate-limit discipline as
    panellist calls. `timeout_floor_s` raises the per-call ceiling for
    patient runs.
    """
    cost: float | None = 0.0
    cost_known = True

    async def _attempt(
        attempt_prompt: str, litellm_id: str, budget: int, timeout: float, provider: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        nonlocal cost, cost_known
        call_kwargs: dict[str, Any] = {
            "model": litellm_id,
            "messages": [{"role": "user", "content": attempt_prompt}],
            "max_completion_tokens": budget,
            "response_format": {"type": "json_object"},
        }
        provider_caps.apply_temperature(call_kwargs, litellm_id, 0.0)
        sems = runner._get_provider_sems()
        sem = sems.get(provider) or sems.get("default")
        try:
            async with sem if sem is not None else contextlib.nullcontext():
                resp = cast(Any, await runner._acompletion_with_retry(timeout=timeout, **call_kwargs))
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
        timeout = float(entry.get("default_timeout_s", 180))
        if timeout_floor_s is not None:
            timeout = max(timeout, timeout_floor_s)
        provider = entry.get("provider", "")
        if idx > 0:
            logger.warning("director %s falling back to %s after: %s", label, litellm_id, last_err)
        data, err = await _attempt(prompt, litellm_id, budget, timeout, provider)
        if err is not None and err != "json_parse_failed":
            last_err = err
            continue
        if data is None:
            data, _retry_err = await _attempt(prompt + _RETRY_NUDGE, litellm_id, budget, timeout, provider)
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
        logger.warning("round %d plan has %d items; truncating to 4", round_num, len(raw_items))
        raw_items = raw_items[:4]
    known_ids = brief.section_ids()
    items: list[WorkItem] = []
    claimed_sections: set[str] = set()
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
        # A hallucinated tier crashed the budget gate with an UnknownModelError
        # before validation existed (review finding, #92) — same fail-closed
        # treatment as an unknown section id.
        if item.tier is not None:
            try:
                registry.resolve_tier(item.tier)
            except KeyError:
                logger.warning(
                    "round %d plan item %s names unknown tier %r; rejecting plan",
                    round_num,
                    item.id,
                    item.tier,
                )
                return None
        # Section ownership must be exclusive per round for panel/consult
        # items — concurrent items sharing a section would race to
        # last-writer-wins. Evidence items only *support* sections, so they
        # don't claim ownership.
        if item.kind != "evidence":
            overlap = claimed_sections & set(item.section_ids)
            if overlap:
                logger.warning(
                    "round %d plan items overlap on sections %s; rejecting plan",
                    round_num,
                    sorted(overlap),
                )
                return None
            claimed_sections |= set(item.section_ids)
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
    sections_with_content: set[str] | None = None,
) -> ResearchVerdict:
    """Salvage-tolerant verdict parser. Unknown sections are ignored, an
    unscored section defaults to "missing" when it has no body yet and
    "draft" when it does (never silently "accepted"), and the reopen
    guard drops any gap whose normalised id was previously resolved."""
    known_ids = brief.section_ids()
    has_content = sections_with_content or set()
    raw_status = data.get("section_status")
    section_status: dict[str, str] = {}
    if isinstance(raw_status, dict):
        for sid, status in raw_status.items():
            key = _norm_id(sid)
            if key in known_ids and status in ("missing", "draft", "accepted"):
                section_status[key] = status
    for sid in known_ids:
        section_status.setdefault(sid, "draft" if sid in has_content else "missing")

    gaps: list[ResearchGap] = []
    raw_gaps = data.get("blocking_gaps")
    if isinstance(raw_gaps, list):
        for i, raw in enumerate(raw_gaps, start=1):
            if not isinstance(raw, dict):
                continue
            # Normalised so a judge emitting "G1-1" or " g1-1 " can't slip a
            # resolved gap past the reopen guard (review finding, #92).
            gap_id = _norm_id(raw.get("id")) or f"g{round_num}-{i}"
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


# ---- Verdict reducer ---------------------------------------------------------


@dataclass
class _LoopState:
    """The verdict-driven loop state, mutated only by `_apply_verdict`.

    One reducer serves both the live loop and journal replay (resume), so
    a resumed run's gap bookkeeping, stall counters, and judge-failure
    tracking are byte-identical to having lived through the rounds.
    """

    statuses: dict[str, str] = field(default_factory=dict)
    resolved_gap_ids: set[str] = field(default_factory=set)
    open_gaps: list[ResearchGap] = field(default_factory=list)
    verdicts: list[ResearchVerdict] = field(default_factory=list)
    stall: int = 0
    judge_failures: int = 0
    progress_prev: tuple[int, int] | None = None


def _apply_verdict(state: _LoopState, verdict: ResearchVerdict, brief: Brief) -> str | None:
    """Fold one verdict into the loop state; return the stop signal it
    implies: "accepted", "stalled", "judge_dead", or None to continue."""
    state.verdicts.append(verdict)
    if not verdict.parsed_ok:
        state.judge_failures += 1
        return "judge_dead" if state.judge_failures >= 2 else None
    state.judge_failures = 0
    state.statuses = dict(verdict.section_status)
    # A gap the judge stopped reporting is resolved — permanently.
    reported = {g.id for g in verdict.blocking_gaps}
    state.resolved_gap_ids |= {g.id for g in state.open_gaps if g.id not in reported}
    state.open_gaps = list(verdict.blocking_gaps)
    if verdict.accepted(brief):
        return "accepted"
    # Stall: no newly accepted section AND no drop in blocking gaps, two
    # rounds running.
    accepted_n = sum(1 for s in state.statuses.values() if s == "accepted")
    progress = (accepted_n, -len(state.open_gaps))
    improved = (
        state.progress_prev is None
        or progress[0] > state.progress_prev[0]
        or progress[1] > state.progress_prev[1]
    )
    state.stall = 0 if improved else state.stall + 1
    state.progress_prev = progress
    return "stalled" if state.stall >= 2 else None


# ---- Journal -----------------------------------------------------------------


def _journal_write(paths: artifacts.RunPaths, record: dict[str, Any]) -> None:
    # Mirrors append_progress_log: a disk hiccup must not destroy a run
    # that has already spent real money. A missing journal line degrades
    # resume fidelity for this run; that is strictly better than losing
    # the in-memory result AND the ledger record (review finding, #92).
    try:
        with (paths.root / "journal.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        logger.warning("research journal write failed (%s); continuing without it", e)


async def _journal(paths: artifacts.RunPaths, **record: Any) -> None:
    await asyncio.to_thread(_journal_write, paths, record)


# ---- Resume (journal replay) -------------------------------------------------


def _load_journal(paths: artifacts.RunPaths) -> list[dict[str, Any]]:
    """Parse journal.jsonl, skipping malformed lines with a warning — one
    corrupt record must not make an expensive run unresumable."""
    journal_path = paths.root / "journal.jsonl"
    if not journal_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for i, line in enumerate(journal_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("journal line %d unparseable (%s); skipping", i, e)
    return records


def _read_subrun_synthesis(run_id: str | None) -> str | None:
    """Best-effort read of a sub-run's synthesis body for section replay.
    A pruned or missing sub-run degrades that section to missing (the
    director re-plans it) rather than failing the resume."""
    if not run_id:
        return None
    try:
        text = (artifacts.load_run(run_id).root / "synthesis.md").read_text()
        return text if text.strip() else None
    except (OSError, ValueError) as e:
        logger.warning("resume: could not read sub-run %s synthesis (%s)", run_id, e)
        return None


def _read_evidence_pack(run_id: str | None, cost_usd: float | None, cost_known: bool):
    """Rebuild an EvidencePack from a pass run dir's evidence JSONL."""
    if not run_id:
        return None
    try:
        lines = (artifacts.load_run(run_id).root / "evidence" / "evidence.jsonl").read_text().splitlines()
        records = []
        for line in lines:
            row = json.loads(line)
            row.pop("gathered_at", None)
            records.append(evidence.EvidenceRecord.model_validate(row))
        if not records:
            return None
        return evidence.EvidencePack(
            run_id=run_id, records=records, cost_usd=cost_usd or 0.0, cost_known=cost_known
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning("resume: could not rebuild evidence pack from %s (%s)", run_id, e)
        return None


@dataclass
class _ResumeState:
    vs: _LoopState
    sections: dict[str, dict[str, Any]]
    packs: list[Any]
    rounds: list[ResearchRound]
    next_round: int


def _replay_journal(
    records: list[dict[str, Any]],
    brief: Brief,
    meter: CostMeter,
    director_meter: CostMeter,
) -> _ResumeState:
    """Reconstruct loop state from the journal, replaying committed rounds
    through the same `_apply_verdict` reducer the live loop uses.

    A round is committed iff its verdict record exists; an in-flight round
    contributes its journaled spend to the meters (the money is gone) but
    none of its state — resume re-plans that round from scratch, losing at
    most one round of work by design. Section bodies re-read from sub-run
    dirs; evidence packs re-read from their pass run dirs; both degrade
    gracefully when artifacts were pruned.
    """
    vs = _LoopState()
    sections: dict[str, dict[str, Any]] = {}
    packs: list[Any] = []
    rounds: list[ResearchRound] = []
    last_round_seen = 0

    def _accrue(rec: dict[str, Any], *, director: bool) -> None:
        cost = rec.get("cost_usd")
        known = bool(rec.get("cost_known", cost is not None))
        meter.add(cost or 0.0, known)
        if director:
            director_meter.add(cost or 0.0, known)

    by_round: dict[int, dict[str, Any]] = {}
    for rec in records:
        phase = rec.get("phase")
        if phase in ("brief", "brief_failed"):
            _accrue(rec, director=True)
            continue
        if phase == "resumed":
            continue
        round_num = int(rec.get("round") or 0)
        last_round_seen = max(last_round_seen, round_num)
        bucket = by_round.setdefault(round_num, {"items": None, "results": [], "verdict": None})
        if phase == "plan":
            _accrue(rec, director=True)
            bucket["items"] = rec.get("items") or []
        elif phase == "item_result":
            result = rec.get("result") or {}
            meter.add(result.get("cost_usd") or 0.0, bool(result.get("cost_known", True)))
            bucket["results"].append(result)
        elif phase == "verdict":
            # Judge cost rides inside the verdict dump.
            v = rec.get("verdict") or {}
            director_meter.add(v.get("cost_usd") or 0.0, bool(v.get("cost_known", True)))
            meter.add(v.get("cost_usd") or 0.0, bool(v.get("cost_known", True)))
            bucket["verdict"] = v

    for round_num in sorted(by_round):
        bucket = by_round[round_num]
        if bucket["verdict"] is None or bucket["items"] is None:
            continue  # in-flight round: spend accrued above, state discarded
        items = [WorkItem.model_validate(raw) for raw in bucket["items"]]
        items_by_id = {i.id: i for i in items}
        results = [WorkItemResult.model_validate(raw) for raw in bucket["results"]]
        for res in results:
            if res.status != "ok":
                continue
            item = items_by_id.get(res.item_id)
            if item is None:
                continue
            if item.kind == "evidence":
                pack = _read_evidence_pack(res.run_id, res.cost_usd, res.cost_known)
                if pack is not None:
                    packs.append(pack)
                continue
            body = _read_subrun_synthesis(res.run_id)
            if body is None:
                continue
            for sid in item.section_ids:
                sections[sid] = {"body": body, "run_id": res.run_id, "round": round_num}
        verdict = ResearchVerdict.model_validate(bucket["verdict"])
        rounds.append(ResearchRound(round=round_num, work_items=items, results=results, verdict=verdict))
        signal = _apply_verdict(vs, verdict, brief)
        if signal in ("accepted", "stalled", "judge_dead"):
            raise ValueError(
                f"nothing to resume: the journal already reaches {signal!r} at round {round_num}"
            )

    committed = rounds[-1].round if rounds else 0
    # An in-flight round (journaled plan, no verdict) restarts AT its own
    # number; otherwise continue after the last committed round.
    next_round = last_round_seen if last_round_seen > committed else committed + 1
    return _ResumeState(vs=vs, sections=sections, packs=packs, rounds=rounds, next_round=next_round)


# ---- Execution ---------------------------------------------------------------


async def _execute_item(
    item: WorkItem,
    *,
    tier: str,
    max_run_usd: float | None,
    max_output_tokens: int | None,
    sem: asyncio.Semaphore,
    evidence_context: str = "",
    timeout_floor_s: float | None = None,
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
                    timeout_floor_s=timeout_floor_s,
                    tail_dropout_s=0.0,
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
                # Patient sub-runs: no slow-tail dropout, floored timeouts —
                # losing a deep model costs more than the wall-clock saved.
                timeout_floor_s=timeout_floor_s,
                tail_dropout_s=0.0,
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
    model_timeout_floor_s: float | None = DEFAULT_MODEL_TIMEOUT_FLOOR_S,
    continuation_id: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> ResearchResult:
    """Run the director loop until the dossier passes the frozen brief.

    `max_run_usd=None` is the explicit uncapped opt-in; the default cap is
    `DEFAULT_MAX_RUN_USD`. Stall detection is always on. `tier` is the
    default worker tier; the director may override per work item.

    Research runs are PATIENT by default: slow-tail dropout is disabled
    for every sub-run and each model's per-call timeout is raised to at
    least `model_timeout_floor_s` (default two hours), so a deep model is
    never cancelled for being slow — losing a costly panellist mid-run
    wastes more than the wall-clock it saves. Pass None to keep the
    registry's interactive-scale timeouts. Heartbeats and the progress
    log keep long waits observable.

    `continuation_id` resumes a CRASHED run from its journal: committed
    rounds replay through the same reducer the live loop uses (section
    bodies re-read from sub-run dirs, evidence packs from their pass
    dirs, both meters restored), and the in-flight round restarts from a
    fresh plan — at most one round of work is lost. Completed runs refuse
    to resume (feedback-driven extension is a later feature), and the
    prompt must match the original goal.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    pricing.ensure_registered()
    director_alias = director or registry.default_synthesiser()
    registry.resolve_model(director_alias)
    registry.resolve_tier(tier)

    start = time.time()
    meter = CostMeter()  # everything: director + sub-runs (the API total)
    director_meter = CostMeter()  # director calls only (the parent manifest's spend)

    resume_state: _ResumeState | None = None
    brief: Brief | None = None
    if continuation_id is not None:
        paths = artifacts.load_run(continuation_id)
        stored_prompt = ""
        if paths.prompt_txt.exists():
            stored_prompt = paths.prompt_txt.read_text()
        if prompt.strip() and stored_prompt.strip() and prompt.strip() != stored_prompt.strip():
            raise ValueError(
                f"continuation_id {continuation_id} was started with a different goal; "
                "resume must re-send the original prompt (or an empty one)"
            )
        brief_path = paths.root / "brief.json"
        if not brief_path.exists():
            raise ValueError(f"run {continuation_id} never produced a brief; start a fresh run instead")
        if paths.manifest_json.exists():
            prior = json.loads(paths.manifest_json.read_text())
            aborted = prior.get("partial_reason") == "run aborted before finalise"
            if prior.get("stop_reason") and not aborted:
                raise ValueError(
                    f"run {continuation_id} completed ({prior['stop_reason']}); "
                    "resume recovers crashed runs only"
                )
        brief = Brief.model_validate(json.loads(brief_path.read_text()))
        resume_state = _replay_journal(_load_journal(paths), brief, meter, director_meter)
        await _journal(paths, phase="resumed", next_round=resume_state.next_round)
        logger.info(
            "resuming run %s at round %d (%d committed round(s), $%.2f replayed spend)",
            continuation_id,
            resume_state.next_round,
            len(resume_state.rounds),
            meter.total,
        )
    else:
        paths = artifacts.create_run()
        await asyncio.to_thread(paths.prompt_txt.write_text, prompt)
        await asyncio.to_thread(
            paths.registry_snapshot.write_text, json.dumps(registry.models_config(), indent=2)
        )

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

    def _write_parent_manifest(payload: dict[str, Any]) -> None:
        # The parent manifest's cost fields carry DIRECTOR-ONLY spend: every
        # sub-run wrote its own manifest, and consult-ledger sums cost_usd
        # across all run dirs, so writing the full total here double-counted
        # (review finding, #92). The full total ships on the API result and
        # rides along as total_cost_usd for humans reading the file.
        artifacts.write_manifest(
            paths,
            {
                "kind": "research",
                **payload,
                "cost_usd": director_meter.total,
                "cost_known": director_meter.known,
                "total_cost_usd": meter.total,
                "total_cost_known": meter.known,
            },
        )

    # ---- Round 0: the frozen brief (skipped on resume) -----------------------
    if resume_state is None:
        # ---- Round 0: the frozen brief ------------------------------------------
        await _emit(ResearchPhase(done=0, total=max_rounds, phase="brief", round=0))
        data, cost, cost_known, err = await _director_json(
            _BRIEF_PROMPT.format(goal=prompt),
            director_alias,
            label="brief",
            timeout_floor_s=model_timeout_floor_s,
        )
        meter.add(cost or 0.0, cost_known)
        director_meter.add(cost or 0.0, cost_known)
        if data is not None:
            try:
                brief = _parse_brief(data)
            except ValueError as e:
                err = f"brief invalid: {e}"
        if brief is None:
            reason = f"director failed to produce a brief: {err}"
            await _journal(paths, phase="brief_failed", error=reason, cost_usd=cost, cost_known=cost_known)
            result = _result(
                brief=None,
                rounds_completed=0,
                partial=True,
                partial_reason=reason,
                stop_reason="director_error",
            )
            # Even a failed brief billed a director call; the manifest must exist
            # so the ledger sees the spend.
            await asyncio.to_thread(_write_parent_manifest, result.model_dump(exclude={"dossier"}))
            return result
        await asyncio.to_thread(
            (paths.root / "brief.json").write_text, json.dumps(brief.model_dump(), indent=2)
        )
        await _journal(paths, phase="brief", brief=brief.model_dump(), cost_usd=cost, cost_known=cost_known)
    assert brief is not None  # both branches above guarantee it

    # ---- Rounds --------------------------------------------------------------
    vs = resume_state.vs if resume_state else _LoopState()
    sections: dict[str, dict[str, Any]] = resume_state.sections if resume_state else {}
    rounds: list[ResearchRound] = list(resume_state.rounds) if resume_state else []
    packs: list[evidence.EvidencePack] = list(resume_state.packs) if resume_state else []
    first_round = resume_state.next_round if resume_state else 1
    stop_reason = "max_rounds"
    partial_reason: str | None = None
    item_sem = asyncio.Semaphore(max(1, max_parallel_items))
    loop_completed = False

    try:
        for round_num in range(first_round, max_rounds + 1):
            # Plan — fail-closed. The dossier state carries an evidence tally so
            # the director knows whether grounding already happened.
            await _emit(ResearchPhase(done=round_num - 1, total=max_rounds, phase="plan", round=round_num))
            dossier_state = _render_dossier_state(brief, sections, vs.statuses)
            gathered = evidence.merged_records(packs)
            if gathered:
                dossier_state += (
                    f"\nEvidence gathered: {len(gathered)} web source(s) from {len(packs)} pass(es)"
                )
            plan_prompt = _PLAN_PROMPT.format(
                round_num=round_num,
                max_rounds=max_rounds,
                brief=_render_brief(brief),
                dossier_state=dossier_state,
                gaps=_render_gaps(vs.open_gaps),
            )
            data, cost, cost_known, err = await _director_json(
                plan_prompt, director_alias, label="plan", timeout_floor_s=model_timeout_floor_s
            )
            meter.add(cost or 0.0, cost_known)
            director_meter.add(cost or 0.0, cost_known)
            items = _parse_plan(data, brief, round_num) if data is not None else None
            if items is None:
                stop_reason = "director_error"
                partial_reason = f"round {round_num} plan unusable: {err or 'failed validation'}"
                await _journal(paths, phase="plan_failed", round=round_num, error=partial_reason)
                break
            await _journal(
                paths,
                phase="plan",
                round=round_num,
                items=[i.model_dump() for i in items],
                cost_usd=cost,
                cost_known=cost_known,
            )

            # Per-round budget gate: accrued + projection + director reserve,
            # BEFORE execution. Also refuses when the ACCRUED total itself has
            # gone unknown — gating a known cap against a known-understated
            # meter is spending blind (review finding, #92).
            if max_run_usd is not None:
                if not meter.known:
                    stop_reason = "budget"
                    partial_reason = (
                        f"accrued spend includes unknown pricing after round {round_num - 1}; "
                        f"refusing further rounds under a strict cap (known spend ${meter.total:.2f}, "
                        f"cap ${max_run_usd:.2f}). Pass max_run_usd=None to run uncapped."
                    )
                    await _journal(paths, phase="budget_refused", round=round_num, reason=partial_reason)
                    break
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
                if meter.total + projection + _DIRECTOR_RESERVE_USD > max_run_usd:
                    stop_reason = "budget"
                    partial_reason = (
                        f"round {round_num} would exceed cap: spent ${meter.total:.2f}, "
                        f"projection ${projection:.2f} plus ${_DIRECTOR_RESERVE_USD:.2f} director "
                        f"reserve, cap ${max_run_usd:.2f}"
                    )
                    await _journal(paths, phase="budget_refused", round=round_num, reason=partial_reason)
                    break

            # Execute — child cap slices; failures trapped per item. Prior
            # rounds' evidence rides along with every panel/consult question.
            await _emit(ResearchPhase(done=round_num - 1, total=max_rounds, phase="execute", round=round_num))
            if max_run_usd is None:
                per_item_cap = None
            else:
                remaining = max(0.0, max_run_usd - meter.total - _DIRECTOR_RESERVE_USD)
                per_item_cap = remaining / len(items)
                if per_item_cap < 0.01:
                    # A near-zero slice guarantees every sub-run refuses at its
                    # own gate — burning latency to produce error entries. Stop
                    # the round cleanly instead (review finding, #92).
                    stop_reason = "budget"
                    partial_reason = (
                        f"round {round_num} per-item budget slice is ${per_item_cap:.4f}; "
                        f"cap ${max_run_usd:.2f} is effectively exhausted at ${meter.total:.2f}"
                    )
                    await _journal(paths, phase="budget_refused", round=round_num, reason=partial_reason)
                    break
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
                        timeout_floor_s=model_timeout_floor_s,
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
                resolved=", ".join(sorted(vs.resolved_gap_ids)) or "(none)",
                prior_gaps=_render_gaps(vs.open_gaps),
            )
            data, cost, cost_known, err = await _director_json(
                judge_prompt, director_alias, label="judge", timeout_floor_s=model_timeout_floor_s
            )
            meter.add(cost or 0.0, cost_known)
            director_meter.add(cost or 0.0, cost_known)
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
                    data,
                    brief,
                    round_num,
                    vs.resolved_gap_ids,
                    cost_usd=cost,
                    cost_known=cost_known,
                    sections_with_content=set(sections),
                )
            rounds.append(ResearchRound(round=round_num, work_items=items, results=results, verdict=verdict))
            await _journal(paths, phase="verdict", round=round_num, verdict=verdict.model_dump())

            signal = _apply_verdict(vs, verdict, brief)
            if signal == "judge_dead":
                stop_reason = "director_error"
                partial_reason = f"judge failed twice in a row (last: {verdict.error})"
                break
            if signal == "accepted":
                stop_reason = "accepted"
                break
            if signal == "stalled":
                stop_reason = "stalled"
                break

        loop_completed = True
    finally:
        if not loop_completed:
            # Crash/cancel path: best-effort persistence so the spend still
            # reaches the ledger and the dossier-so-far survives a mid-run
            # exception or a task-mode cancel (review finding, #92).
            with contextlib.suppress(Exception):
                (paths.root / "dossier.md").write_text(
                    _render_dossier(brief, sections, provenance=True, clip=False)
                )
            with contextlib.suppress(Exception):
                (paths.root / "verdicts.json").write_text(
                    json.dumps([v.model_dump() for v in vs.verdicts], indent=2)
                )
            with contextlib.suppress(Exception):
                _write_parent_manifest(
                    {
                        "run_id": paths.run_id,
                        "rounds_completed": len(rounds),
                        "partial": True,
                        "partial_reason": "run aborted before finalise",
                    }
                )

    # ---- Finalise ------------------------------------------------------------
    dossier = _render_dossier(brief, sections, provenance=True, clip=False)
    await asyncio.to_thread((paths.root / "dossier.md").write_text, dossier)
    await asyncio.to_thread(
        (paths.root / "verdicts.json").write_text,
        json.dumps([v.model_dump() for v in vs.verdicts], indent=2),
    )

    result = _result(
        brief=brief,
        rounds_completed=len(rounds),
        rounds=rounds,
        verdicts=vs.verdicts,
        dossier=dossier,
        converged=stop_reason == "accepted",
        stop_reason=stop_reason,
        open_gaps=vs.open_gaps,
        partial=partial_reason is not None,
        partial_reason=partial_reason,
    )
    await asyncio.to_thread(_write_parent_manifest, result.model_dump(exclude={"dossier"}))
    return result
