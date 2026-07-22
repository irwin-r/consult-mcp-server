"""Per-tool MCP adapter handlers.

Translate the MCP `arguments` dict into typed engine calls, then return
the engine's typed result `model_dump()`'d for the wire. No `mcp.*`
imports here — the wire-shape concerns that *are* MCP-specific
(`TextContent` wrapping, `progressToken` lookup) live in `server.py` and
are passed in as a callback.

The handlers are intentionally thin: anything non-trivial lives in the
engine modules (`runner`, `refine`, `sequence`, `synth`, `orchestrate`)
so non-MCP consumers (CLI, HTTP, library use) can drive the same flows
without going through this adapter.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .. import (
    artifacts,
    attachments,
    capsule,
    orchestrate,
    registry,
    runner,
    synth,
)
from .. import (
    refine as refine_mod,
)
from .. import (
    research as research_mod,
)
from .. import (
    sequence as sequence_mod,
)
from ..exceptions import UnknownModelError
from ..progress import ProgressEvent
from ..types import ModelSpec

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


def _specs_from_args(models_arg: list[dict[str, Any]]) -> list[ModelSpec]:
    return [ModelSpec(**m) for m in models_arg]


def _validate_specs(specs: list[ModelSpec]) -> None:
    """Fail the request BEFORE any provider call when a panellist alias
    resolves nowhere.

    Without this, an unknown alias only surfaces when its own panellist
    task runs — the rest of the panel still fans out, so a typo'd model
    burns a full run's latency and spend to report one bad string
    (FRICTION 2026-07-13: `opus-4.7` alongside two raw-ID near-misses
    produced a three-minute zero-usable-panellist round). Raw LiteLLM IDs
    still pass through: `resolve_model` accepts them by design.
    """
    unknown: list[str] = []
    for spec in specs:
        try:
            registry.resolve_model(spec.model)
        except KeyError:
            unknown.append(spec.model)
    if not unknown:
        return
    cfg = registry.models_config()
    aliases = sorted(cfg.get("models", {}))
    tiers = sorted(cfg.get("tiers", {}))
    described: list[str] = []
    for name in unknown:
        hints = difflib.get_close_matches(name, aliases, n=3, cutoff=0.5)
        suffix = f" (closest: {', '.join(hints)})" if hints else ""
        described.append(f"{name!r}{suffix}")
    raise UnknownModelError(
        f"unknown model alias(es): {'; '.join(described)}. "
        f"Registry aliases: {', '.join(aliases)}. "
        f"Tiers (pass as `tier` instead of `models`): {', '.join(tiers)}. "
        "Raw LiteLLM IDs with a provider prefix (e.g. 'openrouter/x-ai/grok-4.3') "
        "are also accepted."
    )


def _panel_specs(args: dict[str, Any]) -> list[ModelSpec]:
    """Resolve the panel from `models` (explicit list) or `tier` (registry
    tier name), validating every alias up front.

    `models` wins when both are present — an explicit list is more specific
    than a tier name, and silently unioning them would make panel size
    surprising. Neither present is a caller error.
    """
    models_arg = args.get("models")
    if models_arg:
        specs = _specs_from_args(models_arg)
    elif args.get("tier"):
        specs = [ModelSpec(model=alias) for alias in registry.resolve_tier(args["tier"])]
    else:
        raise ValueError("provide `models` (explicit panellist list) or `tier` (registry tier name)")
    _validate_specs(specs)
    return specs


# Statuses that mean the panellist produced a usable body even if the capsule
# was cut short. ERROR/TIMEOUT (and anything else) are hard failures.
_OK_STATUSES = ("OK", "TRUNCATED")
_TRUNCATED_FINISHES = ("length", "max_tokens", "MAX_TOKENS")


def _capsule_is_empty(cap: dict[str, Any] | None) -> bool:
    """True when a capsule carries no content a synthesiser could use.

    Keyed on `kind` because each shape stores its substance in different
    fields. Supplementary or relational fields (caveats, unique_claims,
    sources_cited, confidence, overall_verdict) don't count as value on their
    own: a capsule with only those is a partial extraction, itself worth
    flagging. Unknown or legacy kinds fall back to the decision check, which is
    the conservative direction — a new shape reads as empty until the helper is
    taught it, surfacing as a visible `no_value` entry rather than a silent pass.
    """
    if not isinstance(cap, dict):
        return True
    kind = cap.get("kind", "decision")
    if kind == "review":
        return not (cap.get("findings") or [])
    if kind == "research":
        return not (cap.get("claims") or []) and not (cap.get("evidence") or [])
    return (
        not (cap.get("position") or "").strip()
        and not (cap.get("recommendation") or "").strip()
        and not (cap.get("key_points") or [])
    )


def _summarise_manifest(manifest: list[dict[str, Any]], *, cost_usd: float | None = None) -> dict[str, Any]:
    """Roll up per-panellist health so the invoking agent can see, from the
    result alone, which models contributed and which returned nothing usable —
    the "which models didn't return value?" question that otherwise needs a
    manual manifest dig. `no_value` lists each dud with a concrete reason
    (errored, timed out, truncated before a usable capsule, or empty extraction).

    When `cost_usd` (the run's total, synth and extractor included) is
    given, the summary also carries `cost_per_usable_capsule` — the run's
    truncation-economics headline (issue #55): what one unit of usable
    panel signal cost, with dud panellists priced in rather than hidden.
    """
    from collections import Counter

    status_counts: Counter[str] = Counter()
    findings_total = 0
    no_value: list[dict[str, Any]] = []
    for e in manifest:
        if not isinstance(e, dict):
            continue
        status = e.get("status") or "?"
        status_counts[status] += 1
        cap = e.get("capsule") if isinstance(e.get("capsule"), dict) else None
        # findings_total stays a review-kind metric: count line-anchored
        # findings across the panel.
        if cap is not None and "findings" in cap:
            findings_total += len(cap.get("findings") or [])

        finish = e.get("finish_reason")
        hard_fail = status not in _OK_STATUSES
        # A panellist that returned OK/TRUNCATED but whose capsule has no usable
        # content bought nothing. This was previously caught only for review
        # capsules (via the findings count), so a truncated decision or research
        # panellist looked healthy in the rollup. `cap is not None` gates the
        # check: when extract_capsules=false every capsule is null, and those
        # runs aren't dud panellists.
        empty = cap is not None and _capsule_is_empty(cap)
        if not (hard_fail or empty):
            continue
        if e.get("error"):
            reason = str(e["error"])
        elif status == "TIMEOUT":
            reason = "timed out"
        elif empty and finish in _TRUNCATED_FINISHES:
            reason = "truncated at token cap before emitting a usable capsule"
        elif empty:
            reason = "extractor returned an empty capsule"
        else:
            reason = status
        no_value.append(
            {
                "slug": e.get("slug"),
                "status": status,
                "finish_reason": finish,
                "reason": reason[:200],
            }
        )
    # A panellist is "usable" when it isn't a dud: it returned OK/TRUNCATED
    # and (when capsules were extracted) its capsule carries substance.
    usable = len(manifest) - len(no_value)
    summary: dict[str, Any] = {
        "panellists": len(manifest),
        "status_counts": dict(status_counts),
        "findings_total": findings_total,
        "usable_capsules": usable,
        "no_value": no_value,
    }
    if cost_usd is not None:
        summary["cost_per_usable_capsule"] = round(cost_usd / usable, 4) if usable > 0 else None
    # Truncation is actionable, not just reportable: when any panellist hit
    # the output cap, tell the agent reading this result which knob fixes it
    # (2026-07-13: a long-form prompt truncated 4 of 9 panellists and the
    # only clue was per-entry finish_reasons).
    truncated_n = status_counts.get("TRUNCATED", 0)
    if truncated_n:
        summary["truncation_advice"] = (
            f"{truncated_n} panellist(s) hit their output-token cap; responses "
            "were auto-continued where possible. For long-form deliverables, "
            "pass max_output_tokens to grant more room up front."
        )
    return summary


async def _augment_result(result: dict[str, Any]) -> dict[str, Any]:
    """Surface, in the result the agent actually reads, two things it otherwise
    has to dig for: a `report_url` (file:// link to the rendered HTML feed) and
    a `run_summary` of panellist health, plus `progress_log` (the live JSONL
    progress tail). Best-effort — never let a render/summary hiccup discard the
    engine's real result.

    The render is threaded: it reads every body and builds the whole HTML
    document, which on a wide refine run blocks the event loop for long
    enough to stall heartbeats on any concurrent fanout.
    """
    run_id = result.get("run_id")
    if not run_id:
        return result
    try:
        from .. import viewer

        report_path = await asyncio.to_thread(viewer.render_run, run_id)
        result.setdefault("report_url", report_path.as_uri())
        progress_log = report_path.parent / "_progress.log"
        if progress_log.exists():
            result.setdefault("progress_log", str(progress_log))
    except Exception as e:  # noqa: BLE001 — surfacing is best-effort
        logger.warning("report_url render failed for run %s: %s", run_id, e)

    # Refine results carry `final_manifest` (the last round's panel), not
    # `manifest` — without the fallback they never got a run_summary and a
    # dud panellist was invisible from the result (FRICTION 2026-06-11).
    # On refine, cost_per_usable_capsule prices ALL rounds against the
    # final round's usable panellists, which is the honest read: earlier
    # rounds are what the final capsules cost to produce.
    manifest = result.get("manifest") or result.get("final_manifest")
    if isinstance(manifest, list) and manifest:
        try:
            cost = result.get("cost_usd")
            result.setdefault(
                "run_summary",
                _summarise_manifest(manifest, cost_usd=cost if isinstance(cost, (int, float)) else None),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("run_summary build failed for run %s: %s", run_id, e)
    return result


async def panel(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    specs = _panel_specs(args)
    kind = args.get("capsule_kind", "decision")
    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        dry_run=args.get("dry_run", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=on_progress,
        capsule_kind=kind,
        max_output_tokens=args.get("max_output_tokens"),
    )
    if args.get("extract_capsules", True) and not handle.partial and handle.manifest:
        handle = await capsule.annotate(handle, on_progress=on_progress, kind=kind)
    result = handle.model_dump()
    if args.get("peer_rank", False) and not handle.partial and handle.manifest:
        await _attach_peer_ranking(result, handle, question=args["prompt"])
    # Dry runs have no artifacts to render or summarise.
    return result if args.get("dry_run", False) else await _augment_result(result)


async def _attach_peer_ranking(result: dict[str, Any], handle: Any, *, question: str) -> None:
    """Run the opt-in llm-council peer-rank pass and attach it to the result.

    Mutates `result` in place: adds `peer_ranking` (Borda ranks plus the
    per-ranker forensics, each entry carrying its pairs, drop reason,
    and spend) and rolls the ranking calls' spend into
    `cost_usd`/`cost_known`, on disk too so the ledger sees the true run
    total. The block itemises its own `cost_usd`/`cost_known` so the
    rank pass's spend stays attributable after the roll-up.
    Best-effort: a ranking failure logs and leaves the panel
    result untouched — the user paid for the panel, not the side-car.
    """
    import asyncio as _asyncio

    from .. import artifacts, peer_rank
    from ..types import Status

    try:
        paths = artifacts.load_run(handle.run_id)
        usable = [m for m in handle.manifest if m.status in (Status.OK, Status.TRUNCATED)]
        body_texts = await _asyncio.gather(
            *(_asyncio.to_thread(paths.response_text(m.slug).read_text) for m in usable)
        )
        bodies = {m.slug: text for m, text in zip(usable, body_texts, strict=True)}
        ranking = await peer_rank.peer_rank_run(handle.manifest, bodies, question=question)
        result["peer_ranking"] = {
            "ranks": [[slug, points] for slug, points in ranking.ranks],
            "per_ranker": [
                {
                    "ranker": outcome.slug,
                    "pairs": [[pos, slug] for pos, slug in outcome.pairs],
                    "reason": outcome.reason,
                    "cost_usd": outcome.cost_usd,
                    "cost_known": outcome.cost_known,
                }
                for outcome in ranking.per_ranker
            ],
            "cost_usd": ranking.cost_usd,
            "cost_known": ranking.cost_known,
        }
        result["cost_usd"] = float(result.get("cost_usd") or 0.0) + ranking.cost_usd
        if not ranking.cost_known:
            result["cost_known"] = False
        await artifacts.aaugment_manifest(
            paths, cost_usd=result["cost_usd"], cost_known=result.get("cost_known", True)
        )
    except Exception as e:  # noqa: BLE001 — side-car must not sink the panel
        logger.warning("peer_rank pass failed for run %s: %s", result.get("run_id"), e)


async def synthesise(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> str:
    # Synth's output is a markdown blob. The MCP server wraps the returned
    # string in `TextContent` so clients render it directly; the handler
    # itself stays MCP-free. `on_progress` is accepted for signature
    # uniformity but synth doesn't emit progress events.
    result = await synth.synthesise(
        args["run_id"],
        by_model=args.get("by_model"),
        rubric=args.get("rubric"),
        anonymised=args.get("anonymised", False),
    )
    text = result.text
    # Re-render the feed so the HTML reflects this fresh synthesis, and surface
    # its link inline (synthesise returns a bare string, so there's no envelope
    # to carry report_url).
    try:
        from .. import viewer

        report_path = await asyncio.to_thread(viewer.render_run, args["run_id"])
        text += f"\n\n---\n[View HTML report]({report_path.as_uri()})"
    except Exception as e:  # noqa: BLE001 — surfacing is best-effort
        logger.warning("report_url render failed for run %s: %s", args.get("run_id"), e)
    return text


async def consult(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    # Attachments are forwarded raw to `orchestrate.consult`, which inlines
    # them itself. Two reasons not to pre-inline here: (1) library consumers
    # get the same convenience without re-importing `attachments`, and (2)
    # having one inlining site keeps the rendered shape consistent if it
    # ever changes.
    result = await orchestrate.consult(
        args["prompt"],
        tier=args.get("tier", "standard"),
        roles=args.get("roles"),
        synthesiser=args.get("synthesiser"),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        extract_capsules=args.get("extract_capsules", True),
        capsule_kind=args.get("capsule_kind", "decision"),
        rubric=args.get("rubric"),
        attachments=args.get("attachments"),
        dry_run=args.get("dry_run", False),
        max_output_tokens=args.get("max_output_tokens"),
        gate_synth_at_agreement=args.get("gate_synth_at_agreement"),
        diverse_stances=args.get("diverse_stances", True),
        on_progress=on_progress,
    )
    payload = result.model_dump()
    # Dry runs have no artifacts to render or summarise (parity with panel).
    return payload if args.get("dry_run", False) else await _augment_result(payload)


async def sequence(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    # Each step gets its own inlined-attachments prompt. The top-level
    # `attachments` is the default for every step; a step that's an object
    # can supply its own `attachments` to override (per-step source material
    # — closes the "naïve sequence drops step N's code" trap from the v2 audit).
    raw_prompts = args["prompts"]
    default_attachments = args.get("attachments")
    prompts: list[str] = []
    for item in raw_prompts:
        if isinstance(item, str):
            prompts.append(attachments.inline_attachments(item, default_attachments))
        else:
            step_atts = item.get("attachments")
            effective_atts = step_atts if step_atts is not None else default_attachments
            prompts.append(attachments.inline_attachments(item["prompt"], effective_atts))
    specs = _specs_from_args(args["models"])
    _validate_specs(specs)
    result = await sequence_mod.sequence(
        prompts,
        specs,
        synthesiser=args.get("synthesiser"),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        dry_run=args.get("dry_run", False),
        capsule_kind=args.get("capsule_kind", "decision"),
        rubric=args.get("rubric"),
        max_output_tokens=args.get("max_output_tokens"),
        on_progress=on_progress,
    )
    payload = result.model_dump()
    # Dry runs have no artifacts to render or summarise (parity with panel/consult).
    return payload if args.get("dry_run", False) else await _augment_result(payload)


def _research_summary(result: Any) -> str:
    """Deterministic executive summary for the inline payload — no extra
    model call. The full dossier stays behind `dossier_uri` so a long run
    doesn't bloat the calling agent's context (the product's whole point).
    """
    lines = [
        f"Research stopped: {result.stop_reason or 'unknown'} after "
        f"{result.rounds_completed} round(s); converged={result.converged}."
    ]
    if result.brief is not None:
        if result.brief.assumptions:
            lines.append("Assumptions: " + "; ".join(result.brief.assumptions))
        last = result.verdicts[-1] if result.verdicts else None
        for section in result.brief.sections:
            status = (last.section_status.get(section.id) if last else None) or "missing"
            lines.append(f"- {section.title} [{section.id}]: {status}")
    if result.open_gaps:
        lines.append("Open gaps: " + "; ".join(f"[{g.id}] {g.text}" for g in result.open_gaps))
    prefix = "" if result.cost_known else "≥"
    lines.append(f"Cost: {prefix}${result.cost_usd:.2f}. Full dossier: read the dossier_uri resource.")
    return "\n".join(lines)


async def research(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    # Presence-sensitive cap: an ABSENT max_run_usd takes the engine default;
    # an explicit JSON null arrives as None and means uncapped. `args.get`
    # with a default would collapse the two.
    cap_kwargs: dict[str, Any] = {}
    if "max_run_usd" in args:
        cap_kwargs["max_run_usd"] = args["max_run_usd"]
    result = await research_mod.research(
        prompt,
        tier=args.get("tier", "standard"),
        director=args.get("director"),
        max_rounds=args.get("max_rounds", 6),
        max_output_tokens=args.get("max_output_tokens"),
        on_progress=on_progress,
        **cap_kwargs,
    )
    # The dossier is the one deliberately-large field; it ships as a
    # resource, not inline. Everything else in the result is capsule-sized.
    payload = result.model_dump(exclude={"dossier"})
    payload["dossier_uri"] = f"consult://runs/{result.run_id}/dossier/dossier.md"
    payload["dossier_chars"] = len(result.dossier)
    payload["summary"] = _research_summary(result)
    # Best-effort observability pointers (parity with _augment_result; the
    # panel-manifest summary and viewer render don't apply to a research
    # run until the viewer learns the shape in issue #92 PR 6).
    try:
        root = artifacts.load_run(result.run_id).root
        for filename, key in (("_progress.log", "progress_log"), ("journal.jsonl", "journal_path")):
            path = root / filename
            if path.exists():
                payload[key] = str(path)
    except Exception as e:  # noqa: BLE001 — pointers must never sink the result
        logger.warning("research observability pointers failed for %s: %s", result.run_id, e)
    return payload


async def refine(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    specs = _panel_specs(args)
    result = await refine_mod.refine(
        prompt,
        specs,
        arbiter=args.get("arbiter"),
        threshold=args.get("threshold", 0.85),
        max_rounds=args.get("max_rounds", 3),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        dry_run=args.get("dry_run", False),
        synthesiser=args.get("synthesiser"),
        continuation_id=args.get("continuation_id"),
        rubric=args.get("rubric"),
        # Pass None when the MCP caller didn't specify, so refine inherits
        # from the prior run's bundle (continuation case) instead of
        # silently defaulting back to "decision".
        capsule_kind=args.get("capsule_kind"),
        strategy=args.get("strategy", "default"),
        max_output_tokens=args.get("max_output_tokens"),
        on_progress=on_progress,
    )
    payload = result.model_dump()
    # Dry runs have no artifacts to render or summarise (parity with panel/consult).
    return payload if args.get("dry_run", False) else await _augment_result(payload)
