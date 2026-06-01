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

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .. import (
    attachments,
    capsule,
    orchestrate,
    runner,
    synth,
)
from .. import (
    refine as refine_mod,
)
from .. import (
    sequence as sequence_mod,
)
from ..progress import ProgressEvent
from ..types import ModelSpec

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


def _specs_from_args(models_arg: list[dict[str, Any]]) -> list[ModelSpec]:
    return [ModelSpec(**m) for m in models_arg]


# Statuses that mean the panellist produced a usable body even if the capsule
# was cut short. ERROR/TIMEOUT (and anything else) are hard failures.
_OK_STATUSES = ("OK", "TRUNCATED")
_TRUNCATED_FINISHES = ("length", "max_tokens", "MAX_TOKENS")


def _summarise_manifest(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-panellist health so the invoking agent can see, from the
    result alone, which models contributed and which returned nothing usable —
    the "which models didn't return value?" question that otherwise needs a
    manual manifest dig. `no_value` lists each dud with a concrete reason
    (errored, timed out, truncated before findings, or empty extraction).
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
        has_findings_field = cap is not None and "findings" in cap
        n_findings = len(cap.get("findings") or []) if has_findings_field else None
        if n_findings is not None:
            findings_total += n_findings

        finish = e.get("finish_reason")
        hard_fail = status not in _OK_STATUSES
        empty = has_findings_field and n_findings == 0
        if not (hard_fail or empty):
            continue
        if e.get("error"):
            reason = str(e["error"])
        elif status == "TIMEOUT":
            reason = "timed out"
        elif empty and finish in _TRUNCATED_FINISHES:
            reason = "truncated at token cap before emitting findings"
        elif empty:
            reason = "extractor returned no findings"
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
    return {
        "panellists": len(manifest),
        "status_counts": dict(status_counts),
        "findings_total": findings_total,
        "no_value": no_value,
    }


def _augment_result(result: dict[str, Any]) -> dict[str, Any]:
    """Surface, in the result the agent actually reads, two things it otherwise
    has to dig for: a `report_url` (file:// link to the rendered HTML feed) and
    a `run_summary` of panellist health, plus `progress_log` (the live JSONL
    progress tail). Best-effort — never let a render/summary hiccup discard the
    engine's real result.
    """
    run_id = result.get("run_id")
    if not run_id:
        return result
    try:
        from .. import viewer

        report_path = viewer.render_run(run_id)
        result.setdefault("report_url", report_path.as_uri())
        progress_log = report_path.parent / "_progress.log"
        if progress_log.exists():
            result.setdefault("progress_log", str(progress_log))
    except Exception as e:  # noqa: BLE001 — surfacing is best-effort
        logger.warning("report_url render failed for run %s: %s", run_id, e)

    manifest = result.get("manifest")
    if isinstance(manifest, list) and manifest:
        try:
            result.setdefault("run_summary", _summarise_manifest(manifest))
        except Exception as e:  # noqa: BLE001
            logger.warning("run_summary build failed for run %s: %s", run_id, e)
    return result


async def panel(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    specs = _specs_from_args(args["models"])
    kind = args.get("capsule_kind", "decision")
    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        dry_run=args.get("dry_run", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=on_progress,
        capsule_kind=kind,
    )
    if args.get("extract_capsules", True) and not handle.partial and handle.manifest:
        handle = await capsule.annotate(handle, on_progress=on_progress, kind=kind)
    result = handle.model_dump()
    # Dry runs have no artifacts to render or summarise.
    return result if args.get("dry_run", False) else _augment_result(result)


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

        report_path = viewer.render_run(args["run_id"])
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
        gate_synth_at_agreement=args.get("gate_synth_at_agreement"),
        on_progress=on_progress,
    )
    return _augment_result(result.model_dump())


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
    result = await sequence_mod.sequence(
        prompts,
        specs,
        synthesiser=args.get("synthesiser"),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        capsule_kind=args.get("capsule_kind", "decision"),
        rubric=args.get("rubric"),
        on_progress=on_progress,
    )
    return _augment_result(result.model_dump())


async def refine(args: dict[str, Any], *, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    specs = _specs_from_args(args["models"])
    result = await refine_mod.refine(
        prompt,
        specs,
        arbiter=args.get("arbiter"),
        threshold=args.get("threshold", 0.85),
        max_rounds=args.get("max_rounds", 3),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        synthesiser=args.get("synthesiser"),
        continuation_id=args.get("continuation_id"),
        rubric=args.get("rubric"),
        # Pass None when the MCP caller didn't specify, so refine inherits
        # from the prior run's bundle (continuation case) instead of
        # silently defaulting back to "decision".
        capsule_kind=args.get("capsule_kind"),
        strategy=args.get("strategy", "default"),
        on_progress=on_progress,
    )
    return _augment_result(result.model_dump())
