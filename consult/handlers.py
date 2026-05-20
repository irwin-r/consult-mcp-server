"""Per-tool handler implementations.

The MCP wiring lives in `server.py` (decorator-bound entrypoints + dispatch);
the actual orchestration for each tool lives here. Splitting them keeps the
MCP surface stable while letting the per-tool flows evolve without churning
the wiring file.

`_progress_callback()` uses a lazy import to look up the MCP server's current
request context — avoids a circular import at module load (`server.py`
imports this module to build its dispatch table).
"""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from . import (
    artifacts,
    attachments,
    capsule,
    progress,
    registry,
    runner,
    synth,
)
from . import (
    refine as refine_mod,
)
from . import (
    sequence as sequence_mod,
)
from .types import ModelSpec, RunResult


def _specs_from_args(models_arg: list[dict[str, Any]]) -> list[ModelSpec]:
    return [ModelSpec(**m) for m in models_arg]


def _progress_callback():
    """Build an MCP `notifications/progress` callback for the current request.

    Returns None if the client didn't send a `progressToken` — silent for
    non-subscribers. The wire-format message string is derived from the
    event via `progress.event_message()`; the event's `(done, total)`
    populate the wire `progress` / `total` fields. The token is opaque to
    us; we echo what the client supplied.

    Any notification failure (closed session, etc.) is caught by the
    per-callsite try/except in `runner.fanout` so it never aborts the
    underlying tool call.

    The lazy `from .server import server` is required: server.py imports
    handlers.py at module load time to build its dispatch table, so a
    top-level import here would be circular. The lookup happens once per
    call, which is fine — `_progress_callback()` is called at most a
    handful of times per tool invocation.
    """
    from .server import server  # late import — see docstring
    try:
        ctx = server.request_context
    except LookupError:
        return None
    token = ctx.meta.progressToken if ctx.meta else None
    if token is None:
        return None
    session = ctx.session

    async def notify(event: progress.ProgressEvent) -> None:
        await session.send_progress_notification(
            progress_token=token,
            progress=float(event.done),
            total=float(event.total),
            message=progress.event_message(event),
        )

    return notify


async def panel(args: dict[str, Any]) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    specs = _specs_from_args(args["models"])
    progress_cb = _progress_callback()
    kind = args.get("capsule_kind", "decision")
    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        dry_run=args.get("dry_run", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=progress_cb,
        capsule_kind=kind,
    )
    if args.get("extract_capsules", True) and not handle.partial and handle.manifest:
        handle = await capsule.annotate(handle, on_progress=progress_cb, kind=kind)
    return handle.model_dump()


async def synthesise(args: dict[str, Any]) -> list[TextContent]:
    # Synth's output is a markdown blob — `TextContent` is the natural shape
    # since a structured-content dict would force clients to unwrap the text
    # before rendering. (Every other tool returns a dict so MCP also surfaces
    # `structuredContent` for programmatic callers.)
    result = await synth.synthesise(
        args["run_id"],
        by_model=args.get("by_model"),
        rubric=args.get("rubric"),
        anonymised=args.get("anonymised", False),
    )
    return [TextContent(type="text", text=result.text)]


async def consult(args: dict[str, Any]) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    tier = args.get("tier", "standard")
    tier_models = registry.resolve_tier(tier)
    roles = args.get("roles") or {}
    synth_alias = args.get("synthesiser") or registry.default_synthesiser()

    # Exclude the synthesiser from the panel to avoid self-inclusion bias
    panel_aliases = [m for m in tier_models if m != synth_alias]
    specs = [ModelSpec(model=m, stance=roles.get(m)) for m in panel_aliases]

    # consult has three phases (fanout → capsules → synth). MCP progress
    # is monotonic, so each phase's callback shifts its (done, total) into
    # the consult-wide bucket. Offsets are passed explicitly per phase —
    # easier to follow than rebinding a nonlocal through a factory.
    base = _progress_callback()
    overall_total = len(specs) * 2 + 1  # fanout + capsules + synth
    fanout_offset = 0
    capsule_offset = len(specs)
    synth_offset = len(specs) * 2

    kind = args.get("capsule_kind", "decision")
    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=progress.shift_bucket(base, fanout_offset, overall_total),
        capsule_kind=kind,
    )
    if handle.partial or not handle.manifest:
        # Return a real `RunResult` so the partial response has the same shape
        # as the success path — clients can rely on a single dict schema and
        # branch on `partial` / `partial_reason` rather than two layouts.
        partial = RunResult(
            run_id=handle.run_id,
            synthesis="",
            manifest=handle.manifest,
            cost_usd=handle.cost_usd,
            cost_known=handle.cost_known,
            wall_ms=handle.wall_ms,
            partial=True,
            partial_reason=(
                handle.partial_reason or "no panellists returned usable responses"
            ),
            synthesiser=synth_alias,
        )
        return partial.model_dump()
    extract_capsules = args.get("extract_capsules", True)
    if extract_capsules:
        handle = await capsule.annotate(
            handle,
            on_progress=progress.shift_bucket(base, capsule_offset, overall_total),
            kind=kind,
        )
    # The outer total was sized for fanout + capsules + synth, so synth's
    # `done` starts at the synth offset regardless of whether capsules ran.
    if base is not None:
        await base(progress.SynthStarted(done=synth_offset, total=overall_total))
    synth_result = await synth.synthesise(
        handle.run_id,
        by_model=synth_alias,
        anonymised=args.get("blinded", False),
        rubric=args.get("rubric"),
    )
    if base is not None:
        await base(progress.SynthCompleted(done=overall_total, total=overall_total))
    # Persist the synthesiser choice on disk so `consult-view` can badge it
    # in the header. `RunResult` carries it on the wire, but the manifest
    # written by `runner.fanout` was assembled before synth ran.
    artifacts.augment_manifest(artifacts.load_run(handle.run_id), synthesiser=synth_alias)
    # Roll synth spend into the run total. The synthesiser is often the most
    # expensive call (flagship + big context), so omitting it silently
    # under-reports the run against `max_run_usd`.
    total_cost = handle.cost_usd + synth_result.cost_usd
    total_cost_known = handle.cost_known and synth_result.cost_known
    result = RunResult(
        run_id=handle.run_id,
        synthesis=synth_result.text,
        manifest=handle.manifest,
        cost_usd=total_cost,
        cost_known=total_cost_known,
        wall_ms=handle.wall_ms,
        partial=False,
        synthesiser=synth_alias,
    )
    return result.model_dump()


async def sequence(args: dict[str, Any]) -> dict[str, Any]:
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
        on_progress=_progress_callback(),
    )
    return result.model_dump()


async def refine(args: dict[str, Any]) -> dict[str, Any]:
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
        on_progress=_progress_callback(),
    )
    return result.model_dump()
