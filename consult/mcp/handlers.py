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


async def panel(
    args: dict[str, Any], *, on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
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
    return handle.model_dump()


async def synthesise(
    args: dict[str, Any], *, on_progress: ProgressCallback | None = None
) -> str:
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
    return result.text


async def consult(
    args: dict[str, Any], *, on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
    prompt = attachments.inline_attachments(args["prompt"], args.get("attachments"))
    result = await orchestrate.consult(
        prompt,
        tier=args.get("tier", "standard"),
        roles=args.get("roles"),
        synthesiser=args.get("synthesiser"),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        extract_capsules=args.get("extract_capsules", True),
        capsule_kind=args.get("capsule_kind", "decision"),
        rubric=args.get("rubric"),
        on_progress=on_progress,
    )
    return result.model_dump()


async def sequence(
    args: dict[str, Any], *, on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
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
    return result.model_dump()


async def refine(
    args: dict[str, Any], *, on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
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
        on_progress=on_progress,
    )
    return result.model_dump()
