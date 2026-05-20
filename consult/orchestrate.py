"""Engine-level orchestration of multi-stage tools.

Currently houses the `consult()` hero: fanout → capsule extract → synth,
with progress bucketing, partial-handle short-circuit, synth-cost roll-up,
and on-disk manifest augmentation. Extracted from the MCP adapter so a
non-MCP consumer (CLI, HTTP, library use) can drive the same flow with
typed kwargs instead of a `dict` envelope.

Single-primitive flows (`runner.fanout`, `refine.refine`, `sequence.sequence`,
`synth.synthesise`) don't need wrapping — call them directly.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from . import artifacts, capsule, registry, runner, synth
from .progress import ProgressEvent, SynthCompleted, SynthStarted, shift_bucket
from .types import ModelSpec, RunResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


async def _safe_emit(
    cb: ProgressCallback | None, event: ProgressEvent
) -> None:
    """Emit a progress event, swallowing any callback exception.

    Progress is best-effort: a notification failure (closed transport,
    raising user callback) must never abort the tool call. The runner
    and refine modules wrap their per-call emissions; this helper
    closes the gap for the direct `synth_*` emissions in `consult()`.
    """
    if cb is None:
        return
    try:
        await cb(event)
    except Exception as e:  # noqa: BLE001
        logger.debug("progress callback failed: %s", e)


async def consult(
    prompt: str,
    *,
    tier: str = "standard",
    roles: dict[str, str] | None = None,
    synthesiser: str | None = None,
    blinded: bool = False,
    max_run_usd: float | None = None,
    extract_capsules: bool = True,
    capsule_kind: str = "decision",
    rubric: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> RunResult:
    """Run the 3-phase hero: fanout → capsule extract → synth.

    The synthesiser is excluded from the panel to avoid self-inclusion
    bias. Synth spend rolls into the returned `cost_usd` and is also
    persisted to the on-disk manifest so `consult-ledger` reports the
    true run total rather than the fanout-only figure.

    `on_progress`, when set, receives a single monotonic stream of
    progress events across all three phases — child events from each
    stage are shifted into a consult-wide `(done, total)` bucket.

    Returns a fully-populated `RunResult`. On a partial/failed panel,
    returns a `RunResult` with `partial=True`, empty `synthesis`, and
    `partial_reason` set — same shape as the success path so callers
    can branch on the flag rather than the envelope.
    """
    tier_models = registry.resolve_tier(tier)
    roles = roles or {}
    synth_alias = synthesiser or registry.default_synthesiser()
    # Fail fast on a typo'd synthesiser alias BEFORE we spend the panel
    # cost. `registry.resolve_model` raises KeyError on an unknown alias.
    registry.resolve_model(synth_alias)

    # Exclude the synthesiser from the panel to avoid self-inclusion bias.
    panel_aliases = [m for m in tier_models if m != synth_alias]
    specs = [ModelSpec(model=m, stance=roles.get(m)) for m in panel_aliases]

    # MCP progress is monotonic, so each phase's callback shifts its
    # (done, total) into the consult-wide bucket. Offsets are explicit
    # per phase — easier to follow than rebinding a nonlocal.
    overall_total = len(specs) * 2 + 1  # fanout + capsules + synth
    fanout_offset = 0
    capsule_offset = len(specs)
    synth_offset = len(specs) * 2

    handle = await runner.fanout(
        prompt,
        specs,
        blinded=blinded,
        max_run_usd=max_run_usd,
        on_progress=shift_bucket(on_progress, fanout_offset, overall_total),
        capsule_kind=capsule_kind,
    )
    if handle.partial or not handle.manifest:
        # Return a real `RunResult` so the partial response has the same shape
        # as the success path — clients can rely on a single dict schema and
        # branch on `partial` / `partial_reason` rather than two layouts.
        return RunResult(
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

    if extract_capsules:
        handle = await capsule.annotate(
            handle,
            on_progress=shift_bucket(on_progress, capsule_offset, overall_total),
            kind=capsule_kind,
        )

    # The outer total was sized for fanout + capsules + synth, so synth's
    # `done` starts at the synth offset regardless of whether capsules ran.
    await _safe_emit(
        on_progress, SynthStarted(done=synth_offset, total=overall_total)
    )
    synth_result = await synth.synthesise(
        handle.run_id,
        by_model=synth_alias,
        anonymised=blinded,
        rubric=rubric,
    )
    await _safe_emit(
        on_progress, SynthCompleted(done=overall_total, total=overall_total)
    )

    # Roll synth spend into the run total. The synthesiser is often the most
    # expensive call (flagship + big context), so omitting it silently
    # under-reports the run against `max_run_usd`.
    total_cost = handle.cost_usd + synth_result.cost_usd
    total_cost_known = handle.cost_known and synth_result.cost_known
    # Persist synthesiser + total cost on disk. `RunResult` carries them on
    # the wire, but the manifest written by `runner.fanout` was assembled
    # before synth ran — `consult-ledger` reads from disk and would otherwise
    # under-report by the synth call's spend.
    artifacts.augment_manifest(
        artifacts.load_run(handle.run_id),
        synthesiser=synth_alias,
        cost_usd=total_cost,
        cost_known=total_cost_known,
    )
    return RunResult(
        run_id=handle.run_id,
        synthesis=synth_result.text,
        manifest=handle.manifest,
        cost_usd=total_cost,
        cost_known=total_cost_known,
        wall_ms=handle.wall_ms,
        partial=False,
        synthesiser=synth_alias,
    )
