"""Sequential multi-step consultation.

Runs an ordered list of prompts where each step is a full
fanout → capsule → synth cycle, and step N's synthesis is prepended as
context for step N+1. Each step gets its own run_id; the SequenceResult
collects them all with the final synth (= last step's synth) exposed
directly for callers that don't need per-step detail.

Stops early on cost-cap violation; returns a partial SequenceResult so
the caller can see how far the chain got.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import capsule, registry, runner, synth
from . import progress as progress_mod
from .types import ModelSpec

_STRICT = ConfigDict(extra="forbid")

logger = logging.getLogger(__name__)


class SequenceStep(BaseModel):
    """One step in a sequence — its run_id, prompt-as-sent, and synth."""

    model_config = _STRICT

    step: int = Field(..., ge=1)
    run_id: str
    synthesis: str
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    panel_size: int = Field(..., ge=0)


class SequenceResult(BaseModel):
    """Aggregate result of a sequence run."""

    model_config = _STRICT

    steps: list[SequenceStep] = Field(default_factory=list)
    final_synthesis: str
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    wall_ms: int = Field(..., ge=0)
    partial: bool = False
    partial_reason: str | None = None

    @model_validator(mode="after")
    def _validate_partial(self) -> SequenceResult:
        if self.partial and not self.partial_reason:
            raise ValueError("SequenceResult.partial=True requires partial_reason")
        if not self.partial and self.partial_reason:
            raise ValueError("SequenceResult.partial=False must not carry a partial_reason")
        # cost_known must propagate from the per-step view: if any single
        # step couldn't be priced, the chain's total can't be either. Mirrors
        # the ManifestEntry/RunResult invariant so the cap-enforcement story
        # is uniform across tools.
        if self.cost_known and any(not s.cost_known for s in self.steps):
            raise ValueError(
                "SequenceResult.cost_known=True but a step has cost_known=False"
            )
        return self


def _step_prompt(step_num: int, total: int, prior_synth: str | None, body: str) -> str:
    if prior_synth is None:
        return body
    return (
        f"## Step {step_num - 1} of {total} — prior synthesis\n\n"
        f"{prior_synth}\n\n---\n\n"
        f"## Step {step_num} of {total} prompt\n\n{body}"
    )


async def sequence(
    prompts: list[str],
    specs: list[ModelSpec],
    *,
    synthesiser: str | None = None,
    blinded: bool = False,
    max_run_usd: float | None = None,
    capsule_kind: str = "decision",
    rubric: str | None = None,
    on_progress: runner.ProgressCallback | None = None,
) -> SequenceResult:
    """Run `prompts` as a chain where step i sees step i-1's synthesis."""
    if not prompts:
        raise ValueError("sequence requires at least one prompt")
    if not specs:
        raise ValueError("sequence requires at least one model spec")

    # Resolve `model:N` sugar up front so per-step `estimate_cost` and
    # `panel_n` (used for progress bucketing) see the real expanded panel.
    specs = runner.expand_specs(specs)

    synth_alias = synthesiser or registry.default_synthesiser()
    # Fail fast on a typo'd synthesiser alias BEFORE the first step's
    # fanout spends money. KeyError surfaces as UNKNOWN_MODEL at the MCP
    # boundary; an inline-burned panel would be wasted spend.
    registry.resolve_model(synth_alias)
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()

    start = time.time()
    steps: list[SequenceStep] = []
    cumulative_cost = 0.0
    cost_all_known = True
    partial_reason: str | None = None
    prior_synth: str | None = None
    total = len(prompts)

    # Bucketed progress: roughly 2N+1 ticks per step (fanout panellists +
    # capsules + synth). Coarse but monotonic; the per-step callbacks tick
    # within their bucket.
    panel_n = len(specs)
    progress_total = total * (panel_n * 2 + 1)
    progress_done = 0

    async def emit(event: progress_mod.ProgressEvent) -> None:
        if on_progress is not None:
            try:
                await on_progress(event)
            except Exception as e:  # noqa: BLE001
                logger.debug("sequence on_progress failed: %s", e)

    def phase_cb(base: int) -> runner.ProgressCallback | None:
        """Shift child events into the sequence-wide monotonic bucket.

        Tracks `progress_done` for subsequent direct `emit()` calls; the
        shift itself is delegated to `progress_mod.shift_bucket`.
        """
        if on_progress is None:
            return None
        inner = progress_mod.shift_bucket(emit, base, progress_total)

        async def cb(event: progress_mod.ProgressEvent) -> None:
            nonlocal progress_done
            progress_done = base + event.done
            assert inner is not None
            await inner(event)

        return cb

    for i, body in enumerate(prompts, start=1):
        step_base = (i - 1) * (panel_n * 2 + 1)
        full_prompt = _step_prompt(i, total, prior_synth, body)

        estimate, est_known = runner.estimate_cost(specs, full_prompt)
        if cumulative_cost + estimate > cap:
            partial_reason = (
                f"would exceed cap: spent ${cumulative_cost:.2f}, step {i} estimate "
                f"${estimate:.2f}, cap ${cap:.2f}"
            )
            break
        # Unlike refine — where a runaway arbiter could keep extending rounds —
        # sequence has a fixed, user-supplied step list. Partial-pricing is
        # already handled by the cumulative-cost check above + the per-step
        # `max_run_usd=cap-cumulative_cost` passed into fanout. Refusing on
        # est_known=False (as refine does) was over-conservative — the user
        # asked for N specific steps, surfacing cost_known=False at the end
        # is enough signal.
        if not est_known:
            cost_all_known = False

        await emit(progress_mod.SequenceStepStarted(
            done=step_base, total=progress_total, step=i,
        ))
        handle = await runner.fanout(
            full_prompt,
            specs,
            blinded=blinded,
            max_run_usd=cap - cumulative_cost,
            on_progress=phase_cb(step_base),
            capsule_kind=capsule_kind,
        )
        if handle.partial or not handle.manifest:
            # Roll the partial fanout's spend into the running total before
            # breaking — a zero-usable-panel fanout may have billed for
            # timeouts. Mirrors the refine iter1 fix; without this the
            # SequenceResult silently understates spend on the break.
            cumulative_cost += handle.cost_usd
            if not handle.cost_known:
                cost_all_known = False
            partial_reason = (
                f"step {i} fanout returned partial: {handle.partial_reason}"
            )
            break

        handle = await capsule.annotate(
            handle,
            on_progress=phase_cb(step_base + panel_n),
            kind=capsule_kind,
        )
        cumulative_cost += handle.cost_usd
        if not handle.cost_known:
            cost_all_known = False

        progress_done = step_base + panel_n * 2
        await emit(progress_mod.SynthStarted(done=progress_done, total=progress_total))
        synth_result = await synth.synthesise(
            handle.run_id, by_model=synth_alias, anonymised=blinded, rubric=rubric
        )
        # Synth spend rolls into the per-step total (and so into the
        # cumulative cap check on the next step), mirroring the consult and
        # refine handlers. Previously this was silently dropped.
        step_cost = handle.cost_usd + synth_result.cost_usd
        step_cost_known = handle.cost_known and synth_result.cost_known
        cumulative_cost += synth_result.cost_usd
        if not synth_result.cost_known:
            cost_all_known = False
        progress_done = step_base + panel_n * 2 + 1
        await emit(progress_mod.SequenceStepCompleted(
            done=progress_done, total=progress_total, step=i,
        ))

        steps.append(
            SequenceStep(
                step=i,
                run_id=handle.run_id,
                synthesis=synth_result.text,
                cost_usd=step_cost,
                cost_known=step_cost_known,
                panel_size=len(handle.manifest),
            )
        )
        prior_synth = synth_result.text

    wall_ms = int((time.time() - start) * 1000)
    final = steps[-1].synthesis if steps else "(no steps completed — see partial_reason)"
    return SequenceResult(
        steps=steps,
        final_synthesis=final,
        cost_usd=cumulative_cost,
        cost_known=cost_all_known,
        wall_ms=wall_ms,
        partial=partial_reason is not None,
        partial_reason=partial_reason,
    )
