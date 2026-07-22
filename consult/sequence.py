"""Sequential multi-step consultation.

Runs an ordered list of prompts where each step is a full
fanout → capsule → synth cycle, and EVERY prior step's synthesis is
prepended as context for step N+1. (Only the immediately-prior synthesis
used to be threaded, so a final "synthesise across the prior steps"
prompt silently saw one step; the per-call context fit still trims if a
long chain outgrows a model's window.) Each step gets its own run_id;
the SequenceResult collects them all with the final synth (= last
step's synth) exposed directly for callers that don't need per-step
detail.

Stops early on cost-cap violation; returns a partial SequenceResult so
the caller can see how far the chain got.
"""

from __future__ import annotations

import logging
import time

from pydantic import Field, model_validator

from . import artifacts, capsule, registry, runner, synth
from . import progress as progress_mod
from .cost import CostMeter
from .types import ModelSpec, StrictModel

logger = logging.getLogger(__name__)


class SequenceStep(StrictModel):
    """One step in a sequence — its run_id, prompt-as-sent, and synth."""

    step: int = Field(..., ge=1)
    run_id: str
    synthesis: str
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    panel_size: int = Field(..., ge=0)
    # No per-step invariant validator: `cost_usd` is a non-None `float`, so
    # the `cost_usd=None ⇒ cost_known=False` rule from ManifestEntry doesn't
    # apply. The cross-step "if any step is unknown, the chain is unknown"
    # check lives on SequenceResult.


class SequenceResult(StrictModel):
    """Aggregate result of a sequence run."""

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
            raise ValueError("SequenceResult.cost_known=True but a step has cost_known=False")
        return self


def _step_prompt(step_num: int, total: int, prior_syntheses: list[str] | None, body: str) -> str:
    """Assemble step N's prompt with every prior step's synthesis ahead of it."""
    if not prior_syntheses:
        return body
    sections = [
        f"## Step {i} of {total} — prior synthesis\n\n{synth_text}"
        for i, synth_text in enumerate(prior_syntheses, start=1)
    ]
    sections.append(f"## Step {step_num} of {total} prompt\n\n{body}")
    return "\n\n---\n\n".join(sections)


async def sequence(
    prompts: list[str],
    specs: list[ModelSpec],
    *,
    synthesiser: str | None = None,
    blinded: bool = False,
    max_run_usd: float | None = None,
    dry_run: bool = False,
    capsule_kind: str = "decision",
    rubric: str | None = None,
    max_output_tokens: int | None = None,
    on_progress: runner.ProgressCallback | None = None,
) -> SequenceResult:
    """Run `prompts` as a chain where step i sees every prior step's synthesis."""
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

    # Dry run: price every step's panel and return without spending. The
    # estimate sums per-step fanout cost over the bare step bodies — it
    # excludes per-step synth and the prior-step context that grows later
    # steps, so it's a floor. Before this branch, the MCP `sequence` schema
    # carried no `dry_run`, so a caller's flag was silently dropped and the
    # full multi-step chain ran anyway (FRICTION 2026-06-14). Mirrors the
    # panel/consult dry_run path.
    if dry_run:
        total_est = 0.0
        all_known = True
        for body in prompts:
            est, known = await runner.aestimate_cost(specs, body, capsule_kind=capsule_kind)
            total_est += est
            all_known = all_known and known
        suffix = "" if all_known else " (some prices unknown — actual cost may differ)"
        return SequenceResult(
            steps=[],
            final_synthesis="(dry run — no steps executed; see partial_reason)",
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=(
                f"dry_run: estimated panel cost ~${total_est:.4f} across {len(prompts)} step(s); "
                f"excludes per-step synth and prior-step context growth{suffix}"
            ),
        )

    start = time.time()
    steps: list[SequenceStep] = []
    meter = CostMeter()
    partial_reason: str | None = None
    prior_syntheses: list[str] = []
    total = len(prompts)

    # Bucketed progress: roughly 2N+1 ticks per step (fanout panellists +
    # capsules + synth). Coarse but monotonic; the per-step callbacks tick
    # within their bucket.
    panel_n = len(specs)
    progress_total = total * (panel_n * 2 + 1)
    # Mutable cell so `progress_mod.make_phase_cb` can update it; we read
    # `progress_done[0]` for the direct emits between phases (synth, step
    # completed).
    progress_done = [0]

    async def emit(event: progress_mod.ProgressEvent) -> None:
        if on_progress is not None:
            try:
                await on_progress(event)
            except Exception as e:  # noqa: BLE001
                logger.debug("sequence on_progress failed: %s", e)

    for i, body in enumerate(prompts, start=1):
        step_base = (i - 1) * (panel_n * 2 + 1)
        full_prompt = _step_prompt(i, total, prior_syntheses, body)

        step_est_kwargs: dict = {"capsule_kind": capsule_kind}
        if max_output_tokens is not None:
            step_est_kwargs["max_output_tokens"] = max_output_tokens
        estimate, est_known = await runner.aestimate_cost(
            specs,
            full_prompt,
            **step_est_kwargs,
        )
        if meter.total + estimate > cap:
            partial_reason = (
                f"would exceed cap: spent ${meter.total:.2f}, step {i} estimate "
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
            meter.mark_unknown()

        await emit(
            progress_mod.SequenceStepStarted(
                done=step_base,
                total=progress_total,
                step=i,
            )
        )
        handle = await runner.fanout(
            full_prompt,
            specs,
            blinded=blinded,
            max_run_usd=cap - meter.total,
            on_progress=progress_mod.make_phase_cb(
                emit if on_progress else None,
                step_base,
                progress_total,
                progress_done,
            ),
            capsule_kind=capsule_kind,
            max_output_tokens=max_output_tokens,
        )
        if handle.partial or not handle.manifest:
            # Roll the partial fanout's spend into the running total before
            # breaking — a zero-usable-panel fanout may have billed for
            # timeouts. Mirrors the refine iter1 fix; without this the
            # SequenceResult silently understates spend on the break.
            meter.add(handle.cost_usd, handle.cost_known)
            partial_reason = f"step {i} fanout returned partial: {handle.partial_reason}"
            break

        handle = await capsule.annotate(
            handle,
            on_progress=progress_mod.make_phase_cb(
                emit if on_progress else None,
                step_base + panel_n,
                progress_total,
                progress_done,
            ),
            kind=capsule_kind,
        )
        meter.add(handle.cost_usd, handle.cost_known)

        progress_done[0] = step_base + panel_n * 2
        await emit(progress_mod.SynthStarted(done=progress_done[0], total=progress_total))
        synth_result = await synth.synthesise(
            handle.run_id, by_model=synth_alias, anonymised=blinded, rubric=rubric
        )
        # Synth spend rolls into the per-step total (and so into the
        # cumulative cap check on the next step), mirroring the consult and
        # refine handlers. Previously this was silently dropped.
        step_cost = handle.cost_usd + synth_result.cost_usd
        step_cost_known = handle.cost_known and synth_result.cost_known
        meter.add(synth_result.cost_usd, synth_result.cost_known)

        # Persist + record the step regardless of synth status. Previously
        # the synth-failure break ran BEFORE augment_manifest and
        # steps.append, so a failed step's cost vanished from disk (ledger
        # under-report) and the result's `steps` array omitted the run
        # entirely. The break still fires below — but only after the step
        # has been fully recorded.
        await artifacts.aaugment_manifest(
            artifacts.load_run(handle.run_id),
            synthesiser=synth_alias,
            cost_usd=step_cost,
            cost_known=step_cost_known,
        )

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
        # A non-OK synth means the body is a "# Synthesis unavailable" /
        # "# Synthesis empty" sentinel. Feeding that into the next step as
        # `prior_synth` would make the panel hallucinate continuity from a
        # failure marker — break here so the partial reason is the real
        # cause, not a downstream mystery. The break runs AFTER the step
        # has been recorded so its cost still reaches the ledger.
        if synth_result.status is not synth.SynthStatus.OK:
            partial_reason = (
                f"step {i} synth status={synth_result.status.value}; "
                "stopping chain rather than feeding a sentinel into the next step"
            )
            break

        progress_done[0] = step_base + panel_n * 2 + 1
        await emit(
            progress_mod.SequenceStepCompleted(
                done=progress_done[0],
                total=progress_total,
                step=i,
            )
        )
        prior_syntheses.append(synth_result.text)

    wall_ms = int((time.time() - start) * 1000)
    final = steps[-1].synthesis if steps else "(no steps completed — see partial_reason)"
    return SequenceResult(
        steps=steps,
        final_synthesis=final,
        cost_usd=meter.total,
        cost_known=meter.known,
        wall_ms=wall_ms,
        partial=partial_reason is not None,
        partial_reason=partial_reason,
    )
