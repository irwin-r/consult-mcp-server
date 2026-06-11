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

import litellm

from . import artifacts, capsule, registry, runner, synth, voting
from . import attachments as attachments_mod
from .cost import CostMeter
from .progress import ProgressEvent, SynthCompleted, SynthStarted, shift_bucket
from .types import Capsule, ManifestEntry, ModelSpec, RunResult, Status

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


async def _safe_emit(cb: ProgressCallback | None, event: ProgressEvent) -> None:
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


def _deterministic_aggregate(manifest: list[ManifestEntry], *, heading: str, note: str) -> str:
    """Produce a synthesis-shaped markdown without calling the flagship.

    Two callers: the agreement gate (high consensus means the flagship's
    added cost buys little) and the cost-cap gate (the flagship estimate
    would breach `max_run_usd`). Each supplies its own heading and note
    so the reader can audit why the flagship was skipped. We surface each
    usable panellist's position/recommendation as a bulleted aggregate.

    Only `Capsule` (decision-kind) entries get a structured rendering;
    review/research kinds fall back to their slug + status because their
    "position" semantics differ enough that a one-line summary would
    mislead.
    """
    lines = [
        heading,
        "",
        note,
        "",
        "## Positions",
        "",
    ]
    for entry in manifest:
        if entry.status not in (Status.OK, Status.TRUNCATED) or entry.capsule is None:
            continue
        cap = entry.capsule
        if isinstance(cap, Capsule):
            line = f"- **{entry.slug}** — {cap.position}"
            if cap.recommendation and cap.recommendation != cap.position:
                line += f"; recommends: {cap.recommendation}"
        else:
            # Non-decision capsules: defer to the structured artefact;
            # don't try to one-line them.
            line = f"- **{entry.slug}** — see capsule artefact (`{type(cap).__name__}`)"
        lines.append(line)
    return "\n".join(lines)


def _estimate_synth_cost(synth_alias: str, manifest: list[ManifestEntry]) -> tuple[float, bool]:
    """Rough pre-flight price for the synth call.

    The panel bodies are the synth's input, so their combined output
    tokens approximate its prompt size; the output side uses the same
    budget floor `synth.synthesise` applies. Returns `(0.0, False)` when
    the model can't be priced (CLI synthesisers, table misses) — the
    caller then mirrors fanout's warn-don't-block posture.
    """
    try:
        entry = registry.resolve_model(synth_alias)
        litellm_id = entry["litellm_id"]
        tokens_in = sum(m.tokens_out or 0 for m in manifest)
        budget = max(entry.get("default_budget_tokens", 16000), 16000)
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=litellm_id, prompt_tokens=tokens_in, completion_tokens=budget
        )
        if prompt_cost is None or completion_cost is None:
            return 0.0, False
        return float(prompt_cost) + float(completion_cost), True
    except Exception:  # noqa: BLE001 — estimate failure must not block the run
        return 0.0, False


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
    attachments: list | None = None,
    dry_run: bool = False,
    on_progress: ProgressCallback | None = None,
    gate_synth_at_agreement: float | None = None,
) -> RunResult:
    """Run the 3-phase hero: fanout → capsule extract → synth.

    The synthesiser is excluded from the panel to avoid self-inclusion
    bias. Synth spend rolls into the returned `cost_usd` and is also
    persisted to the on-disk manifest so `consult-ledger` reports the
    true run total rather than the fanout-only figure.

    `attachments`, when set, is the same shape the MCP `consult` tool
    accepts — bare strings, `{path, label?, kind?}` dicts, or
    `{source: "git_diff", base, head, repo_path?, label?}` — and gets
    inlined into the prompt under an `--- ATTACHMENTS ---` separator.
    Library consumers can pass file references without separately
    importing `attachments.inline_attachments`.

    `dry_run=True` estimates cost without invoking any model — useful
    for pre-flight cap validation. Returns a partial RunResult with
    `cost_usd=0` and a `dry_run:` partial_reason.

    `on_progress`, when set, receives a single monotonic stream of
    progress events across all three phases — child events from each
    stage are shifted into a consult-wide `(done, total)` bucket.

    `gate_synth_at_agreement` (default None = always synth): when set
    and the post-capsule panel `disagreement` score is BELOW this value
    (i.e. the panel agreed strongly), the flagship synth is skipped and
    a deterministic per-panellist aggregate is returned instead. The
    returned RunResult has `synth_gated=True`. Useful for cost-aware
    cascades (MAgICoRe / FrugalGPT pattern): pay flagship only when the
    panel disagrees, accept aggregation when it doesn't. A reasonable
    starting threshold is 0.15-0.25 — calibrate against your panel.
    Requires `extract_capsules=True` to compute the score; with
    `extract_capsules=False` the gate is a no-op.

    Returns a fully-populated `RunResult`. On a partial/failed panel,
    returns a `RunResult` with `partial=True`, empty `synthesis`, and
    `partial_reason` set — same shape as the success path so callers
    can branch on the flag rather than the envelope.
    """
    if gate_synth_at_agreement is not None and not (0.0 <= gate_synth_at_agreement <= 1.0):
        raise ValueError(f"gate_synth_at_agreement must be in [0,1] or None; got {gate_synth_at_agreement!r}")
    tier_models = registry.resolve_tier(tier)
    roles = roles or {}
    synth_alias = synthesiser or registry.default_synthesiser()
    # Fail fast on a typo'd synthesiser alias BEFORE we spend the panel
    # cost. `registry.resolve_model` raises KeyError on an unknown alias.
    registry.resolve_model(synth_alias)

    # Inline attachments inside the engine so library consumers don't
    # have to do it. MCP handlers pre-inline before calling us, which
    # makes this a no-op for that path (attachments will be None).
    if attachments is not None:
        prompt = attachments_mod.inline_attachments(prompt, attachments)

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
        dry_run=dry_run,
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
            partial_reason=(handle.partial_reason or "no panellists returned usable responses"),
            synthesiser=synth_alias,
        )

    if extract_capsules:
        handle = await capsule.annotate(
            handle,
            on_progress=shift_bucket(on_progress, capsule_offset, overall_total),
            kind=capsule_kind,
        )

    # Spend so far: fanout plus (post-annotate) capsule extraction. The
    # meter owns the None-means-unknown propagation from here on.
    meter = CostMeter()
    meter.add(handle.cost_usd, handle.cost_known)

    # Compute disagreement post-capsule-extraction. None when fewer than
    # two usable capsules to compare (e.g. extract_capsules=False, or a
    # panel where most entries failed).
    disagreement = voting.panel_disagreement(handle.manifest)

    # Gating decision: skip the flagship synth when the panel converged
    # tightly. Requires both a configured threshold AND a computable
    # disagreement score (the None-case happens when capsules are off or
    # most panellists failed — gating in those cases would obscure the
    # real signal).
    gated = (
        gate_synth_at_agreement is not None
        and disagreement is not None
        and disagreement < gate_synth_at_agreement
    )

    if gated:
        # Deterministic aggregate. No model call, no synth spend rolled in.
        # We still write `synthesis.md` so the on-disk artifact dir stays
        # consistent (consult-view + consult-ledger work the same way).
        paths = artifacts.load_run(handle.run_id)
        synth_text = _deterministic_aggregate(
            handle.manifest,
            heading="# Synthesis (gated — high consensus)",
            note=(
                f"The panel converged with low disagreement (score "
                f"{disagreement:.2f}). The flagship synth was skipped to "
                f"save cost; each panellist's position is listed below."
            ),
        )
        (paths.root / "synthesis.md").write_text(synth_text)
        await _safe_emit(on_progress, SynthCompleted(done=overall_total, total=overall_total))
        await artifacts.aaugment_manifest(
            paths,
            synthesiser="(gated)",  # marker so the ledger entry is unambiguous
            cost_usd=meter.total,
            cost_known=meter.known,
        )
        return RunResult(
            run_id=handle.run_id,
            synthesis=synth_text,
            manifest=handle.manifest,
            cost_usd=meter.total,
            cost_known=meter.known,
            wall_ms=handle.wall_ms,
            partial=False,
            synthesiser="(gated)",
            disagreement=disagreement,
            synth_gated=True,
        )

    # Cost-cap gate for the synth stage. The fanout gate only covered the
    # panel: the flagship synth is often the most expensive single call in
    # the run and previously went unchecked, so a run could sail past
    # `max_run_usd` after the panel had already passed its own gate. When
    # the estimate can't be priced we mirror fanout's warn-don't-block
    # posture, unless spend has already reached the cap.
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()
    synth_est, synth_est_known = _estimate_synth_cost(synth_alias, handle.manifest)
    synth_over_cap = (synth_est_known and meter.total + synth_est > cap) or (
        not synth_est_known and meter.total >= cap
    )
    if not synth_est_known and not synth_over_cap:
        logger.warning(
            "synth cost for %s can't be estimated; proceeding (spent $%.2f of $%.2f cap)",
            synth_alias,
            meter.total,
            cap,
        )
    if synth_over_cap:
        paths = artifacts.load_run(handle.run_id)
        est_text = f"${synth_est:.2f}" if synth_est_known else "unknown"
        synth_text = _deterministic_aggregate(
            handle.manifest,
            heading="# Synthesis (aggregate — cost cap reached)",
            note=(
                f"The flagship synthesis was skipped: ${meter.total:.2f} of the "
                f"${cap:.2f} cap is already spent and the synthesiser estimate is "
                f"{est_text}. Each usable panellist's position is listed below. "
                f"Run `synthesise(run_id)` with a higher cap or a cheaper model "
                f"for a full synthesis."
            ),
        )
        (paths.root / "synthesis.md").write_text(synth_text)
        await _safe_emit(on_progress, SynthCompleted(done=overall_total, total=overall_total))
        await artifacts.aaugment_manifest(
            paths,
            synthesiser="(cap-skipped)",
            cost_usd=meter.total,
            cost_known=meter.known,
        )
        return RunResult(
            run_id=handle.run_id,
            synthesis=synth_text,
            manifest=handle.manifest,
            cost_usd=meter.total,
            cost_known=meter.known,
            wall_ms=handle.wall_ms,
            partial=True,
            partial_reason=(
                f"synthesis skipped at cost cap: spent ${meter.total:.2f}, "
                f"synth estimate {est_text}, cap ${cap:.2f}"
            ),
            synthesiser="(cap-skipped)",
            disagreement=disagreement,
        )

    # The outer total was sized for fanout + capsules + synth, so synth's
    # `done` starts at the synth offset regardless of whether capsules ran.
    await _safe_emit(on_progress, SynthStarted(done=synth_offset, total=overall_total))
    synth_result = await synth.synthesise(
        handle.run_id,
        by_model=synth_alias,
        anonymised=blinded,
        rubric=rubric,
    )
    await _safe_emit(on_progress, SynthCompleted(done=overall_total, total=overall_total))

    # Roll synth spend into the run total. The synthesiser is often the most
    # expensive call (flagship + big context), so omitting it silently
    # under-reports the run against `max_run_usd`.
    meter.add(synth_result.cost_usd, synth_result.cost_known)
    # Persist synthesiser + total cost on disk. `RunResult` carries them on
    # the wire, but the manifest written by `runner.fanout` was assembled
    # before synth ran — `consult-ledger` reads from disk and would otherwise
    # under-report by the synth call's spend.
    await artifacts.aaugment_manifest(
        artifacts.load_run(handle.run_id),
        synthesiser=synth_alias,
        cost_usd=meter.total,
        cost_known=meter.known,
    )
    # A non-OK synth status means `.text` is a sentinel ("# Synthesis
    # unavailable" / "# Synthesis empty"), not a real answer. Surface that as
    # a partial result instead of a clean success — otherwise the caller sees
    # partial=False with an error string in `synthesis`. Mirrors sequence.py.
    synth_failed = synth_result.status is not synth.SynthStatus.OK
    return RunResult(
        run_id=handle.run_id,
        synthesis=synth_result.text,
        manifest=handle.manifest,
        cost_usd=meter.total,
        cost_known=meter.known,
        wall_ms=handle.wall_ms,
        partial=synth_failed,
        partial_reason=(f"synthesis status={synth_result.status.value}" if synth_failed else None),
        synthesiser=synth_alias,
        disagreement=disagreement,
    )
