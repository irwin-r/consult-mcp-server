"""Refine — iterative consortium-style consultation.

Workflow per the v1.1 spec voted by the v1 panel:
- Hard cap at `max_rounds` (default 3) rounds, no exceptions.
- Each round: fanout → capsule extraction → arbiter scores sufficiency.
- If arbiter score >= threshold, converge and synthesise.
- Otherwise build a refinement prompt using prior positions + gaps and run another round.
- Per-round transcripts live as MCP resources via round-suffixed slugs (`<slug>.r<n>`).
- Cumulative cost tracked against `max_run_usd`; refuses further rounds if a
  subsequent round would push past the cap (returns `partial=True`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import litellm

from . import artifacts, capsule, context, registry, runner, synth
from . import progress as progress_mod
from .jsonparse import extract_json
from .types import (
    ArbiterVerdict,
    Capsule,
    ManifestEntry,
    ModelSpec,
    RefineResult,
    ResearchCapsule,
    ReviewCapsule,
    Status,
)

logger = logging.getLogger(__name__)

_ARBITER_PROMPT = """\
You are evaluating whether a multi-model panel has reached sufficient \
agreement to ship a final answer. You are scoring **sufficiency for action**, \
not absolute truth.

Original question:
{question}

Panel health: {usable_count} of {total_count} panellists returned usable \
responses ({health_breakdown}). Down-weight your sufficiency score if a \
significant fraction of the panel failed — consensus from half a panel is \
weaker evidence than consensus from a full panel.

Round {round_num} panel capsules:
{capsules}

How positions changed since the prior round:
{position_diff}

Return EXACTLY this JSON object (no commentary, no markdown fences):

{{
  "score": 0.0,
  "gaps": ["..."],
  "next_round_focus": "...",
  "reasoning": "..."
}}

Where:
- score: 0.0-1.0. 1.0 = strong consensus, ready to ship the final answer.
                  0.5 = useful signal but material disagreement or missing detail.
                  0.0 = panellists contradict each other or miss the question.
- gaps: 1-4 specific items the panel hasn't resolved
- next_round_focus: one sentence telling the next round what to address
- reasoning: 1-3 sentences explaining the score
"""

_REFINEMENT_PROMPT_TEMPLATE = """\
This is round {round_num} of a multi-round consultation. \
You are refining your answer based on the prior round.

Original question:
{question}

Positions from the prior round:
{positions}

Gaps the arbiter flagged that the next round should address:
{gaps}

Specifically focus on: {focus}

Now give your refined answer to the original question, addressing the gaps. \
Be concrete; don't simply restate the prior position."""

def _capsule_summary(cap: Capsule | ReviewCapsule | ResearchCapsule) -> str:
    """One-line summary of any capsule kind. Used in arbiter prompts where
    a position-like signal is needed regardless of `capsule_kind`."""
    if isinstance(cap, ReviewCapsule):
        n = len(cap.findings)
        if not n:
            return cap.overall_verdict
        sev_counts: dict[str, int] = {}
        for f in cap.findings:
            sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
        sev_str = ", ".join(f"{v}× {k}" for k, v in sorted(sev_counts.items()))
        return f"{cap.overall_verdict} ({n} findings: {sev_str})"
    if isinstance(cap, ResearchCapsule):
        return (
            f"{len(cap.claims)} claims, "
            f"{len(cap.uncertainties)} uncertainties, "
            f"{len(cap.evidence)} evidence items"
        )
    # Decision capsule
    return cap.position or "(no position)"


def _capsule_detail(cap: Capsule | ReviewCapsule | ResearchCapsule) -> str:
    """Multi-line detail rendering for the arbiter's capsules section.

    Each capsule kind gets a different layout — review surfaces top
    findings, research surfaces top claims, decision keeps the original
    recommendation/key_points shape.
    """
    if isinstance(cap, ReviewCapsule):
        lines: list[str] = [f"  overall_verdict: {cap.overall_verdict}"]
        if cap.findings:
            lines.append(f"  findings ({len(cap.findings)}; showing up to 5):")
            for f in cap.findings[:5]:
                loc = f.file or "(no file)"
                if f.line_range:
                    loc += f":{f.line_range[0]}-{f.line_range[1]}"
                lines.append(f"    - [{f.severity}/{f.category}] {loc} — {f.summary}")
        return "\n".join(lines)
    if isinstance(cap, ResearchCapsule):
        parts: list[str] = []
        if cap.claims:
            parts.append("  claims: " + "; ".join(cap.claims[:3]))
        if cap.uncertainties:
            parts.append("  uncertainties: " + "; ".join(cap.uncertainties[:3]))
        if cap.evidence:
            parts.append("  evidence: " + "; ".join(cap.evidence[:3]))
        return "\n".join(parts) or "  (no claims extracted)"
    # Decision capsule
    bullets = "; ".join(cap.key_points[:3]) if cap.key_points else "(no key points)"
    return (
        f"  recommendation: {cap.recommendation}\n"
        f"  key_points: {bullets}"
    )


def _format_capsules(manifest: list[ManifestEntry]) -> str:
    lines = []
    for m in manifest:
        if m.status not in (Status.OK, Status.TRUNCATED):
            lines.append(f"- {m.slug} [status={m.status.value}, no usable response]")
            continue
        c = m.capsule
        if not c:
            lines.append(f"- {m.slug}: (no capsule extracted)")
            continue
        conf = f" conf={c.confidence:.2f}" if c.confidence is not None else ""
        lines.append(
            f"- {m.slug}{conf}: {_capsule_summary(c)}\n{_capsule_detail(c)}"
        )
    return "\n".join(lines)


def _format_positions(manifest: list[ManifestEntry]) -> str:
    lines = []
    for m in manifest:
        if not m.capsule or m.status not in (Status.OK, Status.TRUNCATED):
            continue
        lines.append(f"- {m.slug}: {_capsule_summary(m.capsule)}")
    return "\n".join(lines) or "(none extracted)"


_ROUND_SUFFIX_RE = re.compile(r"\.r\d+$")


def _base_slug(slug: str) -> str:
    """Strip the `.r<n>` round suffix so a panellist matches across rounds."""
    return _ROUND_SUFFIX_RE.sub("", slug)


def _format_position_diff(
    prior: list[ManifestEntry] | None,
    current: list[ManifestEntry],
) -> str:
    """Per-panellist position changes between rounds, by base slug.

    Sized for the arbiter (NOT bodies — bodies would dilute the arbiter's
    attention per the v2 review). Shows what each panellist's stance was
    before vs after, so the arbiter can score whether the round actually
    moved the needle.
    """
    if not prior:
        return "(first round — no prior to diff against)"

    def _by_base(manifest: list[ManifestEntry]) -> dict[str, ManifestEntry]:
        return {
            _base_slug(m.slug): m
            for m in manifest
            if m.capsule and m.status in (Status.OK, Status.TRUNCATED)
        }

    prior_by_base = _by_base(prior)
    current_by_base = _by_base(current)

    lines: list[str] = []
    for base, cur in current_by_base.items():
        assert cur.capsule is not None  # _by_base filters None capsules
        cur_summary = _capsule_summary(cur.capsule)
        old = prior_by_base.get(base)
        if old and old.capsule:
            old_summary = _capsule_summary(old.capsule)
            if old_summary == cur_summary:
                lines.append(f"- {base}: unchanged — {cur_summary}")
            else:
                lines.append(
                    f"- {base}:\n"
                    f"    before: {old_summary}\n"
                    f"    after:  {cur_summary}"
                )
        else:
            lines.append(f"- {base} (new this round): {cur_summary}")
    # Surface panellists that dropped out this round — their disappearance
    # is signal the arbiter should weigh ("3 of 5 now agree, but 2 of 5
    # are missing this round so consensus is weaker than it looks").
    for base, old in prior_by_base.items():
        if base not in current_by_base and old.capsule:
            lines.append(
                f"- {base} (dropped this round, last position): {_capsule_summary(old.capsule)}"
            )
    return "\n".join(lines) or "(no comparable positions)"


def _build_refinement_prompt(
    question: str, round_num: int, prior_manifest: list[ManifestEntry], verdict: ArbiterVerdict
) -> str:
    return _REFINEMENT_PROMPT_TEMPLATE.format(
        round_num=round_num,
        question=question,
        positions=_format_positions(prior_manifest),
        gaps="\n".join(f"- {g}" for g in verdict.gaps) if verdict.gaps else "(none flagged)",
        focus=verdict.next_round_focus or "any remaining ambiguity",
    )


async def _ask_arbiter(
    question: str,
    round_num: int,
    manifest: list[ManifestEntry],
    arbiter_alias: str,
    prior_manifest: list[ManifestEntry] | None = None,
) -> ArbiterVerdict:
    entry = registry.resolve_model(arbiter_alias)
    litellm_id = entry["litellm_id"]
    timeout = entry.get("default_timeout_s", 180)

    usable_count = sum(1 for m in manifest if m.status in (Status.OK, Status.TRUNCATED))
    counts: dict[str, int] = {}
    for m in manifest:
        counts[m.status.value] = counts.get(m.status.value, 0) + 1
    health_breakdown = ", ".join(f"{v}× {k}" for k, v in sorted(counts.items()))

    prompt = _ARBITER_PROMPT.format(
        question=question,
        round_num=round_num,
        capsules=_format_capsules(manifest),
        position_diff=_format_position_diff(prior_manifest, manifest),
        usable_count=usable_count,
        total_count=len(manifest),
        health_breakdown=health_breakdown,
    )

    # 1) Call — exception here means we never got text back
    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=litellm_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.0,
            ),
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("arbiter call failed: %s: %s", type(e).__name__, e)
        return ArbiterVerdict(
            round=round_num,
            score=0.0,
            gaps=[],
            reasoning="arbiter call failed; refine must abort or retry",
            cost_usd=None,
            cost_known=False,
            parsed_ok=False,
            error=f"{type(e).__name__}: {e!s:.150}",
        )

    text = resp.choices[0].message.content or ""

    # 2) Cost — independent of parsing
    try:
        cost = litellm.completion_cost(completion_response=resp)
        cost_known = cost is not None
    except Exception as e:
        logger.warning("arbiter cost lookup failed: %s", e)
        cost = None
        cost_known = False

    # 3) JSON parse — failure here is real signal (don't pollute gaps with an
    # exception string; the next round's prompt would silently include it)
    data = extract_json(text)
    if data is None:
        logger.warning(
            "arbiter returned non-JSON; sample=%r", text[:120].replace("\n", " ")
        )
        return ArbiterVerdict(
            round=round_num,
            score=0.0,
            gaps=[],
            reasoning="arbiter returned non-JSON output",
            cost_usd=cost,
            cost_known=cost_known,
            parsed_ok=False,
            error="json_parse_failed",
        )

    # 4) Score coercion — strings like "high" must not silently float-fail
    try:
        score = float(data.get("score", 0.0))
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"out of range: {score}")
    except (TypeError, ValueError) as e:
        logger.warning("arbiter score not parseable: %s", e)
        return ArbiterVerdict(
            round=round_num,
            score=0.0,
            gaps=[],
            reasoning="arbiter score field malformed",
            cost_usd=cost,
            cost_known=cost_known,
            parsed_ok=False,
            error=f"bad_score: {e!s:.100}",
        )

    return ArbiterVerdict(
        round=round_num,
        score=score,
        gaps=list(data.get("gaps") or []),
        next_round_focus=str(data.get("next_round_focus") or ""),
        reasoning=str(data.get("reasoning") or ""),
        cost_usd=cost,
        cost_known=cost_known,
        parsed_ok=True,
    )


def _suffix_specs(specs: list[ModelSpec], round_num: int) -> list[ModelSpec]:
    """Suffix slugs with `.r<n>` so each round writes to distinct artifact files
    within the same run directory.
    """
    out = []
    for i, s in enumerate(specs):
        base = s.slug or s.model.split("/")[-1].lower()
        out.append(
            ModelSpec(model=s.model, stance=s.stance, slug=f"{base}-{i}.r{round_num}")
        )
    return out


def _apply_continuation(prompt: str, continuation_id: str | None) -> str:
    """Prepend the prior run's original question + synthesis to `prompt`.

    Order: stable prefix (prior question + prior synthesis) first, then the
    new (volatile) question last. Anthropic + OpenAI prompt caches key on
    leading bytes, so this ordering means every panellist in this turn
    reuses the cache from the first panellist. Including the prior question
    (not just the synthesis) restores fidelity — without it, the new panel
    would only see a summary of what was asked previously, not the source
    material.

    Raises `ValueError` on a missing run dir or missing `synthesis.md` — a
    typo in `continuation_id` must not silently drop the prior context
    (caller would think the new round had it, but it wouldn't). Empty
    string is treated the same as None.
    """
    if not continuation_id:
        return prompt
    try:
        prior_paths = artifacts.load_run(continuation_id)
    except FileNotFoundError as e:
        raise ValueError(f"continuation_id not found: {continuation_id}") from e
    synth_path = prior_paths.root / "synthesis.md"
    if not synth_path.exists():
        raise ValueError(
            f"continuation_id {continuation_id} has no synthesis.md "
            "(was the prior run partial, dry-run, or pre-synth?)"
        )

    # Load the prior question — prefer the bundle (canonical post-Phase 1)
    # and fall back to prompt.txt for legacy runs created before contexts
    # were a thing.
    prior_bundle = context.load_or_none(prior_paths)
    if prior_bundle is not None:
        prior_question = prior_bundle.prompt
    elif prior_paths.prompt_txt.exists():
        prior_question = prior_paths.prompt_txt.read_text()
    else:
        prior_question = "(prior question unavailable)"

    prior_synth = synth_path.read_text()
    return (
        "## Prior consultation — original question\n\n"
        f"{prior_question}\n\n"
        "## Prior consultation — synthesis\n\n"
        f"{prior_synth}\n\n---\n\n"
        "## Follow-up question\n\n"
        f"{prompt}"
    )


async def refine(
    prompt: str,
    specs: list[ModelSpec],
    *,
    arbiter: str | None = None,
    threshold: float = 0.85,
    max_rounds: int = 3,
    blinded: bool = False,
    max_run_usd: float | None = None,
    synthesiser: str | None = None,
    continuation_id: str | None = None,
    rubric: str | None = None,
    capsule_kind: str | None = None,
    on_progress: runner.ProgressCallback | None = None,
) -> RefineResult:
    if max_rounds < 1 or max_rounds > 5:
        raise ValueError("max_rounds must be between 1 and 5")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.0, 1.0]")

    # Resolve capsule_kind precedence: explicit caller value > inherited
    # from prior run's ContextBundle (when continuation_id is set) >
    # "decision" default. Without this, a continuation that started as
    # review/research silently switches back to decision on the next
    # round, producing wrong-shape capsules.
    resolved_kind = capsule_kind
    if resolved_kind is None and continuation_id:
        try:
            prior_paths = artifacts.load_run(continuation_id)
            prior_bundle = context.load_or_none(prior_paths)
            if prior_bundle is not None:
                resolved_kind = prior_bundle.capsule_kind
        except FileNotFoundError:
            # `_apply_continuation` raises with a clearer message below;
            # don't pre-empt that here.
            pass
    if resolved_kind is None:
        resolved_kind = "decision"

    prompt = _apply_continuation(prompt, continuation_id)
    # Resolve `model:N` sugar here too so `estimate_cost` (called before
    # `fanout` in each round) sees the real expanded panel.
    specs = runner.expand_specs(specs)
    arbiter_alias = arbiter or registry.default_synthesiser()
    synth_alias = synthesiser or arbiter_alias

    paths = artifacts.create_run()
    paths.prompt_txt.write_text(prompt)
    paths.registry_snapshot.write_text(json.dumps(registry.models_config(), indent=2))
    # Per-run context bundle. Refine creates its own run dir then calls
    # `runner.fanout` with `existing_paths=` so we own the bundle write
    # here — runner skips it when given an existing path.
    context.write(
        paths,
        context.build(prompt, blinded=blinded, capsule_kind=resolved_kind),
    )
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()

    start = time.time()
    verdicts: list[ArbiterVerdict] = []
    final_manifest: list[ManifestEntry] = []
    cumulative_cost = 0.0
    cost_all_known = True
    converged = False
    partial_reason: str | None = None

    # Bucketed progress across all rounds. Per round we tick once per
    # panellist (fanout), once per capsule, then once for the arbiter;
    # final synth ticks once at the end. The counter is monotonic across
    # the whole refine call so the client never sees `done` go backwards.
    panel_n = len(specs)
    progress_total = max_rounds * (panel_n * 2 + 1) + 1  # rounds × (fanout+capsule+arbiter) + synth
    progress_done = 0

    async def emit(event: progress_mod.ProgressEvent) -> None:
        if on_progress is not None:
            try:
                await on_progress(event)
            except Exception as e:  # noqa: BLE001
                logger.debug("refine on_progress failed: %s", e)

    def make_phase_cb(base: int) -> runner.ProgressCallback | None:
        """Wrap a child event by shifting its `done`/`total` into the outer
        refine-wide bucket. Event identity (kind + payload) is preserved.
        """
        if on_progress is None:
            return None

        async def cb(event: progress_mod.ProgressEvent) -> None:
            nonlocal progress_done
            progress_done = base + event.done
            shifted = event.model_copy(update={"done": progress_done, "total": progress_total})
            await emit(shifted)

        return cb

    round_prompt = prompt
    prior_manifest: list[ManifestEntry] | None = None
    for round_num in range(1, max_rounds + 1):
        # Estimate next-round cost; refuse if it'd blow the cap.
        # If pricing is unknown for any spec, refuse conservatively past the
        # first round to avoid an unbounded bill.
        estimate, est_known = runner.estimate_cost(specs, round_prompt)
        if cumulative_cost + estimate > cap:
            partial_reason = (
                f"would exceed cap: spent ${cumulative_cost:.2f}, next round estimate "
                f"${estimate:.2f}, cap ${cap:.2f}"
            )
            break
        if not est_known and round_num > 1:
            partial_reason = (
                "refusing further rounds: per-model pricing unknown for at least one "
                f"panellist, can't validate cap (${cumulative_cost:.2f} spent / ${cap:.2f} cap)"
            )
            break

        round_base = (round_num - 1) * (panel_n * 2 + 1)
        round_specs = _suffix_specs(specs, round_num)
        handle = await runner.fanout(
            round_prompt,
            round_specs,
            blinded=blinded,
            existing_paths=paths,
            on_progress=make_phase_cb(round_base),
        )
        handle = await capsule.annotate(
            handle,
            on_progress=make_phase_cb(round_base + panel_n),
            kind=resolved_kind,
        )
        final_manifest = handle.manifest
        cumulative_cost += handle.cost_usd
        if not handle.cost_known:
            cost_all_known = False

        verdict = await _ask_arbiter(
            prompt, round_num, handle.manifest, arbiter_alias, prior_manifest
        )
        progress_done = round_base + panel_n * 2 + 1
        await emit(progress_mod.ArbiterScored(
            done=progress_done, total=progress_total,
            round=round_num, score=verdict.score,
        ))
        verdicts.append(verdict)
        if verdict.cost_usd:
            cumulative_cost += verdict.cost_usd
        paths.arbiter_for(round_num).write_text(verdict.model_dump_json(indent=2))

        if not verdict.parsed_ok:
            # Don't continue: an arbiter call/parse failure means we can't trust
            # the verdict to drive a next-round prompt. Stop the loop and let
            # the caller decide what to do with the final manifest we have.
            partial_reason = (
                f"arbiter failed at round {round_num} ({verdict.error}); "
                "loop aborted to avoid feeding error text into next-round prompt"
            )
            break

        if verdict.score >= threshold:
            converged = True
            break

        if round_num < max_rounds:
            round_prompt = _build_refinement_prompt(prompt, round_num + 1, handle.manifest, verdict)
        # Snapshot for the next round's arbiter position-diff. Updated
        # after the verdict so an aborted round (parse failure above)
        # leaves prior_manifest pointing at the last fully-scored round.
        prior_manifest = handle.manifest

    # Synthesise from the final round
    if final_manifest:
        progress_done = progress_total - 1
        await emit(progress_mod.SynthStarted(done=progress_done, total=progress_total))
        text = await synth.synthesise(
            paths.run_id, by_model=synth_alias, rubric=rubric
        )
        progress_done = progress_total
        await emit(progress_mod.SynthCompleted(done=progress_done, total=progress_total))
        # Persist the synthesiser so consult-view can badge it in the
        # header; matches the consult handler. Refine writes the manifest
        # once per round from `runner.fanout`, so this lands on the final
        # version after the loop has stopped.
        artifacts.augment_manifest(paths, synthesiser=synth_alias)
    else:
        text = "(no rounds completed — see partial_reason)"

    wall_ms = int((time.time() - start) * 1000)
    return RefineResult(
        run_id=paths.run_id,
        rounds_completed=len(verdicts),
        final_manifest=final_manifest,
        verdicts=verdicts,
        synthesis=text,
        converged=converged,
        threshold=threshold,
        cost_usd=cumulative_cost,
        cost_known=cost_all_known,
        wall_ms=wall_ms,
        partial=partial_reason is not None,
        partial_reason=partial_reason,
        continuation_of=continuation_id or None,
    )
