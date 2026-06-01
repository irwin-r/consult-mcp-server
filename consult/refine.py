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
import random
import re
import time
from typing import Any

import litellm

from . import artifacts, capsule, context, provider_caps, registry, runner, strategies, synth
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

# v2 arbiter prompt: per-dimension 1-5 Likert with explicit anchors per
# level (Prometheus-style). The five dimensions decompose "sufficiency"
# into orthogonal axes so the next-round prompt can target the *weakest*
# axis rather than rephrasing a single global score. The shipped overall
# `score` is derived (averaged from the five dimensions normalised to
# [0,1]); the arbiter does not emit it directly to avoid the model
# double-counting its own per-dimension scoring.
#
# Capsules are presented in shuffled order each round — position-bias
# mitigation from the MT-Bench / "When Identity Skews Debate" literature.
# The shuffle happens at the formatter call site; the prompt text just
# reminds the arbiter not to infer importance from order.
_ARBITER_PROMPT = """\
You are evaluating whether a multi-model panel has reached sufficient \
agreement to ship a final answer. You are scoring **sufficiency for action**, \
not absolute truth.

Original question:
{question}

Panel health: {usable_count} of {total_count} panellists returned usable \
responses ({health_breakdown}). Down-weight your scoring if a significant \
fraction of the panel failed — consensus from half a panel is weaker evidence \
than consensus from a full panel.

Round {round_num} panel capsules (presented in randomised order — do NOT \
infer importance or model identity from position):
{capsules}

How positions changed since the prior round:
{position_diff}

Score the panel on FIVE dimensions, each on the same 1-5 Likert scale:

  1 = critical gap   — panellists missed the question, contradict each other, or no usable signal
  2 = weak           — substantial disagreement or hand-wavy assertions; not ready to act on
  3 = mixed          — useful signal but real gaps; another round would likely help
  4 = strong         — good consensus, or principled disagreement well-explained; minor gaps
  5 = ship-ready     — panellists agree on the load-bearing point and the action is clear

The five dimensions:
- coverage      — did the panel address all aspects of the question?
- agreement     — how aligned are the panellists' recommendations? (productive disagreement is not the same as contradiction)
- depth         — is the reasoning substantiated rather than asserted?
- calibration   — do panellists' stated confidence levels match how well-backed their claims are?
- actionability — is the panel converging on a concrete recommendation a caller can act on?

For each dimension, also give a one-sentence note explaining the score — \
focused and localised (point at a specific panellist or claim), not generic.

Return EXACTLY this JSON object (no commentary, no markdown fences):

{{
  "dimensions": {{
    "coverage": <int 1-5>,
    "agreement": <int 1-5>,
    "depth": <int 1-5>,
    "calibration": <int 1-5>,
    "actionability": <int 1-5>
  }},
  "dimension_notes": {{
    "coverage": "<one sentence — why this score>",
    "agreement": "<one sentence>",
    "depth": "<one sentence>",
    "calibration": "<one sentence>",
    "actionability": "<one sentence>"
  }},
  "gaps": ["<one specific item the panel hasn't resolved>", "..."],
  "next_round_focus": "<one sentence telling the next round what to address>",
  "reasoning": "<1-3 sentences summarising the verdict>"
}}

Notes:
- gaps: 1-4 specific items. dimensions tell us *how* the panel is short; gaps enumerate *what* exactly.
- The engine derives the overall sufficiency score from your dimensions (normalised average). Do NOT output an overall score yourself.
"""

_REFINEMENT_PROMPT_TEMPLATE = """\
This is round {round_num} of a multi-round consultation. \
You are refining your answer based on the prior round.

Original question:
{question}

Positions from the prior round:
{positions}

Per-dimension critique from the arbiter (focus on the weakest axes):
{dim_critique}

Gaps the arbiter flagged that the next round should address:
{gaps}

Specifically focus on: {focus}

Now give your refined answer to the original question, addressing the \
weakest dimensions and the gaps. Be concrete; don't simply restate the \
prior position."""


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
    return f"  recommendation: {cap.recommendation}\n  key_points: {bullets}"


def _shuffled(manifest: list[ManifestEntry]) -> list[ManifestEntry]:
    """Return a shuffled copy of `manifest`.

    Position-bias mitigation: judges (the arbiter, and panellists reading
    prior-round positions for refinement) systematically over-weight items
    they see first. MT-Bench measured 75% first-position bias in Claude-v1
    judging; the "When Identity Skews Debate" paper showed similar effects
    in multi-agent settings. Shuffling per-call (not per-run) is enough —
    we just need the order the judge sees to be uncorrelated with anything
    the judge could use as a heuristic shortcut.
    """
    out = list(manifest)
    random.shuffle(out)
    return out


def _format_capsules(manifest: list[ManifestEntry]) -> str:
    lines = []
    for m in _shuffled(manifest):
        if m.status not in (Status.OK, Status.TRUNCATED):
            lines.append(f"- {m.slug} [status={m.status.value}, no usable response]")
            continue
        c = m.capsule
        if not c:
            lines.append(f"- {m.slug}: (no capsule extracted)")
            continue
        conf = f" conf={c.confidence:.2f}" if c.confidence is not None else ""
        lines.append(f"- {m.slug}{conf}: {_capsule_summary(c)}\n{_capsule_detail(c)}")
    return "\n".join(lines)


def _format_positions(manifest: list[ManifestEntry]) -> str:
    lines = []
    for m in _shuffled(manifest):
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
            _base_slug(m.slug): m for m in manifest if m.capsule and m.status in (Status.OK, Status.TRUNCATED)
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
                lines.append(f"- {base}:\n    before: {old_summary}\n    after:  {cur_summary}")
        else:
            lines.append(f"- {base} (new this round): {cur_summary}")
    # Surface panellists that dropped out this round — their disappearance
    # is signal the arbiter should weigh ("3 of 5 now agree, but 2 of 5
    # are missing this round so consensus is weaker than it looks").
    for base, old in prior_by_base.items():
        if base not in current_by_base and old.capsule:
            lines.append(f"- {base} (dropped this round, last position): {_capsule_summary(old.capsule)}")
    return "\n".join(lines) or "(no comparable positions)"


def _format_dim_critique(verdict: ArbiterVerdict) -> str:
    """Render the three weakest dimensions with their localised notes.

    The next-round prompt focuses panellists on the axes the arbiter scored
    lowest — sharper guidance than the generic `gaps` list because each
    note points at a specific panellist or claim. Falls back to a
    placeholder for legacy verdicts that have no `dimensions`.
    """
    if not verdict.dimensions:
        return "(no per-dimension critique — legacy arbiter prompt)"
    weakest = sorted(verdict.dimensions.items(), key=lambda kv: kv[1])
    lines: list[str] = []
    for dim, dim_score in weakest[:3]:
        note = verdict.dimension_notes.get(dim, "").strip()
        if note:
            lines.append(f"- {dim} (scored {dim_score:.2f}): {note}")
        else:
            lines.append(f"- {dim} (scored {dim_score:.2f})")
    return "\n".join(lines) or "(no critique recorded)"


def _build_refinement_prompt(
    question: str, round_num: int, prior_manifest: list[ManifestEntry], verdict: ArbiterVerdict
) -> str:
    return _REFINEMENT_PROMPT_TEMPLATE.format(
        round_num=round_num,
        question=question,
        positions=_format_positions(prior_manifest),
        dim_critique=_format_dim_critique(verdict),
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

    # 1) Call — exception here means we never got text back.
    # `temperature` is only set on providers that accept it. claude-opus-4-7
    # and Gemini both reject the kwarg; falling back to the provider default
    # is fine for an arbiter call that's just producing a JSON verdict.
    call_kwargs: dict[str, Any] = {
        "model": litellm_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 2000,
    }
    provider_caps.apply_temperature(call_kwargs, litellm_id, 0.0)
    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(**call_kwargs),
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
        logger.warning("arbiter returned non-JSON; sample=%r", text[:120].replace("\n", " "))
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

    # 4) Score derivation. v2 prompt: arbiter emits per-dimension 1-5
    # scores; engine clamps to [1,5], normalises each to [0,1], averages.
    # v1 fallback: legacy arbiters or off-rubric responses may emit a bare
    # `score` field — accept it for backwards compatibility, but mark the
    # verdict so the viewer can show "legacy" vs "per-dim". A response that
    # has neither dimensions nor a parseable score is a parse failure.
    dimensions_normalised: dict[str, float] = {}
    dimensions_raw = data.get("dimensions")
    if isinstance(dimensions_raw, dict):
        for k, v in dimensions_raw.items():
            try:
                n = float(v)
            except (TypeError, ValueError):
                # Skip non-numeric entries — a single bad value shouldn't
                # nuke the whole verdict. If ALL are non-numeric the empty
                # `dimensions_normalised` triggers the v1 fallback below.
                continue
            n = max(1.0, min(5.0, n))
            dimensions_normalised[str(k)] = (n - 1.0) / 4.0

    dimension_notes_raw = data.get("dimension_notes")
    dimension_notes: dict[str, str] = {}
    if isinstance(dimension_notes_raw, dict):
        dimension_notes = {str(k): str(v) for k, v in dimension_notes_raw.items() if v}

    if dimensions_normalised:
        score = sum(dimensions_normalised.values()) / len(dimensions_normalised)
    else:
        # v1 fallback: legacy `score` 0..1
        try:
            score = float(data.get("score", 0.0))
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"out of range: {score}")
        except (TypeError, ValueError) as e:
            logger.warning(
                "arbiter emitted neither dimensions nor a parseable score: %s",
                e,
            )
            return ArbiterVerdict(
                round=round_num,
                score=0.0,
                gaps=[],
                reasoning="arbiter returned no parseable dimensions or score",
                cost_usd=cost,
                cost_known=cost_known,
                parsed_ok=False,
                error=f"no_dimensions_or_score: {e!s:.100}",
            )

    return ArbiterVerdict(
        round=round_num,
        score=score,
        dimensions=dimensions_normalised,
        dimension_notes=dimension_notes,
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
        # Sanitise model-derived bases — a raw LiteLLM ID like
        # `openrouter/meta-llama/llama-3.1-8b:free` would otherwise carry
        # the `:` straight into the slug and trip ModelSpec's safe-id
        # field validator. User-supplied slugs are already constrained.
        base = s.slug or runner.sanitise_derived_slug(s.model.split("/")[-1].lower())
        out.append(ModelSpec(model=s.model, stance=s.stance, slug=f"{base}-{i}.r{round_num}"))
    return out


def _apply_continuation(prompt: str, continuation_id: str | None) -> tuple[str, list[dict[str, Any]] | None]:
    """Resolve the continuation and split it from the follow-up prompt.

    Returns `(prompt_for_storage, prior_turns)`:
    - `prompt_for_storage` is what gets persisted to `prompt.txt` and the
      context bundle. It includes the prior question + synthesis as
      markdown-headed sections so disk artifacts remain self-describing
      and `consult-view` can render the full conversation in one place.
    - `prior_turns`, when not `None`, is a `[user(prior_question),
      assistant(prior_synthesis)]` pair fed to `runner.fanout` as message
      history. The model sees the prior consultation as a proper exchange,
      not a single user blob — better role boundaries, and the prefix
      becomes a stable cache target across follow-ups from the same run.

    Returns `(prompt, None)` when continuation isn't set, so non-continued
    refine calls are unchanged.

    Raises `ValueError` on a missing run dir or missing `synthesis.md` — a
    typo must not silently drop the prior context. Empty string is treated
    the same as None.
    """
    if not continuation_id:
        return prompt, None
    # Let `FileNotFoundError` from `artifacts.load_run` propagate so the MCP
    # dispatcher in server.py maps it to the dedicated RUN_NOT_FOUND envelope
    # — the previous wrap-as-ValueError funnelled the same failure through
    # INVALID_INPUT, which agents can't distinguish from a malformed prompt.
    prior_paths = artifacts.load_run(continuation_id)
    synth_path = prior_paths.root / "synthesis.md"
    if not synth_path.exists():
        raise ValueError(
            f"continuation_id {continuation_id} has no synthesis.md "
            "(was the prior run partial, dry-run, or pre-synth?)"
        )
    # Reject the synth sentinels written by `synth.synthesise` when the
    # prior run had no usable panel, a failed flagship call, or returned
    # empty content. Feeding "# Synthesis unavailable" to a new panel as
    # "prior consultation" produces hallucinated follow-ups that pretend
    # the sentinel was a real conclusion.
    prior_synth_text = synth_path.read_text()
    _SYNTH_SENTINELS = (
        "# Synthesis skipped",
        "# Synthesis unavailable",
        "# Synthesis empty",
    )
    # `lstrip()` first — a future synth model might emit a BOM, leading
    # newline, or shell-prompt-style preamble before the sentinel heading.
    # Without it, a single stray whitespace would slip the sentinel past
    # the check and feed "the prior run failed" verbatim to the new panel.
    _sentinel_head = prior_synth_text.lstrip()
    for sentinel in _SYNTH_SENTINELS:
        if _sentinel_head.startswith(sentinel):
            raise ValueError(
                f"continuation_id {continuation_id} has a sentinel synthesis "
                f"({sentinel!r}) — the prior run did not produce real synthesis. "
                "Re-run the prior consultation before continuing."
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

    combined_for_storage = (
        "## Prior consultation — original question\n\n"
        f"{prior_question}\n\n"
        "## Prior consultation — synthesis\n\n"
        f"{prior_synth_text}\n\n---\n\n"
        "## Follow-up question\n\n"
        f"{prompt}"
    )
    prior_turns: list[dict[str, Any]] = [
        {"role": "user", "content": prior_question},
        {"role": "assistant", "content": prior_synth_text},
    ]
    return combined_for_storage, prior_turns


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
    strategy: str = "default",
    on_progress: runner.ProgressCallback | None = None,
) -> RefineResult:
    if max_rounds < 1 or max_rounds > 5:
        raise ValueError("max_rounds must be between 1 and 5")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.0, 1.0]")
    # Resolve the strategy now so a typo on the caller's `strategy="…"`
    # surfaces as ValueError before the panel spends any money.
    strategy_inst = strategies.strategy_for(strategy)

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

    # `prompt_for_storage` includes the prior conversation as markdown (used
    # for `prompt.txt`, the context bundle, and the arbiter's "original
    # question"); `prior_turns` is the proper user/assistant exchange fed
    # to the panellists so they see role boundaries, not stitched text.
    # `followup_only` is what the panellist's *user* turn carries — just
    # the new question, since the prior is already in prior_turns.
    followup_only = prompt
    prompt, prior_turns = _apply_continuation(prompt, continuation_id)
    # Resolve `model:N` sugar here too so `estimate_cost` (called before
    # `fanout` in each round) sees the real expanded panel.
    specs = runner.expand_specs(specs)
    arbiter_alias = arbiter or registry.default_synthesiser()
    synth_alias = synthesiser or arbiter_alias
    # Fail fast on a typo'd arbiter/synthesiser alias BEFORE we spend on
    # fanout + capsule extraction. Without this the run burns through the
    # parallel panel and only crashes when the round-1 arbiter call
    # reaches `registry.resolve_model`. KeyError surfaces as
    # ErrorCode.UNKNOWN_MODEL at the MCP boundary.
    registry.resolve_model(arbiter_alias)
    if synth_alias != arbiter_alias:
        registry.resolve_model(synth_alias)

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
    # Mutable cell so `progress_mod.make_phase_cb` can update it; we read
    # `progress_done[0]` for the direct emits between phases (arbiter, synth).
    progress_done = [0]

    async def emit(event: progress_mod.ProgressEvent) -> None:
        if on_progress is not None:
            try:
                await on_progress(event)
            except Exception as e:  # noqa: BLE001
                logger.debug("refine on_progress failed: %s", e)

    # Pre-resolve the arbiter spec so its cost estimate can roll into the
    # per-round budget check below. The arbiter call is sequential after
    # fanout — flagship arbiters on 3-5 rounds were the source of the
    # silent cap overshoot.
    arbiter_spec = ModelSpec(model=arbiter_alias)

    # Panellist's user turn = just the follow-up question when continuation
    # is active (the prior is already in `prior_turns`). With no
    # continuation, `followup_only` equals `prompt` so this is a no-op.
    round_prompt = followup_only if prior_turns else prompt
    prior_manifest: list[ManifestEntry] | None = None

    # Per-panellist conversation history across rounds. Keyed by *base*
    # slug (the part before `.r<n>`) so a panellist's slug-suffixed
    # round-N spec maps to its base's accumulated history. Round 1's
    # answer becomes round 2's assistant turn; round 2's refinement
    # prompt + answer become rounds 3+'s context. The first-turn prefix
    # stays byte-identical across rounds, which is what Anthropic's
    # prompt cache keys on — round-2 and round-3 calls reuse the cached
    # round-1 prefix for a ~50% input-token discount + faster TTFT.
    #
    # Seeded with the continuation `prior_turns` (if any) so a refine
    # follow-up's panellists still see the prior consultation as their
    # first turns. Reads of this dict in round-1 fall through to the
    # global `prior_turns` (since the dict is empty); round-2+ uses the
    # accumulated per-slug history exclusively.
    panel_conversations: dict[str, list[dict[str, Any]]] = {}

    def _base_for_slug(slug: str) -> str:
        return _base_slug(slug)

    if prior_turns:
        for s in _suffix_specs(specs, 1):
            panel_conversations[_base_for_slug(s.slug)] = list(prior_turns)

    for round_num in range(1, max_rounds + 1):
        # Strategy decides which panellists run this round. The default
        # strategy passes `specs` through unchanged; `elimination` drops
        # the most-divergent panellist from round 2 onwards.
        round_base_specs = strategy_inst.before_round(
            round_num=round_num,
            base_specs=specs,
            prior_manifest=prior_manifest,
        )
        if not round_base_specs:
            partial_reason = (
                f"strategy {strategy!r} returned an empty panel for round "
                f"{round_num}; aborting to avoid a zero-panel fanout"
            )
            break
        # Estimate next-round cost (fanout + arbiter); refuse if it'd blow
        # the cap. The arbiter's prompt isn't known until after fanout, but
        # token_counter on the round prompt is a reasonable proxy — the
        # arbiter's input is roughly "round prompt + capsule summaries"
        # which scales with the prompt size for code-review / long-context
        # work where the cap actually matters.
        # Mirror runner.fanout's view: when a continuation is active, the
        # prior_turns text is part of every panellist call's input. Omitting
        # it here lets refine wave a round through that fanout would then
        # reject as cap-exceeded — and fanout's early-return path would
        # clobber the prior round's manifest because we share `paths`.
        fanout_cost_input = round_prompt
        if prior_turns:
            fanout_cost_input = runner.concat_turn_text(prior_turns) + "\n" + round_prompt
        fanout_est, fanout_known = await runner.aestimate_cost(
            round_base_specs,
            fanout_cost_input,
            capsule_kind=resolved_kind,
        )
        # Arbiter call has its own hardcoded max_completion_tokens=2000 (see _ask_arbiter);
        # "decision" matches that budget so the estimate is honest.
        arbiter_est, arbiter_known = await runner.aestimate_cost(
            [arbiter_spec],
            round_prompt,
            capsule_kind="decision",
        )
        estimate = fanout_est + arbiter_est
        est_known = fanout_known and arbiter_known
        if cumulative_cost + estimate > cap:
            partial_reason = (
                f"would exceed cap: spent ${cumulative_cost:.2f}, next round estimate "
                f"${estimate:.2f} (fanout ${fanout_est:.2f} + arbiter ${arbiter_est:.2f}), "
                f"cap ${cap:.2f}"
            )
            break
        # Refuse further rounds only when partial pricing AND spend is
        # already most of the way to the cap. Earlier behaviour was an
        # asymmetric "round 1 with unknown pricing proceeds, round 2
        # refuses" which surprised callers with mixed-provider panels
        # (FRICTION pass #14 saw this on sequence). Conservative threshold
        # of 80% leaves headroom for one more bounded round.
        cap_warning_floor = cap * 0.8
        if not est_known and cumulative_cost > cap_warning_floor:
            partial_reason = (
                f"refusing further rounds: per-model pricing unknown for at least one "
                f"panellist and spend is past 80% of cap "
                f"(${cumulative_cost:.2f} spent / ${cap:.2f} cap)"
            )
            break

        round_base = (round_num - 1) * (panel_n * 2 + 1)
        round_specs = _suffix_specs(round_base_specs, round_num)
        # Per-panellist conversation history for round 2+. Map each
        # round-N slug to its base's accumulated turns. Round 1 falls
        # back to the global `prior_turns` (continuation context) since
        # `panel_conversations` only has continuation seeds at this point.
        round_prior_by_slug: dict[str, list[dict[str, Any]]] | None
        if round_num == 1:
            round_prior_by_slug = None
        else:
            round_prior_by_slug = {}
            for rspec in round_specs:
                base = _base_for_slug(rspec.slug)
                if base in panel_conversations:
                    round_prior_by_slug[rspec.slug] = panel_conversations[base]
        # Pass the remaining budget so fanout's internal cap matches the
        # refine cap — without this the nested call falls back to
        # `registry.default_max_run_usd()` and a caller's higher refine
        # cap (e.g. $20) is silently downgraded to the default ($5).
        handle = await runner.fanout(
            round_prompt,
            round_specs,
            blinded=blinded,
            max_run_usd=cap - cumulative_cost,
            existing_paths=paths,
            on_progress=progress_mod.make_phase_cb(
                emit if on_progress else None,
                round_base,
                progress_total,
                progress_done,
            ),
            # Round 1 uses the global continuation; round 2+ uses the
            # per-slug accumulated history.
            prior_turns=prior_turns if round_num == 1 else None,
            prior_turns_by_slug=round_prior_by_slug,
        )
        # Short-circuit when fanout itself is partial. Running the arbiter
        # on a zero-usable-panel manifest just burns the arbiter's price for
        # a verdict that can only say "no signal" — and on a cap-exceeded
        # fanout, it would push the spend further over. Surface the fanout
        # reason verbatim so the caller knows it wasn't refine that aborted.
        #
        # Preserve `final_manifest` from the last good round on partial
        # break: a round-2 cap-exceeded fanout returns `manifest=[]`, and
        # the previous unconditional `final_manifest = handle.manifest`
        # would clobber round 1's good consensus with the empty failure
        # manifest before the break fired.
        if handle.partial:
            cumulative_cost += handle.cost_usd
            if not handle.cost_known:
                cost_all_known = False
            partial_reason = f"round {round_num} fanout partial: {handle.partial_reason}"
            break
        final_manifest = handle.manifest

        # Append this round's (user turn, assistant turn) to each
        # panellist's per-slug conversation history. The next round's
        # fanout will pass these as `prior_turns_by_slug` so the
        # panellist sees its prior answer as a proper assistant turn
        # rather than receiving the question fresh. The round-1 user
        # turn (`round_prompt`) is byte-stable across rounds 2+'s
        # prefix — that's what Anthropic's prompt cache keys on.
        for entry in handle.manifest:
            if entry.status not in (Status.OK, Status.TRUNCATED):
                continue
            try:
                body_text = (paths.root / "responses" / f"{entry.slug}.txt").read_text()
            except OSError:
                # Body file missing — skip this panellist's history
                # update. The next round will fall back to a fresh
                # turn for this slug (no per-slug entry in the dict).
                logger.debug("could not read body for %s; skipping history", entry.slug)
                continue
            base = _base_for_slug(entry.slug)
            history = panel_conversations.setdefault(base, [])
            history.append({"role": "user", "content": round_prompt})
            history.append({"role": "assistant", "content": body_text})

        handle = await capsule.annotate(
            handle,
            on_progress=progress_mod.make_phase_cb(
                emit if on_progress else None,
                round_base + panel_n,
                progress_total,
                progress_done,
            ),
            kind=resolved_kind,
        )
        # Accumulate AFTER capsule.annotate — it mutates handle.cost_usd in
        # place to add extractor spend. Pre-capsule accumulation silently
        # dropped the extractor cost (one bug-fix landed in iter1 of the
        # refine loop). cost_known likewise needs the post-capsule view: an
        # extractor pricing miss flips handle.cost_known False.
        cumulative_cost += handle.cost_usd
        if not handle.cost_known:
            cost_all_known = False

        # Arbiter sees the follow-up question alone when a continuation is
        # active. Passing the full `prompt` (which `_apply_continuation`
        # rewrote to include the entire prior synthesis blob) confuses the
        # sufficiency-scoring — the arbiter ends up grading the panel against
        # a multi-page conversation history rather than the actual question.
        arbiter_question = followup_only if prior_turns else prompt
        verdict = await _ask_arbiter(
            arbiter_question, round_num, handle.manifest, arbiter_alias, prior_manifest
        )
        progress_done[0] = round_base + panel_n * 2 + 1
        await emit(
            progress_mod.ArbiterScored(
                done=progress_done[0],
                total=progress_total,
                round=round_num,
                score=verdict.score,
            )
        )
        verdicts.append(verdict)
        # `is not None` rather than truthy: a successful arbiter call that
        # returned a $0.00 cost is semantically different from no-cost-known.
        # Functionally equivalent for zero but reads correctly when the
        # invariant is "None ⇒ unknown".
        if verdict.cost_usd is not None:
            cumulative_cost += verdict.cost_usd
        # An arbiter pricing miss must propagate to the top-level cost_known.
        # Previously only the truthy-cost branch fed into the totals — an
        # unmapped-price arbiter (verdict.cost_usd=None, cost_known=False)
        # left cost_all_known wrongly True, breaching the same invariant
        # that bit the consult success path.
        if not verdict.cost_known:
            cost_all_known = False
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
            # Refinement prompt's "Original question" = follow-up only when
            # continuation is active. The prior consultation context lives
            # in `prior_turns`, not in this user turn — duplicating it
            # would dilute the model's attention on the actual question.
            base_q = followup_only if prior_turns else prompt
            round_prompt = _build_refinement_prompt(base_q, round_num + 1, handle.manifest, verdict)
        # Snapshot for the next round's arbiter position-diff. Updated
        # after the verdict so an aborted round (parse failure above)
        # leaves prior_manifest pointing at the last fully-scored round.
        prior_manifest = handle.manifest

    # Synthesise from the final round
    if final_manifest:
        progress_done[0] = progress_total - 1
        await emit(progress_mod.SynthStarted(done=progress_done[0], total=progress_total))
        # `anonymised=blinded` so a refine-with-blinded-True doesn't leak the
        # raw original prompt into synth_input.txt — without it the synth
        # call defaults to `anonymised=False`, which makes the context bundle
        # return the un-scrubbed prompt regardless of the blinded flag.
        synth_result = await synth.synthesise(
            paths.run_id,
            by_model=synth_alias,
            anonymised=blinded,
            rubric=rubric,
        )
        text = synth_result.text
        cumulative_cost += synth_result.cost_usd
        if not synth_result.cost_known:
            cost_all_known = False
        progress_done[0] = progress_total
        await emit(progress_mod.SynthCompleted(done=progress_done[0], total=progress_total))
        # Persist synthesiser + cumulative cost. Refine writes the manifest
        # once per round from `runner.fanout`, which only knows that round's
        # fanout spend; capsule + arbiter + synth costs were rolled into
        # `cumulative_cost` in memory but never reached disk. Without this
        # `consult-ledger` reads the last fanout's cost and silently
        # under-reports the run total.
        await artifacts.aaugment_manifest(
            paths,
            synthesiser=synth_alias,
            cost_usd=cumulative_cost,
            cost_known=cost_all_known,
        )
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
