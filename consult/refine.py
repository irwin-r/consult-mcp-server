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
from typing import Any

import litellm

from . import artifacts, capsule, registry, runner, synth
from .types import (
    ArbiterVerdict,
    ManifestEntry,
    ModelSpec,
    RefineResult,
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

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


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
        bullets = "; ".join(c.key_points[:3]) if c.key_points else "(no key points)"
        lines.append(
            f"- {m.slug}{conf}: {c.position}\n  recommendation: {c.recommendation}\n  key_points: {bullets}"
        )
    return "\n".join(lines)


def _format_positions(manifest: list[ManifestEntry]) -> str:
    lines = []
    for m in manifest:
        if not m.capsule or m.status not in (Status.OK, Status.TRUNCATED):
            continue
        lines.append(f"- {m.slug}: {m.capsule.position}")
    return "\n".join(lines) or "(none extracted)"


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
    question: str, round_num: int, manifest: list[ManifestEntry], arbiter_alias: str
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
    data = _extract_json(text)
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
) -> RefineResult:
    if max_rounds < 1 or max_rounds > 3:
        raise ValueError("max_rounds must be between 1 and 3 (hard cap from v1.1 spec)")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.0, 1.0]")

    arbiter_alias = arbiter or registry.default_synthesiser()
    synth_alias = synthesiser or arbiter_alias

    paths = artifacts.create_run()
    paths.prompt_txt.write_text(prompt)
    paths.registry_snapshot.write_text(json.dumps(registry.models_config(), indent=2))
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()

    start = time.time()
    verdicts: list[ArbiterVerdict] = []
    final_manifest: list[ManifestEntry] = []
    cumulative_cost = 0.0
    cost_all_known = True
    converged = False
    partial_reason: str | None = None

    round_prompt = prompt
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

        round_specs = _suffix_specs(specs, round_num)
        handle = await runner.fanout(
            round_prompt, round_specs, blinded=blinded, existing_paths=paths
        )
        handle = await capsule.annotate(handle)
        final_manifest = handle.manifest
        cumulative_cost += handle.cost_usd
        if not handle.cost_known:
            cost_all_known = False

        verdict = await _ask_arbiter(prompt, round_num, handle.manifest, arbiter_alias)
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

    # Synthesise from the final round
    if final_manifest:
        text = await synth.synthesise(paths.run_id, by_model=synth_alias)
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
    )
