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

_ARBITER_PROMPT = """\
You are evaluating whether a multi-model panel has reached sufficient \
agreement to ship a final answer. You are scoring **sufficiency for action**, \
not absolute truth.

Original question:
{question}

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

    prompt = _ARBITER_PROMPT.format(
        question=question, round_num=round_num, capsules=_format_capsules(manifest)
    )
    cost: float | None = None
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
        text = resp.choices[0].message.content or ""
        data = _extract_json(text) or {}
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = None
        return ArbiterVerdict(
            round=round_num,
            score=float(data.get("score", 0.0)),
            gaps=list(data.get("gaps") or []),
            next_round_focus=str(data.get("next_round_focus") or ""),
            reasoning=str(data.get("reasoning") or ""),
            cost_usd=cost,
        )
    except Exception as e:
        # Arbiter failure: assume not converged so caller decides whether to retry
        return ArbiterVerdict(
            round=round_num,
            score=0.0,
            gaps=[f"arbiter error: {type(e).__name__}: {e!s:.150}"],
            reasoning="arbiter call failed; treating as non-convergent",
            cost_usd=None,
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
    converged = False
    partial_reason: str | None = None

    round_prompt = prompt
    for round_num in range(1, max_rounds + 1):
        # Estimate next-round cost; refuse if it'd blow the cap
        estimate = runner.estimate_cost(specs, round_prompt)
        if cumulative_cost + estimate > cap:
            partial_reason = (
                f"would exceed cap: spent ${cumulative_cost:.2f}, next round estimate "
                f"${estimate:.2f}, cap ${cap:.2f}"
            )
            break

        round_specs = _suffix_specs(specs, round_num)
        handle = await runner.fanout(
            round_prompt, round_specs, blinded=blinded, existing_paths=paths
        )
        handle = await capsule.annotate(handle)
        final_manifest = handle.manifest
        cumulative_cost += handle.cost_usd

        verdict = await _ask_arbiter(prompt, round_num, handle.manifest, arbiter_alias)
        verdicts.append(verdict)
        if verdict.cost_usd:
            cumulative_cost += verdict.cost_usd
        paths.arbiter_for(round_num).write_text(verdict.model_dump_json(indent=2))

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
        wall_ms=wall_ms,
        partial=partial_reason is not None,
        partial_reason=partial_reason,
    )
