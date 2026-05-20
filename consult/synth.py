"""Synthesise a finished run with a flagship model.

Default synthesiser is `gemini-pro`. The synthesiser is excluded from
panellist composition where possible (`consult` hero tool handles this).
Anonymised mode strips real model identities from the synthesis input.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import litellm

from . import artifacts, context, registry
from .runner import _build_messages
from .types import Status

logger = logging.getLogger(__name__)

# Defensive fallback if the package's consensus.md is missing on disk
# (broken install, etc). Kept in sync with consult/config/rubrics/consensus.md.
_FALLBACK_CONSENSUS_RUBRIC = """\
You have {n} expert responses below. Synthesise them under this rubric:

# Consensus
Claims appearing across multiple responses. Cite source slugs in [brackets].

# Dissent
Specific places models disagree, with the reason (and persona where assigned). Quote briefly.

# Minority Report
The single most coherent disagreeing view, even if held by one model. Steel-man it.

# Weighted Recommendation
Your judgement, weighing the responses by confidence and persona relevance. State it directly.

# Risks if the consensus is wrong
What breaks if the majority view turns out to be incorrect.

# Next Steps
3–5 concrete actions in order.

Be specific. Quote when it helps. Down-weight responses tagged TRUNCATED or with confidence < 0.4.
"""


def _resolve_rubric(rubric: str | None) -> str:
    """Pick a rubric: named lookup, literal passthrough, or the fallback."""
    if rubric is None:
        rubric = "consensus"
    resolved = registry.resolve_rubric(rubric)
    # `resolve_rubric` returns the input string verbatim when no file matched.
    # For the "consensus" default, fall back to the in-process constant so a
    # missing rubrics dir doesn't ship a literal "consensus" string to the
    # synth model. Custom literals (multi-line strings the caller composed
    # themselves) pass through unchanged.
    if rubric == "consensus" and resolved == "consensus":
        return _FALLBACK_CONSENSUS_RUBRIC
    return resolved


def _build_input(
    manifest: list[dict[str, Any]],
    bodies: dict[str, str],
    *,
    rubric: str,
    anonymised: bool,
    original_prompt: str | None = None,
) -> str:
    usable = [m for m in manifest if m["status"] in (Status.OK.value, Status.TRUNCATED.value)]
    # `str.replace` (not `str.format`) so a user-supplied rubric in
    # ~/.consult/rubrics/ that contains literal `{` / `}` characters (a JSON
    # example, a template marker for another tool) doesn't crash with
    # `KeyError`. Only the `{n}` placeholder is meaningful here.
    header = rubric.replace("{n}", str(len(usable)))
    blocks = []
    for entry in usable:
        slug = entry["slug"]
        persona = entry.get("persona") or "neutral"
        conf = entry.get("confidence")
        status = entry["status"]
        if anonymised:
            label = f"[{slug} | persona={persona} | confidence={conf} | status={status}]"
        else:
            mid = entry.get("model_id") or "unknown"
            label = f"[{slug} ({mid}) | persona={persona} | confidence={conf} | status={status}]"
        body = bodies.get(slug, "")
        blocks.append(f"{label}\n{body.strip()}")
    parts: list[str] = []
    if original_prompt:
        parts.append("## Original question / source\n\n" + original_prompt + "\n\n---\n\n")
    parts.append(header)
    parts.append("\n\n---\nRESPONSES:\n\n" + "\n\n".join(blocks))
    return "".join(parts)


async def synthesise(
    run_id: str,
    *,
    by_model: str | None = None,
    rubric: str | None = None,
    anonymised: bool = False,
) -> str:
    paths = artifacts.load_run(run_id)
    manifest_payload = json.loads(paths.manifest_json.read_text())
    manifest = manifest_payload["manifest"]
    bodies = {
        m["slug"]: paths.response_text(m["slug"]).read_text()
        for m in manifest
        if m["status"] in (Status.OK.value, Status.TRUNCATED.value)
    }
    rub = _resolve_rubric(rubric)
    # Load the per-run context bundle so the synthesiser can fact-check
    # panellist claims against the source. Legacy runs (pre-Phase 1)
    # have no context.json; in that case the prompt is simply omitted
    # and the synthesis proceeds with bodies only (pre-Phase-1 behaviour).
    bundle = context.load_or_none(paths)
    original_prompt = (
        bundle.prompt_for_downstream(anonymised=anonymised) if bundle else None
    )
    # Apply the per-stage input budget — trims the longest bodies first,
    # then the original prompt as a last resort. Silently passing 1MB+
    # of source material to a model with a 200K context would either
    # fail at the API or drop the response, neither of which we want.
    original_prompt, bodies = context.trim_synth_input(
        original_prompt=original_prompt, bodies=bodies
    )
    synth_input = _build_input(
        manifest,
        bodies,
        rubric=rub,
        anonymised=anonymised,
        original_prompt=original_prompt,
    )
    # Persist for reproducibility
    (paths.root / "synth_input.txt").write_text(synth_input)

    synth_alias = by_model or registry.default_synthesiser()
    entry = registry.resolve_model(synth_alias)
    litellm_id = entry["litellm_id"]
    budget = max(entry.get("default_budget_tokens", 16000), 16000)
    timeout = entry.get("default_timeout_s", 300)
    provider = entry.get("provider", "")

    # Containment: a synthesiser failure must not tear down the parent request
    # (consult / refine). Persist a clear sentinel to synthesis.md so the run
    # artifact directory remains consistent.
    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=litellm_id,
                messages=_build_messages(synth_input, provider),
                max_tokens=budget,
            ),
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("synth call failed (%s): %s", litellm_id, e)
        text = (
            f"# Synthesis unavailable\n\n"
            f"The synthesiser (`{litellm_id}`) failed: `{type(e).__name__}: {e!s:.300}`.\n\n"
            f"The panel manifest is still available at the run's artifacts. "
            f"Retry `synthesise(run_id, by_model=...)` with a different model."
        )
        (paths.root / "synthesis.md").write_text(text)
        return text

    content = resp.choices[0].message.content
    if not content or not content.strip():
        finish = getattr(resp.choices[0], "finish_reason", None)
        logger.warning(
            "synth produced empty content (finish_reason=%s) for %s",
            finish,
            litellm_id,
        )
        text = (
            f"# Synthesis empty\n\n"
            f"The synthesiser (`{litellm_id}`) returned no content "
            f"(finish_reason=`{finish}`). "
            f"Retry with a larger budget or a different model."
        )
        (paths.root / "synthesis.md").write_text(text)
        return text

    text = content.strip()
    (paths.root / "synthesis.md").write_text(text)
    return text
