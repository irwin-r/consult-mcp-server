"""Synthesise a finished run with a flagship model.

Default synthesiser is `gemini-pro`. The synthesiser is excluded from
panellist composition where possible (`consult` hero tool handles this).
Anonymised mode strips real model identities from the synthesis input.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm

from . import artifacts, context, registry
from .runner import _build_messages
from .types import Status

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthResult:
    """Synthesis output plus its own pricing.

    `synthesise` was returning only `str`; callers (`consult`, `refine`,
    `sequence`) silently dropped the synthesiser's spend from `cost_usd`,
    breaching the `max_run_usd` accounting. Returning the cost alongside
    the text keeps the sentinel-write path (skipped/unavailable/empty)
    cost-neutral while letting successful calls roll into the run total.
    """

    text: str
    cost_usd: float = 0.0
    cost_known: bool = True


def _resolve_rubric(rubric: str | None) -> str:
    """Pick a rubric: named lookup or literal passthrough.

    The shipped package contains `consult/config/rubrics/consensus.md` as
    the single source of truth for the default rubric. If that file is
    missing we fail loudly — a previous in-process duplicate constant was
    a drift trap (kept in sync by hand) and silently shipped stale text
    when the on-disk file changed.
    """
    if rubric is None:
        rubric = "consensus"
    resolved = registry.resolve_rubric(rubric)
    # `resolve_rubric` returns the input string verbatim when no file matched.
    # For the default "consensus" rubric, that means the package install is
    # broken — refuse to ship the literal word "consensus" to the synth model.
    if rubric == "consensus" and resolved == "consensus":
        # RuntimeError (not FileNotFoundError) — the latter is reserved by
        # `server.handle_call_tool` for `run_not_found`, so a missing rubric
        # would otherwise surface as a misleading "the run_id is bad". A
        # missing default rubric is an install-time defect; INTERNAL_ERROR
        # is the right category for that.
        raise RuntimeError(
            "consult/config/rubrics/consensus.md is missing — the package "
            "install is broken. Reinstall or restore the rubrics directory."
        )
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
) -> SynthResult:
    paths = artifacts.load_run(run_id)
    manifest_payload = json.loads(paths.manifest_json.read_text())
    manifest = manifest_payload["manifest"]
    bodies = {
        m["slug"]: paths.response_text(m["slug"]).read_text()
        for m in manifest
        if m["status"] in (Status.OK.value, Status.TRUNCATED.value)
    }
    # Zero-usable-body guard. Without this, a dry-run or fully-failed run
    # would proceed to call the synthesiser with an empty RESPONSES block —
    # a billable call whose only possible output is hallucinated content.
    # Persist a clear sentinel so consult-view and downstream callers can
    # branch on the same disk artifact path as the success case.
    if not bodies:
        text = (
            "# Synthesis skipped\n\n"
            f"Run `{run_id}` has zero usable panellist responses "
            f"({len(manifest)} entries; none OK or TRUNCATED). "
            "No synthesiser call was made — there was nothing to synthesise. "
            "Inspect per-panellist artifacts to see why the panel failed."
        )
        (paths.root / "synthesis.md").write_text(text)
        return SynthResult(text=text)
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
        # No completion was returned, so there's nothing reliable to price.
        # cost_known=False mirrors the partial-cost convention elsewhere.
        return SynthResult(text=text, cost_usd=0.0, cost_known=False)

    # Look up cost even on the empty-content path: provider billed for the
    # call regardless of whether output was usable. Independent try block so
    # a price-table miss never discards the synthesis text.
    try:
        cost = litellm.completion_cost(completion_response=resp)
        cost_known = cost is not None
        cost_value = float(cost) if cost is not None else 0.0
    except Exception as ce:  # noqa: BLE001
        logger.warning("synth cost lookup failed for %s: %s", litellm_id, ce)
        cost_value = 0.0
        cost_known = False

    # Defensive extraction: most providers follow OpenAI's `choices[0].message`
    # shape, but a non-conformant response (or a future SDK regression) would
    # otherwise raise AttributeError/IndexError straight out of synthesise,
    # bypassing the unavailable-sentinel path that callers expect.
    try:
        content = resp.choices[0].message.content
        finish = getattr(resp.choices[0], "finish_reason", None)
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        logger.warning("synth response shape unexpected for %s: %s", litellm_id, e)
        text = (
            f"# Synthesis unavailable\n\n"
            f"The synthesiser (`{litellm_id}`) returned an unexpected response "
            f"shape: `{type(e).__name__}: {e!s:.200}`. "
            f"Retry `synthesise(run_id, by_model=...)` with a different model."
        )
        (paths.root / "synthesis.md").write_text(text)
        return SynthResult(text=text, cost_usd=cost_value, cost_known=cost_known)
    if not content or not content.strip():
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
        return SynthResult(text=text, cost_usd=cost_value, cost_known=cost_known)

    text = content.strip()
    (paths.root / "synthesis.md").write_text(text)
    return SynthResult(text=text, cost_usd=cost_value, cost_known=cost_known)
