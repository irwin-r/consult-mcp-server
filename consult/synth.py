"""Synthesise a finished run with a flagship model.

Default synthesiser is `gemini-pro`. The synthesiser is excluded from
panellist composition where possible (`consult` hero tool handles this).

The synthesiser's *internal view* of the manifest is always blinded: each
panellist is presented as `Alpha`, `Beta`, `Gamma`, ... (or `P12`, `P13`,
... for panels bigger than the Greek alphabet) instead of by slug or
model_id. The synth's output text is then de-anonymised by word-boundary
regex before being persisted. This is the "When Identity Skews Debate"
(arxiv 2510.07517) finding: full-pipeline anonymisation drops conformity
bias ~96% on benchmark tasks, whereas *partial* anonymisation is worse
than none (the judge picks up identity from style cues). Panellist-level
blinding is unconditional — independent of the `anonymised` flag.

The `anonymised` flag on `synthesise()` controls a different axis: whether
the synth sees the brand-scrubbed version of the *original prompt*
(`bundle.prompt_scrubbed`) or the raw one. Callers set it via
`prompt_for_downstream(anonymised=...)`; engine entry points
(`orchestrate.consult`, `refine`, `sequence`) pass `anonymised=blinded`
so a blinded run also keeps brand names out of the synth's view of the
question. Standalone synth callers can override per-call.

Capsule order is shuffled per call too — position bias in LLM judging
is well-documented (MT-Bench measured 75% first-position preference on
Claude-v1; consult's synth is structurally a judge over the panel).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import litellm

from . import artifacts, context, registry
from .runner import build_messages
from .types import Status

logger = logging.getLogger(__name__)


class SynthStatus(StrEnum):
    """Outcome category for a `synth.synthesise` call.

    Callers (`sequence`, refine continuation) need to distinguish a real
    synthesised answer from one of the three sentinel paths — feeding
    "# Synthesis unavailable" verbatim into the next step's prompt as
    "prior synthesis" makes the panel hallucinate continuity that doesn't
    exist. With a status enum the caller can short-circuit cleanly.
    """

    OK = "OK"
    """The synthesiser returned real content."""

    SKIPPED_EMPTY = "SKIPPED_EMPTY"
    """Zero usable bodies — no synth call was made (no cost)."""

    FAILED = "FAILED"
    """The synthesiser call raised — sentinel text was written instead."""

    EMPTY = "EMPTY"
    """The synthesiser returned no content — sentinel text was written."""


@dataclass(frozen=True)
class SynthResult:
    """Synthesis output, pricing, and outcome status.

    `synthesise` was returning only `str`; callers silently dropped the
    synthesiser's spend from `cost_usd`, breaching the `max_run_usd`
    accounting. Returning the cost alongside the text keeps the
    sentinel-write path (skipped/unavailable/empty) cost-neutral while
    letting successful calls roll into the run total. `status` lets
    callers distinguish real output from a sentinel rather than
    string-matching the body.
    """

    text: str
    cost_usd: float = 0.0
    cost_known: bool = True
    status: SynthStatus = SynthStatus.OK


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


# Greek letters give human-friendly blind labels up to a 12-panellist
# panel. Beyond that fall back to numeric IDs `P13`, `P14`, ... — none of
# the shipped tiers go past 14 (`deep`), and a P-prefix word-boundary
# matches are uncollidable with English prose.
_BLIND_LABELS: tuple[str, ...] = (
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
    "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu",
)


def _blind_label(index: int) -> str:
    return _BLIND_LABELS[index] if index < len(_BLIND_LABELS) else f"P{index + 1}"


def _deblind(text: str, label_to_slug: dict[str, str]) -> str:
    """Replace blind labels with display slugs using word-boundary regex.

    Case-sensitive: the synth is told to use the exact "Alpha" / "Beta"
    tokens, and casual prose words like "alpha release" (lowercase) MUST
    NOT be rewritten. Labels matched longest-first so panel-specific
    numeric labels like `P10` don't get partial-matched by `P1`.

    Single combined regex pass: a chained per-label substitution could
    rewrite something twice if a slug happened to contain another label
    name (e.g. user-supplied slug "Alpha-1").
    """
    if not label_to_slug:
        return text
    sorted_labels = sorted(label_to_slug, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(l) for l in sorted_labels) + r")\b"
    )
    return pattern.sub(lambda m: label_to_slug[m.group(0)], text)


def _build_input(
    manifest: list[dict[str, Any]],
    bodies: dict[str, str],
    *,
    rubric: str,
    original_prompt: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the synth's input text and return the de-anonymisation map.

    Returns `(synth_input_text, label_to_slug)`. The synth sees blind
    labels (`Alpha`, `Beta`, ...) — never the real slug or model_id. The
    `label_to_slug` mapping is used post-call to de-anonymise the synth's
    output text. Empty when no usable panellists (caller handles).

    Panellist order is shuffled per call (position-bias mitigation).
    """
    usable = [m for m in manifest if m["status"] in (Status.OK.value, Status.TRUNCATED.value)]
    # Shuffle once per call. The synth's view of the panel is uncorrelated
    # with manifest order; position-as-importance shortcuts are removed.
    random.shuffle(usable)
    label_to_slug: dict[str, str] = {}
    # `str.replace` (not `str.format`) so a user-supplied rubric in
    # ~/.consult/rubrics/ that contains literal `{` / `}` characters (a JSON
    # example, a template marker for another tool) doesn't crash with
    # `KeyError`. Only the `{n}` placeholder is meaningful here.
    header = rubric.replace("{n}", str(len(usable)))
    blocks = []
    for i, entry in enumerate(usable):
        slug = entry["slug"]
        label = _blind_label(i)
        label_to_slug[label] = slug
        persona = entry.get("persona") or "neutral"
        conf = entry.get("confidence")
        status = entry["status"]
        label_str = (
            f"[{label} | persona={persona} | confidence={conf} | status={status}]"
        )
        body = bodies.get(slug, "")
        blocks.append(f"{label_str}\n{body.strip()}")
    parts: list[str] = []
    if original_prompt:
        parts.append("## Original question / source\n\n" + original_prompt + "\n\n---\n\n")
    parts.append(header)
    # Prepend a one-line note telling the synth what the blind labels
    # mean — without this it might invent free-form references to
    # panellists ("the first model said..."). Word-boundary case-sensitive
    # de-anonymisation post-call only catches the exact Alpha/Beta/...
    # tokens, so it's worth nudging the synth to use them in prose.
    parts.append(
        "\n\n---\nRESPONSES (panellists are referred to as Alpha, Beta, etc.; "
        "use these exact labels in your synthesis):\n\n"
        + "\n\n".join(blocks)
    )
    return "".join(parts), label_to_slug


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
        return SynthResult(text=text, status=SynthStatus.SKIPPED_EMPTY)
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
    synth_input, label_to_slug = _build_input(
        manifest,
        bodies,
        rubric=rub,
        original_prompt=original_prompt,
    )
    # Persist for reproducibility — the synth input is what the model
    # *actually* saw (blind labels in place of slugs).
    (paths.root / "synth_input.txt").write_text(synth_input)
    # Persist the blind→slug mapping so the viewer (and any forensic
    # tooling) can reconstruct exactly which panellist each Alpha/Beta
    # corresponded to on this call. Cheap on disk and uncomplicates
    # debugging if a de-anonymised synthesis looks wrong.
    if label_to_slug:
        (paths.root / "blind_map.json").write_text(
            json.dumps(label_to_slug, indent=2, sort_keys=True)
        )

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
                messages=build_messages(synth_input, provider),
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
        return SynthResult(
            text=text, cost_usd=0.0, cost_known=False, status=SynthStatus.FAILED,
        )

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
        return SynthResult(
            text=text, cost_usd=cost_value, cost_known=cost_known,
            status=SynthStatus.FAILED,
        )
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
        return SynthResult(
            text=text, cost_usd=cost_value, cost_known=cost_known,
            status=SynthStatus.EMPTY,
        )

    text = content.strip()
    # De-anonymise the synth's output: each `\bAlpha\b` / `\bBeta\b` /
    # ... token reverts to the panellist's display slug. The synth's
    # internal view stays bias-mitigated; the on-disk synthesis.md still
    # carries the real slugs the user expects to see in references.
    text = _deblind(text, label_to_slug)
    (paths.root / "synthesis.md").write_text(text)
    # Persist synthesiser badge + spend to the manifest so the standalone
    # `synthesise` tool's cost reaches `consult-ledger`. `orchestrate.consult`,
    # `refine`, and `sequence` already roll their cumulative cost into the
    # manifest themselves (they need to combine fanout + capsule + synth);
    # the direct-synth path was the missing one. Best-effort: a partial
    # manifest or failed lookup silently no-ops (matches augment_manifest's
    # contract) so a re-synth of a legacy run doesn't surface a new error.
    try:
        await artifacts.aaugment_manifest(
            paths,
            synthesiser=synth_alias,
            cost_usd=cost_value,
            cost_known=cost_known,
        )
    except Exception as e:  # noqa: BLE001 — augment is best-effort, never load-bearing
        logger.debug("augment_manifest skipped on direct synth: %s", e)
    return SynthResult(
        text=text, cost_usd=cost_value, cost_known=cost_known,
        status=SynthStatus.OK,
    )
