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

logger = logging.getLogger(__name__)

from . import artifacts, registry
from .runner import _build_messages
from .types import Status

_DEFAULT_RUBRIC = """\
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


def _build_input(
    manifest: list[dict[str, Any]],
    bodies: dict[str, str],
    *,
    rubric: str,
    anonymised: bool,
) -> str:
    usable = [m for m in manifest if m["status"] in (Status.OK.value, Status.TRUNCATED.value)]
    header = rubric.format(n=len(usable))
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
    return header + "\n\n---\nRESPONSES:\n\n" + "\n\n".join(blocks)


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
    rub = rubric or _DEFAULT_RUBRIC
    synth_input = _build_input(manifest, bodies, rubric=rub, anonymised=anonymised)
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
