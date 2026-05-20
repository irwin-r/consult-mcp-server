"""Capsule extractor — turn a raw panellist body into a compact structured
Capsule via a cheap model returning strict JSON.

Runs in parallel across the panel after fanout completes. Failed extractions
return a near-empty Capsule (carrying body-parsed confidence if present)
rather than failing the whole run; the body is still available as a resource.
The extractor call, JSON build, and cost lookup are independently wrapped so
a failure in one stage doesn't poison the others.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import litellm

from . import artifacts, registry
from .jsonparse import extract_json
from .progress import CapsuleExtracted
from .runner import ProgressCallback, _append_progress_log
from .types import Capsule, ManifestEntry, RunHandle, Status

logger = logging.getLogger(__name__)

_CAPSULE_PROMPT = """\
You will be given one panellist's response from a multi-model consultation.
Extract a structured capsule. Return EXACTLY a JSON object with these keys:

{
  "position": "one-line summary of stance/conclusion",
  "recommendation": "what the panellist recommends, one sentence",
  "key_points": ["...", "...", "..."],
  "unique_claims": ["claims this panellist made that others might not"],
  "caveats": ["assumptions or conditions the recommendation depends on"],
  "agrees_with": [],
  "disagrees_with": [],
  "confidence": 0.0
}

Rules:
- key_points: 2-5 entries, each ≤ 20 words
- unique_claims and caveats: 0-3 entries each
- agrees_with / disagrees_with: leave empty here (filled later by the orchestrator)
- confidence: parse from a "CONFIDENCE:" line in the body if present, else null
- Output JSON only, no commentary, no markdown fences.

PANELLIST RESPONSE:
"""

_CONFIDENCE = re.compile(r"^\s*CONFIDENCE\s*:\s*([0-9.]+)", re.M | re.I)


async def _extract_one(
    body: str, extractor_id: str, timeout: int
) -> tuple[Capsule, float | None, bool]:
    """Returns (capsule, cost_usd, cost_known).

    The extractor call and the cost lookup are kept in separate try blocks so
    that a price-table miss for the extractor model never discards a
    successfully extracted capsule. `cost_known=False` distinguishes "we don't
    know" from 0.0.
    """
    if not body or not body.strip():
        return Capsule(), None, True  # zero cost is known: we made no call

    # 1) Extraction call — exceptions here mean we couldn't build a capsule.
    # `response_format=Capsule` asks LiteLLM to enforce the Pydantic schema on
    # supporting providers (OpenAI strict mode, Anthropic tool-use emulation,
    # Gemini responseSchema). On providers that don't support it,
    # litellm.drop_params silently drops the param and we fall back to the
    # prompt + regex JSON recovery below.
    #
    # Temperature: most providers want temperature=0.0 for deterministic JSON
    # extraction. Gemini-3 specifically warns that temperature < 1.0 "can
    # cause infinite loops, degraded reasoning, and failure on complex tasks"
    # and recommends omitting the parameter — so for Gemini we leave it
    # unset and trust the provider default. Detect by substring so the
    # openrouter-routed Gemini path (`openrouter/google/gemini-...`) is
    # caught alongside the direct `gemini/...` path.
    prompt = _CAPSULE_PROMPT + body
    kwargs: dict[str, Any] = {
        "model": extractor_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "response_format": Capsule,
    }
    if "gemini" not in extractor_id.lower():
        kwargs["temperature"] = 0.0
    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(**kwargs),
            timeout=timeout,
        )
    except Exception as e:
        logger.warning(
            "capsule extractor call failed for extractor=%s: %s", extractor_id, e
        )
        m = _CONFIDENCE.search(body)
        conf = float(m.group(1)) if m else None
        return Capsule(confidence=conf), None, False

    # 2) Capsule build — failures here are JSON shape or Pydantic validation
    try:
        text = resp.choices[0].message.content or ""
        data = extract_json(text) or {}
        if data.get("confidence") in (None, "null"):
            m = _CONFIDENCE.search(body)
            if m:
                data["confidence"] = float(m.group(1))
        capsule = Capsule(
            **{k: v for k, v in data.items() if k in Capsule.model_fields}
        )
    except Exception as e:
        logger.warning("capsule JSON build failed for extractor=%s: %s", extractor_id, e)
        m = _CONFIDENCE.search(body)
        conf = float(m.group(1)) if m else None
        capsule = Capsule(confidence=conf)

    # 3) Cost lookup — never let a pricing miss discard a successful capsule
    try:
        cost = litellm.completion_cost(completion_response=resp)
        cost_known = cost is not None
    except Exception as e:
        logger.warning(
            "capsule cost lookup failed for extractor=%s: %s", extractor_id, e
        )
        cost = None
        cost_known = False

    return capsule, cost, cost_known


async def annotate(
    handle: RunHandle,
    *,
    extractor: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> RunHandle:
    """Populate `capsule` and `confidence` on each manifest entry in place.

    Returns the same handle for chainability. Cost from extraction is added to
    the handle's `cost_usd`. `on_progress(done, total, msg)` is invoked once
    per capsule as it lands; failures inside the callback are swallowed.
    """
    ext_alias = extractor or registry.default_capsule_extractor()
    ext_entry = registry.resolve_model(ext_alias)
    ext_id = ext_entry["litellm_id"]
    timeout = ext_entry.get("default_timeout_s", 60)

    paths = artifacts.load_run(handle.run_id)
    targets: list[ManifestEntry] = []
    bodies: list[str] = []
    for entry in handle.manifest:
        if entry.status not in (Status.OK, Status.TRUNCATED):
            continue
        bodies.append(paths.response_text(entry.slug).read_text())
        targets.append(entry)

    if not targets:
        return handle

    total = len(targets)
    done = 0

    async def _run(body: str, slug: str) -> tuple[Capsule, float | None, bool]:
        nonlocal done
        result = await _extract_one(body, ext_id, timeout)
        done += 1
        event = CapsuleExtracted(done=done, total=total, slug=slug)
        _append_progress_log(paths.root, event)
        if on_progress is not None:
            try:
                await on_progress(event)
            except Exception as e:  # noqa: BLE001 — best-effort
                logger.debug("capsule on_progress failed: %s", e)
        return result

    results = await asyncio.gather(
        *(_run(body, entry.slug) for body, entry in zip(bodies, targets, strict=True))
    )
    extra_cost = 0.0
    extractor_cost_all_known = True
    for entry, (capsule, cost, cost_known) in zip(targets, results, strict=True):
        entry.capsule = capsule
        if capsule.confidence is not None:
            entry.confidence = capsule.confidence
        if cost:
            extra_cost += cost
        if not cost_known:
            extractor_cost_all_known = False
        paths.capsule_for(entry.slug).write_text(capsule.model_dump_json(indent=2))

    handle.cost_usd += extra_cost
    if not extractor_cost_all_known:
        handle.cost_known = False
    artifacts.write_manifest(paths, handle.model_dump())
    return handle
