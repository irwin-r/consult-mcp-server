"""Capsule extractor — turn a raw panellist body into a ~200-token
structured Capsule via a cheap model returning strict JSON.

Runs in parallel across the panel after fanout completes. Failed extractions
return an empty Capsule rather than failing the whole run; the body is still
available as a resource.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import litellm

from . import artifacts, registry
from .types import Capsule, ManifestEntry, RunHandle, Status

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

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_CONFIDENCE = re.compile(r"^\s*CONFIDENCE\s*:\s*([0-9.]+)", re.M | re.I)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    # Strip markdown fences if present
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


async def _extract_one(
    body: str, extractor_id: str, timeout: int
) -> tuple[Capsule, float | None]:
    if not body or not body.strip():
        return Capsule(), None
    prompt = _CAPSULE_PROMPT + body
    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=extractor_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.0,
            ),
            timeout=timeout,
        )
        text = resp.choices[0].message.content or ""
        data = _extract_json(text) or {}
        # Coerce confidence — model may emit null
        if data.get("confidence") in (None, "null"):
            m = _CONFIDENCE.search(body)
            if m:
                data["confidence"] = float(m.group(1))
        return Capsule(**{k: v for k, v in data.items() if k in Capsule.model_fields}), getattr(
            litellm, "completion_cost", lambda **_: 0.0
        )(completion_response=resp) if hasattr(litellm, "completion_cost") else None
    except Exception:
        # Fallback: parse confidence from body, leave the rest blank
        m = _CONFIDENCE.search(body)
        conf = float(m.group(1)) if m else None
        return Capsule(confidence=conf), None


async def annotate(handle: RunHandle, *, extractor: str | None = None) -> RunHandle:
    """Populate `capsule` and `confidence` on each manifest entry in place.

    Returns the same handle for chainability. Cost from extraction is added to
    the handle's `cost_usd`.
    """
    ext_alias = extractor or registry.default_capsule_extractor()
    ext_entry = registry.resolve_model(ext_alias)
    ext_id = ext_entry["litellm_id"]
    timeout = ext_entry.get("default_timeout_s", 60)

    paths = artifacts.load_run(handle.run_id)
    tasks = []
    targets: list[ManifestEntry] = []
    for entry in handle.manifest:
        if entry.status not in (Status.OK, Status.TRUNCATED):
            continue
        body = paths.response_text(entry.slug).read_text()
        tasks.append(_extract_one(body, ext_id, timeout))
        targets.append(entry)

    if not tasks:
        return handle

    results = await asyncio.gather(*tasks)
    extra_cost = 0.0
    for entry, (capsule, cost) in zip(targets, results, strict=True):
        entry.capsule = capsule
        if capsule.confidence is not None:
            entry.confidence = capsule.confidence
        if cost:
            extra_cost += cost
        # Persist capsule artifact
        paths.capsule_for(entry.slug).write_text(capsule.model_dump_json(indent=2))

    handle.cost_usd += extra_cost
    artifacts.write_manifest(paths, handle.model_dump())
    return handle
