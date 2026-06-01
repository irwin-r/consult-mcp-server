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

from . import artifacts, context, provider_caps, registry
from .jsonparse import extract_json
from .progress import CapsuleExtracted, PhaseStarted, ProgressCallback, append_progress_log
from .types import AnyCapsule, Capsule, ManifestEntry, ResearchCapsule, ReviewCapsule, RunHandle, Status

logger = logging.getLogger(__name__)

_CAPSULE_PROMPT_HEAD_DECISION = """\
You will be given one panellist's response from a multi-model consultation.
Extract a structured capsule. Return EXACTLY a JSON object with these keys:

{
  "kind": "decision",
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

"""

_CAPSULE_PROMPT_HEAD_REVIEW = """\
You will be given one panellist's review of a code artefact (PR diff, file, or codebase).
Your job is to ENUMERATE every distinct finding the panellist raised — one Finding object per issue, suggestion, or praise item. Do not summarise, do not merge similar items, do not drop items because they "seem minor". Aim for completeness — if the panellist listed 12 issues, return 12 findings.

Scan the body for any of:
- Bulleted or numbered lists of issues
- Markdown headers naming files, sections, or severities (e.g. "## Blockers", "### Bug:", "🔴 path.py:42")
- Severity prefixes: "Blocker:", "Critical:", "Major:", "Minor:", "Nit:", "Praise:", "🔴", "🟡", "🟢"
- Phrases naming specific code: "in file X", "function Y", "lines A-B", "the foo helper"
- Recommendations: "should", "consider", "suggest", "would improve"
- A final verdict line: SHIP / CHANGES_REQUESTED / DISCUSS

Return EXACTLY a JSON object with these keys:

{
  "kind": "review",
  "findings": [
    {
      "severity": "blocker | major | minor | nit | praise",
      "file": "path/to/file.py or null",
      "line_range": [42, 58] or null,
      "category": "security | performance | correctness | style | maintainability | tests | docs",
      "summary": "≤30 words",
      "suggestion": "≤30 words on the specific change"
    }
  ],
  "overall_verdict": "ship | changes_requested | discuss",
  "confidence": 0.0
}

Rules:
- findings: 0 to 30 entries. One per distinct item — DO NOT collapse multiple findings into one. Praise items go in findings with severity="praise".
- file: null if the finding is general, otherwise the path verbatim from the panellist (e.g. "consult/refine.py")
- line_range: [start, end] when given; null otherwise. Use start=end for a single line.
- severity: pick the closest match. If the panellist labels something "🔴" or "Critical" it's "blocker"; "🟡" or "Major" is "major"; "🟢" or "Minor"/"Nit" is "minor"/"nit"; positive remarks are "praise".
- category: pick the most apt label. Default to "correctness" if unsure.
- summary and suggestion: ≤ 30 words each. Prefer concrete suggestions over vague hand-waving. If the panellist did not propose a fix, leave suggestion empty.
- overall_verdict: the panellist's overall recommendation if stated explicitly; default "discuss" if unstated.
- confidence: parse from a "CONFIDENCE:" line in the body if present, else null
- Output JSON only, no commentary, no markdown fences.

"""

_CAPSULE_PROMPT_HEAD_RESEARCH = """\
You will be given one panellist's response to a research question.
Extract a structured research capsule. Return EXACTLY a JSON object with these keys:

{
  "kind": "research",
  "claims": ["..."],
  "evidence": ["..."],
  "uncertainties": ["..."],
  "sources_cited": ["..."],
  "confidence": 0.0
}

Rules:
- claims: 2-5 entries, each ≤ 20 words. The panellist's main assertions.
- evidence: 2-5 entries, each ≤ 30 words. What backs each claim (study, doc, first-principles reasoning, vendor claim).
- uncertainties: 0-3 entries, each ≤ 20 words. Where the panellist is genuinely unsure.
- sources_cited: 0-5 entries — URLs, papers, vendor docs the panellist named verbatim.
- confidence: parse from a "CONFIDENCE:" line in the body if present, else null
- Output JSON only, no commentary, no markdown fences.

"""

_HEAD_BY_KIND = {
    "decision": _CAPSULE_PROMPT_HEAD_DECISION,
    "review": _CAPSULE_PROMPT_HEAD_REVIEW,
    "research": _CAPSULE_PROMPT_HEAD_RESEARCH,
}

_RESPONSE_FORMAT_BY_KIND: dict[str, type] = {
    "decision": Capsule,
    "review": ReviewCapsule,
    "research": ResearchCapsule,
}

# Per-kind max-output budget. Sized for the panellist body (free-form prose
# enumerating findings/claims), not just the structured capsule — a review
# body needs room for ~20-30 findings with reasoning. The capsule extractor
# reuses the same budget; overcapping the extractor is harmless (it stops at
# the actual end-of-output). History: review at 4000 truncated claude-haiku
# mid-review (FRICTION #16), so the dimension was flipped from per-model to
# per-kind at 8000; 8000 then truncated verbose/reasoning models (gpt-5.5,
# glm, mimo, gemini-pro) at finish_reason=length — reasoning tokens eat the
# budget before findings are emitted — so review is now 16000.
MAX_TOKENS_BY_KIND: dict[str, int] = {
    "decision": 2000,
    "review": 16000,
    "research": 4000,
}

_CAPSULE_PROMPT_RESPONSE_MARKER = "PANELLIST RESPONSE:\n"


def _build_capsule_prompt(body: str, original_question: str | None, *, kind: str = "decision") -> str:
    """Construct the extractor prompt, optionally with original-question context.

    Without the question, the extractor sees only the body and may flatten
    precise references ("section 3.2", "lines 42-58") to abstract bullets
    because it has no idea what they refer to. Including the question
    grounds the extraction.

    `kind` selects the extraction shape (decision/review/research). Unknown
    kinds fall back to decision so a typo doesn't silently produce empty
    capsules.
    """
    head = _HEAD_BY_KIND.get(kind, _CAPSULE_PROMPT_HEAD_DECISION)
    parts: list[str] = [head]
    if original_question:
        parts.append("ORIGINAL QUESTION (context — extract claims from the response below, not from this):\n")
        parts.append(original_question.strip())
        parts.append("\n\n")
    parts.append(_CAPSULE_PROMPT_RESPONSE_MARKER)
    parts.append(body)
    return "".join(parts)


_CONFIDENCE = re.compile(r"^\s*CONFIDENCE\s*:\s*([0-9.]+)", re.M | re.I)


def _body_confidence(body: str) -> float | None:
    """Extract a CONFIDENCE: value from the body, clamped to [0.0, 1.0].

    The body-level CONFIDENCE footer is documented as 0.0-1.0, but models
    sometimes emit "75" or "0.85.5" (instructions ignored / typo / decimal
    drift). Out-of-range or unparseable values must NOT propagate as
    Pydantic validation errors out of `_extract_one`'s fallback paths —
    that was what crashed an entire refine run (asyncio.gather without
    return_exceptions). Return None for anything unusable.
    """
    m = _CONFIDENCE.search(body)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if not (0.0 <= v <= 1.0):
        return None
    return v


def _body_has_findings(body: str) -> bool:
    # Gate for the empty-findings retry: only re-ask when the body is substantial
    # and looks like it enumerates issues, so we don't burn a retry on a model
    # that genuinely abstained or returned a short non-answer.
    if len(body.strip()) < 300:
        return False
    low = body.lower()
    return any(m in low for m in ("severity", "finding", "fix:", "issue", "\n- ", "\n1.", "\n* ", "\n#"))


async def _extract_one(
    body: str,
    extractor_id: str,
    timeout: int,
    original_question: str | None = None,
    *,
    kind: str = "decision",
) -> tuple[AnyCapsule, float | None, bool]:
    """Returns (capsule, cost_usd, cost_known).

    `kind` selects the capsule shape (decision/review/research) — pass
    "review" for line-anchored PR review findings, "research" for
    claims-and-evidence research briefs, or "decision" (default) for the
    original general-purpose capsule.

    The extractor call and the cost lookup are kept in separate try blocks so
    that a price-table miss for the extractor model never discards a
    successfully extracted capsule. `cost_known=False` distinguishes "we don't
    know" from 0.0.
    """
    capsule_cls: type = _RESPONSE_FORMAT_BY_KIND.get(kind, Capsule)

    if not body or not body.strip():
        return capsule_cls(), None, True  # zero cost is known: we made no call

    # 1) Extraction call — exceptions here mean we couldn't build a capsule.
    # `response_format=<CapsuleClass>` asks LiteLLM to enforce the Pydantic
    # schema on supporting providers (OpenAI strict mode, Anthropic tool-use
    # emulation, Gemini responseSchema). On providers that don't support it,
    # litellm.drop_params silently drops the param and we fall back to the
    # prompt + regex JSON recovery below.
    #
    # Temperature: most providers want 0.0 for deterministic JSON extraction.
    # A handful reject the parameter (Gemini-3 warns it causes infinite
    # loops; claude-opus-4-7 errors outright). `provider_caps` centralises
    # the deny list so capsule/synth/arbiter share the same source of truth.
    # Trim overlong bodies to keep the cheap extractor's input bounded.
    body = context.trim_capsule_body(body)
    prompt = _build_capsule_prompt(body, original_question, kind=kind)
    kwargs: dict[str, Any] = {
        "model": extractor_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MAX_TOKENS_BY_KIND.get(kind, MAX_TOKENS_BY_KIND["decision"]),
        "response_format": capsule_cls,
    }
    provider_caps.apply_temperature(kwargs, extractor_id, 0.0)
    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(**kwargs),
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("capsule extractor call failed for extractor=%s: %s", extractor_id, e)
        return capsule_cls(confidence=_body_confidence(body)), None, False

    # 2) Capsule build — failures here are JSON shape or Pydantic validation
    try:
        text = resp.choices[0].message.content or ""
        data = extract_json(text) or {}
        # Trust the body fallback over an absent/null confidence in JSON.
        if data.get("confidence") in (None, "null"):
            body_conf = _body_confidence(body)
            if body_conf is not None:
                data["confidence"] = body_conf
        # Filter to known fields so an extractor adding an extra key doesn't
        # break the strict (`extra="forbid"`) Pydantic model.
        capsule = capsule_cls(**{k: v for k, v in data.items() if k in capsule_cls.model_fields})
    except Exception as e:
        logger.warning("capsule JSON build failed for extractor=%s: %s", extractor_id, e)
        # Pydantic validation on the body-fallback path must NOT propagate:
        # `annotate`'s `asyncio.gather` doesn't pass `return_exceptions=True`,
        # so a confidence-out-of-range or any other build failure here would
        # tear down the whole panel's capsule pass and any tool that wraps it.
        try:
            capsule = capsule_cls(confidence=_body_confidence(body))
        except Exception as ce:
            logger.warning(
                "capsule fallback construction failed for extractor=%s: %s (returning empty capsule)",
                extractor_id,
                ce,
            )
            capsule = capsule_cls()

    # 2b) Retry once when the extractor returned an empty findings list on a
    # body that clearly enumerates findings. The cheap extractor occasionally
    # emits a verdict + confidence but zero findings (a stochastic miss — seen
    # with well-formatted grok/llama review bodies); a single sharper re-ask
    # usually recovers them. Only fires for the finding-bearing kinds.
    # `billed_responses` accrues every extractor call we actually made so the
    # cost lookup below prices all of them. The retry call is billed by the
    # provider whether or not we end up adopting its capsule, so it must be
    # counted either way — overwriting `resp` here would silently drop the
    # first call's cost.
    billed_responses = [resp]

    if kind in ("review", "research") and not getattr(capsule, "findings", None) and _body_has_findings(body):
        retry_kwargs = dict(kwargs)
        retry_kwargs["messages"] = [
            {
                "role": "user",
                "content": prompt + "\n\nIMPORTANT: the panellist response above DOES "
                "contain findings. Enumerate every one as a separate object — "
                "returning an empty findings list is incorrect.",
            }
        ]
        try:
            retry_resp = await asyncio.wait_for(litellm.acompletion(**retry_kwargs), timeout=timeout)
            billed_responses.append(retry_resp)
            retry_data = extract_json(retry_resp.choices[0].message.content or "") or {}
            if retry_data.get("confidence") in (None, "null"):
                bc = _body_confidence(body)
                if bc is not None:
                    retry_data["confidence"] = bc
            retry_capsule = capsule_cls(
                **{k: v for k, v in retry_data.items() if k in capsule_cls.model_fields}
            )
            if getattr(retry_capsule, "findings", None):
                capsule = retry_capsule
        except Exception as e:
            logger.warning("capsule empty-findings retry failed for extractor=%s: %s", extractor_id, e)

    # 3) Cost lookup — sum every extractor call we made (first + any retry).
    # A pricing miss on any call flips cost_known False but never discards a
    # successful capsule.
    cost: float | None = None
    cost_known = True
    for billed in billed_responses:
        try:
            c = litellm.completion_cost(completion_response=billed)
        except Exception as e:
            logger.warning("capsule cost lookup failed for extractor=%s: %s", extractor_id, e)
            c = None
        if c is None:
            cost_known = False
        else:
            cost = (cost or 0.0) + c

    return capsule, cost, cost_known


async def annotate(
    handle: RunHandle,
    *,
    extractor: str | None = None,
    on_progress: ProgressCallback | None = None,
    kind: str = "decision",
) -> RunHandle:
    """Populate `capsule` and `confidence` on each manifest entry in place.

    Returns the same handle for chainability. Cost from extraction is added to
    the handle's `cost_usd`. `on_progress(done, total, msg)` is invoked once
    per capsule as it lands; failures inside the callback are swallowed.

    `kind` selects the capsule shape produced:
    - `"decision"` (default) — the general-purpose position/recommendation capsule
    - `"review"` — line-anchored Finding[] for code/PR review
    - `"research"` — claims/evidence/uncertainties for research-question panels
    """
    ext_alias = extractor or registry.default_capsule_extractor()
    ext_entry = registry.resolve_model(ext_alias)
    ext_id = ext_entry["litellm_id"]
    timeout = ext_entry.get("default_timeout_s", 60)

    paths = artifacts.load_run(handle.run_id)
    # Load the per-run context bundle so the extractor sees the original
    # question alongside the panellist body. Legacy runs without a bundle
    # fall back to body-only extraction (pre-Phase-1 behaviour).
    bundle = context.load_or_none(paths)
    original_question = bundle.prompt_for_downstream() if bundle else None

    targets: list[ManifestEntry] = [
        entry for entry in handle.manifest if entry.status in (Status.OK, Status.TRUNCATED)
    ]

    if not targets:
        return handle

    # Read all panellist bodies in parallel off the event loop. Previously
    # this was a sequential blocking `read_text()` per entry, which on a
    # 10-panellist run with sizable bodies could stall heartbeats and
    # other awaits for hundreds of ms. `asyncio.gather` over
    # `asyncio.to_thread` keeps the loop free while still preserving
    # per-target ordering.
    bodies: list[str] = await asyncio.gather(
        *(asyncio.to_thread(paths.response_text(entry.slug).read_text) for entry in targets)
    )

    total = len(targets)
    done = 0

    # PhaseStarted("capsules"): so the parent sees the capsule phase begin
    # rather than only learning when the first extraction completes. The
    # consult flow's progress wrapper shifts done/total into the overall
    # bucket; callers that don't wrap get the per-phase counter.
    phase_event = PhaseStarted(done=0, total=total, phase="capsules")
    append_progress_log(paths.root, phase_event)
    if on_progress is not None:
        try:
            await on_progress(phase_event)
        except Exception as e:  # noqa: BLE001
            logger.debug("capsule phase on_progress failed: %s", e)

    async def _run(body: str, slug: str) -> tuple[AnyCapsule, float | None, bool]:
        nonlocal done
        try:
            result = await _extract_one(body, ext_id, timeout, original_question, kind=kind)
        except Exception as e:  # noqa: BLE001
            # `_extract_one` already swallows the documented failure modes
            # (extractor call, JSON build, Pydantic validation), but a path
            # we haven't anticipated must not poison the whole panel — the
            # capsule pass is a best-effort enrichment on top of bodies
            # the panellists already produced.
            logger.warning(
                "capsule extraction crashed for slug=%s: %s (returning empty capsule)",
                slug,
                e,
            )
            cls = _RESPONSE_FORMAT_BY_KIND.get(kind, Capsule)
            result = (cls(), None, False)
        done += 1
        event = CapsuleExtracted(done=done, total=total, slug=slug)
        append_progress_log(paths.root, event)
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
