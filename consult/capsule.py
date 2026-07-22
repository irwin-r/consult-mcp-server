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
from typing import Any, cast

import litellm

from . import artifacts, citations, context, provider_caps, registry
from .jsonparse import extract_json
from .progress import CapsuleExtracted, PhaseStarted, ProgressCallback, append_progress_log
from .redact import redact_exc
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
  "confidence": 0.0
}

Rules:
- key_points: 2-5 entries, each ≤ 20 words
- unique_claims and caveats: 0-3 entries each
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
- sources_cited: 0-5 entries, copied exactly as they appear in the body: URLs, paper titles, or bare bracket markers like "[3]". Do not resolve markers to URLs and do not invent sources.
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
    # Gate for the empty-extraction retry: only re-ask when the body is substantial
    # and looks like it carries extractable structure, so we don't burn a retry on
    # a model that genuinely abstained or returned a short non-answer.
    if len(body.strip()) < 300:
        return False
    low = body.lower()
    return any(m in low for m in ("severity", "finding", "fix:", "issue", "\n- ", "\n1.", "\n* ", "\n#"))


_ENVELOPE_KEYS = ("parameter", "parameters", "input", "arguments", "capsule", "response", "properties")

# Extractor models drift off the Literal enums under pressure ("architecture",
# "compliance", "critical"...). Coerce the common aliases; anything still
# invalid is dropped PER FINDING rather than discarding the whole capsule —
# observed live 2026-07-13 run 20260713-095511-17732: ONE
# `category: "architecture"` finding nuked a 6-finding gemini capsule to
# empty via the all-or-nothing Pydantic build.
_SEVERITY_ALIASES = {
    "critical": "blocker",
    "high": "major",
    "medium": "minor",
    "moderate": "minor",
    "low": "nit",
    "info": "nit",
    "suggestion": "nit",
    "positive": "praise",
}
_CATEGORY_ALIASES = {
    "architecture": "maintainability",
    "design": "maintainability",
    "infrastructure": "maintainability",
    "config": "maintainability",
    "configuration": "maintainability",
    "deployment": "maintainability",
    "compliance": "correctness",
    "legal": "correctness",
    "bug": "correctness",
    "reliability": "correctness",
    "data-loss": "correctness",
    "testing": "tests",
    "test": "tests",
    "documentation": "docs",
    "perf": "performance",
}
_LINE_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:[-–:]\s*(\d+))?\s*$")


def _salvage_review_findings(data: dict[str, Any], capsule_cls: type) -> dict[str, Any]:
    """Per-finding validation for review capsules: coerce, keep valid, drop bad.

    Also normalises `overall_verdict` case/hyphens. Non-review kinds pass
    through untouched.
    """
    if capsule_cls is not ReviewCapsule:
        return data
    from .types import Finding

    out = dict(data)
    verdict = out.get("overall_verdict")
    if isinstance(verdict, str):
        out["overall_verdict"] = verdict.strip().lower().replace("-", "_").replace(" ", "_")

    raw = out.get("findings")
    if not isinstance(raw, list):
        return out
    kept: list[dict[str, Any]] = []
    dropped = 0
    for item in raw:
        if not isinstance(item, dict):
            dropped += 1
            continue
        candidate = dict(item)
        sev = str(candidate.get("severity", "")).strip().lower()
        candidate["severity"] = _SEVERITY_ALIASES.get(sev, sev)
        cat = str(candidate.get("category", "")).strip().lower()
        candidate["category"] = _CATEGORY_ALIASES.get(cat, cat)
        lr = candidate.get("line_range")
        if isinstance(lr, str):
            m = _LINE_RANGE_RE.match(lr)
            candidate["line_range"] = (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None
        candidate = {k: v for k, v in candidate.items() if k in Finding.model_fields}
        try:
            kept.append(Finding(**candidate).model_dump())
        except Exception:
            dropped += 1
    if dropped:
        logger.warning("capsule salvage dropped %d invalid finding(s), kept %d", dropped, len(kept))
    out["findings"] = kept
    return out


def _unwrap_capsule_data(data: Any, capsule_cls: type) -> dict[str, Any]:
    """Undo tool-call envelope nesting around extracted capsule JSON.

    LiteLLM's `response_format` emulation on some providers wraps the payload
    in an envelope key — observed live 2026-07-13: claude-haiku via Anthropic
    tool-use returned `{"parameter": {"kind": "review", "findings": [...]}}`.
    The known-fields filter below then dropped the single unknown key and
    built a perfectly VALID empty capsule: five panellists' findings vanished
    with no log line anywhere, and the round-1 arbiter blamed the panellists
    ("failed to extract parseable findings").

    Descend (bounded) while the dict carries none of the capsule's fields and
    either a known envelope key or exactly one dict value points deeper.
    """
    fields = set(capsule_cls.model_fields)
    for _ in range(3):
        if not isinstance(data, dict) or (fields & data.keys()):
            break
        nxt = None
        for key in _ENVELOPE_KEYS:
            candidate = data.get(key)
            if isinstance(candidate, dict):
                nxt = candidate
                break
        if nxt is None and len(data) == 1:
            (only,) = data.values()
            if isinstance(only, dict):
                nxt = only
        if nxt is None:
            break
        data = nxt
    return data if isinstance(data, dict) else {}


def _capsule_lacks_content(capsule: AnyCapsule, kind: str) -> bool:
    """True when the capsule carries nothing a synthesiser could use.

    Kind-aware, mirroring the `no_value` check in mcp/handlers: review
    substance is findings, research substance is claims/evidence, decision
    substance is position/recommendation/key_points. The old retry gate
    checked `findings` for research too — a field ResearchCapsule doesn't
    have — so the retry fired on every substantial research body and its
    result was never adopted (one wasted extractor call per panellist).
    """
    if kind == "review":
        return not getattr(capsule, "findings", None)
    if kind == "research":
        return not getattr(capsule, "claims", None) and not getattr(capsule, "evidence", None)
    return (
        not getattr(capsule, "position", "").strip()
        and not getattr(capsule, "recommendation", "").strip()
        and not getattr(capsule, "key_points", None)
    )


async def _extract_one(
    body: str,
    extractor_id: str,
    timeout: float,
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
    # The Sources footer (web panellist citations, issue #61) is split off
    # first and re-attached so it survives regardless of the trim policy —
    # head+tail today, but a future change must not silently drop the one
    # block that resolves the body's [n] markers.
    prose, sources_footer = citations.split_sources_footer(body)
    body = context.trim_capsule_body(prose) + sources_footer
    prompt = _build_capsule_prompt(body, original_question, kind=kind)
    kwargs: dict[str, Any] = {
        "model": extractor_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": MAX_TOKENS_BY_KIND.get(kind, MAX_TOKENS_BY_KIND["decision"]),
        "response_format": capsule_cls,
    }
    provider_caps.apply_temperature(kwargs, extractor_id, 0.0)
    try:
        resp = cast(
            Any,
            await asyncio.wait_for(
                litellm.acompletion(**kwargs),
                timeout=timeout,
            ),
        )
    except Exception as e:
        logger.warning("capsule extractor call failed for extractor=%s: %s", extractor_id, redact_exc(e))
        return capsule_cls(confidence=_body_confidence(body)), None, False

    # 2) Capsule build — failures here are JSON shape or Pydantic validation
    try:
        text = resp.choices[0].message.content or ""
        data = _salvage_review_findings(
            _unwrap_capsule_data(extract_json(text) or {}, capsule_cls), capsule_cls
        )
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

    # 2b) Retry once when the extractor returned a content-free capsule from a
    # body that clearly carries structure. The cheap extractor occasionally
    # emits a verdict + confidence but no substance (a stochastic miss — seen
    # with well-formatted grok/llama review bodies, and with claude-opus /
    # gpt-codex decision bodies on the 2026-06-11 design panel); a single
    # sharper re-ask usually recovers it. Kind-aware on both the fire and
    # adopt sides — the old `findings` check fired for every research body
    # and could never adopt the result (ResearchCapsule has no findings).
    # `billed_responses` accrues every extractor call we actually made so the
    # cost lookup below prices all of them. The retry call is billed by the
    # provider whether or not we end up adopting its capsule, so it must be
    # counted either way — overwriting `resp` here would silently drop the
    # first call's cost.
    billed_responses = [resp]

    if _capsule_lacks_content(capsule, kind) and _body_has_findings(body):
        retry_kwargs = dict(kwargs)
        retry_kwargs["messages"] = [
            {
                "role": "user",
                "content": prompt + "\n\nIMPORTANT: the panellist response above DOES "
                "contain extractable content. Populate the substantive fields — "
                "returning them empty is incorrect.",
            }
        ]
        try:
            retry_resp = cast(
                Any, await asyncio.wait_for(litellm.acompletion(**retry_kwargs), timeout=timeout)
            )
            billed_responses.append(retry_resp)
            retry_data = _salvage_review_findings(
                _unwrap_capsule_data(
                    extract_json(retry_resp.choices[0].message.content or "") or {},
                    capsule_cls,
                ),
                capsule_cls,
            )
            if retry_data.get("confidence") in (None, "null"):
                bc = _body_confidence(body)
                if bc is not None:
                    retry_data["confidence"] = bc
            retry_capsule = capsule_cls(
                **{k: v for k, v in retry_data.items() if k in capsule_cls.model_fields}
            )
            if not _capsule_lacks_content(retry_capsule, kind):
                capsule = retry_capsule
        except Exception as e:
            logger.warning(
                "capsule empty-extraction retry failed for extractor=%s: %s",
                extractor_id,
                redact_exc(e),
            )

    # 2c) Resolve bare-marker sources against the body's Sources footer.
    # Runs once on the adopted capsule (first attempt or retry) — the
    # extractor prompt asks for verbatim copying only, and this Python pass
    # owns the marker-to-URL lookup, since cheap extractor models are
    # unreliable at deterministic string mapping (issue #61).
    cited = getattr(capsule, "sources_cited", None)
    if cited:
        capsule.sources_cited = citations.resolve_marker_sources(cited, body)

    # 3) Cost lookup — sum every extractor call we made (first + any retry).
    # A pricing miss on any call flips cost_known False but never discards a
    # successful capsule.
    cost: float | None = None
    # bool(billed_responses): an empty list (shouldn't happen on this path, but
    # be defensive) means no priced call was made, so cost is unknown — not a
    # silent "known $0".
    cost_known = bool(billed_responses)
    for billed in billed_responses:
        try:
            c = litellm.completion_cost(completion_response=billed)
        except Exception as e:
            logger.warning("capsule cost lookup failed for extractor=%s: %s", extractor_id, redact_exc(e))
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
    ext_id = ext_entry.get("litellm_id")
    if not ext_id:
        # A CLI panellist has no litellm_id; using one as the extractor was
        # a guaranteed KeyError deep in the annotate pass.
        raise ValueError(f"capsule extractor {ext_alias!r} must be an API model, not a CLI panellist")
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
