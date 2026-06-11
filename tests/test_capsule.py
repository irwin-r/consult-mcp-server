"""Capsule extractor and JSON salvage.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import pytest

from consult import artifacts
from consult.types import Capsule, ManifestEntry, RunHandle, Status


@pytest.mark.asyncio
async def test_capsule_annotate_emits_phase_started(tmp_path, monkeypatch):
    """`capsule.annotate` emits `PhaseStarted(phase="capsules")` so the
    parent sees the capsule phase begin rather than only learning when
    the first extraction completes.
    """
    from consult import capsule as capsule_mod
    from consult.progress import CapsuleExtracted, PhaseStarted, ProgressEvent
    from consult.types import Capsule

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.prompt_txt.write_text("hello")

    entry = ManifestEntry(
        slug="alpha",
        model_id="x/y",
        persona=None,
        status=Status.OK,
        finish_reason="stop",
        resource_uri=paths.resource_uri("alpha"),
        body_path=str(paths.response_text("alpha")),
        latency_ms=10,
        cost_usd=0.0,
        cost_known=True,
    )
    paths.response_text("alpha").write_text("body content")

    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=[entry],
        cost_usd=0.0,
        cost_known=True,
        wall_ms=10,
        partial=False,
        blinded=False,
    )

    async def fake_extract_one(body, ext_id, timeout, original_question, *, kind="decision"):
        return Capsule(position="x", recommendation="y", confidence=0.5), 0.0, True

    monkeypatch.setattr(capsule_mod, "_extract_one", fake_extract_one)

    events: list[ProgressEvent] = []

    async def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    await capsule_mod.annotate(handle, on_progress=on_progress)

    assert isinstance(events[0], PhaseStarted)
    assert events[0].phase == "capsules"
    # Followed by the per-capsule extraction events.
    assert any(isinstance(e, CapsuleExtracted) for e in events)


def test_capsule_extract_json_recovers_prose_and_fences():
    """The capsule contract depends on this — one regex change breaks all callers."""
    from consult.jsonparse import extract_json

    assert extract_json('{"score": 0.5}') == {"score": 0.5}
    assert extract_json('```json\n{"k": "v"}\n```') == {"k": "v"}
    assert extract_json('prefix\n{"k": 1}\nsuffix') == {"k": 1}
    assert extract_json("definitely not json") is None
    assert extract_json("") is None


def test_capsule_build_prompt_includes_original_question():
    """The capsule extractor now sees the original question above the
    panellist body so precise refs ("section 3.2") aren't flattened."""
    from consult.capsule import _build_capsule_prompt

    out = _build_capsule_prompt("BODY", "QUESTION-TEXT")
    assert "QUESTION-TEXT" in out
    assert "BODY" in out
    assert out.index("QUESTION-TEXT") < out.index("BODY")


def test_capsule_build_prompt_without_question_is_legacy():
    """Omitting original_question preserves the pre-Phase-1 shape."""
    from consult.capsule import _build_capsule_prompt

    out = _build_capsule_prompt("BODY", None)
    assert "ORIGINAL QUESTION" not in out
    assert out.endswith("BODY")


def test_context_bundle_persists_capsule_kind(tmp_path, monkeypatch):
    """ContextBundle records `capsule_kind` so a continuation can inherit
    it without the caller having to re-specify."""
    from consult import context as ctx

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    ctx.write(paths, ctx.build("the prompt", blinded=False, capsule_kind="review"))

    loaded = ctx.load_or_none(paths)
    assert loaded is not None
    assert loaded.capsule_kind == "review"


def test_capsule_review_extraction_prompt_directs_enumeration():
    """The review-kind extraction prompt must explicitly tell the extractor
    to enumerate every distinct finding (regression: cheap extractors
    returned `findings=[]` when given detailed reviews)."""
    from consult.capsule import _CAPSULE_PROMPT_HEAD_REVIEW

    head = _CAPSULE_PROMPT_HEAD_REVIEW.lower()
    assert "enumerate" in head
    assert "every distinct" in head
    assert "🔴" in _CAPSULE_PROMPT_HEAD_REVIEW
    assert "blocker" in head


def test_capsule_review_kind_uses_larger_token_budget():
    """ReviewCapsule body+extraction needs more output tokens than decision
    (a thorough review can produce 20+ findings, each ~150 chars)."""
    from consult.capsule import MAX_TOKENS_BY_KIND

    assert MAX_TOKENS_BY_KIND["review"] >= 2000
    assert MAX_TOKENS_BY_KIND["review"] > MAX_TOKENS_BY_KIND["decision"]


def test_legacy_capsule_dict_without_kind_loads_as_decision():
    """Pre-M2 manifest entries don't have `capsule.kind`. The ManifestEntry
    pre-validator must inject it so they parse as decision capsules."""
    from consult.types import ManifestEntry

    legacy_payload = {
        "slug": "alpha",
        "status": "OK",
        "resource_uri": "consult://runs/x/responses/alpha",
        "body_path": "/x/alpha",
        "capsule": {
            "position": "ship it",
            "recommendation": "merge",
            "key_points": [],
            "unique_claims": [],
            "caveats": [],
            "agrees_with": [],
            "disagrees_with": [],
            "confidence": 0.8,
        },
    }
    entry = ManifestEntry.model_validate(legacy_payload)
    assert entry.capsule is not None
    assert entry.capsule.kind == "decision"
    assert entry.capsule.position == "ship it"


def test_capsule_kind_picks_correct_prompt_head():
    """The extractor's prompt head varies by kind — verify the dispatch."""
    from consult.capsule import (
        _CAPSULE_PROMPT_HEAD_DECISION,
        _CAPSULE_PROMPT_HEAD_RESEARCH,
        _CAPSULE_PROMPT_HEAD_REVIEW,
        _build_capsule_prompt,
    )

    body = "BODY-TEXT"
    decision_p = _build_capsule_prompt(body, None, kind="decision")
    review_p = _build_capsule_prompt(body, None, kind="review")
    research_p = _build_capsule_prompt(body, None, kind="research")
    assert _CAPSULE_PROMPT_HEAD_DECISION.split("\n")[0] in decision_p
    assert _CAPSULE_PROMPT_HEAD_REVIEW.split("\n")[0] in review_p
    assert _CAPSULE_PROMPT_HEAD_RESEARCH.split("\n")[0] in research_p
    # Unknown kinds default to decision (so a typo doesn't silently produce
    # zero-data capsules).
    assert _CAPSULE_PROMPT_HEAD_DECISION.split("\n")[0] in _build_capsule_prompt(body, None, kind="nonsense")


async def test_capsule_extractor_out_of_range_confidence_doesnt_crash(tmp_path, monkeypatch):
    """A panellist body with `CONFIDENCE: 75.0` (model wrote percent instead
    of fraction) must NOT propagate a Pydantic validation error out of
    `_extract_one`. Pre-fix this crashed an entire refine run because
    `capsule.annotate`'s `asyncio.gather` had no `return_exceptions=True`.
    """
    import litellm

    from consult import capsule as capsule_mod

    async def fake_acompletion(**kwargs):
        # Return malformed JSON (out-of-range confidence) so the JSON-build
        # path takes the except branch, then the body-fallback would also
        # hit the same out-of-range value.
        class Resp:
            class _Choice:
                class _Msg:
                    content = '{"kind":"decision","confidence":75.0}'

                message = _Msg()
                finish_reason = "stop"

            choices = [_Choice()]
            usage = None

        return Resp()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.0)
    body = "stuff stuff\n\nCONFIDENCE: 75.0\nKEY_REASON: whatever"
    # Decision kind — confidence is ge=0 le=1 in Capsule.
    cap, cost, cost_known = await capsule_mod._extract_one(
        body,
        extractor_id="anthropic/claude-haiku-test",
        timeout=30,
        kind="decision",
    )
    # Did not crash. Out-of-range body confidence was discarded.
    assert cap.confidence is None


async def test_capsule_annotate_isolates_per_slug_failure(tmp_path, monkeypatch):
    """A single panellist's capsule extraction crash must not abort the
    whole panel — annotate's per-slug wrapper now swallows unexpected
    failures and returns an empty capsule for that slug.
    """
    from consult import capsule as capsule_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.prompt_txt.write_text("q")

    manifest = []
    for slug, body in [("good", "Real answer.\nCONFIDENCE: 0.7"), ("bad", "boom")]:
        paths.response_text(slug).write_text(body)
        manifest.append(
            ManifestEntry(
                slug=slug,
                model_id="m/x",
                status=Status.OK,
                resource_uri=paths.resource_uri(slug),
                body_path=str(paths.response_text(slug)),
                latency_ms=0,
                cost_usd=0.0,
                cost_known=True,
                confidence=None,
                capsule=None,
            )
        )
    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=manifest,
        cost_usd=0.0,
        cost_known=True,
        wall_ms=0,
    )
    artifacts.write_manifest(paths, handle.model_dump())

    call_count = {"i": 0}

    async def fake_extract_one(body, extractor_id, timeout, original_question=None, *, kind="decision"):
        call_count["i"] += 1
        if call_count["i"] == 2:
            # Simulate an extractor crash on the second slug.
            raise RuntimeError("explosion")
        return Capsule(position="ok"), 0.0, True

    monkeypatch.setattr(capsule_mod, "_extract_one", fake_extract_one)

    annotated = await capsule_mod.annotate(handle, kind="decision")
    # No exception propagated. Both slugs annotated; "bad" got an empty capsule.
    assert annotated.manifest[0].capsule is not None
    assert annotated.manifest[1].capsule is not None
    # Second entry's capsule is the empty fallback (no position set).
    assert annotated.manifest[1].capsule.position == ""


def test_extract_json_rejects_non_dict_root():
    """LLMs occasionally return a JSON array instead of an object. Upstream
    callers do `data.get(...)`, which raises AttributeError on a list.
    `extract_json` must return None for any non-dict root.
    """
    from consult.jsonparse import extract_json

    assert extract_json("[1, 2, 3]") is None
    assert extract_json('[{"score": 1.0}]') is None
    assert extract_json("42") is None
    assert extract_json('"hello"') is None
    # Real dict still parses.
    assert extract_json('{"score": 0.5}') == {"score": 0.5}


def test_extract_inlined_blocks_finds_attachments():
    """Parser recognises the deterministic block format that
    `render_attachment` produces, ignores code blocks in the prose
    section above the separator."""
    from consult.attachments import (
        ATTACHMENT_SEPARATOR,
        extract_inlined_blocks,
    )

    prompt = (
        "Here's some prose with a fenced code sample:\n"
        "```py\nprint('not an attachment')\n```\n"
        + ATTACHMENT_SEPARATOR
        + "\n# /Users/x/foo.py\n```python\ndef foo(): pass\n```\n"
        + "\n## label: /Users/x/bar.py\n```python\nclass B: pass\n```\n"
    )
    blocks = extract_inlined_blocks(prompt)
    assert len(blocks) == 2
    assert blocks[0].path == "/Users/x/foo.py"
    assert blocks[0].label is None
    assert "def foo()" in blocks[0].content
    assert blocks[1].path == "/Users/x/bar.py"
    assert blocks[1].label == "label"


@pytest.mark.asyncio
async def test_capsule_retry_counts_both_extractor_calls_cost(monkeypatch):
    """Regression: when the empty-findings re-ask fires, the cost of BOTH
    extractor calls must be counted. A previous version overwrote `resp`
    with the retry response and priced only the retry, silently dropping
    the first (already-billed) call's cost — understating spend against
    `max_run_usd`.
    """
    import litellm

    from consult import capsule as capsule_mod

    # Long, finding-shaped body so `_body_has_findings` opens the retry gate.
    body = "Severity: major. " + ("There is a real correctness finding here. " * 12)

    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        # 1st call: a valid ReviewCapsule with no findings (triggers retry).
        # 2nd (retry): a capsule that does enumerate a finding.
        if calls["n"] == 1:
            content = '{"overall_verdict": "discuss", "findings": []}'
        else:
            content = (
                '{"overall_verdict": "changes_requested", "findings": '
                '[{"severity": "major", "category": "correctness", "summary": "real bug"}]}'
            )

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

            def model_dump(self):
                return {}

        return _Resp()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)

    capsule, cost, cost_known = await capsule_mod._extract_one(
        body, "anthropic/claude-haiku-4-5", 30, kind="review"
    )

    assert calls["n"] == 2  # the retry fired
    assert [f.summary for f in capsule.findings] == ["real bug"]  # retry capsule adopted
    assert cost_known is True
    assert cost == pytest.approx(0.002)  # both calls counted, not just the retry
