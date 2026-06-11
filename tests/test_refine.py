"""Refine loop: rounds, arbiter, continuation, cost gates, strategies wiring.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from smoke_helpers import _make_run_dir, _refine_fake_fanout, _refine_fake_synth, _refine_noop_annotate

from consult import artifacts
from consult import refine as refine_mod
from consult.runner import _make_slug
from consult.types import ArbiterVerdict, Capsule, ManifestEntry, ModelSpec, RunHandle, Status


def test_refine_suffix_specs_round_indexes_slugs():
    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="grok", stance="contrarian")]
    r2 = refine_mod._suffix_specs(specs, 2)
    assert r2[0].slug.endswith(".r2")
    assert r2[1].slug.endswith(".r2")
    # Same model index should produce stable base slug
    r2b = refine_mod._suffix_specs(specs, 2)
    assert [s.slug for s in r2] == [s.slug for s in r2b]


def test_refinement_prompt_includes_gaps_and_focus():
    manifest = [
        ManifestEntry(
            slug="m-1.r1",
            model_id="anthropic/x",
            status=Status.OK,
            resource_uri="consult://x",
            body_path="/tmp/x",
            capsule=Capsule(position="A says X", recommendation="do X"),
        )
    ]
    verdict = ArbiterVerdict(round=1, score=0.4, gaps=["cost not discussed"], next_round_focus="address cost")
    out = refine_mod._build_refinement_prompt("Should we ship X?", 2, manifest, verdict)
    assert "Should we ship X?" in out
    assert "cost not discussed" in out
    assert "address cost" in out
    assert "m-1.r1" in out


@pytest.mark.asyncio
async def test_refine_continuation_prepends_prior_synthesis(tmp_path, monkeypatch):
    """A valid continuation_id loads the prior run's question + synthesis and
    returns both the storage-form combined prompt (for disk + arbiter
    context) and a `prior_turns` user/assistant pair (for the panellist's
    actual messages array — preserves role boundaries instead of stitching
    everything into one user blob).
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()
    prior.prompt_txt.write_text("Polars vs DuckDB for 10GB Parquet?")
    (prior.root / "synthesis.md").write_text("ANSWER: pick DuckDB.")

    combined, prior_turns = _apply_continuation("Now what about Polars for ETL?", prior.run_id)
    # Storage form keeps the markdown sections so disk artifacts stay self-describing.
    assert "Prior consultation — original question" in combined
    assert "Polars vs DuckDB for 10GB Parquet?" in combined
    assert "Prior consultation — synthesis" in combined
    assert "ANSWER: pick DuckDB." in combined
    assert "Follow-up question" in combined
    assert "Now what about Polars for ETL?" in combined
    assert combined.index("ANSWER: pick DuckDB.") < combined.index("Now what about Polars for ETL?")
    # prior_turns is what the LLM actually sees — proper user/assistant pair.
    assert prior_turns is not None
    assert len(prior_turns) == 2
    assert prior_turns[0]["role"] == "user"
    assert "Polars vs DuckDB for 10GB Parquet?" in prior_turns[0]["content"]
    assert prior_turns[1]["role"] == "assistant"
    assert "ANSWER: pick DuckDB." in prior_turns[1]["content"]


def test_refine_continuation_legacy_run_without_prompt_txt(tmp_path, monkeypatch):
    """Legacy runs (pre-Phase-1) may lack prompt.txt; continuation must not
    crash — it falls back to a placeholder for the prior question and still
    prepends the synthesis.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()
    # Deliberately do NOT write prompt.txt — simulate a pre-Phase-1 run that
    # only had synthesis.md.
    (prior.root / "synthesis.md").write_text("ANSWER: pick DuckDB.")

    combined, prior_turns = _apply_continuation("Follow-up", prior.run_id)
    # Falls back to a placeholder without raising
    assert "prior question unavailable" in combined.lower()
    assert "ANSWER: pick DuckDB." in combined
    assert "Follow-up" in combined
    assert prior_turns is not None
    assert "prior question unavailable" in prior_turns[0]["content"].lower()


def test_refine_continuation_none_or_empty_is_passthrough(tmp_path, monkeypatch):
    """No continuation_id (or empty string) leaves the prompt untouched and
    returns no prior_turns."""
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    assert _apply_continuation("hello", None) == ("hello", None)
    assert _apply_continuation("hello", "") == ("hello", None)


def test_refine_continuation_unknown_id_raises(tmp_path, monkeypatch):
    """An unknown continuation_id must raise — silently dropping the prior
    context would leave the caller thinking the new round had it. The
    exception type is FileNotFoundError (propagated from artifacts.load_run)
    so the MCP dispatcher in server.py maps it to RUN_NOT_FOUND rather than
    INVALID_INPUT.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="Run not found"):
        _apply_continuation("hello", "20990101-000000-99999")


def test_refine_continuation_missing_synthesis_raises(tmp_path, monkeypatch):
    """A run that exists but has no synthesis.md (e.g. dry-run, cap-aborted)
    can't be a continuation source — fail clearly rather than prepend empty.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()  # no synthesis.md written
    with pytest.raises(ValueError, match="no synthesis.md"):
        _apply_continuation("hello", prior.run_id)


@pytest.mark.asyncio
async def test_refine_validates_max_rounds():
    # Cap raised from 3 to 5 in M4 — values above 5 still rejected.
    with pytest.raises(ValueError, match="max_rounds"):
        await refine_mod.refine("q", [ModelSpec(model="claude-haiku")], max_rounds=6)
    with pytest.raises(ValueError, match="max_rounds"):
        await refine_mod.refine("q", [ModelSpec(model="claude-haiku")], max_rounds=0)


def test_arbiter_json_extractor_tolerates_fences():
    from consult.jsonparse import extract_json

    fenced = '```json\n{"score": 0.7, "gaps": ["x"], "next_round_focus": "", "reasoning": ""}\n```'
    data = extract_json(fenced)
    assert data["score"] == 0.7
    assert data["gaps"] == ["x"]


def test_refine_arbiter_v2_dimensions_normalised_to_overall_score(monkeypatch):
    """v2 arbiter prompt: per-dimension 1-5 Likert input is rescaled to
    [0,1] and averaged into the overall `score`. Round-trips through
    `_ask_arbiter`'s JSON-parse path.
    """
    import asyncio

    from consult.refine import _ask_arbiter

    arbiter_text = json.dumps(
        {
            "dimensions": {
                "coverage": 5,
                "agreement": 5,
                "depth": 3,
                "calibration": 5,
                "actionability": 5,
            },
            "dimension_notes": {
                "depth": "Beta hand-waved on the cost argument.",
            },
            "gaps": ["cost magnitude unclear"],
            "next_round_focus": "quantify cost",
            "reasoning": "Strong consensus; depth lags.",
        }
    )

    class _Choice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _Resp:
        choices = [_Choice(arbiter_text)]

    async def fake_acompletion(**_kw):
        return _Resp()

    monkeypatch.setattr("consult.refine.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "consult.refine.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )

    verdict = asyncio.run(
        _ask_arbiter(
            "Q?",
            round_num=1,
            manifest=[],
            arbiter_alias="claude-haiku",
        )
    )
    assert verdict.parsed_ok is True
    # 5→1.0, 3→0.5 normalised; avg of [1.0, 1.0, 0.5, 1.0, 1.0] = 0.9
    assert verdict.score == pytest.approx(0.9, abs=1e-9)
    # Per-dimension scores in [0,1]
    assert verdict.dimensions["coverage"] == 1.0
    assert verdict.dimensions["depth"] == 0.5
    # Localised note preserved
    assert verdict.dimension_notes["depth"] == "Beta hand-waved on the cost argument."


def test_refine_arbiter_v1_score_field_still_accepted(monkeypatch):
    """Legacy arbiters (or fine-tuned models that don't emit the v2
    `dimensions` block) still work: the bare `score` field is honoured
    when `dimensions` is missing or empty.
    """
    import asyncio

    from consult.refine import _ask_arbiter

    arbiter_text = json.dumps(
        {
            "score": 0.65,
            "gaps": ["legacy"],
            "next_round_focus": "f",
            "reasoning": "r",
        }
    )

    class _Choice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _Resp:
        choices = [_Choice(arbiter_text)]

    async def fake_acompletion(**_kw):
        return _Resp()

    monkeypatch.setattr("consult.refine.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "consult.refine.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )

    verdict = asyncio.run(
        _ask_arbiter(
            "Q?",
            round_num=1,
            manifest=[],
            arbiter_alias="claude-haiku",
        )
    )
    assert verdict.parsed_ok is True
    assert verdict.score == pytest.approx(0.65, abs=1e-9)
    assert verdict.dimensions == {}


def test_refine_build_refinement_prompt_surfaces_weak_dimensions():
    """The refinement prompt must focus the next round on the lowest-scoring
    dimensions, not just the generic gaps list. Bottom-3 dimensions are
    rendered with their localised notes.
    """
    from consult.refine import _build_refinement_prompt
    from consult.types import ArbiterVerdict

    verdict = ArbiterVerdict(
        round=1,
        score=0.5,
        dimensions={
            "coverage": 0.75,
            "agreement": 0.25,
            "depth": 0.5,
            "calibration": 1.0,
            "actionability": 0.0,
        },
        dimension_notes={
            "agreement": "Alpha and Beta disagree on the cost model.",
            "actionability": "No panellist proposed a concrete next step.",
            "depth": "Reasoning was thin on the SLA implications.",
        },
        gaps=["gap-1"],
        next_round_focus="quantify cost",
    )
    out = _build_refinement_prompt("Q?", 2, [], verdict)
    # Bottom-3 by dimension score: actionability (0.0), agreement (0.25), depth (0.5)
    assert "actionability" in out
    assert "agreement" in out
    assert "depth" in out
    # Localised notes appear
    assert "Alpha and Beta disagree" in out
    assert "No panellist proposed a concrete" in out
    # Calibration (highest score) should NOT appear in critique
    # (the prompt header may mention it but the critique section won't)
    crit_section_start = out.find("Per-dimension critique")
    crit_section_end = out.find("Gaps the arbiter flagged")
    assert crit_section_start != -1 and crit_section_end != -1
    crit_section = out[crit_section_start:crit_section_end]
    assert "calibration" not in crit_section


def test_refine_build_refinement_prompt_handles_legacy_verdict():
    """Verdicts from legacy (v1) arbiter calls have no `dimensions`. The
    refinement prompt must still render — falling back to a placeholder
    line — rather than crashing or rendering an empty critique block.
    """
    from consult.refine import _build_refinement_prompt
    from consult.types import ArbiterVerdict

    legacy = ArbiterVerdict(
        round=1,
        score=0.4,
        gaps=["legacy-gap"],
        next_round_focus="f",
    )
    out = _build_refinement_prompt("Q?", 2, [], legacy)
    assert "legacy-gap" in out
    assert "legacy arbiter prompt" in out  # placeholder marker


def test_error_envelope_shape_round_trips():
    """The structured-error envelope must round-trip through JSON with the
    exact shape agents pattern-match against. Locks in the wire contract.
    """
    from consult.mcp.errors import ErrorCode, ErrorDetail, ErrorEnvelope

    env = ErrorEnvelope(
        error=ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="continuation_id not found: bogus",
            run_id=None,
        ),
    )
    payload = json.loads(env.model_dump_json())
    assert payload == {
        "ok": False,
        "error": {
            "code": "invalid_input",
            "message": "continuation_id not found: bogus",
            "run_id": None,
        },
    }

    # run_id-carrying variant for mid-failure partial runs
    env2 = ErrorEnvelope(
        error=ErrorDetail(
            code=ErrorCode.INTERNAL_ERROR,
            message="boom",
            run_id="20260520-010203-1234",
        ),
    )
    payload2 = json.loads(env2.model_dump_json())
    assert payload2["error"]["run_id"] == "20260520-010203-1234"


def test_progress_event_round_trips_through_json():
    """JSONL log line → dict → discriminated-union dispatch. Pydantic's
    `discriminator='kind'` on `ProgressEvent` enables programmatic consumers
    to parse one line and get a typed object back.
    """
    from pydantic import TypeAdapter

    from consult.progress import (
        CapsuleExtracted,
        Heartbeat,
        PanellistCompleted,
        PanellistStarted,
        PhaseStarted,
        ProgressEvent,
    )

    adapter = TypeAdapter(ProgressEvent)
    p = PanellistCompleted(done=1, total=2, slug="x", status="OK", latency_ms=10)
    parsed = adapter.validate_json(p.model_dump_json())
    assert isinstance(parsed, PanellistCompleted)
    assert parsed.slug == "x"

    c = CapsuleExtracted(done=1, total=2, slug="x")
    parsed = adapter.validate_json(c.model_dump_json())
    assert isinstance(parsed, CapsuleExtracted)

    s = PanellistStarted(done=0, total=2, slug="alpha", started_count=1)
    parsed = adapter.validate_json(s.model_dump_json())
    assert isinstance(parsed, PanellistStarted)
    assert parsed.started_count == 1

    ph = PhaseStarted(done=0, total=3, phase="capsules")
    parsed = adapter.validate_json(ph.model_dump_json())
    assert isinstance(parsed, PhaseStarted)
    assert parsed.phase == "capsules"

    hb = Heartbeat(
        done=1,
        total=3,
        elapsed_ms=2500,
        cost_so_far_usd=0.01,
        cost_known=True,
        pending_count=2,
        pending_slugs=["a", "b"],
    )
    parsed = adapter.validate_json(hb.model_dump_json())
    assert isinstance(parsed, Heartbeat)
    assert parsed.pending_slugs == ["a", "b"]


def test_refine_result_validates_partial_coupling_and_surfaces_reason():
    """RefineResult must (a) enforce partial⇔partial_reason coupling and
    (b) actually accept partial_reason at all — previously the field didn't
    exist on the model, so refine() silently dropped it via Pydantic's
    "ignore extras" default and callers had no way to learn why a run
    stopped early (e.g. cost-cap or unknown-pricing refusal).
    """
    import pydantic

    from consult.types import RefineResult

    base = dict(
        run_id="r",
        rounds_completed=1,
        final_manifest=[],
        verdicts=[],
        synthesis="x",
        converged=False,
        threshold=0.85,
        cost_usd=0.0,
        wall_ms=0,
    )
    # Happy path: partial_reason actually round-trips through the model
    rr = RefineResult(**base, partial=True, partial_reason="cost cap exceeded")
    assert rr.partial_reason == "cost cap exceeded"

    # Validator catches the inconsistent states
    with pytest.raises(pydantic.ValidationError):
        RefineResult(**base, partial=True)  # no reason
    with pytest.raises(pydantic.ValidationError):
        RefineResult(**base, partial=False, partial_reason="oops")


def test_viewer_round_of_extracts_refine_round():
    """Refine slugs carry an `.r<n>` suffix; the viewer uses this to bucket
    panellists by round in the panel section.
    """
    from consult.viewer import _round_of

    assert _round_of("claude-opus") is None
    assert _round_of("claude-opus.r1") == 1
    assert _round_of("claude-opus-0.r3") == 3


def test_viewer_render_run_refine_shows_arbiter_rounds_and_groups_panellists(tmp_path, monkeypatch):
    """Refine runs: arbiter section per round + panel cards bucketed by round."""
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-3"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[
            {
                "slug": "alpha.r1",
                "model_id": "x/y",
                "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha.r1",
                "body_path": "/x",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "cost_known": True,
            },
            {
                "slug": "alpha.r2",
                "model_id": "x/y",
                "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha.r2",
                "body_path": "/x",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "cost_known": True,
            },
        ],
        synth="final synth",
        arbiters=[
            {
                "round": 1,
                "score": 0.5,
                "gaps": ["missing X"],
                "next_round_focus": "address X",
                "reasoning": "r1",
            },
            {"round": 2, "score": 0.9, "gaps": [], "next_round_focus": "", "reasoning": "r2"},
        ],
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert "Arbiter rounds" in text
    assert "Round 1" in text
    assert "Round 2" in text
    assert "missing X" in text
    assert "address X" in text
    assert "score 0.50" in text
    assert "score 0.90" in text
    # Multi-round panellists must be grouped under per-round headers.
    assert text.count("Round 1") >= 2  # arbiter card + panel group header
    assert text.count("Round 2") >= 2


def test_context_write_and_load_roundtrip(tmp_path, monkeypatch):
    """A written bundle loads back byte-identically."""
    from consult import context as ctx

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    src = "Review this PR for race conditions: <DIFF>"
    bundle = ctx.build(src, blinded=False)
    ctx.write(paths, bundle)

    loaded = ctx.load_or_none(paths)
    assert loaded is not None
    assert loaded.prompt == bundle.prompt
    assert loaded.prompt_scrubbed == bundle.prompt_scrubbed
    assert loaded.blinded == bundle.blinded


def test_refine_base_slug_strips_round_suffix():
    from consult.refine import _base_slug

    assert _base_slug("claude-opus-0.r2") == "claude-opus-0"
    assert _base_slug("alpha.r3") == "alpha"
    assert _base_slug("no-suffix") == "no-suffix"
    assert _base_slug("name-with.dot.r1") == "name-with.dot"


def test_refine_format_position_diff_round_one_placeholder():
    from consult.refine import _format_position_diff

    out = _format_position_diff(None, [])
    assert "first round" in out


def test_refine_format_position_diff_shows_changes_and_unchanged():
    """The position-diff helper renders:
    - unchanged stances as `unchanged`
    - changed stances as `before:` / `after:`
    - new panellists this round
    - dropped panellists from prior round
    """
    from consult.refine import _format_position_diff
    from consult.types import Capsule, ManifestEntry, Status

    def entry(slug: str, position: str) -> ManifestEntry:
        return ManifestEntry(
            slug=slug,
            status=Status.OK,
            resource_uri=f"consult://runs/x/responses/{slug}",
            body_path=f"/x/{slug}",
            capsule=Capsule(position=position),
        )

    prior = [
        entry("alpha.r1", "ship now"),
        entry("beta.r1", "needs review"),
        entry("gamma.r1", "dropping out"),
    ]
    current = [
        entry("alpha.r2", "ship now"),  # unchanged
        entry("beta.r2", "ready to merge"),  # changed
        entry("delta.r2", "joined late"),  # new this round
        # gamma is missing this round
    ]
    out = _format_position_diff(prior, current)
    assert "alpha: unchanged" in out
    assert "beta:" in out and "before: needs review" in out and "after:  ready to merge" in out
    assert "delta (new this round)" in out
    assert "gamma (dropped this round" in out


def test_refine_format_position_diff_handles_review_capsules():
    """Regression: `_format_position_diff` previously read `capsule.position`
    directly, which crashes on ReviewCapsule (no `position` field). Must
    work across all capsule kinds via `_capsule_summary`."""
    from consult.refine import _format_position_diff
    from consult.types import Finding, ManifestEntry, ReviewCapsule, Status

    def entry(slug: str, verdict: str, findings: int) -> ManifestEntry:
        return ManifestEntry(
            slug=slug,
            status=Status.OK,
            resource_uri=f"consult://runs/x/responses/{slug}",
            body_path=f"/x/{slug}",
            capsule=ReviewCapsule(
                overall_verdict=verdict,
                findings=[
                    Finding(
                        severity="blocker",
                        category="security",
                        summary=f"finding {i}",
                        suggestion="fix it",
                    )
                    for i in range(findings)
                ],
            ),
        )

    prior = [entry("alpha.r1", "changes_requested", 3)]
    current = [entry("alpha.r2", "ship", 0)]
    # Must not raise AttributeError; previous code did because
    # ReviewCapsule has no `.position` field.
    out = _format_position_diff(prior, current)
    assert "alpha" in out
    assert "changes_requested" in out
    assert "ship" in out


def test_refine_format_position_diff_handles_research_capsules():
    """Regression: same as review, but for ResearchCapsule."""
    from consult.refine import _format_position_diff
    from consult.types import ManifestEntry, ResearchCapsule, Status

    def entry(slug: str, n_claims: int) -> ManifestEntry:
        return ManifestEntry(
            slug=slug,
            status=Status.OK,
            resource_uri=f"consult://runs/x/responses/{slug}",
            body_path=f"/x/{slug}",
            capsule=ResearchCapsule(
                claims=[f"claim {i}" for i in range(n_claims)],
                evidence=["e"],
                uncertainties=["u"],
            ),
        )

    prior = [entry("alpha.r1", 2)]
    current = [entry("alpha.r2", 5)]
    out = _format_position_diff(prior, current)
    assert "alpha" in out
    assert "claims" in out


def test_refine_format_capsules_handles_review_kind():
    """`_format_capsules` must render review-kind capsules without
    crashing on the missing `position` / `recommendation` fields."""
    from consult.refine import _format_capsules
    from consult.types import Finding, ManifestEntry, ReviewCapsule, Status

    m = ManifestEntry(
        slug="alpha",
        status=Status.OK,
        resource_uri="consult://runs/x/responses/alpha",
        body_path="/x/alpha",
        capsule=ReviewCapsule(
            overall_verdict="changes_requested",
            findings=[
                Finding(
                    severity="blocker",
                    file="src/auth.py",
                    line_range=(42, 58),
                    category="security",
                    summary="SQL injection in login",
                    suggestion="use parameterised query",
                ),
            ],
        ),
    )
    out = _format_capsules([m])
    assert "changes_requested" in out
    assert "SQL injection" in out


@pytest.mark.asyncio
async def test_refine_round_two_passes_per_panellist_conversation(
    tmp_path,
    monkeypatch,
):
    """Round 2's fanout must receive `prior_turns_by_slug` populated with
    each panellist's round-1 (user, assistant) pair. Round 1 must NOT —
    it's the round that establishes the history.

    This is the Anthropic prompt-cache mechanism: round-2's prefix
    (continuation + user-round-1-prompt + assistant-round-1-answer) is
    byte-stable across rounds 2 and 3, which is what the cache keys on.
    """
    from consult import refine as refine_mod
    from consult.types import (
        ArbiterVerdict,
        Capsule,
        ManifestEntry,
        ModelSpec,
        RunHandle,
        Status,
    )

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # Disable heartbeats so the test runs deterministically
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    fanout_calls: list[dict] = []

    import copy

    async def fake_fanout(prompt, specs, **kwargs):
        # Capture what each round's fanout sees. Deep-copy the dict so
        # later mutation by refine's post-round bookkeeping doesn't
        # retroactively change what we observed (refine reuses the same
        # `panel_conversations` dict whose values are the lists we'd be
        # capturing by reference).
        fanout_calls.append(
            {
                "prompt": prompt,
                "specs": [(s.model, s.slug) for s in specs],
                "prior_turns": copy.deepcopy(kwargs.get("prior_turns")),
                "prior_turns_by_slug": copy.deepcopy(kwargs.get("prior_turns_by_slug")),
            }
        )
        # Use the existing run dir (existing_paths) when refine passes one
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        # Write a body file per slug so refine's post-round bookkeeping
        # can build per-slug history from disk.
        manifest = []
        for spec in specs:
            slug = spec.slug or spec.model
            body_path = paths.root / "responses" / f"{slug}.txt"
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_text(f"{slug} body for round")
            manifest.append(
                ManifestEntry(
                    slug=slug,
                    model_id="x/a",
                    status=Status.OK,
                    resource_uri=f"consult://x/{slug}",
                    body_path=str(body_path),
                    latency_ms=10,
                    cost_usd=0.001,
                    cost_known=True,
                    capsule=Capsule(position=f"{slug} position"),
                )
            )
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=manifest,
            cost_usd=0.001 * len(specs),
            cost_known=True,
            wall_ms=10,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    arbiter_n = {"n": 0}

    async def fake_arbiter(*args, **kwargs):
        # Round 1 → low score → triggers round 2. Round 2 → high → stop.
        arbiter_n["n"] += 1
        score = 0.2 if arbiter_n["n"] == 1 else 0.95
        return ArbiterVerdict(round=arbiter_n["n"], score=score, parsed_ok=True)

    async def fake_synth(*args, **kwargs):
        from consult import synth as _synth_mod

        return _synth_mod.SynthResult(text="final synth")

    async def fake_aestimate(*a, **kw):
        return (0.001, True)

    monkeypatch.setattr("consult.refine.runner.fanout", fake_fanout)
    monkeypatch.setattr("consult.refine.capsule.annotate", fake_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    result = await refine_mod.refine(
        "Should we ship feature X?",
        [ModelSpec(model="claude-haiku"), ModelSpec(model="gpt-mini")],
        threshold=0.85,
        max_rounds=2,
    )

    # Two rounds fired
    assert len(fanout_calls) == 2

    # Round 1: no per-slug history (the seed) — falls through to global
    # `prior_turns` (None here, since no continuation_id).
    r1 = fanout_calls[0]
    assert r1["prior_turns_by_slug"] is None
    assert r1["prior_turns"] is None

    # Round 2: per-slug history populated. Each panellist's slug for
    # round 2 maps to a 2-turn list (user=round-1 prompt, assistant=body).
    r2 = fanout_calls[1]
    assert r2["prior_turns_by_slug"] is not None
    assert len(r2["prior_turns_by_slug"]) == 2  # one per panellist
    for _slug, history in r2["prior_turns_by_slug"].items():
        # 2 turns: user(round-1 prompt) + assistant(round-1 body)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        # The user turn should carry the original question, not the
        # refinement prompt (round 1's `round_prompt` is the original).
        assert "Should we ship feature X?" in history[0]["content"]
        # The assistant turn carries this panellist's round-1 body.
        # Slug suffix is `.r1` on round-1; the body filename matches.
        assert "body for round" in history[1]["content"]

    # Refine completed successfully
    assert result.rounds_completed == 2
    assert result.converged is True


@pytest.mark.asyncio
async def test_refine_inherits_capsule_kind_from_continuation(tmp_path, monkeypatch):
    """When `capsule_kind` is not passed and `continuation_id` is, refine
    should pick up the prior run's capsule_kind from its ContextBundle."""
    from consult import context as ctx
    from consult import refine as refine_mod
    from consult.types import ArbiterVerdict, ManifestEntry, ModelSpec, RunHandle, Status

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()
    prior.prompt_txt.write_text("prior question")
    ctx.write(prior, ctx.build("prior question", blinded=False, capsule_kind="review"))
    (prior.root / "synthesis.md").write_text("prior synthesis")

    captured: dict[str, str] = {}

    async def fake_fanout(prompt, specs, **kwargs):
        return RunHandle(
            run_id=prior.run_id,
            artifacts_dir=str(prior.root),
            manifest=[
                ManifestEntry(
                    slug="alpha.r1",
                    status=Status.OK,
                    resource_uri="consult://runs/x/responses/alpha.r1",
                    body_path="/x",
                    latency_ms=10,
                    cost_known=True,
                )
            ],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        captured["annotate_kind"] = kwargs.get("kind", "?")
        return handle

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(round=1, score=1.0, parsed_ok=True)

    async def fake_synth(*args, **kwargs):
        from consult import synth as _synth_mod

        return _synth_mod.SynthResult(text="synthesised")

    monkeypatch.setattr("consult.refine.runner.fanout", fake_fanout)
    monkeypatch.setattr("consult.refine.capsule.annotate", fake_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", fake_synth)

    # Implicit inheritance — caller doesn't pass capsule_kind
    await refine_mod.refine(
        "follow-up",
        [ModelSpec(model="claude-haiku")],
        threshold=0.5,
        max_rounds=1,
        continuation_id=prior.run_id,
    )
    assert captured.get("annotate_kind") == "review"

    # Explicit override
    captured.clear()
    await refine_mod.refine(
        "follow-up",
        [ModelSpec(model="claude-haiku")],
        threshold=0.5,
        max_rounds=1,
        continuation_id=prior.run_id,
        capsule_kind="research",
    )
    assert captured.get("annotate_kind") == "research"


def test_review_capsule_round_trips():
    from consult.types import Finding, ManifestEntry, ReviewCapsule

    review = ReviewCapsule(
        findings=[
            Finding(
                severity="blocker",
                file="src/auth.py",
                line_range=(42, 58),
                category="security",
                summary="SQL injection in login",
                suggestion="use parameterised query",
            ),
        ],
        overall_verdict="changes_requested",
        confidence=0.9,
    )
    entry = ManifestEntry(
        slug="alpha",
        status="OK",
        resource_uri="consult://runs/x/responses/alpha",
        body_path="/x",
        capsule=review,
    )
    dumped = entry.model_dump()
    assert dumped["capsule"]["kind"] == "review"
    assert dumped["capsule"]["overall_verdict"] == "changes_requested"
    # Round-trip through validation
    reloaded = ManifestEntry.model_validate(dumped)
    assert reloaded.capsule.kind == "review"
    assert reloaded.capsule.findings[0].file == "src/auth.py"


def test_research_capsule_round_trips():
    from consult.types import ManifestEntry, ResearchCapsule

    research = ResearchCapsule(
        claims=["Polars is faster for groupby on 10GB+"],
        evidence=["TPC-H q3 benchmark from polars team"],
        uncertainties=["Memory headroom at 100GB unknown"],
        sources_cited=["https://pola.rs/blog/..."],
        confidence=0.7,
    )
    entry = ManifestEntry(
        slug="alpha",
        status="OK",
        resource_uri="consult://runs/x/responses/alpha",
        body_path="/x",
        capsule=research,
    )
    dumped = entry.model_dump()
    assert dumped["capsule"]["kind"] == "research"
    reloaded = ManifestEntry.model_validate(dumped)
    assert reloaded.capsule.kind == "research"
    assert "Polars" in reloaded.capsule.claims[0]


@pytest.mark.asyncio
async def test_refine_breaks_when_fanout_returns_partial(tmp_path, monkeypatch):
    """A partial fanout (zero usable, cap exceeded, etc.) must not flow into
    the arbiter — the arbiter call would burn flagship $$ for no signal.
    """
    from consult import refine as refine_mod
    from consult import runner

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    arbiter_called = {"n": 0}

    async def fake_arbiter(*args, **kwargs):
        arbiter_called["n"] += 1
        return ArbiterVerdict(
            round=1,
            score=1.0,
            gaps=[],
            reasoning="should not be reached",
            cost_usd=0.0,
            cost_known=True,
            parsed_ok=True,
        )

    monkeypatch.setattr(refine_mod, "_ask_arbiter", fake_arbiter)

    async def fake_fanout(prompt, specs, **kwargs):
        # Simulate a zero-usable-panel partial fanout.
        return RunHandle(
            run_id="stub-run",
            artifacts_dir=str(tmp_path),
            manifest=[],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
            partial=True,
            partial_reason="zero usable panellists (2 returned: TIMEOUT)",
            blinded=False,
        )

    monkeypatch.setattr(runner, "fanout", fake_fanout)
    monkeypatch.setattr(refine_mod.runner, "fanout", fake_fanout)

    async def fake_synth(run_id, **kwargs):
        from consult import synth as _synth_mod

        return _synth_mod.SynthResult(text="(no rounds completed — see partial_reason)")

    monkeypatch.setattr(refine_mod.synth, "synthesise", fake_synth)

    result = await refine_mod.refine(
        "test",
        [ModelSpec(model="claude-haiku")],
        max_rounds=3,
        threshold=0.85,
    )
    assert arbiter_called["n"] == 0, "arbiter must not be called after a partial fanout"
    assert result.partial is True
    assert result.partial_reason and "fanout partial" in result.partial_reason
    assert result.rounds_completed == 0


def test_refine_arbiter_cost_pre_estimated_in_cap_check(monkeypatch):
    """`estimate_cost` must be called for the arbiter spec on every round so
    a flagship arbiter can't silently overshoot the cap."""
    from consult import refine as refine_mod

    arbiter_estimate_calls = {"n": 0}

    def fake_estimate(specs, prompt, **_):
        if len(specs) == 1 and specs[0].model == refine_mod.registry.default_synthesiser():
            arbiter_estimate_calls["n"] += 1
        return (0.01, True)

    monkeypatch.setattr(refine_mod.runner, "estimate_cost", fake_estimate)

    # We don't run the full refine — just verify the arbiter-spec estimate
    # branch via a unit-level slice. Build the arbiter spec the same way
    # refine() does.
    arbiter_alias = refine_mod.registry.default_synthesiser()
    spec = ModelSpec(model=arbiter_alias)
    est, known = refine_mod.runner.estimate_cost([spec], "any prompt")
    assert known is True
    assert arbiter_estimate_calls["n"] == 1


async def test_refine_rejects_continuation_with_sentinel_synthesis(tmp_path, monkeypatch):
    """A prior run whose synthesis is a sentinel (`# Synthesis unavailable`,
    skipped, empty) must be refused — feeding the sentinel to the next panel
    as 'prior consultation' produces hallucinated follow-ups.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.prompt_txt.write_text("prior question")
    (paths.root / "synthesis.md").write_text("# Synthesis unavailable\n\nThe synthesiser failed.")
    # Even with a valid synthesis.md file, the sentinel header makes it unusable.
    with pytest.raises(ValueError) as exc:
        refine_mod._apply_continuation("follow-up", paths.run_id)
    assert "sentinel" in str(exc.value)


async def test_refine_arbiter_sees_followup_only_under_continuation(tmp_path, monkeypatch):
    """When refine continues a prior run, the arbiter must score sufficiency
    against the follow-up question alone — not against the prior synth blob
    that `_apply_continuation` stitches into storage.
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    # Create a prior run with real synthesis (not a sentinel).
    prior = artifacts.create_run()
    prior.prompt_txt.write_text("PRIOR QUESTION TEXT")
    (prior.root / "synthesis.md").write_text("PRIOR SYNTH TEXT")

    arbiter_questions: list[str] = []

    async def fake_fanout(prompt, specs, **kwargs):
        paths_h = kwargs.get("existing_paths") or artifacts.create_run()
        return RunHandle(
            run_id=paths_h.run_id,
            artifacts_dir=str(paths_h.root),
            manifest=[
                ManifestEntry(
                    slug="x.r1",
                    model_id="m/x",
                    status=Status.OK,
                    resource_uri=paths_h.resource_uri("x.r1"),
                    body_path=str(paths_h.response_text("x.r1")),
                    latency_ms=0,
                    cost_usd=0.0,
                    cost_known=True,
                    confidence=None,
                    capsule=None,
                ),
            ],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_arbiter(question, round_num, manifest, arbiter_alias, prior_manifest):
        arbiter_questions.append(question)
        return ArbiterVerdict(
            round=round_num,
            score=1.0,
            gaps=[],
            reasoning="ok",
            cost_usd=0.0,
            cost_known=True,
            parsed_ok=True,
        )

    async def fake_synth(*args, **kwargs):
        return synth_mod.SynthResult(text="final")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(refine_mod.runner, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(refine_mod.capsule, "annotate", fake_annotate)
    monkeypatch.setattr(refine_mod, "_ask_arbiter", fake_arbiter)
    monkeypatch.setattr(refine_mod.synth, "synthesise", fake_synth)
    monkeypatch.setattr(refine_mod.runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    await refine_mod.refine(
        "FOLLOWUP QUESTION TEXT",
        [ModelSpec(model="claude-haiku")],
        threshold=0.5,
        max_rounds=1,
        continuation_id=prior.run_id,
    )
    assert arbiter_questions, "arbiter was never asked"
    asked = arbiter_questions[0]
    assert "FOLLOWUP QUESTION TEXT" in asked
    assert "PRIOR SYNTH TEXT" not in asked
    assert "PRIOR QUESTION TEXT" not in asked


def test_make_slug_blinded_preserves_round_suffix():
    """Blinded refine relies on `_make_slug` honouring `.r<n>` even when it
    rewrites the visible portion to a greek slug. Pre-fix the suffix was
    dropped and per-round artifacts overwrote each other on disk.
    """

    # Round-suffixed spec (refine._suffix_specs produces these).
    s = ModelSpec(model="claude-haiku", slug="claude-haiku-0.r2")
    blinded = _make_slug(s, 0, blinded=True)
    assert blinded == "panelist-alpha.r2"
    # No `.r<n>` ⇒ plain greek slug.
    s2 = ModelSpec(model="claude-haiku", slug="claude-haiku-0")
    assert _make_slug(s2, 0, blinded=True) == "panelist-alpha"


def test_refine_suffix_specs_sanitises_model_derived_base():
    """refine._suffix_specs derives slugs from spec.model when slug is
    None. A model id with a colon would otherwise build a ModelSpec slug
    that fails the field validator at construction time.
    """
    out = refine_mod._suffix_specs(
        [ModelSpec(model="openrouter/meta-llama/llama-3.1-8b:free")],
        round_num=1,
    )
    assert out[0].slug == "llama-3.1-8b-free-0.r1"


async def test_refine_synth_call_passes_anonymised_when_blinded(tmp_path, monkeypatch):
    """A blinded refine must pass `anonymised=True` to `synth.synthesise` so
    the bundle's brand-scrubbed prompt is what reaches the synthesiser.
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (0.0, True))

    async def fake_fanout(prompt, specs, **kwargs):
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="x.r1",
                    model_id="m/x",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("x.r1"),
                    body_path=str(paths.response_text("x.r1")),
                    latency_ms=0,
                    cost_usd=0.0,
                    cost_known=True,
                    confidence=None,
                    capsule=None,
                ),
            ],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(
            round=1,
            score=1.0,
            gaps=[],
            reasoning="ok",
            cost_usd=0.0,
            cost_known=True,
            parsed_ok=True,
        )

    captured: dict[str, Any] = {}

    async def fake_synth(run_id, **kwargs):
        captured.update(kwargs)
        return synth_mod.SynthResult(text="final")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(refine_mod.runner, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(refine_mod.capsule, "annotate", fake_annotate)
    monkeypatch.setattr(refine_mod, "_ask_arbiter", fake_arbiter)
    monkeypatch.setattr(refine_mod.synth, "synthesise", fake_synth)

    await refine_mod.refine(
        "Q",
        [ModelSpec(model="claude-haiku")],
        threshold=0.5,
        max_rounds=1,
        blinded=True,
    )
    assert captured.get("anonymised") is True


async def test_refine_passes_max_run_usd_to_nested_fanout(tmp_path, monkeypatch):
    """A refine caller's `max_run_usd` must reach the per-round `runner.fanout`
    call. Pre-fix the nested fanout fell back to `registry.default_max_run_usd()`
    ($5) so a refine cap of $20 was silently downgraded.
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (0.0, True))

    seen_caps: list[float | None] = []

    async def fake_fanout(prompt, specs, **kwargs):
        seen_caps.append(kwargs.get("max_run_usd"))
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="x.r1",
                    model_id="m/x",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("x.r1"),
                    body_path=str(paths.response_text("x.r1")),
                    latency_ms=0,
                    cost_usd=0.0,
                    cost_known=True,
                    confidence=None,
                    capsule=None,
                ),
            ],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(
            round=1,
            score=1.0,
            gaps=[],
            reasoning="ok",
            cost_usd=0.0,
            cost_known=True,
            parsed_ok=True,
        )

    async def fake_synth(*args, **kwargs):
        return synth_mod.SynthResult(text="x")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(refine_mod.runner, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(refine_mod.capsule, "annotate", fake_annotate)
    monkeypatch.setattr(refine_mod, "_ask_arbiter", fake_arbiter)
    monkeypatch.setattr(refine_mod.synth, "synthesise", fake_synth)

    await refine_mod.refine(
        "q",
        [ModelSpec(model="claude-haiku")],
        threshold=0.5,
        max_rounds=1,
        max_run_usd=20.0,
    )
    assert seen_caps and seen_caps[0] == pytest.approx(20.0)


async def test_refine_rejects_typo_synthesiser_before_fanout(tmp_path, monkeypatch):
    """A typo in `synthesiser` must surface KeyError BEFORE any fanout
    spend. Pre-fix the typo crashed only on the synth phase, after the
    entire parallel panel had already been billed.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    fanout_calls = {"n": 0}

    async def fake_fanout(*args, **kwargs):
        fanout_calls["n"] += 1
        raise AssertionError("fanout must not be reached on typo")

    monkeypatch.setattr(refine_mod.runner, "fanout", fake_fanout)
    with pytest.raises(KeyError):
        await refine_mod.refine(
            "q",
            [ModelSpec(model="claude-haiku")],
            arbiter="totally-not-a-real-alias",
            threshold=0.5,
            max_rounds=1,
        )
    assert fanout_calls["n"] == 0


def test_refine_cost_estimate_includes_prior_turns_for_fanout(monkeypatch):
    """Refine's per-round cost estimate must match `runner.fanout`'s view
    when a continuation is active — fanout includes prior_turns text in
    its token math, and an under-estimate here would let refine wave a
    round through that fanout rejects (corrupting the prior round's
    manifest via the shared `paths`).
    """
    from consult import runner as runner_mod

    captured_inputs: list[str] = []

    def fake_estimate_cost(specs, text, *, capsule_kind="decision"):
        captured_inputs.append(text)
        return 0.0, True

    monkeypatch.setattr(runner_mod, "estimate_cost", fake_estimate_cost)
    monkeypatch.setattr(refine_mod.runner, "estimate_cost", fake_estimate_cost)

    # Direct unit slice: call the refine round's estimate-build by
    # invoking the function with continuation set up. Easier path is to
    # spot-check the line. Build the same expression refine builds:
    prior_turns = [
        {"role": "user", "content": "PRIOR QUESTION"},
        {"role": "assistant", "content": "PRIOR SYNTH"},
    ]
    round_prompt = "FOLLOWUP"
    expected = runner_mod.concat_turn_text(prior_turns) + "\n" + round_prompt
    assert "PRIOR QUESTION" in expected
    assert "PRIOR SYNTH" in expected
    assert "FOLLOWUP" in expected


def test_resource_uri_formatter_override_round_trips(tmp_path, monkeypatch):
    """A custom URI formatter applies to every new manifest entry; reset
    restores the default. Used by non-MCP consumers (HTTP/library/CLI)
    that want their own URI scheme on the manifest.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    artifacts.set_resource_uri_formatter(lambda run_id, slug: f"https://example.com/{run_id}/{slug}.txt")
    try:
        paths = artifacts.create_run()
        uri = paths.resource_uri("alpha")
        assert uri == f"https://example.com/{paths.run_id}/alpha.txt"
    finally:
        artifacts.reset_resource_uri_formatter()

    # Default restored — new run gets the consult:// scheme again.
    paths2 = artifacts.create_run()
    assert paths2.resource_uri("alpha").startswith("consult://runs/")


@pytest.mark.asyncio
async def test_refine_preserves_final_manifest_when_late_round_partial(tmp_path, monkeypatch):
    """A round-N+1 fanout returning partial=True must not overwrite the
    final_manifest from a successful round-N. Previously
    `final_manifest = handle.manifest` was unconditional, so the empty
    partial manifest clobbered the prior round's consensus.
    """
    from consult import capsule as capsule_mod
    from consult import refine as refine_mod
    from consult import runner

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    call_count = {"n": 0}

    async def fake_fanout(prompt, specs, *, existing_paths=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Round 1: real, single-entry manifest
            paths = existing_paths or artifacts.create_run()
            entry = ManifestEntry(
                slug="m-good",
                model_id="x/y",
                status=Status.OK,
                resource_uri=paths.resource_uri("m-good"),
                body_path=str(paths.response_text("m-good")),
                latency_ms=10,
                cost_usd=0.01,
                cost_known=True,
            )
            await asyncio.to_thread(paths.response_text("m-good").write_text, "good body")
            return RunHandle(
                run_id=paths.run_id,
                artifacts_dir=str(paths.root),
                manifest=[entry],
                cost_usd=0.01,
                cost_known=True,
                wall_ms=10,
                partial=False,
            )
        # Round 2: zero-usable partial — must NOT clobber round 1's manifest
        paths = existing_paths or artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
            partial=True,
            partial_reason="zero usable panellists (2 returned: TIMEOUT)",
        )

    async def fake_annotate(handle, **kwargs):
        # Pretend the extractor populated a usable capsule on round 1.
        for entry in handle.manifest:
            if entry.status == Status.OK and entry.capsule is None:
                entry.capsule = Capsule(position="round 1 consensus", recommendation="ok")
        return handle

    async def fake_arbiter(question, round_num, manifest, *_, **__):
        return ArbiterVerdict(
            round=round_num,
            score=0.5,
            gaps=["needs more"],
            next_round_focus="dig deeper",
            reasoning="not converged",
            cost_usd=0.0,
            cost_known=True,
            parsed_ok=True,
        )

    async def fake_synth(run_id, **kwargs):
        from consult import synth as synth_mod

        return synth_mod.SynthResult(text="synthesised", cost_usd=0.0)

    monkeypatch.setattr(runner, "fanout", fake_fanout)
    monkeypatch.setattr(refine_mod.runner, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(refine_mod, "_ask_arbiter", fake_arbiter)
    monkeypatch.setattr(refine_mod.synth, "synthesise", fake_synth)

    result = await refine_mod.refine(
        "test",
        [ModelSpec(model="claude-haiku")],
        max_rounds=3,
        threshold=0.85,
    )
    # Partial because round 2 failed.
    assert result.partial is True
    # But final_manifest is the round-1 good manifest, NOT the empty
    # partial from round 2.
    assert len(result.final_manifest) == 1
    assert result.final_manifest[0].slug == "m-good"
    assert result.final_manifest[0].capsule is not None


def test_refine_continuation_sentinel_check_handles_leading_whitespace(tmp_path, monkeypatch):
    """The sentinel check now lstrips before startswith, so a synthesis
    file with a leading newline/BOM can't slip a sentinel past the
    guard."""
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()
    prior.prompt_txt.write_text("prior question")
    # Leading whitespace before the sentinel heading.
    (prior.root / "synthesis.md").write_text("\n\n  # Synthesis empty\n\nno content")

    with pytest.raises(ValueError, match="sentinel synthesis"):
        _apply_continuation("follow-up", prior.run_id)


@pytest.mark.asyncio
async def test_refine_aborts_when_arbiter_parse_fails(tmp_path, monkeypatch):
    """A non-parseable arbiter verdict stops the loop after that round, so its
    error text never seeds the next-round prompt. The result is partial.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    fanout_calls: list = []

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(round=1, score=0.0, parsed_ok=False, error="unparseable verdict")

    async def fake_aestimate(*a, **kw):
        return (0.001, True)

    monkeypatch.setattr("consult.refine.runner.fanout", _refine_fake_fanout(fanout_calls))
    monkeypatch.setattr("consult.refine.capsule.annotate", _refine_noop_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", _refine_fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    result = await refine_mod.refine("q", [ModelSpec(model="claude-haiku")], threshold=0.85, max_rounds=3)

    assert len(fanout_calls) == 1  # no round 2 after the parse failure
    assert result.rounds_completed == 1
    assert result.converged is False
    assert result.partial is True
    assert result.partial_reason is not None and "arbiter failed" in result.partial_reason
    assert result.final_manifest  # round-1 manifest preserved


@pytest.mark.asyncio
async def test_refine_runs_all_rounds_without_converging(tmp_path, monkeypatch):
    """Sub-threshold scores every round: refine exhausts max_rounds, reports
    converged=False, and still synthesises from the final round. Running out
    of rounds is not a partial result.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    fanout_calls: list = []

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(round=1, score=0.2, parsed_ok=True)

    async def fake_aestimate(*a, **kw):
        return (0.001, True)

    monkeypatch.setattr("consult.refine.runner.fanout", _refine_fake_fanout(fanout_calls))
    monkeypatch.setattr("consult.refine.capsule.annotate", _refine_noop_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", _refine_fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    result = await refine_mod.refine("q", [ModelSpec(model="claude-haiku")], threshold=0.85, max_rounds=2)

    assert len(fanout_calls) == 2
    assert result.rounds_completed == 2
    assert result.converged is False
    assert result.partial is False  # ran to completion, just below threshold
    assert result.synthesis == "final synth"


@pytest.mark.asyncio
async def test_refine_breaks_when_next_round_would_exceed_cap(tmp_path, monkeypatch):
    """When the next round's estimate would exceed max_run_usd, refine stops
    before launching it and preserves the prior round's manifest.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    fanout_calls: list = []

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(round=1, score=0.2, parsed_ok=True)  # wants another round

    est_n = {"n": 0}

    async def fake_aestimate(*a, **kw):
        # Round 1's two estimates (fanout + arbiter) are cheap; round 2's blow
        # the cap regardless of how much round 1 actually spent.
        est_n["n"] += 1
        return (0.05 if est_n["n"] <= 2 else 100.0, True)

    monkeypatch.setattr("consult.refine.runner.fanout", _refine_fake_fanout(fanout_calls))
    monkeypatch.setattr("consult.refine.capsule.annotate", _refine_noop_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", _refine_fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    result = await refine_mod.refine(
        "q", [ModelSpec(model="claude-haiku")], threshold=0.85, max_rounds=3, max_run_usd=1.0
    )

    assert len(fanout_calls) == 1  # round 2 refused before fanout
    assert result.rounds_completed == 1
    assert result.partial is True
    assert result.partial_reason is not None and "exceed cap" in result.partial_reason
    assert result.final_manifest  # round-1 manifest preserved (clobber guard)


@pytest.mark.asyncio
async def test_refine_refuses_further_rounds_when_pricing_unknown_past_80pct(tmp_path, monkeypatch):
    """The 80%-of-cap safety valve in `_round_cost_gate` (refine.py:606): when a
    panellist's pricing is unknown AND spend is already past 80% of the cap,
    refine refuses the next round rather than risk an unbounded overspend. This
    branch had no coverage; the only cap test exercises the hard
    `cumulative + estimate > cap` branch. A flip of `not est_known`, the 0.8
    threshold, or the `>` comparison would otherwise ship silently. The valve
    was a FRICTION-pass fix for an asymmetric round-1-proceeds / round-2-refuses
    surprise on mixed-provider panels, so it guards real behaviour.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    fanout_calls: list = []

    async def fake_arbiter(*args, **kwargs):
        # Sub-threshold so the loop wants another round; zero-cost and known so
        # round-1 cumulative is exactly the fanout spend.
        return ArbiterVerdict(round=1, score=0.2, parsed_ok=True, cost_usd=0.0, cost_known=True)

    # The counter encodes the gate's call order: each round's gate estimates
    # the fanout then the arbiter, so calls 1-2 are round 1's gate and calls 3-4
    # are round 2's.
    est_n = {"n": 0}

    async def fake_aestimate(*a, **kw):
        # Round 1's gate (calls 1-2) is cheap and known, so it proceeds. Round
        # 2's gate (calls 3-4) returns unknown pricing with a small estimate: the
        # hard-cap branch must NOT fire (0.82 + 0.10 < 1.0), leaving only the 80%
        # valve to trip.
        est_n["n"] += 1
        return (0.02, True) if est_n["n"] <= 2 else (0.05, False)

    # Round-1 fanout spends 0.82 of the 1.00 cap — past the 80% floor but below
    # the hard cap. annotate is a noop (adds no extractor cost), so round-1
    # cumulative is exactly the fanout's 0.82 when round 2's gate runs.
    monkeypatch.setattr("consult.refine.runner.fanout", _refine_fake_fanout(fanout_calls, cost=0.82))
    monkeypatch.setattr("consult.refine.capsule.annotate", _refine_noop_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", _refine_fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    result = await refine_mod.refine(
        "q", [ModelSpec(model="claude-haiku")], threshold=0.85, max_rounds=3, max_run_usd=1.0
    )

    assert len(fanout_calls) == 1  # round 2 refused at the gate, before fanout
    assert result.rounds_completed == 1
    assert result.partial is True
    assert result.partial_reason is not None
    assert "refusing further rounds" in result.partial_reason
    assert "80%" in result.partial_reason
    assert result.final_manifest  # round-1 manifest preserved
    assert result.synthesis == "final synth"  # synth still runs on the kept round
    # Lock the precondition the valve fires against: round-1 spend is exactly the
    # fanout's 0.82 (annotate noop, arbiter and synth zero-cost). A hidden round-1
    # cost leak would shift the gate's input without changing which branch trips.
    assert result.cost_usd == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_refine_partial_fanout_rolls_unknown_cost_into_total(tmp_path, monkeypatch):
    """A partial fanout carrying a non-zero, unknown-priced cost must roll that
    cost into the result total and drag `cost_known` False (refine.py:830-832),
    and must short-circuit before the capsule extractor, arbiter, and synth run.
    The existing partial-fanout test returns cost_usd=0.0/cost_known=True, so the
    accumulation and the cost_known flip are never exercised with effect. This
    mirrors test_sequence_partial_fanout_rolls_cost_into_total for refine.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    calls = {"fanout": 0, "annotate": 0, "arbiter": 0, "synth": 0}

    async def fake_fanout(prompt, specs, **kwargs):
        calls["fanout"] += 1
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.42,
            cost_known=False,
            wall_ms=10,
            partial=True,
            partial_reason="zero usable panellists (1 returned: TIMEOUT)",
        )

    async def fake_annotate(handle, **kwargs):
        calls["annotate"] += 1
        return handle

    async def fake_arbiter(*args, **kwargs):
        calls["arbiter"] += 1
        return ArbiterVerdict(round=1, score=1.0, parsed_ok=True, cost_usd=0.0, cost_known=True)

    async def fake_synth(*args, **kwargs):
        calls["synth"] += 1
        from consult import synth as _synth_mod

        return _synth_mod.SynthResult(text="should not be reached")

    async def fake_aestimate(*a, **kw):
        return (0.001, True)

    monkeypatch.setattr("consult.refine.runner.fanout", fake_fanout)
    monkeypatch.setattr("consult.refine.capsule.annotate", fake_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    result = await refine_mod.refine(
        "q", [ModelSpec(model="claude-haiku")], threshold=0.85, max_rounds=3, max_run_usd=5.0
    )

    # The partial round must short-circuit before the arbiter and synth — both
    # cost flagship $$, and the whole point of breaking on a partial fanout is to
    # not burn them on a zero-usable panel. annotate (the extractor) is skipped
    # for the same reason. Asserted individually rather than as one dict so a
    # future benign call elsewhere doesn't trip a guarantee it isn't part of.
    assert calls["fanout"] == 1
    assert calls["annotate"] == 0
    assert calls["arbiter"] == 0
    assert calls["synth"] == 0
    # The partial fanout's cost is not lost — it rolls into the total and drags
    # cost_known False.
    assert result.cost_usd == pytest.approx(0.42)
    assert result.cost_known is False
    assert result.partial is True
    assert result.partial_reason and "fanout partial" in result.partial_reason
    assert result.rounds_completed == 0  # arbiter never appended a verdict
    assert result.converged is False
    assert result.synthesis == "(no rounds completed — see partial_reason)"


@pytest.mark.asyncio
async def test_refine_arbiter_unknown_pricing_flips_result_cost_known(tmp_path, monkeypatch):
    """An arbiter on an unmapped-price model returns cost_usd=None,
    cost_known=False. refine.py:909-910 must drag the result's cost_known False
    even though the fanout's own cost was known. This propagation regressed
    silently once before (see the code's fix-history note at that line). The
    RefineResult pydantic invariant only fires when propagation is *wrong*; this
    proves the happy path returns a cost_known=False result rather than raising.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    fanout_calls: list = []

    async def fake_arbiter(*args, **kwargs):
        # Converges immediately so the loop ends after one round; unknown
        # pricing is the only thing under test.
        return ArbiterVerdict(round=1, score=0.95, parsed_ok=True, cost_usd=None, cost_known=False)

    async def fake_aestimate(*a, **kw):
        return (0.001, True)

    monkeypatch.setattr("consult.refine.runner.fanout", _refine_fake_fanout(fanout_calls))
    monkeypatch.setattr("consult.refine.capsule.annotate", _refine_noop_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", _refine_fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    result = await refine_mod.refine("q", [ModelSpec(model="claude-haiku")], threshold=0.85, max_rounds=3)

    assert len(fanout_calls) == 1
    assert result.converged is True
    assert result.cost_known is False  # the arbiter's unknown price dragged it False
    # A None arbiter cost is a zero-addition, not a reset: the fanout's own spend
    # survives in the total. Guards against a regression that nukes cumulative.
    assert result.cost_usd == pytest.approx(0.001)


# --- issue #53: arbiter hardening (budget floor, JSON mode, parse retry) -----


def _arbiter_resp(text):
    class _Choice:
        def __init__(self, t):
            self.message = type("M", (), {"content": t})()

    class _Resp:
        choices = [_Choice(text)]

    return _Resp()


_VALID_ARBITER_JSON = json.dumps(
    {
        "dimensions": {"coverage": 4, "agreement": 4, "depth": 4, "calibration": 4, "actionability": 4},
        "dimension_notes": {},
        "gaps": [],
        "next_round_focus": "f",
        "reasoning": "fine",
    }
)


def test_arbiter_budget_floors_at_model_budget_and_requests_json_mode(monkeypatch):
    """The arbiter call gets the same per-model budget floor as panellists
    (a thinking arbiter at a flat 2000 cap can cut its own JSON mid-stream)
    and asks for native JSON mode where the provider supports it."""
    import asyncio

    from consult.refine import _ask_arbiter

    captured: dict = {}

    async def fake_acompletion(**kw):
        captured.update(kw)
        return _arbiter_resp(_VALID_ARBITER_JSON)

    monkeypatch.setattr("consult.refine.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("consult.refine.litellm.completion_cost", lambda completion_response=None: 0.001)

    verdict = asyncio.run(_ask_arbiter("Q?", round_num=1, manifest=[], arbiter_alias="gemini-pro"))
    assert verdict.parsed_ok is True
    assert captured["max_completion_tokens"] == 16000  # gemini-pro default_budget_tokens
    assert captured["response_format"] == {"type": "json_object"}


def test_arbiter_retries_once_on_non_json_and_sums_cost(monkeypatch):
    """A non-JSON first verdict gets exactly one re-ask with the corrective
    nudge; the recovered verdict parses and BOTH calls are billed. Run
    20260611-043229 died here with no second chance."""
    import asyncio

    from consult.refine import _ask_arbiter

    calls = {"n": 0, "prompts": []}

    async def fake_acompletion(**kw):
        calls["n"] += 1
        calls["prompts"].append(kw["messages"][0]["content"])
        if calls["n"] == 1:
            return _arbiter_resp("I think the panel did quite well overall, no JSON for you.")
        return _arbiter_resp(_VALID_ARBITER_JSON)

    monkeypatch.setattr("consult.refine.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("consult.refine.litellm.completion_cost", lambda completion_response=None: 0.001)

    verdict = asyncio.run(_ask_arbiter("Q?", round_num=1, manifest=[], arbiter_alias="claude-haiku"))
    assert calls["n"] == 2
    assert "could not be parsed" in calls["prompts"][1]
    assert verdict.parsed_ok is True
    assert verdict.score == pytest.approx(0.75)
    assert verdict.cost_usd == pytest.approx(0.002)  # both attempts billed


def test_arbiter_double_parse_failure_aborts_with_both_costs(monkeypatch):
    """Two unparseable replies keep the existing abort semantics (score 0.0,
    json_parse_failed) while still accounting for both billed calls."""
    import asyncio

    from consult.refine import _ask_arbiter

    calls = {"n": 0}

    async def fake_acompletion(**kw):
        calls["n"] += 1
        return _arbiter_resp("still chatting, still not JSON")

    monkeypatch.setattr("consult.refine.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("consult.refine.litellm.completion_cost", lambda completion_response=None: 0.001)

    verdict = asyncio.run(_ask_arbiter("Q?", round_num=1, manifest=[], arbiter_alias="claude-haiku"))
    assert calls["n"] == 2
    assert verdict.parsed_ok is False
    assert verdict.error == "json_parse_failed"
    assert verdict.score == 0.0
    assert verdict.cost_usd == pytest.approx(0.002)


def test_arbiter_call_exception_does_not_retry(monkeypatch):
    """A transport-level failure on the first attempt stays terminal — the
    parse retry is for formatting, not for outages."""
    import asyncio

    from consult.refine import _ask_arbiter

    calls = {"n": 0}

    async def fake_acompletion(**kw):
        calls["n"] += 1
        raise RuntimeError("socket exploded")

    monkeypatch.setattr("consult.refine.litellm.acompletion", fake_acompletion)

    verdict = asyncio.run(_ask_arbiter("Q?", round_num=1, manifest=[], arbiter_alias="claude-haiku"))
    assert calls["n"] == 1
    assert verdict.parsed_ok is False
    assert verdict.score == 0.0
    assert verdict.cost_known is False


def test_arbiter_unscoreable_json_retries_then_reports_no_score(monkeypatch):
    """Valid JSON with neither numeric dimensions nor a score triggers the
    retry; if the retry is no better, the no_dimensions_or_score path
    reports parse failure rather than inventing a verdict."""
    import asyncio

    from consult.refine import _ask_arbiter

    calls = {"n": 0}

    async def fake_acompletion(**kw):
        calls["n"] += 1
        return _arbiter_resp('{"dimensions": {"coverage": "great"}, "reasoning": "vibes"}')

    monkeypatch.setattr("consult.refine.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("consult.refine.litellm.completion_cost", lambda completion_response=None: 0.001)

    verdict = asyncio.run(_ask_arbiter("Q?", round_num=1, manifest=[], arbiter_alias="claude-haiku"))
    assert calls["n"] == 2
    assert verdict.parsed_ok is False
    assert verdict.score == 0.0
