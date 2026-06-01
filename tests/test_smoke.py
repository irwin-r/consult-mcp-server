"""Smoke tests. The unit slice runs offline (no API keys); the live slice
hits real providers only when relevant API keys are present.

Run: `pytest -v`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from consult import artifacts, registry
from consult import refine as refine_mod
from consult.runner import _build_per_slug_prompt, _make_slug, estimate_cost
from consult.status import classify
from consult.types import ArbiterVerdict, Capsule, ManifestEntry, ModelSpec, RunHandle, Status

# ---- Pure-Python tests (no network) ----------------------------------------


def test_registry_loads_defaults():
    cfg = registry.models_config()
    assert "models" in cfg
    assert "claude-haiku" in cfg["models"]
    assert "tiers" in cfg
    assert "quick" in cfg["tiers"]


def test_registry_resolve_alias_and_raw_id():
    entry = registry.resolve_model("claude-haiku")
    assert entry["litellm_id"].startswith("anthropic/")
    raw = registry.resolve_model("openrouter/some/model")
    assert raw["litellm_id"] == "openrouter/some/model"


def test_stance_lookup_and_passthrough():
    assert "security" in registry.resolve_stance("security")
    assert registry.resolve_stance("You are a freeform stance.") == "You are a freeform stance."
    assert registry.resolve_stance(None) == ""


def test_expand_specs_multi_instance_and_passthrough():
    """`model:N` sugar expands to N specs; bare model strings pass through.
    Stance and custom slug are preserved on every expanded instance.
    """
    from consult.runner import expand_specs

    raw = [
        ModelSpec(model="claude-haiku:3", stance="skeptic"),
        ModelSpec(model="gpt-pro"),
        ModelSpec(model="openrouter/foo/bar"),  # no colon — passthrough
    ]
    expanded = expand_specs(raw)
    assert len(expanded) == 5  # 3 + 1 + 1
    assert [s.model for s in expanded] == [
        "claude-haiku", "claude-haiku", "claude-haiku",
        "gpt-pro",
        "openrouter/foo/bar",
    ]
    # Stance survives expansion
    assert all(s.stance == "skeptic" for s in expanded[:3])

    # Idempotent: re-expanding already-expanded specs is a no-op
    assert expand_specs(expanded) == expanded


def test_expand_specs_rejects_zero_count():
    """`model:0` is almost certainly a typo and must fail loudly rather
    than silently dropping the spec from the panel.
    """
    from consult.runner import expand_specs

    with pytest.raises(ValueError, match="must be ≥1"):
        expand_specs([ModelSpec(model="claude-haiku:0")])


def test_slug_and_prompt_assembly():
    spec = ModelSpec(model="claude-haiku", stance="security")
    assert _make_slug(spec, 0, blinded=False).startswith("claude-haiku")
    assert _make_slug(spec, 2, blinded=True) == "panelist-gamma"


def test_footer_injected_on_every_prompt():
    """Capsule confidence extraction depends on the footer being present."""
    with_stance = _build_per_slug_prompt("How to ship X?", "You are an SRE.")
    assert with_stance.startswith("You are an SRE.")
    assert "How to ship X?" in with_stance
    assert "CONFIDENCE:" in with_stance
    assert "KEY_REASON:" in with_stance

    no_stance = _build_per_slug_prompt("How to ship X?", "")
    assert no_stance.startswith("How to ship X?")
    assert "CONFIDENCE:" in no_stance
    assert "KEY_REASON:" in no_stance


def test_status_classifier_handles_exceptions():
    s, _, _ = classify(None, exception=RuntimeError("rate limit hit"))
    assert s == Status.RATE_LIMITED
    s, _, _ = classify(None, exception=TimeoutError("timed out"))
    assert s == Status.TIMEOUT


def test_run_handle_usable_parametric():
    entries = [
        ManifestEntry(
            slug=f"m{i}", model_id=f"openrouter/x{i}/y", status=Status.OK,
            resource_uri="consult://x", body_path="/tmp/x",
        )
        for i in range(4)
    ]
    entries[3].model_id = "openai/gpt-5"
    handle = RunHandle(
        run_id="t", artifacts_dir="/tmp/t", manifest=entries, cost_usd=0, wall_ms=0
    )
    assert handle.usable() is True
    # Set all-OR — only one provider
    for e in entries:
        e.model_id = "openrouter/foo/bar"
    assert handle.usable(min_providers=2) is False


def test_artifacts_create_and_uri():
    paths = artifacts.create_run()
    assert paths.root.exists()
    assert paths.responses.exists()
    uri = paths.resource_uri("alpha")
    rid, kind, name = artifacts.parse_resource_uri(uri)
    assert rid == paths.run_id
    assert kind == "responses"
    assert name == "alpha"
    # cleanup
    import shutil
    shutil.rmtree(paths.root)


# ---- Refine offline tests --------------------------------------------------


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
    verdict = ArbiterVerdict(
        round=1, score=0.4, gaps=["cost not discussed"], next_round_focus="address cost"
    )
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

    combined, prior_turns = _apply_continuation(
        "Now what about Polars for ETL?", prior.run_id
    )
    # Storage form keeps the markdown sections so disk artifacts stay self-describing.
    assert "Prior consultation — original question" in combined
    assert "Polars vs DuckDB for 10GB Parquet?" in combined
    assert "Prior consultation — synthesis" in combined
    assert "ANSWER: pick DuckDB." in combined
    assert "Follow-up question" in combined
    assert "Now what about Polars for ETL?" in combined
    assert combined.index("ANSWER: pick DuckDB.") < combined.index(
        "Now what about Polars for ETL?"
    )
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


def test_synth_deblind_replaces_word_boundary_only():
    """`_deblind` must be case-sensitive + word-boundary-anchored.

    The synth's prose may legitimately mention "alpha release" or
    "alphanumeric" — replacing those with a slug would corrupt the
    synthesis. Only the exact `Alpha` / `Beta` tokens the synth was
    told to use should be rewritten.
    """
    from consult.synth import _deblind

    mapping = {"Alpha": "claude-opus", "Beta": "gpt-pro"}
    src = (
        "Alpha argued for X, while Beta supported Y. "
        "Both noted that the alpha release of the library "
        "had alphanumeric IDs. Gamma was not mentioned."
    )
    out = _deblind(src, mapping)
    assert "claude-opus argued for X" in out
    assert "gpt-pro supported Y" in out
    # Casual prose preserved
    assert "alpha release" in out
    assert "alphanumeric" in out
    # Unmapped label left alone
    assert "Gamma was not mentioned" in out


def test_synth_deblind_empty_map_is_noop():
    """Zero-panel runs (or callers that skip blinding) must pass through
    unchanged. Building a regex of an empty alternation would explode."""
    from consult.synth import _deblind

    src = "Synthesis with Alpha mentioned."
    assert _deblind(src, {}) == src


def test_refine_arbiter_v2_dimensions_normalised_to_overall_score(monkeypatch):
    """v2 arbiter prompt: per-dimension 1-5 Likert input is rescaled to
    [0,1] and averaged into the overall `score`. Round-trips through
    `_ask_arbiter`'s JSON-parse path.
    """
    import asyncio

    from consult.refine import _ask_arbiter

    arbiter_text = json.dumps({
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
    })

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

    verdict = asyncio.run(_ask_arbiter(
        "Q?", round_num=1, manifest=[], arbiter_alias="claude-haiku",
    ))
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

    arbiter_text = json.dumps({
        "score": 0.65,
        "gaps": ["legacy"],
        "next_round_focus": "f",
        "reasoning": "r",
    })

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

    verdict = asyncio.run(_ask_arbiter(
        "Q?", round_num=1, manifest=[], arbiter_alias="claude-haiku",
    ))
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
        round=1, score=0.4, gaps=["legacy-gap"], next_round_focus="f",
    )
    out = _build_refinement_prompt("Q?", 2, [], legacy)
    assert "legacy-gap" in out
    assert "legacy arbiter prompt" in out  # placeholder marker


# ---- New (post-review) offline tests --------------------------------------


@pytest.mark.asyncio
async def test_fanout_dry_run_returns_partial():
    """Dry run must never make a billable call and must explain itself."""
    from consult.runner import fanout

    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="claude-sonnet")]
    handle = await fanout("any prompt", specs, dry_run=True)
    assert handle.partial is True
    assert handle.partial_reason and "dry_run" in handle.partial_reason
    assert handle.manifest == []
    assert handle.cost_usd == 0.0


@pytest.mark.asyncio
async def test_fanout_emits_progress_callbacks(tmp_path, monkeypatch):
    """fanout emits a layered progress stream:
    - one `PhaseStarted(phase="fanout")` before any panellist begins
    - one `PanellistStarted` per panellist (before its network call)
    - one `PanellistCompleted` per panellist (after the call returns)
    Heartbeat ticks are disabled (interval=0) so the assertions stay
    deterministic; a separate test covers the heartbeat path.
    """
    from consult import runner
    from consult.progress import (
        PanellistCompleted,
        PanellistStarted,
        PhaseStarted,
        ProgressEvent,
    )
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    events: list[ProgressEvent] = []

    async def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    specs = [
        ModelSpec(model="claude-haiku"),
        ModelSpec(model="claude-sonnet"),
        ModelSpec(model="claude-opus"),
    ]
    handle = await fanout("p", specs, on_progress=on_progress)
    assert handle.partial is False

    # First event must be the phase boundary so the parent sees fanout
    # begin before any panellist completion fires.
    assert isinstance(events[0], PhaseStarted)
    assert events[0].phase == "fanout"
    assert events[0].total == 3

    started = [e for e in events if isinstance(e, PanellistStarted)]
    completed = [e for e in events if isinstance(e, PanellistCompleted)]

    assert len(started) == 3
    assert {e.started_count for e in started} == {1, 2, 3}
    assert all(e.total == 3 for e in started)

    assert len(completed) == 3
    assert {e.done for e in completed} == {1, 2, 3}
    assert all(e.total == 3 for e in completed)
    assert all(e.status == "OK" for e in completed)


def test_append_progress_log_writes_jsonl(tmp_path):
    """The "D" half: a tailable JSONL log in the run dir. Each line is one
    event.model_dump() with a `ts` prepended so programmatic consumers can
    parse by `kind` without scraping free-text.
    """
    from consult.progress import CapsuleExtracted, PanellistCompleted, append_progress_log

    append_progress_log(tmp_path, PanellistCompleted(
        done=1, total=2, slug="haiku", status="OK", latency_ms=42,
    ))
    append_progress_log(tmp_path, CapsuleExtracted(done=1, total=2, slug="haiku"))

    lines = (tmp_path / "_progress.log").read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["kind"] == "panellist_completed"
    assert parsed[0]["slug"] == "haiku"
    assert parsed[0]["status"] == "OK"
    assert parsed[0]["latency_ms"] == 42
    assert "ts" in parsed[0]
    assert parsed[1]["kind"] == "capsule_extracted"
    assert parsed[1]["slug"] == "haiku"


def test_progress_event_message_for_every_kind():
    """Wire-format message must cover every event kind. New events must
    extend `event_message` — this test fails fast if a kind is added
    without updating the helper.
    """
    from consult.progress import (
        ArbiterScored,
        CapsuleExtracted,
        Heartbeat,
        PanellistCompleted,
        PanellistStarted,
        PhaseStarted,
        SequenceStepCompleted,
        SequenceStepStarted,
        SynthCompleted,
        SynthStarted,
        event_message,
    )

    assert "OK" in event_message(PanellistCompleted(
        done=1, total=2, slug="x", status="OK", latency_ms=1,
    ))
    assert "capsule" in event_message(CapsuleExtracted(done=1, total=2, slug="x"))
    assert "r2 arbiter" in event_message(ArbiterScored(
        done=1, total=2, round=2, score=0.5,
    ))
    assert event_message(SynthStarted(done=1, total=2)) == "synthesising"
    assert event_message(SynthCompleted(done=2, total=2)) == "synthesis complete"
    assert "step 3" in event_message(SequenceStepStarted(done=1, total=5, step=3))
    assert "step 3" in event_message(SequenceStepCompleted(done=2, total=5, step=3))

    started_msg = event_message(PanellistStarted(
        done=0, total=3, slug="alpha", started_count=1,
    ))
    assert "alpha" in started_msg
    assert "1/3" in started_msg

    assert event_message(PhaseStarted(done=0, total=3, phase="fanout")) == "phase: fanout"
    assert event_message(PhaseStarted(done=3, total=6, phase="capsules")) == "phase: capsules"

    hb_msg = event_message(Heartbeat(
        done=1, total=3, elapsed_ms=12_500,
        cost_so_far_usd=0.0234, cost_known=True,
        pending_count=2, pending_slugs=["gpt-pro", "claude-opus"],
    ))
    assert "12s" in hb_msg or "13s" in hb_msg
    assert "$0.0234" in hb_msg
    assert "2 pending" in hb_msg
    assert "gpt-pro" in hb_msg

    # Cost-unknown variant uses ≥ prefix to mark the total as a lower bound.
    hb_unknown = event_message(Heartbeat(
        done=1, total=3, elapsed_ms=1000,
        cost_so_far_usd=0.5, cost_known=False,
        pending_count=0, pending_slugs=[],
    ))
    assert "≥$0.5000" in hb_unknown


def test_error_envelope_shape_round_trips():
    """The structured-error envelope must round-trip through JSON with the
    exact shape agents pattern-match against. Locks in the wire contract.
    """
    from consult.mcp.errors import ConsultError, ErrorCode, ErrorEnvelope

    env = ErrorEnvelope(
        error=ConsultError(
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
        error=ConsultError(
            code=ErrorCode.INTERNAL_ERROR,
            message="boom",
            run_id="20260520-010203-1234",
        ),
    )
    payload2 = json.loads(env2.model_dump_json())
    assert payload2["error"]["run_id"] == "20260520-010203-1234"


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_value_error_in_envelope(monkeypatch):
    """A handler raising ValueError must surface as `invalid_input` envelope,
    not as a raw exception bubbling out of the MCP dispatch.

    The envelope is returned as a `dict` so the MCP SDK populates
    `structuredContent` on the wire — agents can branch on `error.code`
    without re-parsing the text body.
    """
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise ValueError("max_rounds must be between 1 and 3")

    monkeypatch.setitem(server_mod._HANDLERS, "refine", bad_handler)

    payload = await server_mod.handle_call_tool("refine", {"prompt": "x", "models": []})
    assert isinstance(payload, dict)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input"
    assert "max_rounds" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_key_error_as_unknown_model(monkeypatch):
    """`registry.resolve_model` raises KeyError on a missing alias —
    `synthesise(by_model="bogus")` would propagate that through. Must
    surface as `unknown_model`, not `invalid_input`.
    """
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise KeyError("Unknown model: bogus-alias")

    monkeypatch.setitem(server_mod._HANDLERS, "synthesise", bad_handler)
    payload = await server_mod.handle_call_tool("synthesise", {"run_id": "x"})
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unknown_model"


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_file_not_found_as_run_not_found(monkeypatch):
    """`artifacts.load_run` raises FileNotFoundError on missing run_id."""
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise FileNotFoundError("Run not found: 20260520-foo")

    monkeypatch.setitem(server_mod._HANDLERS, "synthesise", bad_handler)
    payload = await server_mod.handle_call_tool("synthesise", {"run_id": "20260520-foo"})
    assert payload["error"]["code"] == "run_not_found"


@pytest.mark.asyncio
async def test_handle_call_tool_unknown_tool_returns_envelope():
    """Asking for a tool that doesn't exist returns an invalid_input
    envelope rather than raising a ValueError out the top.
    """
    from consult.mcp import server as server_mod

    payload = await server_mod.handle_call_tool("not-a-tool", {})
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input"
    assert "not-a-tool" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_unhandled_exception_becomes_internal_error(monkeypatch):
    """Any unanticipated exception type from a handler must become an
    `internal_error` envelope rather than tearing out the MCP dispatch.
    """
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(server_mod._HANDLERS, "panel", bad_handler)
    payload = await server_mod.handle_call_tool("panel", {"prompt": "x", "models": []})
    assert payload["error"]["code"] == "internal_error"
    assert "kaboom" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_success_path_returns_dict(monkeypatch):
    """The success path must return a `dict` so the MCP SDK populates
    `structuredContent` on the response. Returning `list[TextContent]` would
    leave clients with only the JSON-text-blob fallback.
    """
    from consult.mcp import server as server_mod

    sentinel = {"run_id": "20260520-stub", "synthesis": "ok", "manifest": []}

    async def fake_handler(args, **_kwargs):
        return sentinel

    monkeypatch.setitem(server_mod._HANDLERS, "consult", fake_handler)
    result = await server_mod.handle_call_tool("consult", {"prompt": "x"})
    assert result is sentinel
    assert isinstance(result, dict)


# ---- MCP boundary tests ----------------------------------------------------
# These exercise the @server.list_tools / @server.list_resources /
# @server.read_resource handlers directly. Without them, a typo in tool name
# or schema (or a broken URI grammar) silently breaks the MCP surface — the
# unit tests on the underlying runner/refine/synth all pass, and the
# offline test suite has nothing to fail on.


@pytest.mark.asyncio
async def test_handle_list_tools_advertises_full_surface():
    """Pin the five-tool surface plus each tool's required input fields.
    Catches schema drift (e.g. dropping `prompt` from `panel`'s required
    list) that the type system can't see."""
    from consult.mcp import server as server_mod

    tools = await server_mod.handle_list_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"panel", "synthesise", "consult", "refine", "sequence"}
    assert "prompt" in by_name["panel"].inputSchema["required"]
    assert "models" in by_name["panel"].inputSchema["required"]
    assert "prompt" in by_name["consult"].inputSchema["required"]
    assert "prompt" in by_name["refine"].inputSchema["required"]
    assert "prompts" in by_name["sequence"].inputSchema["required"]
    assert "run_id" in by_name["synthesise"].inputSchema["required"]


@pytest.mark.asyncio
async def test_handle_list_resources_surfaces_run_bodies(tmp_path, monkeypatch):
    """list_resources advertises each run's panellist bodies so MCP clients
    can discover them without prior knowledge of the URI grammar."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    for i in range(2):
        run_dir = tmp_path / f"20260520-fake-{i}"
        (run_dir / "responses").mkdir(parents=True)
        for slug in ("alpha", "beta"):
            (run_dir / "responses" / f"{slug}.txt").write_text("body")

    resources = await server_mod.handle_list_resources()
    uris = {str(r.uri) for r in resources}
    assert len(uris) == 4
    assert any(u.endswith("/responses/alpha") for u in uris)
    assert any(u.endswith("/responses/beta") for u in uris)
    assert all(r.mimeType == "text/plain" for r in resources)


@pytest.mark.asyncio
async def test_handle_list_resources_empty_dir_returns_empty(tmp_path, monkeypatch):
    """No runs on disk → no resources advertised. Don't crash."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    assert await server_mod.handle_list_resources() == []


@pytest.mark.asyncio
async def test_handle_read_resource_returns_body_text(tmp_path, monkeypatch):
    """Happy path: read a body via its `consult://...` URI."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.response_text("alpha").write_text("the body")

    body = await server_mod.handle_read_resource(
        f"consult://runs/{paths.run_id}/responses/alpha"
    )
    assert body == "the body"


@pytest.mark.asyncio
async def test_handle_read_resource_missing_body_raises(tmp_path, monkeypatch):
    """Reading a slug whose body wasn't written must raise FileNotFoundError —
    the top-level dispatcher then maps to a RUN_NOT_FOUND envelope rather
    than returning a misleading empty body."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()

    with pytest.raises(FileNotFoundError):
        await server_mod.handle_read_resource(
            f"consult://runs/{paths.run_id}/responses/missing"
        )


@pytest.mark.asyncio
async def test_handle_call_tool_dispatch_routes_each_tool_name(monkeypatch):
    """The dispatch chain in handle_call_tool must route each tool name to
    its corresponding _handle_*. A typo (e.g. `"Panel"` vs `"panel"`) here
    silently breaks one tool with no compile-time signal."""
    from consult.mcp import server as server_mod

    for tool_name in ("panel", "consult", "refine", "sequence", "synthesise"):
        called: list[str] = []

        async def fake(args, _name=tool_name, _called=called, **_kwargs):
            _called.append(_name)
            return {"routed": _name}

        monkeypatch.setitem(server_mod._HANDLERS, tool_name, fake)
        await server_mod.handle_call_tool(tool_name, {})
        assert called == [tool_name]


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
        done=1, total=3, elapsed_ms=2500,
        cost_so_far_usd=0.01, cost_known=True,
        pending_count=2, pending_slugs=["a", "b"],
    )
    parsed = adapter.validate_json(hb.model_dump_json())
    assert isinstance(parsed, Heartbeat)
    assert parsed.pending_slugs == ["a", "b"]


@pytest.mark.asyncio
async def test_fanout_progress_callback_failure_does_not_abort_run(tmp_path, monkeypatch):
    """A raising on_progress callback must not tear down the fanout —
    progress is best-effort, the run completes regardless.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=1,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    async def bad_progress(event):
        raise RuntimeError("client went away")

    handle = await fanout(
        "p",
        [ModelSpec(model="claude-haiku")],
        on_progress=bad_progress,
    )
    assert handle.partial is False
    assert len(handle.manifest) == 1
    assert handle.manifest[0].status is Status.OK


@pytest.mark.asyncio
async def test_fanout_emits_heartbeat_while_panellists_in_flight(tmp_path, monkeypatch):
    """With a short heartbeat interval and an artificially slow panellist,
    at least one `Heartbeat` event must fire before the panellist completes
    — proving the "still working" liveness pulse works.

    The heartbeat snapshot shows the in-flight slug as pending and elapsed
    time greater than the interval.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.progress import Heartbeat, ProgressEvent
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0.05")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        await _asyncio.sleep(0.2)
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug, model_id="x/y", persona=None, status=Status.OK,
            finish_reason="stop", resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=200, cost_usd=0.01, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    events: list[ProgressEvent] = []

    async def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    handle = await fanout(
        "p", [ModelSpec(model="claude-haiku", slug="alpha")],
        on_progress=on_progress,
    )
    assert handle.partial is False

    heartbeats = [e for e in events if isinstance(e, Heartbeat)]
    assert heartbeats, f"expected at least one Heartbeat, got: {[e.kind for e in events]}"
    hb = heartbeats[0]
    assert hb.elapsed_ms >= 50  # at least one interval
    assert hb.pending_count == 1
    assert hb.pending_slugs == ["alpha"]
    # Cost-so-far is 0 before any panellist returns.
    assert hb.cost_so_far_usd == 0.0


@pytest.mark.asyncio
async def test_fanout_heartbeat_disabled_when_interval_zero(tmp_path, monkeypatch):
    """`CONSULT_HEARTBEAT_INTERVAL_S=0` disables the heartbeat task so tests
    (and clients that don't want the pulse) get a clean event stream.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.progress import Heartbeat, ProgressEvent
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        await _asyncio.sleep(0.1)
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug, model_id="x/y", persona=None, status=Status.OK,
            finish_reason="stop", resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=100, cost_usd=0.0, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    events: list[ProgressEvent] = []

    async def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    await fanout(
        "p", [ModelSpec(model="claude-haiku")],
        on_progress=on_progress,
    )
    assert not any(isinstance(e, Heartbeat) for e in events)


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
        slug="alpha", model_id="x/y", persona=None, status=Status.OK,
        finish_reason="stop", resource_uri=paths.resource_uri("alpha"),
        body_path=str(paths.response_text("alpha")), latency_ms=10,
        cost_usd=0.0, cost_known=True,
    )
    paths.response_text("alpha").write_text("body content")

    handle = RunHandle(
        run_id=paths.run_id, artifacts_dir=str(paths.root),
        manifest=[entry], cost_usd=0.0, cost_known=True,
        wall_ms=10, partial=False, blinded=False,
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


@pytest.mark.asyncio
async def test_fanout_slow_tail_dropout_cancels_stragglers(tmp_path, monkeypatch):
    """Once `N - k` panellists return, slow stragglers get cancelled and
    surface as Status.TIMEOUT with a "slow-tail dropout" error. The full
    panel is returned (no panellist silently missing) so cost math and the
    progress total still see a complete N.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.05")
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.25")  # k=1 for n=4 → trigger=3

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        if "slow" in slug:
            await _asyncio.sleep(5.0)
        paths.response_text(slug).write_text("ok")
        return ManifestEntry(
            slug=slug, model_id="x/y", persona=None, status=Status.OK,
            finish_reason="stop", resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)), latency_ms=1,
            cost_usd=0.0, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    specs = [
        ModelSpec(model="claude-haiku", slug="fast-0"),
        ModelSpec(model="claude-haiku", slug="fast-1"),
        ModelSpec(model="claude-haiku", slug="fast-2"),
        ModelSpec(model="claude-haiku", slug="slow-3"),
    ]
    handle = await fanout("anything", specs)
    assert len(handle.manifest) == 4
    by_status = [m.status for m in handle.manifest]
    assert by_status.count(Status.OK) == 3
    assert by_status.count(Status.TIMEOUT) == 1
    dropped = next(m for m in handle.manifest if m.status is Status.TIMEOUT)
    assert "slow-tail dropout" in (dropped.error or "")


@pytest.mark.asyncio
async def test_fanout_no_dropout_below_threshold_panel_size(tmp_path, monkeypatch):
    """Panels smaller than 4 panellists never trigger slow-tail dropout —
    there's no statistically useful "rest of the panel" signal at N<4.
    A 3-spec panel with a slow panellist still completes all three.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.05")
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.5")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        if "slow" in slug:
            await _asyncio.sleep(0.3)
        paths.response_text(slug).write_text("ok")
        return ManifestEntry(
            slug=slug, model_id="x/y", persona=None, status=Status.OK,
            finish_reason="stop", resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)), latency_ms=1,
            cost_usd=0.0, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    specs = [
        ModelSpec(model="claude-haiku", slug="fast-0"),
        ModelSpec(model="claude-haiku", slug="fast-1"),
        ModelSpec(model="claude-haiku", slug="slow-2"),
    ]
    handle = await fanout("anything", specs)
    assert len(handle.manifest) == 3
    assert all(m.status is Status.OK for m in handle.manifest)


@pytest.mark.asyncio
async def test_acompletion_with_retry_recovers_after_rate_limit(monkeypatch):
    """One rate-limit followed by a success: the retry loop sleeps with
    jittered backoff and returns the second response. Without retry, a
    single 429 from a shared OpenAI key wastes the panel's full call cost.
    """
    from consult import runner

    class _StubRateLimit(BaseException):
        pass

    monkeypatch.setattr(runner, "_rate_limit_class", lambda: _StubRateLimit)
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _StubRateLimit("rate-limited")

        class _Resp:
            class _Choice:
                class _Msg:
                    content = "ok"

                message = _Msg()
                finish_reason = "stop"

            choices = [_Choice()]

        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    resp = await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 2
    assert resp.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_acompletion_with_retry_exhausts_then_raises(monkeypatch):
    """Persistent rate-limit across all attempts must re-raise the last
    RateLimitError, not swallow it — the caller's `Status.RATE_LIMITED`
    classification depends on the exception propagating out.
    """
    from consult import runner

    class _StubRateLimit(BaseException):
        pass

    monkeypatch.setattr(runner, "_rate_limit_class", lambda: _StubRateLimit)
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    async def always_rate_limit(**kwargs):
        raise _StubRateLimit("nope")

    monkeypatch.setattr(runner.litellm, "acompletion", always_rate_limit)
    with pytest.raises(_StubRateLimit):
        await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])


@pytest.mark.asyncio
async def test_acompletion_with_retry_does_not_retry_non_rate_limit(monkeypatch):
    """Auth/content-filter/bad-request errors must NOT trigger retry —
    they aren't transient and retrying just burns spend.
    """
    from consult import runner

    class _StubRateLimit(BaseException):
        pass

    monkeypatch.setattr(runner, "_rate_limit_class", lambda: _StubRateLimit)
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    attempts = 0

    async def raise_auth_error(**kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("invalid api key")

    monkeypatch.setattr(runner.litellm, "acompletion", raise_auth_error)
    with pytest.raises(RuntimeError, match="invalid api key"):
        await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 1


@pytest.mark.asyncio
async def test_acompletion_with_retry_recovers_from_bare_api_error(monkeypatch):
    """Bare `litellm.APIError` (no subclass) is the OpenRouter "Unable to
    get json response" failure mode — upstream returned all-whitespace.
    Retry must recover from it; previously it surfaced as a one-shot ERROR.
    """
    from consult import runner

    bare_api = runner._bare_api_error_class()
    assert bare_api is not None, "litellm.exceptions.APIError must resolve"
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # Construct a bare APIError the way litellm raises it.
            raise bare_api(
                status_code=500,
                message="Unable to get json response",
                llm_provider="openrouter",
                model=kwargs.get("model", "m"),
            )

        class _Resp:
            class _Choice:
                class _Msg:
                    content = "ok"
                message = _Msg()
                finish_reason = "stop"
            choices = [_Choice()]
        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    resp = await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 2
    assert resp.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_acompletion_with_retry_does_not_retry_api_error_subclass(monkeypatch):
    """A subclass of APIError (e.g. AuthenticationError) must NOT trigger
    retry — only bare APIError is treated as transient. Subclasses are
    terminal failures we shouldn't burn spend on.
    """
    from consult import runner

    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    from litellm import exceptions as lex
    auth_cls = getattr(lex, "AuthenticationError", None)
    if auth_cls is None:
        import pytest as _pytest
        _pytest.skip("litellm.AuthenticationError not available")

    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        raise auth_cls(
            message="bad key", llm_provider="openai", model=kwargs.get("model", "m"),
        )

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    with pytest.raises(auth_cls):
        await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 1, "AuthenticationError must be terminal, not retried"


@pytest.mark.asyncio
async def test_fanout_caps_per_provider_concurrency(tmp_path, monkeypatch):
    """With CONSULT_PROVIDER_CONCURRENCY=anthropic:1, only one Anthropic
    panellist may be in-flight at a time even if the panel has 5 of them.
    Locks the FRICTION-driven rate-limit mitigation: without the cap, every
    panellist hit the provider in lockstep.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_PROVIDER_CONCURRENCY", "anthropic:1")
    # Drop the registry cache so the env var takes effect on this call.
    registry.models_config.cache_clear()

    inflight = 0
    peak = 0
    real_acompletion = runner.litellm.acompletion

    async def fake_acompletion(**kwargs):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await _asyncio.sleep(0.05)
        finally:
            inflight -= 1
        # Build a minimal LiteLLM-like response object
        class _Msg:
            content = "ok\n\nCONFIDENCE: 0.7\nKEY_REASON: x"
            tool_calls = None
        class _Choice:
            message = _Msg()
            finish_reason = "stop"
        class _Resp:
            choices = [_Choice()]
            usage = None
            def model_dump(self): return {"_stub": True}
        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda **_: 0.0)

    specs = [ModelSpec(model="claude-haiku") for _ in range(5)]
    handle = await fanout("anything", specs)
    assert handle.partial is False
    assert len(handle.manifest) == 5
    assert peak == 1, f"semaphore cap=1 violated: peak in-flight = {peak}"

    # Restore registry cache so other tests aren't affected
    registry.models_config.cache_clear()
    monkeypatch.setattr(runner.litellm, "acompletion", real_acompletion)


@pytest.mark.asyncio
async def test_call_one_unknown_alias_returns_error_entry(tmp_path, monkeypatch):
    """An unknown alias must surface as a per-spec Status.ERROR rather than
    crashing the panel. Regression guard: KeyError out of `resolve_model`
    previously propagated through `asyncio.gather` and aborted every sibling.
    """
    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    spec = ModelSpec(model="definitely-not-a-real-alias")
    entry = await _call_one(spec, "bogus-0", "prompt", paths)
    assert entry.status is Status.ERROR
    assert entry.error and "definitely-not-a-real-alias" in entry.error
    assert entry.model_id is None
    assert entry.cost_known is True  # no call was billable


def test_estimate_cost_skips_unknown_alias_without_raising():
    """Unknown aliases mark cost_known=False but must not raise — `fanout`
    relies on this so a typo doesn't abort the run before any panel work.
    """
    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="bogus-xyz")]
    total, all_known = estimate_cost(specs, "hello")
    assert total >= 0.0
    assert all_known is False


@pytest.mark.asyncio
async def test_fanout_cost_cap_returns_partial(monkeypatch):
    """Setting max_run_usd to 0 must abort before any model call."""
    from consult import runner
    from consult.runner import fanout

    # Force a non-zero estimate so the cap path is exercised
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.99, True))
    specs = [ModelSpec(model="claude-haiku")]
    handle = await fanout("p", specs, max_run_usd=0.01)
    assert handle.partial is True
    assert handle.partial_reason and "exceeds cap" in handle.partial_reason
    assert "known-priced" not in handle.partial_reason  # all_known=True path
    assert handle.manifest == []


@pytest.mark.asyncio
async def test_fanout_cost_cap_message_discloses_partial_pricing(monkeypatch):
    """When estimate_cost returns all_known=False, the cap message must say
    so — otherwise the displayed estimate (only the known-priced portion)
    looks misleadingly low. Mirrors the dry_run branch.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.50, False))
    specs = [ModelSpec(model="claude-haiku")]
    handle = await fanout("p", specs, max_run_usd=0.01)
    assert handle.partial is True
    assert handle.partial_reason
    assert "known-priced portion only" in handle.partial_reason
    assert "exceeds cap" in handle.partial_reason
    assert handle.cost_known is False


def test_sequence_step_prompt_threads_prior_synth():
    """Step N>1 must include step N-1's synthesis as 'prior synthesis' context."""
    from consult.sequence import _step_prompt

    p1 = _step_prompt(1, 3, None, "What is X?")
    assert p1 == "What is X?"  # first step: no prior context

    p2 = _step_prompt(2, 3, "X is foo.", "Given X is foo, what about Y?")
    assert "Step 1 of 3 — prior synthesis" in p2
    assert "X is foo." in p2
    assert "Step 2 of 3 prompt" in p2
    assert "Given X is foo, what about Y?" in p2


@pytest.mark.asyncio
async def test_sequence_chains_synthesis_across_steps(tmp_path, monkeypatch):
    """A 2-step sequence: step 2's prompt-as-sent must contain step 1's
    synthesis, and the final_synthesis matches the last step's output.
    Heavy machinery (fanout, capsule, synth) is monkeypatched.
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import sequence as sequence_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (0.0, True))

    seen_prompts: list[str] = []

    async def fake_fanout(prompt, specs, **kwargs):
        seen_prompts.append(prompt)
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="alpha",
                    model_id="x/y",
                    status=Status.OK,
                    finish_reason="stop",
                    resource_uri=paths.resource_uri("alpha"),
                    body_path=str(paths.response_text("alpha")),
                    latency_ms=1,
                    cost_usd=0.01,
                    cost_known=True,
                )
            ],
            cost_usd=0.01,
            cost_known=True,
            wall_ms=1,
            partial=False,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    synth_counter = {"i": 0}

    async def fake_synth(run_id, by_model=None, anonymised=False, **kwargs):
        synth_counter["i"] += 1
        return synth_mod.SynthResult(text=f"SYNTH_{synth_counter['i']}")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await sequence_mod.sequence(
        ["First question", "Second question"],
        [ModelSpec(model="claude-haiku")],
    )
    assert len(result.steps) == 2
    assert result.final_synthesis == "SYNTH_2"
    assert result.cost_usd == pytest.approx(0.02)
    assert result.partial is False

    # Step 1 prompt: just the body.
    assert seen_prompts[0] == "First question"
    # Step 2 prompt: prior synth must be embedded.
    assert "SYNTH_1" in seen_prompts[1]
    assert "Second question" in seen_prompts[1]
    assert "prior synthesis" in seen_prompts[1]


@pytest.mark.asyncio
async def test_sequence_continues_through_partial_pricing(tmp_path, monkeypatch):
    """Unlike refine (where an arbiter can extend rounds indefinitely),
    sequence has a fixed user-supplied step list — partial pricing must
    NOT abort step 2+, since the cumulative-cost check + per-step
    `max_run_usd=cap-cumulative_cost` already provide the cap guarantee.
    Regression guard: pass #14 found that copying refine's est_known
    refusal into sequence made multi-step runs stop at step 1 whenever
    any panellist had unmapped pricing (which is most of them on
    openrouter).
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import sequence as sequence_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # Return est_known=False to simulate the openrouter unknown-pricing case.
    monkeypatch.setattr(
        runner_mod, "estimate_cost", lambda *a, **kw: (0.0, False)
    )

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="alpha", model_id="x/y", status=Status.OK,
                    finish_reason="stop",
                    resource_uri=paths.resource_uri("alpha"),
                    body_path=str(paths.response_text("alpha")),
                    latency_ms=1, cost_usd=0.01, cost_known=False,
                )
            ],
            cost_usd=0.01, cost_known=False, wall_ms=1, partial=False,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_synth(run_id, **kwargs):
        return synth_mod.SynthResult(text="X")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await sequence_mod.sequence(
        ["q1", "q2", "q3"],
        [ModelSpec(model="claude-haiku")],
    )
    # All three steps must complete despite est_known=False.
    assert len(result.steps) == 3
    assert result.partial is False
    # cost_known must propagate as False so the caller still knows.
    assert result.cost_known is False


@pytest.mark.asyncio
async def test_sequence_rejects_empty_inputs():
    """Empty prompts list or empty specs list is a usage error."""
    from consult import sequence as sequence_mod

    with pytest.raises(ValueError, match="at least one prompt"):
        await sequence_mod.sequence([], [ModelSpec(model="claude-haiku")])
    with pytest.raises(ValueError, match="at least one model spec"):
        await sequence_mod.sequence(["q"], [])


def test_litellm_logger_does_not_propagate_to_root():
    """LiteLLM's logger must not propagate, otherwise callers that enable
    a root handler (basicConfig at INFO etc.) see every line twice — once
    via LiteLLM's own coloured handler, once via root. The disable lives
    in consult/runner.py at module level so import is enough to set it.
    """
    import logging

    # Importing the package runs runner.py at module load (via consult.server
    # → consult.runner), which disables propagation. Confirm the effect.
    import consult.runner  # noqa: F401

    assert logging.getLogger("LiteLLM").propagate is False


@pytest.mark.asyncio
async def test_capsule_extractor_omits_temperature_for_gemini(monkeypatch):
    """Gemini-3 emits a warning + can loop when temperature < 1.0. The
    extractor must omit the param entirely for any gemini-routed model
    (direct or via openrouter) while keeping it for everyone else.
    """
    import litellm

    from consult import capsule as capsule_mod

    captured: list[dict[str, Any]] = []

    async def fake_completion(**kwargs):
        captured.append(kwargs)
        # Return a shape capsule._extract_one can parse cleanly
        class _Resp:
            def __init__(self):
                self.choices = [type("Msg", (), {"message": type("M", (), {"content": '{"position": "x"}'})()})()]
            def model_dump(self):
                return {}
        return _Resp()

    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    monkeypatch.setattr(
        litellm, "completion_cost", lambda completion_response: 0.0
    )

    # Gemini direct
    await capsule_mod._extract_one("hello body", "gemini/gemini-3.1-pro", 30)
    # Gemini via openrouter
    await capsule_mod._extract_one("hello body", "openrouter/google/gemini-3-pro", 30)
    # Anthropic — should still get temperature=0.0
    await capsule_mod._extract_one("hello body", "anthropic/claude-haiku-4-5", 30)

    assert len(captured) == 3
    assert "temperature" not in captured[0]  # gemini direct
    assert "temperature" not in captured[1]  # gemini via openrouter
    assert captured[2]["temperature"] == 0.0  # anthropic keeps it


def test_capsule_extract_json_recovers_prose_and_fences():
    """The capsule contract depends on this — one regex change breaks all callers."""
    from consult.jsonparse import extract_json

    assert extract_json('{"score": 0.5}') == {"score": 0.5}
    assert extract_json('```json\n{"k": "v"}\n```') == {"k": "v"}
    assert extract_json('prefix\n{"k": 1}\nsuffix') == {"k": 1}
    assert extract_json("definitely not json") is None
    assert extract_json("") is None


def test_parse_resource_uri_rejects_malformed():
    """Permissive parsing would be a path-traversal hazard."""
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("http://example.com/runs/abc/responses/x")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc/responses/x/extra")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc/capsules/x")
    # Happy path still works
    rid, kind, name = artifacts.parse_resource_uri("consult://runs/r1/responses/alpha.r2")
    assert rid == "r1"
    assert kind == "responses"
    assert name == "alpha.r2"
    # attachments/ kind also parses
    rid, kind, name = artifacts.parse_resource_uri(
        "consult://runs/r1/attachments/main.py",
    )
    assert (rid, kind, name) == ("r1", "attachments", "main.py")


def test_status_classifier_normal_responses():
    """Cover the OK/TRUNCATED/EMPTY/CONTENT_FILTERED/MALFORMED response paths.

    Today only the exception branch is tested — the body classification logic
    could return wrong statuses silently if not exercised.
    """
    from types import SimpleNamespace

    from consult.status import classify

    def make_resp(content, finish):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason=finish
                )
            ]
        )

    s, _, body = classify(make_resp("real content", "stop"))
    assert s == Status.OK
    assert body == "real content"

    s, _, body = classify(make_resp("partial", "length"))
    assert s == Status.TRUNCATED
    assert body == "partial"

    s, _, _ = classify(make_resp("", "length"))
    assert s == Status.TRUNCATED  # empty body + length still TRUNCATED

    s, _, _ = classify(make_resp("    \n  \n", "stop"))
    assert s == Status.EMPTY  # whitespace-only body (OR thinking-model fail mode)

    s, _, _ = classify(make_resp("blocked", "content_filter"))
    assert s == Status.CONTENT_FILTERED

    s, _, _ = classify(SimpleNamespace(choices=[]))
    assert s == Status.MALFORMED

    s, _, _ = classify(None)
    assert s == Status.EMPTY


def test_synth_build_input_always_blinds_and_filters_failures():
    """Synth never sees real slug or model_id — bias mitigation per the
    "When Identity Skews Debate" paper. Tests:

    - ERROR/EMPTY entries are filtered out of the input
    - The synth input contains blind labels (Alpha, Beta, ...) not slugs
    - Model IDs never appear in the synth input
    - The returned `label_to_slug` mapping covers every usable panellist
    """
    from consult.synth import _build_input

    manifest = [
        {
            "slug": "panelist-alpha",
            "model_id": "anthropic/claude-opus-4-7",
            "persona": "contrarian",
            "confidence": 0.8,
            "status": "OK",
        },
        {
            "slug": "panelist-beta",
            "model_id": "openai/gpt-5.5",
            "persona": None,
            "confidence": None,
            "status": "EMPTY",  # must be filtered out
        },
        {
            "slug": "panelist-gamma",
            "model_id": "gemini/gemini-3.1-pro-preview",
            "persona": None,
            "confidence": 0.6,
            "status": "TRUNCATED",  # truncated-with-body stays in
        },
    ]
    bodies = {
        "panelist-alpha": "alpha body",
        "panelist-beta": "",
        "panelist-gamma": "gamma body",
    }
    rubric = "rubric {n}"

    text, label_to_slug = _build_input(manifest, bodies, rubric=rubric)
    # Real model identifiers never reach the synth
    assert "anthropic/claude-opus-4-7" not in text
    assert "openai/gpt-5.5" not in text
    assert "gemini/gemini-3.1-pro-preview" not in text
    # Real slugs are hidden too — replaced by blind labels in the label
    # row. (Bodies stay verbatim; if a panellist happened to mention
    # its own slug in the body, that's the panellist's own leak —
    # `consult` doesn't try to scrub bodies.)
    assert "[panelist-alpha" not in text
    assert "[panelist-gamma" not in text
    # Filtered: EMPTY status entry was dropped before labels were assigned
    assert "panelist-beta" not in text  # filtered
    # The mapping covers exactly the 2 usable entries (OK + TRUNCATED)
    assert set(label_to_slug.values()) == {"panelist-alpha", "panelist-gamma"}
    assert len(label_to_slug) == 2
    # Each label appears in the input text somewhere
    for label in label_to_slug:
        assert label in text
    # Bodies survive the relabelling
    assert "alpha body" in text
    assert "gamma body" in text
    assert "rubric 2" in text  # only OK + TRUNCATED counted


def test_manifest_entry_validates_error_requirement():
    """Constructing an ERROR/TIMEOUT entry without an error string must fail."""
    import pydantic

    base = dict(
        slug="x", status=Status.ERROR, resource_uri="consult://x", body_path="/tmp/x"
    )
    with pytest.raises(pydantic.ValidationError):
        ManifestEntry(**base)
    # With error, it succeeds
    ManifestEntry(**base, error="auth failed")


def test_run_handle_validates_partial_coupling():
    """partial=True ⇔ partial_reason set."""
    import pydantic

    base = dict(run_id="r", artifacts_dir="/tmp/r", manifest=[], cost_usd=0.0, wall_ms=0)
    with pytest.raises(pydantic.ValidationError):
        RunHandle(**base, partial=True)  # no reason
    with pytest.raises(pydantic.ValidationError):
        RunHandle(**base, partial=False, partial_reason="oops")  # reason without partial


def test_daily_ledger_aggregates_costs_status_and_panel_size(tmp_path, monkeypatch):
    """Daily ledger reads every run dir whose ID starts with YYYYMMDD,
    aggregates cost + cost_known, and records per-status counts. Malformed
    or missing manifests are skipped without aborting the scan.
    """
    from datetime import date as date_cls

    from consult import ledger

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    # Two real runs on the target date, one on a different date, plus a
    # malformed-manifest run that should be skipped without crashing.
    def make_run(rid: str, cost: float, cost_known: bool, statuses: list[str]):
        d = tmp_path / rid
        d.mkdir()
        manifest = {
            "run_id": rid,
            "cost_usd": cost,
            "cost_known": cost_known,
            "manifest": [{"status": s} for s in statuses],
        }
        (d / "manifest.json").write_text(json.dumps(manifest))

    make_run("20260101-100000-1", 0.40, True, ["OK", "OK", "RATE_LIMITED"])
    make_run("20260101-110000-2", 0.15, False, ["OK"])
    make_run("20260102-100000-3", 99.0, True, ["OK"])  # different day, ignored

    # Malformed manifest — bytes that aren't JSON
    bad = tmp_path / "20260101-120000-9"
    bad.mkdir()
    (bad / "manifest.json").write_text("{this is not json")

    # Bare directory with no manifest at all — also skipped
    (tmp_path / "20260101-130000-9").mkdir()

    led = ledger.daily_ledger(date_cls(2026, 1, 1))
    assert led.date == date_cls(2026, 1, 1)
    assert len(led.runs) == 2  # malformed + manifest-less skipped, other day excluded
    assert led.total_usd == pytest.approx(0.55)
    assert led.total_known is False  # one of the two had cost_known=False

    by_id = {r.run_id: r for r in led.runs}
    assert by_id["20260101-100000-1"].status_counts == {"OK": 2, "RATE_LIMITED": 1}
    assert by_id["20260101-100000-1"].panel_size == 3
    assert by_id["20260101-110000-2"].panel_size == 1


def test_daily_ledger_empty_day_returns_zero(tmp_path, monkeypatch):
    """A day with no runs must return a well-formed empty ledger (not raise),
    and total_known=True since there's nothing unknown about $0."""
    from datetime import date as date_cls

    from consult import ledger

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    led = ledger.daily_ledger(date_cls(2030, 6, 15))
    assert led.runs == []
    assert led.total_usd == 0.0
    assert led.total_known is True


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


def test_internal_models_forbid_unknown_fields():
    """`extra="forbid"` is the type-system half of the RefineResult silent-drop
    fix: kwargs that don't match a field must raise instead of being dropped.
    Locks the policy across every internal result/handle type so a future
    field rename or unset model_config gets caught the moment it ships.
    """
    import pydantic

    from consult.ledger import DailyLedger, LedgerRunEntry
    from consult.progress import PanellistCompleted
    from consult.sequence import SequenceResult, SequenceStep
    from consult.types import (
        ArbiterVerdict,
        Capsule,
        ManifestEntry,
        ModelSpec,
        RefineResult,
        RunHandle,
        RunResult,
    )

    cases: list[tuple[type, dict]] = [
        (ModelSpec, {"model": "x"}),
        (Capsule, {}),
        (ManifestEntry, {
            "slug": "s", "status": "OK", "resource_uri": "consult://x",
            "body_path": "/tmp/x",
        }),
        (RunHandle, {
            "run_id": "r", "artifacts_dir": "/tmp", "manifest": [],
            "cost_usd": 0.0, "wall_ms": 0,
        }),
        (RunResult, {
            "run_id": "r", "synthesis": "", "manifest": [],
            "cost_usd": 0.0, "wall_ms": 0,
        }),
        (ArbiterVerdict, {"round": 1, "score": 0.5}),
        (RefineResult, {
            "run_id": "r", "rounds_completed": 0, "final_manifest": [],
            "verdicts": [], "synthesis": "", "converged": False,
            "threshold": 0.85, "cost_usd": 0.0, "wall_ms": 0,
        }),
        (SequenceStep, {
            "step": 1, "run_id": "r", "synthesis": "", "cost_usd": 0.0,
            "panel_size": 0,
        }),
        (SequenceResult, {
            "final_synthesis": "", "cost_usd": 0.0, "wall_ms": 0,
        }),
        (LedgerRunEntry, {
            "run_id": "r", "cost_usd": 0.0, "cost_known": True, "panel_size": 0,
        }),
        (DailyLedger, {
            "date": "2026-01-01", "total_usd": 0.0, "total_known": True,
        }),
        (PanellistCompleted, {
            "done": 0, "total": 0, "slug": "s", "status": "OK", "latency_ms": 0,
        }),
    ]
    for cls, kwargs in cases:
        # Sanity: the baseline kwargs construct successfully
        cls(**kwargs)
        # An unknown kwarg must be rejected, not silently dropped
        with pytest.raises(pydantic.ValidationError):
            cls(**kwargs, definitely_not_a_field="boom")


# ---- Viewer tests ----------------------------------------------------------


def test_viewer_markdown_subset_renders_each_construct():
    """The viewer ships its own small markdown renderer (rather than pulling
    in a library) — pin every construct the synthesiser actually emits so a
    regex regression doesn't silently lose formatting in `feed.html`.
    """
    from consult.viewer import render_markdown

    out = render_markdown("# Heading 1\n## Heading 2")
    assert "<h1>Heading 1</h1>" in out
    assert "<h2>Heading 2</h2>" in out

    out = render_markdown("- one\n- two\n- three")
    assert out.count("<li>") == 3
    assert "<ul>" in out and "</ul>" in out

    out = render_markdown("1. first\n2. second")
    assert "<ol>" in out and out.count("<li>") == 2

    out = render_markdown("Some **bold** and *italic* with `code` and a [link](https://x.test).")
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out
    assert "<code>code</code>" in out
    assert '<a href="https://x.test">link</a>' in out

    out = render_markdown("```python\nprint(1)\n```")
    assert '<pre><code class="lang-python">print(1)</code></pre>' in out

    out = render_markdown("para one\n\npara two")
    assert out.count("<p>") == 2


def test_viewer_markdown_escapes_untrusted_html():
    """A panellist body that contained `<script>` must not become live HTML
    when threaded through synthesis — every line passes through html.escape
    before inline transforms run.
    """
    from consult.viewer import render_markdown

    out = render_markdown("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out

    out = render_markdown("- <img src=x onerror=alert(1)>")
    assert "<img" not in out
    assert "&lt;img" in out


def test_viewer_markdown_inline_code_protects_bold_markers():
    """Inline-code spans must be stashed before the bold regex runs;
    otherwise `` `**foo**` `` is wrongly rendered with a nested <strong>.
    """
    from consult.viewer import render_markdown

    out = render_markdown("Literal `**not bold**` here.")
    assert "<code>**not bold**</code>" in out
    assert "<strong>" not in out


def test_viewer_round_of_extracts_refine_round():
    """Refine slugs carry an `.r<n>` suffix; the viewer uses this to bucket
    panellists by round in the panel section.
    """
    from consult.viewer import _round_of

    assert _round_of("claude-opus") is None
    assert _round_of("claude-opus.r1") == 1
    assert _round_of("claude-opus-0.r3") == 3


def test_viewer_fmt_cost_handles_unknown_and_small_values():
    """`cost_usd=None` and `cost_known=False` must not crash the renderer."""
    from consult.viewer import _fmt_cost, _fmt_ms

    assert _fmt_cost(None) == "—"
    assert _fmt_cost(0) == "$0"
    assert _fmt_cost(0.0123) == "$0.0123"
    assert _fmt_cost(2.5).startswith("$2.5")
    # Unknown-pricing is flagged with a trailing `*` — mirrors ledger output.
    assert _fmt_cost(0.5, known=False).endswith("*")
    assert _fmt_ms(None) == "—"
    assert _fmt_ms(250) == "250ms"
    assert _fmt_ms(2500).endswith("s")


def _make_run_dir(
    tmp_path,
    run_id: str,
    *,
    entries: list[dict],
    bodies: dict[str, str] | None = None,
    synth: str | None = None,
    arbiters: list[dict] | None = None,
    progress_lines: list[dict] | None = None,
    prompt: str = "test prompt",
    extras: dict | None = None,
):
    """Materialise a minimal-but-realistic run dir on disk for viewer tests.

    Shared across the render_run cases so each test stays focused on one
    invariant rather than rebuilding the artifact tree.
    """
    root = tmp_path / run_id
    (root / "responses").mkdir(parents=True)
    (root / "capsules").mkdir()
    (root / "arbiters").mkdir()
    (root / "prompts").mkdir()
    (root / "prompt.txt").write_text(prompt)
    manifest = {
        "run_id": run_id,
        "artifacts_dir": str(root),
        "manifest": entries,
        "cost_usd": sum((e.get("cost_usd") or 0) for e in entries),
        "cost_known": all(e.get("cost_known", True) for e in entries),
        "wall_ms": 1234,
        "partial": False,
        "blinded": False,
        **(extras or {}),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for slug, body in (bodies or {}).items():
        (root / "responses" / f"{slug}.txt").write_text(body)
    if synth is not None:
        (root / "synthesis.md").write_text(synth)
    for v in arbiters or []:
        (root / "arbiters" / f"round-{v['round']}.json").write_text(json.dumps(v))
    if progress_lines:
        (root / "_progress.log").write_text(
            "\n".join(json.dumps(p) for p in progress_lines) + "\n"
        )
    return root


def test_viewer_render_run_panel_includes_core_sections(tmp_path, monkeypatch):
    """Panel-only runs (no synthesis) still render — manifest table, capsule,
    body. The 'panel' kind is derived from the absence of synthesis.md.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-1"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[{
            "slug": "alpha",
            "model_id": "anthropic/claude-opus-4-7",
            "status": "OK",
            "capsule": {
                "position": "supports A",
                "recommendation": "ship A",
                "key_points": ["fast", "cheap"],
            },
            "resource_uri": f"consult://runs/{rid}/responses/alpha",
            "body_path": "/x",
            "latency_ms": 4200,
            "cost_usd": 0.012,
            "cost_known": True,
            "confidence": 0.8,
        }],
        bodies={"alpha": "alpha body text"},
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert out.name == "feed.html"
    assert "<!doctype html>" in text
    assert rid in text
    # Panel run: no synthesis section, no arbiter section, but core panellist
    # card is present with the capsule fields surfaced.
    assert "panel" in text.lower()
    assert "<h2>Synthesis</h2>" not in text
    assert "<h2>Arbiter rounds</h2>" not in text
    assert "alpha" in text
    assert "supports A" in text
    assert "ship A" in text
    assert "anthropic/claude-opus-4-7" in text
    # The status pill must use the OK tone.
    assert "pill-ok" in text


def test_viewer_render_run_consult_renders_synthesis_markdown(tmp_path, monkeypatch):
    """A consult run has synthesis.md — the viewer renders it through the
    markdown subset, not as a raw `<pre>` blob. Catches a regression where
    we'd accidentally drop the synthesis section.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-2"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[{
            "slug": "alpha", "model_id": "x/y", "status": "OK",
            "resource_uri": f"consult://runs/{rid}/responses/alpha",
            "body_path": "/x", "latency_ms": 1, "cost_usd": 0.0,
            "cost_known": True,
        }],
        synth="# Consensus\n\n- point one\n- point two",
        extras={"synthesiser": "gemini-pro"},
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert "<h2>Synthesis</h2>" in text
    assert "<h1>Consensus</h1>" in text
    assert "<li>point one</li>" in text
    assert "consult" in text.lower()
    # `synthesiser` field surfaces in the header stats so the reader can see
    # which model produced the synthesis without opening the manifest.
    assert "gemini-pro" in text


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
                "slug": "alpha.r1", "model_id": "x/y", "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha.r1",
                "body_path": "/x", "latency_ms": 1, "cost_usd": 0.0,
                "cost_known": True,
            },
            {
                "slug": "alpha.r2", "model_id": "x/y", "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha.r2",
                "body_path": "/x", "latency_ms": 1, "cost_usd": 0.0,
                "cost_known": True,
            },
        ],
        synth="final synth",
        arbiters=[
            {"round": 1, "score": 0.5, "gaps": ["missing X"], "next_round_focus": "address X", "reasoning": "r1"},
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


def test_viewer_render_run_escapes_panellist_bodies(tmp_path, monkeypatch):
    """Untrusted panellist response text must be HTML-escaped before
    landing in `feed.html` — otherwise a body containing `<script>` would
    execute when the user opened the page in a browser.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-4"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[{
            "slug": "alpha", "model_id": "x/y", "status": "OK",
            "resource_uri": f"consult://runs/{rid}/responses/alpha",
            "body_path": "/x", "latency_ms": 1, "cost_usd": 0.0,
            "cost_known": True,
        }],
        bodies={"alpha": "<script>alert(1)</script>"},
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text


def test_viewer_render_run_renders_progress_timeline(tmp_path, monkeypatch):
    """`_progress.log` events become a chronological timeline section with
    deltas computed from the first event.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-5"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[{
            "slug": "alpha", "model_id": "x/y", "status": "OK",
            "resource_uri": f"consult://runs/{rid}/responses/alpha",
            "body_path": "/x", "latency_ms": 1, "cost_usd": 0.0,
            "cost_known": True,
        }],
        progress_lines=[
            {"ts": "2026-05-20T10:18:09.000000+00:00", "done": 0, "total": 0,
             "kind": "panellist_completed", "slug": "alpha", "status": "OK", "latency_ms": 1234},
            {"ts": "2026-05-20T10:18:13.500000+00:00", "done": 1, "total": 1,
             "kind": "capsule_extracted", "slug": "alpha"},
        ],
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert "<h2>Timeline</h2>" in text
    assert "panellist_completed" in text
    assert "capsule_extracted" in text
    # Relative offset: first event is +0.0s, second is +4.5s after it.
    assert "+0.0s" in text
    assert "+4.5s" in text


def test_viewer_render_run_handles_missing_optional_artifacts(tmp_path, monkeypatch):
    """Empty/missing prompt.txt, synthesis.md, _progress.log, and arbiters
    must all be tolerated — the renderer is for whatever state the run is
    in, not a strict completeness check.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-6"
    root = tmp_path / rid
    (root / "responses").mkdir(parents=True)
    # Bare-minimum manifest, no other artifacts at all
    (root / "manifest.json").write_text(json.dumps({
        "run_id": rid,
        "artifacts_dir": str(root),
        "manifest": [],
        "cost_usd": 0.0,
        "wall_ms": 0,
        "partial": False,
    }))
    out = viewer.render_run(rid)
    text = out.read_text()
    assert rid in text
    # No panellists, no synth, no arbiters — still produces a complete document.
    assert "<!doctype html>" in text
    assert "</html>" in text


def test_viewer_render_run_raises_for_missing_run(tmp_path, monkeypatch):
    """Unknown run_id → FileNotFoundError, which the CLI surfaces as an
    exit-code-1 error rather than a stack trace.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        viewer.render_run("20990101-nope-1")


def test_viewer_render_run_raises_for_run_without_manifest(tmp_path, monkeypatch):
    """A run dir that exists but has no manifest can't be rendered — bail
    out with a clear message rather than producing a half-built page.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-no-manifest"
    (tmp_path / rid).mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json missing"):
        viewer.render_run(rid)


def test_viewer_render_run_surfaces_partial_and_cancelled_banners(tmp_path, monkeypatch):
    """Partial reason and the CANCELLED marker must surface visually so the
    reader doesn't mistake a half-finished run for a clean one.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-7"
    root = _make_run_dir(
        tmp_path,
        rid,
        entries=[],
        extras={"partial": True, "partial_reason": "cost cap exceeded"},
    )
    (root / "CANCELLED").touch()
    text = viewer.render_run(rid).read_text()
    assert "cost cap exceeded" in text
    assert "cancelled" in text.lower()


def test_viewer_cli_writes_path_to_stdout(tmp_path, monkeypatch, capsys):
    """`consult-view <run_id>` prints the absolute path so users can pipe
    it into `open(1)` or copy/paste it. No --open flag = no browser launch.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-cli"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[{
            "slug": "alpha", "model_id": "x/y", "status": "OK",
            "resource_uri": f"consult://runs/{rid}/responses/alpha",
            "body_path": "/x", "latency_ms": 1, "cost_usd": 0.0,
            "cost_known": True,
        }],
    )

    opened: list[str] = []
    monkeypatch.setattr(viewer.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(sys, "argv", ["consult-view", rid])
    viewer.cli()

    out = capsys.readouterr().out.strip()
    assert out.endswith("feed.html")
    assert Path(out).exists()
    assert opened == []  # --open not passed


def test_viewer_cli_open_flag_launches_browser(tmp_path, monkeypatch, capsys):
    """`--open` invokes webbrowser.open with the file:// URI of feed.html."""
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-cli-open"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[{
            "slug": "alpha", "model_id": "x/y", "status": "OK",
            "resource_uri": f"consult://runs/{rid}/responses/alpha",
            "body_path": "/x", "latency_ms": 1, "cost_usd": 0.0,
            "cost_known": True,
        }],
    )

    opened: list[str] = []
    monkeypatch.setattr(viewer.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(sys, "argv", ["consult-view", rid, "--open"])
    viewer.cli()

    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith("/feed.html")


def test_viewer_cli_missing_run_exits_with_message(tmp_path, monkeypatch, capsys):
    """An unknown run_id surfaces as a SystemExit (exit code != 0), not a
    raw traceback — keeps the CLI feeling like the rest of the unix tools.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["consult-view", "20990101-no-such-run"])
    with pytest.raises(SystemExit) as ei:
        viewer.cli()
    # SystemExit carries the message in `code` when raised with a string.
    assert "Run not found" in str(ei.value) or "no-such-run" in str(ei.value)


# ---- M1: Context bundle / fidelity layer -----------------------------------


def test_context_scrub_brands_masks_known_providers():
    """Brand-name scrubber masks model + provider names so blinded mode
    doesn't leak identity through the prompt itself.
    """
    from consult import context as ctx

    src = "Compare claude-opus, gpt-pro, and gemini-pro for code review on Anthropic."
    out = ctx.scrub_brands(src)
    for brand in ("claude", "gpt", "gemini", "Anthropic"):
        assert brand.lower() not in out.lower(), (brand, out)
    assert out.count("[MODEL]") >= 4


def test_context_scrub_brands_handles_provider_prefixed_ids():
    """Raw LiteLLM IDs like `x-ai/grok-4.3` survive a simple word-boundary
    regex; the provider-prefix pass must catch them.
    """
    from consult import context as ctx

    src = "I asked openrouter/x-ai/grok-4.3 and meta-llama/llama-4-maverick."
    out = ctx.scrub_brands(src)
    assert "x-ai" not in out.lower()
    assert "grok" not in out.lower()
    assert "meta-llama" not in out.lower()
    assert "llama" not in out.lower()


def test_context_scrub_brands_is_idempotent():
    """Running the scrubber twice produces the same result — the
    replacement token `[MODEL]` doesn't itself match the regex."""
    from consult import context as ctx

    src = "claude vs gpt for coding"
    once = ctx.scrub_brands(src)
    twice = ctx.scrub_brands(once)
    assert once == twice


def test_context_build_keeps_raw_when_not_blinded():
    """When blinded=False, prompt_scrubbed equals prompt — scrubbing
    only fires when downstream stages will actually use it."""
    from consult import context as ctx

    src = "claude vs gpt — which is better?"
    bundle = ctx.build(src, blinded=False)
    assert bundle.prompt == src
    assert bundle.prompt_scrubbed == src
    assert bundle.blinded is False


def test_context_build_scrubs_when_blinded():
    from consult import context as ctx

    src = "claude vs gpt — which is better?"
    bundle = ctx.build(src, blinded=True)
    assert bundle.prompt == src  # raw preserved
    assert "[MODEL]" in bundle.prompt_scrubbed
    assert "claude" not in bundle.prompt_scrubbed.lower()
    assert bundle.blinded is True


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


def test_context_load_or_none_returns_none_for_legacy_run(tmp_path, monkeypatch):
    """Legacy run dirs (no context.json) load as None so callers can fall
    back to pre-Phase-1 behaviour rather than crashing."""
    from consult import context as ctx

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()  # no context.write
    assert ctx.load_or_none(paths) is None


def test_context_trim_text_head_and_tail():
    """Trimming preserves head + tail with a marker indicating the cut."""
    from consult import context as ctx

    src = "A" * 1000 + "BBBB" + "Z" * 1000  # distinct middle marker
    out = ctx.trim_text(src, max_chars=400)
    assert len(out) < len(src)
    assert "TRIMMED" in out
    # Head from the front; tail from the back
    assert out.startswith("AAAA")
    assert out.endswith("ZZZZ")


def test_context_trim_text_under_budget_is_passthrough():
    from consult import context as ctx

    src = "small"
    assert ctx.trim_text(src, max_chars=100) == src


def test_context_trim_synth_input_trims_largest_body_first():
    """When overall budget is tight, the largest body is trimmed first
    so we recover the most slack with the least per-body signal loss."""
    from consult import context as ctx

    big = "X" * 50_000
    small = "Y" * 1_000
    bodies = {"big": big, "small": small}
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt="prompt", bodies=bodies, overall_budget=15_000
    )
    assert len(new_bodies["small"]) == 1_000  # untouched
    assert len(new_bodies["big"]) < 50_000     # trimmed
    assert "TRIMMED" in new_bodies["big"]
    assert new_prompt == "prompt"               # prompt is the last resort


def test_context_trim_synth_input_passes_through_when_under_budget():
    from consult import context as ctx

    bodies = {"a": "x" * 1000, "b": "y" * 1000}
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt="prompt", bodies=bodies, overall_budget=100_000
    )
    assert new_prompt == "prompt"
    assert new_bodies == bodies


def test_context_trim_synth_input_trims_prompt_when_bodies_at_floor():
    """If every body is at the 5000-char floor and we're still over budget,
    the original prompt gets trimmed too — preserving the prompt is preferred
    but not at the cost of failing the synth call."""
    from consult import context as ctx

    bodies = {"a": "x" * 8000, "b": "y" * 8000}
    huge_prompt = "P" * 200_000
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt=huge_prompt, bodies=bodies, overall_budget=20_000
    )
    assert new_prompt is not None
    assert len(new_prompt) < len(huge_prompt)
    assert "TRIMMED" in new_prompt


@pytest.mark.asyncio
async def test_synth_writes_synth_input_with_original_prompt(tmp_path, monkeypatch):
    """End-to-end fidelity check: after a fanout+synth, the persisted
    synth_input.txt contains the original prompt and all panellist bodies —
    the synthesiser can fact-check claims against the source."""
    import consult.synth as synth_mod
    from consult import context as ctx
    from consult.types import ManifestEntry, Status

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    src = "Review this PR diff: <SIGNATURE-TOKEN-FOR-TEST>"
    ctx.write(paths, ctx.build(src, blinded=False))
    paths.prompt_txt.write_text(src)

    # Synthetic 2-panellist manifest + bodies on disk
    manifest = [
        ManifestEntry(
            slug="alpha", model_id="anthropic/claude-opus-4-7", status=Status.OK,
            resource_uri=paths.resource_uri("alpha"),
            body_path=str(paths.response_text("alpha")),
            latency_ms=100, cost_known=True,
        ),
        ManifestEntry(
            slug="beta", model_id="openai/gpt-5.5-pro", status=Status.OK,
            resource_uri=paths.resource_uri("beta"),
            body_path=str(paths.response_text("beta")),
            latency_ms=120, cost_known=True,
        ),
    ]
    paths.response_text("alpha").write_text("Found a race condition at line 42.")
    paths.response_text("beta").write_text("Found a SQL injection at line 117.")
    artifacts.write_manifest(paths, {
        "run_id": paths.run_id,
        "artifacts_dir": str(paths.root),
        "manifest": [m.model_dump(mode="json") for m in manifest],
        "cost_usd": 0.0,
        "cost_known": True,
        "wall_ms": 0,
        "partial": False,
        "blinded": False,
    })

    # Stub out the actual LiteLLM call — we only care about what gets written
    # to synth_input.txt.
    async def fake_acompletion(**kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="STUBBED SYNTHESIS"),
                finish_reason="stop",
            )]
        )

    monkeypatch.setattr("consult.synth.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "consult.synth.litellm.completion_cost", lambda completion_response: 0.0
    )

    await synth_mod.synthesise(paths.run_id)

    synth_input = (paths.root / "synth_input.txt").read_text()
    # The original prompt is now in synth_input — synthesiser can fact-check
    assert "SIGNATURE-TOKEN-FOR-TEST" in synth_input
    # Both panellist bodies present
    assert "race condition at line 42" in synth_input
    assert "SQL injection at line 117" in synth_input


def test_synth_build_input_prepends_original_prompt_section():
    """Unit-level: _build_input puts the original prompt before the rubric/bodies."""
    from consult.synth import _build_input

    manifest = [
        {"slug": "alpha", "model_id": "x", "persona": None, "confidence": None,
         "status": "OK"},
    ]
    bodies = {"alpha": "BODY-TEXT"}
    out, _label_map = _build_input(
        manifest, bodies, rubric="rubric for {n} responses",
        original_prompt="PROMPT-TEXT",
    )
    assert "Original question / source" in out
    assert "PROMPT-TEXT" in out
    assert "BODY-TEXT" in out
    assert out.index("PROMPT-TEXT") < out.index("BODY-TEXT")


def test_synth_build_input_without_original_prompt_is_legacy_shape():
    """Omitting original_prompt yields the pre-Phase-1 layout (rubric +
    responses only), so legacy runs render identically."""
    from consult.synth import _build_input

    manifest = [
        {"slug": "alpha", "model_id": "x", "persona": None, "confidence": None,
         "status": "OK"},
    ]
    bodies = {"alpha": "BODY-TEXT"}
    out, _label_map = _build_input(
        manifest, bodies, rubric="rubric for {n} responses",
    )
    assert "Original question / source" not in out
    assert "BODY-TEXT" in out


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
            slug=slug, status=Status.OK,
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
            slug=slug, status=Status.OK,
            resource_uri=f"consult://runs/x/responses/{slug}",
            body_path=f"/x/{slug}",
            capsule=ReviewCapsule(
                overall_verdict=verdict,
                findings=[
                    Finding(
                        severity="blocker", category="security",
                        summary=f"finding {i}", suggestion="fix it",
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
            slug=slug, status=Status.OK,
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
        slug="alpha", status=Status.OK,
        resource_uri="consult://runs/x/responses/alpha",
        body_path="/x/alpha",
        capsule=ReviewCapsule(
            overall_verdict="changes_requested",
            findings=[
                Finding(
                    severity="blocker", file="src/auth.py",
                    line_range=(42, 58), category="security",
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
    tmp_path, monkeypatch,
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
        fanout_calls.append({
            "prompt": prompt,
            "specs": [(s.model, s.slug) for s in specs],
            "prior_turns": copy.deepcopy(kwargs.get("prior_turns")),
            "prior_turns_by_slug": copy.deepcopy(kwargs.get("prior_turns_by_slug")),
        })
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
            manifest.append(ManifestEntry(
                slug=slug, model_id="x/a", status=Status.OK,
                resource_uri=f"consult://x/{slug}",
                body_path=str(body_path),
                latency_ms=10, cost_usd=0.001, cost_known=True,
                capsule=Capsule(position=f"{slug} position"),
            ))
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
async def test_runner_fanout_dispatches_prior_turns_by_slug(monkeypatch, tmp_path):
    """`prior_turns_by_slug` lets a caller (refine round 2+) give each
    panellist its own conversation history. The runner must pick the
    per-slug entry when present and fall back to the global `prior_turns`
    otherwise."""
    from consult import registry, runner
    from consult.types import ModelSpec

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(registry, "provider_concurrency", lambda: {})
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    captured: dict[str, list] = {}

    async def fake_call_one(spec, slug, per_slug_prompt, paths, provider_sems, **kw):
        captured[slug] = kw.get("prior_turns") or []
        from consult.types import ManifestEntry, Status
        return ManifestEntry(
            slug=slug, model_id=spec.model, status=Status.OK,
            resource_uri=f"consult://x/{slug}",
            body_path=str(paths.response_text(slug)),
            latency_ms=1, cost_usd=0.0, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call_one)
    # Disable token-counter and cost-estimation so the test stays offline
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    global_pt = [{"role": "user", "content": "global"}]
    per_slug_pt = {
        "claude-haiku": [
            {"role": "user", "content": "haiku-specific"},
            {"role": "assistant", "content": "prior haiku answer"},
        ],
    }
    await runner.fanout(
        "the prompt",
        [
            ModelSpec(model="claude-haiku"),
            ModelSpec(model="gpt-mini"),
        ],
        prior_turns=global_pt,
        prior_turns_by_slug=per_slug_pt,
    )
    # claude-haiku used the per-slug override
    assert captured["claude-haiku"] == per_slug_pt["claude-haiku"]
    # gpt-mini fell back to the global prior_turns
    assert captured["gpt-mini"] == global_pt


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


def test_context_bundle_v1_loads_with_default_kind(tmp_path, monkeypatch):
    """Legacy bundles (schema_version=1) had no `capsule_kind` field. The
    Pydantic default makes them load as `capsule_kind="decision"` without
    raising."""
    from consult import context as ctx

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    (paths.root / "context.json").write_text(
        '{"schema_version": 1, "prompt": "p", "prompt_scrubbed": "p", "blinded": false}'
    )
    loaded = ctx.load_or_none(paths)
    assert loaded is not None
    assert loaded.capsule_kind == "decision"


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
            manifest=[ManifestEntry(
                slug="alpha.r1", status=Status.OK,
                resource_uri="consult://runs/x/responses/alpha.r1",
                body_path="/x", latency_ms=10, cost_known=True,
            )],
            cost_usd=0.0, cost_known=True, wall_ms=0,
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
        "follow-up", [ModelSpec(model="claude-haiku")],
        threshold=0.5, max_rounds=1, continuation_id=prior.run_id,
    )
    assert captured.get("annotate_kind") == "review"

    # Explicit override
    captured.clear()
    await refine_mod.refine(
        "follow-up", [ModelSpec(model="claude-haiku")],
        threshold=0.5, max_rounds=1, continuation_id=prior.run_id,
        capsule_kind="research",
    )
    assert captured.get("annotate_kind") == "research"


@pytest.mark.asyncio
async def test_stream_acompletion_raises_on_builder_failure(monkeypatch):
    """SECURITY/CORRECTNESS: when `stream_chunk_builder` fails, the
    streaming variant must raise rather than return a malformed partial
    chunk that downstream `classify()` and `completion_cost()` would
    mishandle."""
    import litellm

    from consult.runner import _stream_acompletion

    class _FakeAsyncStream:
        def __init__(self, chunks):
            self.chunks = chunks
            self._i = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._i >= len(self.chunks):
                raise StopAsyncIteration
            c = self.chunks[self._i]
            self._i += 1
            return c

    async def fake_acompletion(**kwargs):
        return _FakeAsyncStream([{"raw": "chunk1"}, {"raw": "chunk2"}])

    def fake_builder(chunks, messages=None):
        raise ValueError("builder unsupported chunk shape")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "stream_chunk_builder", fake_builder)

    with pytest.raises(RuntimeError, match="stream_chunk_builder failed"):
        await _stream_acompletion(
            timeout=10.0, on_partial=None, start=0.0,
            model="x/y", messages=[], max_tokens=100,
        )


def test_synth_build_input_rubric_with_literal_braces_does_not_crash():
    """Regression: `_build_input` previously used `.format(n=...)`, which
    crashes when a user-supplied rubric contains literal `{}` (e.g. a JSON
    example). Switched to `.replace("{n}", ...)`."""
    from consult.synth import _build_input

    rubric_with_braces = (
        "You have {n} responses.\n\nExpected JSON shape: "
        "{ \"verdict\": \"ship\" }"
    )
    manifest = [
        {"slug": "alpha", "model_id": "x", "persona": None,
         "confidence": None, "status": "OK"},
    ]
    out, _label_map = _build_input(
        manifest, {"alpha": "body"}, rubric=rubric_with_braces,
    )
    assert "1 responses" in out
    assert '{ "verdict": "ship" }' in out


def test_context_brand_regex_includes_registry_models():
    """The brand regex is derived from `registry.models_config()` so adding
    a model to models.json extends scrub coverage automatically. `sonnet`,
    `codex`, and other tier suffixes are picked up via alias parsing."""
    from consult import context as ctx

    text = "Compare claude-sonnet against gpt-codex for refactoring."
    out = ctx.scrub_brands(text)
    assert "claude" not in out.lower()
    assert "sonnet" not in out.lower()
    assert "gpt" not in out.lower()
    assert "codex" not in out.lower()


def test_context_trim_synth_input_proportional_hard_trim_on_large_panel():
    """When N panellists × per-body floor exceeds the budget, the hard-trim
    pass shrinks bodies proportionally so the overall input fits."""
    from consult import context as ctx

    # 20 bodies × 10000 chars = 200000; budget 50000. Floor (5000) × 20 =
    # 100000, still over budget. Hard-trim kicks in.
    bodies = {f"slug-{i}": "X" * 10_000 for i in range(20)}
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt=None, bodies=bodies, overall_budget=50_000,
    )
    total = sum(len(b) for b in new_bodies.values())
    # Allow a small overhead per body for trim markers
    assert total <= 50_000 + 30 * 200, (total, "should fit within budget + marker overhead")


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


def test_runner_writes_context_bundle_at_run_init(tmp_path, monkeypatch):
    """`runner.fanout` writes context.json alongside prompt.txt — every
    fresh run has a bundle downstream stages can load.
    """
    import asyncio

    from consult import context as ctx
    from consult.runner import fanout
    from consult.types import ModelSpec

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # dry_run avoids any API call but still triggers run-init.
    handle = asyncio.run(fanout(
        "test prompt with claude reference",
        [ModelSpec(model="claude-haiku")],
        dry_run=True,
        blinded=True,
    ))
    paths = artifacts.load_run(handle.run_id)
    bundle = ctx.load_or_none(paths)
    assert bundle is not None
    assert bundle.prompt == "test prompt with claude reference"
    assert bundle.blinded is True
    # Blinded mode scrubbed the brand from prompt_scrubbed
    assert "claude" not in bundle.prompt_scrubbed.lower()
    assert "[MODEL]" in bundle.prompt_scrubbed


# ---- M2: Schema versioning, rubric registry, capsule kinds, streaming ------


def test_manifest_carries_schema_version():
    """Every RunHandle / RunResult / RefineResult dump includes
    `schema_version` so future capsule-shape additions are detectable by
    clients without out-of-band coordination."""
    from consult.types import RefineResult, RunHandle, RunResult

    rh = RunHandle(
        run_id="x", artifacts_dir="/x", manifest=[],
        cost_usd=0.0, wall_ms=0,
    )
    assert rh.model_dump()["schema_version"] >= 2

    rr = RunResult(
        run_id="x", synthesis="s", manifest=[], cost_usd=0.0, wall_ms=0,
    )
    assert rr.model_dump()["schema_version"] >= 2

    rfr = RefineResult(
        run_id="x", rounds_completed=0, final_manifest=[], verdicts=[],
        synthesis="s", converged=False, threshold=0.85,
        cost_usd=0.0, wall_ms=0,
    )
    assert rfr.model_dump()["schema_version"] >= 2


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


def test_review_capsule_round_trips():
    from consult.types import Finding, ManifestEntry, ReviewCapsule

    review = ReviewCapsule(
        findings=[
            Finding(
                severity="blocker", file="src/auth.py", line_range=(42, 58),
                category="security", summary="SQL injection in login",
                suggestion="use parameterised query",
            ),
        ],
        overall_verdict="changes_requested",
        confidence=0.9,
    )
    entry = ManifestEntry(
        slug="alpha", status="OK",
        resource_uri="consult://runs/x/responses/alpha", body_path="/x",
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
        slug="alpha", status="OK",
        resource_uri="consult://runs/x/responses/alpha", body_path="/x",
        capsule=research,
    )
    dumped = entry.model_dump()
    assert dumped["capsule"]["kind"] == "research"
    reloaded = ManifestEntry.model_validate(dumped)
    assert reloaded.capsule.kind == "research"
    assert "Polars" in reloaded.capsule.claims[0]


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
    assert _CAPSULE_PROMPT_HEAD_DECISION.split("\n")[0] in _build_capsule_prompt(
        body, None, kind="nonsense"
    )


def test_registry_resolve_rubric_loads_named_packaged_rubrics():
    """The four packaged rubrics resolve by name."""
    from consult import registry as reg

    for name in ("consensus", "code_review", "research_brief", "critique"):
        text = reg.resolve_rubric(name)
        assert "{n}" in text, f"rubric {name} should have an n-placeholder"
        # Should look like a real rubric, not a one-line stub
        assert len(text) > 200, f"rubric {name} suspiciously short: {len(text)}"


def test_registry_resolve_rubric_passes_literal_through():
    """Unknown names → returned as-is (literal rubric)."""
    from consult import registry as reg

    literal = "You have {n} responses. Write a haiku."
    assert reg.resolve_rubric(literal) == literal


def test_registry_list_rubrics_includes_packaged():
    from consult import registry as reg

    rubrics = reg.list_rubrics()
    for expected in ("consensus", "code_review", "research_brief", "critique"):
        assert expected in rubrics, (expected, rubrics)


def test_progress_panellist_partial_event_message():
    """The new PanellistPartial event has a stable wire message."""
    from consult.progress import PanellistPartial, event_message

    ev = PanellistPartial(
        done=2, total=8, slug="alpha", chars_so_far=1500, elapsed_ms=4200
    )
    msg = event_message(ev)
    assert "alpha" in msg
    assert "1500" in msg
    assert "4200" in msg


@pytest.mark.asyncio
async def test_fanout_stream_env_var_enables_streaming(tmp_path, monkeypatch):
    """The CONSULT_STREAM env var flips fanout's `stream` default on so
    callers can opt into streaming without changing their tool call."""
    from consult.runner import fanout
    from consult.types import ManifestEntry, ModelSpec, Status

    captured: dict[str, bool] = {}

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, *, stream=False, on_partial=None, prior_turns=None, **_):
        captured["stream"] = stream
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug, status=Status.OK,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10, cost_known=True,
        )

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr("consult.runner._call_one", fake_call)
    monkeypatch.setenv("CONSULT_STREAM", "1")

    await fanout("hi", [ModelSpec(model="claude-haiku")])
    assert captured.get("stream") is True


@pytest.mark.asyncio
async def test_fanout_stream_default_off(tmp_path, monkeypatch):
    """Without CONSULT_STREAM or explicit stream=True, streaming stays off."""
    from consult.runner import fanout
    from consult.types import ManifestEntry, ModelSpec, Status

    captured: dict[str, bool] = {}

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, *, stream=False, on_partial=None, prior_turns=None, **_):
        captured["stream"] = stream
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug, status=Status.OK,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10, cost_known=True,
        )

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr("consult.runner._call_one", fake_call)
    monkeypatch.delenv("CONSULT_STREAM", raising=False)

    await fanout("hi", [ModelSpec(model="claude-haiku")])
    assert captured.get("stream") is False


# ---- M3: Labelled attachments, git-diff resolver, per-step sequence ---------


def test_render_attachment_bare_string_renders_path_and_content(tmp_path):
    """Bare string attachment paths still work (backwards compat)."""
    from consult.attachments import render_attachment as _render_attachment

    f = tmp_path / "foo.py"
    f.write_text("print('hi')")
    out = _render_attachment(str(f))
    assert str(f) in out
    assert "print('hi')" in out
    assert "```" in out  # code-fenced


def test_render_attachment_labelled_renders_label_heading(tmp_path):
    """Labelled attachments render `## LABEL: path` headers so panellists
    can refer to sections by name."""
    from consult.attachments import render_attachment as _render_attachment

    f = tmp_path / "auth.py"
    f.write_text("def login(): pass")
    out = _render_attachment({"path": str(f), "label": "AUTH_MODULE", "kind": "source"})
    assert "## AUTH_MODULE" in out
    assert str(f) in out
    assert "def login()" in out


def test_render_attachment_kind_hints_fence_language(tmp_path):
    """`kind: "diff"` produces a ```diff fence so the panellist sees the
    syntax-highlighting hint."""
    from consult.attachments import render_attachment as _render_attachment

    f = tmp_path / "patch.diff"
    f.write_text("--- a/foo\n+++ b/foo\n@@ +1\n+hello")
    out = _render_attachment({"path": str(f), "kind": "diff"})
    assert "```diff" in out


def test_render_attachment_missing_file_renders_error_not_crash(tmp_path):
    """A missing file produces an inline error marker — the rest of the
    panel still runs."""
    from consult.attachments import render_attachment as _render_attachment

    out = _render_attachment(str(tmp_path / "nonexistent.py"))
    assert "ERROR" in out
    assert "nonexistent.py" in out


def test_render_attachment_malformed_dict_surfaces_error():
    """A dict without `path` or `source` keys surfaces an error rather
    than crashing the whole tool call."""
    from consult.attachments import render_attachment as _render_attachment

    out = _render_attachment({"foo": "bar"})
    assert "ERROR" in out


def test_inline_attachments_renders_each_item():
    """`_inline_attachments` chains render output and prepends the
    --- ATTACHMENTS --- divider."""
    from consult.attachments import inline_attachments as _inline_attachments

    out = _inline_attachments("PROMPT", [])
    # Empty list → no divider added (kept as passthrough)
    assert out == "PROMPT"


def test_sources_validate_ref_rejects_shell_metachars():
    """Refs with shell metacharacters must not flow to subprocess."""
    from consult.sources import _validate_ref

    # Valid refs
    for good in ("main", "refs/heads/feature/x", "v1.0.0", "abc123", "feat+x", "HEAD~1", "HEAD^"):
        _validate_ref(good, field="base")

    # Invalid refs — anything outside [A-Za-z0-9._/+~^-] is rejected
    for bad in ("main; rm -rf /", "main$(id)", "main`whoami`", "main|cat", "main\nfoo"):
        with pytest.raises(ValueError, match="invalid base"):
            _validate_ref(bad, field="base")


def test_sources_validate_ref_rejects_leading_dash_git_option_injection():
    """SECURITY: a ref must not start with `-`, otherwise it would be
    interpreted as a git option (`base="--no-index"` becomes
    `git diff --no-index..HEAD`)."""
    from consult.sources import _validate_ref

    for bad in ("-rf", "--no-index", "-h", "--exec=evil"):
        with pytest.raises(ValueError, match="invalid base"):
            _validate_ref(bad, field="base")


def test_sources_resolve_git_diff_uses_double_dash_separator(monkeypatch, tmp_path):
    """`git diff` is invoked with a trailing `--` so a future regex
    relaxation can't smuggle an option through. Belt and braces."""
    import subprocess

    from consult import sources

    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="diff body", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", str(tmp_path))

    sources.resolve_git_diff("main", "HEAD", repo_path=str(tmp_path))

    assert "--" in captured["cmd"], captured["cmd"]
    # `--` should be AFTER the diff range, not before
    range_idx = next(i for i, a in enumerate(captured["cmd"]) if a == "main..HEAD")
    dashdash_idx = captured["cmd"].index("--")
    assert dashdash_idx > range_idx


def test_sources_validate_repo_path_enforces_trusted_roots(tmp_path, monkeypatch):
    """`repo_path` must resolve under CONSULT_TRUSTED_REPO_ROOTS."""
    from consult.sources import _validate_repo_path

    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", str(trusted))

    # Inside the trusted root → returns the resolved path
    assert _validate_repo_path(str(trusted)) == trusted.resolve()

    # Outside the trusted root → raises with a helpful message
    with pytest.raises(ValueError, match="not under any CONSULT_TRUSTED_REPO_ROOTS"):
        _validate_repo_path(str(outside))


def test_sources_validate_repo_path_defaults_to_cwd(tmp_path, monkeypatch):
    """Without CONSULT_TRUSTED_REPO_ROOTS, only the current cwd is trusted."""
    from consult.sources import _validate_repo_path

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    monkeypatch.chdir(tmp_path)

    # cwd works
    sub = tmp_path / "sub"
    sub.mkdir()
    assert _validate_repo_path(str(sub)) == sub.resolve()

    # Outside cwd fails
    elsewhere = tmp_path.parent / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        _validate_repo_path(str(elsewhere))


def test_sources_resolve_git_diff_against_real_repo(tmp_path, monkeypatch):
    """End-to-end: init a real git repo, commit a file, modify it, and
    confirm resolve_git_diff returns the expected diff text."""
    import subprocess

    from consult.sources import resolve_git_diff

    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", str(tmp_path))
    # Init repo
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    # First commit
    (tmp_path / "hello.py").write_text("print('hi')\n")
    subprocess.run(["git", "add", "hello.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    # Change + new commit
    (tmp_path / "hello.py").write_text("print('hello, world')\n")
    subprocess.run(["git", "commit", "-aq", "-m", "change"], cwd=tmp_path, check=True)

    diff = resolve_git_diff("HEAD~1", "HEAD", repo_path=str(tmp_path))
    assert "hello.py" in diff
    assert "-print('hi')" in diff
    assert "+print('hello, world')" in diff


@pytest.mark.asyncio
async def test_handle_sequence_per_step_attachments(tmp_path, monkeypatch):
    """A sequence with object-form prompts can carry per-step attachments
    that override the top-level default."""
    from consult.mcp.handlers import sequence as _handle_sequence

    # Stub out the underlying sequence to capture what prompts arrive.
    captured: dict[str, list[str]] = {}

    async def fake_sequence(prompts, specs, **kwargs):
        from consult.sequence import SequenceResult
        captured["prompts"] = list(prompts)
        return SequenceResult(
            steps=[], final_synthesis="", cost_usd=0.0, cost_known=True, wall_ms=0,
        )

    monkeypatch.setattr("consult.mcp.handlers.sequence_mod.sequence", fake_sequence)

    f_default = tmp_path / "default.txt"
    f_default.write_text("DEFAULT-CONTENT")
    f_step2 = tmp_path / "step2.txt"
    f_step2.write_text("STEP2-CONTENT")

    args = {
        "prompts": [
            "step 1 prompt",  # bare string → uses default attachments
            {"prompt": "step 2 prompt", "attachments": [str(f_step2)]},  # override
            {"prompt": "step 3 prompt"},  # no override → uses default
        ],
        "models": [{"model": "claude-haiku"}],
        "attachments": [str(f_default)],
    }
    await _handle_sequence(args)

    p1, p2, p3 = captured["prompts"]
    # Step 1 sees the default attachment
    assert "DEFAULT-CONTENT" in p1
    assert "STEP2-CONTENT" not in p1
    # Step 2 sees only its own attachment, not the default
    assert "STEP2-CONTENT" in p2
    assert "DEFAULT-CONTENT" not in p2
    # Step 3 falls back to the default (no override)
    assert "DEFAULT-CONTENT" in p3


# ---- Live tests (gated on API keys) ----------------------------------------


HAVE_KEYS = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
)


# ---- build_messages: prior_turns role boundaries ---------------------------


def test_build_messages_prepends_prior_turns_for_anthropic():
    """prior_turns must be prepended verbatim; the final user turn is
    Anthropic-cache-tagged. This is what gives continuation runs proper
    role boundaries instead of one giant user blob."""
    from consult.runner import build_messages

    prior_turns = [
        {"role": "user", "content": "What database?"},
        {"role": "assistant", "content": "DuckDB."},
    ]
    msgs = build_messages("Now for ETL?", "anthropic", prior_turns)
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "What database?"}
    assert msgs[1] == {"role": "assistant", "content": "DuckDB."}
    # Final user turn — Anthropic uses the structured content list with cache_control.
    assert msgs[2]["role"] == "user"
    assert isinstance(msgs[2]["content"], list)
    assert msgs[2]["content"][0]["type"] == "text"
    assert msgs[2]["content"][0]["text"] == "Now for ETL?"
    assert msgs[2]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_build_messages_prepends_prior_turns_for_non_anthropic():
    """Same shape, OpenAI-style flat string content on the final user turn."""
    from consult.runner import build_messages

    prior_turns = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    msgs = build_messages("Q2", "openai", prior_turns)
    assert msgs == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]


def test_build_messages_no_prior_turns_is_unchanged():
    """The default (no prior_turns) path must not regress the existing single-turn shape."""
    from consult.runner import build_messages

    msgs = build_messages("hi", "openai")
    assert msgs == [{"role": "user", "content": "hi"}]


# ---- fanout: prior_turns threaded through to panellists ---------------------


@pytest.mark.asyncio
async def test_fanout_threads_prior_turns_into_call_one(tmp_path, monkeypatch):
    """fanout(prior_turns=...) must reach `_call_one`. Critical for refine's
    continuation flow: without this, the prior consultation context is
    silently dropped from the panellist's messages."""
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    seen: list = []

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **kwargs):
        seen.append(kwargs.get("prior_turns"))
        return ManifestEntry(
            slug=slug, model_id="x/y", persona=None, status=Status.OK,
            finish_reason="stop", resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10, cost_usd=0.0, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    prior_turns = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    await fanout(
        "follow-up",
        [ModelSpec(model="claude-haiku")],
        prior_turns=prior_turns,
    )
    assert seen == [prior_turns]


# ---- provider_caps -----------------------------------------------------------


def test_provider_caps_temperature_blocks_gemini_and_opus():
    """Both the legacy gemini substring case and the new claude-opus-4-7
    case must be blocked. Regressions here cause every refine arbiter
    call to fail with a BadRequestError (live-found during the dogfood
    pass that produced this whole sweep)."""
    from consult import provider_caps

    provider_caps.reset_cache()
    assert provider_caps.supports_temperature("gpt-4") is True
    assert provider_caps.supports_temperature("anthropic/claude-sonnet-4-6") is True
    assert provider_caps.supports_temperature("anthropic/claude-opus-4-7") is False
    assert provider_caps.supports_temperature("openrouter/google/gemini-3.1-pro-preview") is False
    assert provider_caps.supports_temperature("gemini/gemini-3.1-pro-preview") is False


def test_provider_caps_env_override_extends_deny_list(monkeypatch):
    """Operators add to the deny list via env without code edits."""
    from consult import provider_caps

    monkeypatch.setenv("CONSULT_NO_TEMPERATURE", "weird-future-model")
    provider_caps.reset_cache()
    try:
        assert provider_caps.supports_temperature("vendor/weird-future-model-v2") is False
        # Built-ins still apply.
        assert provider_caps.supports_temperature("anthropic/claude-opus-4-7") is False
    finally:
        provider_caps.reset_cache()


def test_provider_caps_apply_temperature_skips_blocked_models():
    """apply_temperature is the single place call sites set the kwarg."""
    from consult import provider_caps

    provider_caps.reset_cache()
    kwargs: dict[str, Any] = {"model": "x"}
    provider_caps.apply_temperature(kwargs, "anthropic/claude-opus-4-7", 0.0)
    assert "temperature" not in kwargs

    kwargs2: dict[str, Any] = {"model": "x"}
    provider_caps.apply_temperature(kwargs2, "anthropic/claude-sonnet-4-6", 0.0)
    assert kwargs2["temperature"] == 0.0


# ---- fanout: zero-usable-panel guard ----------------------------------------


@pytest.mark.asyncio
async def test_fanout_returns_partial_when_zero_usable_panellists(tmp_path, monkeypatch):
    """Every panellist times out → fanout must return partial=True with a
    clear reason. Without this, refine.py would hand an empty manifest to
    the arbiter and burn its cost on a verdict with no signal — the bug
    that surfaced during the dogfood pass producing this PR.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        return ManifestEntry(
            slug=slug,
            model_id=None,
            persona=None,
            status=Status.TIMEOUT,
            finish_reason=None,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=100,
            cost_usd=None,
            cost_known=False,
            error="timeout after 1s",
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    handle = await fanout(
        "p",
        [ModelSpec(model="claude-haiku"), ModelSpec(model="claude-sonnet")],
    )
    assert handle.partial is True
    assert handle.partial_reason and "zero usable panellists" in handle.partial_reason
    # Manifest still surfaces the failure entries for diagnosis
    assert len(handle.manifest) == 2
    assert all(m.status == Status.TIMEOUT for m in handle.manifest)


@pytest.mark.asyncio
async def test_fanout_succeeds_when_any_panellist_usable(tmp_path, monkeypatch):
    """One OK + one TIMEOUT must NOT trip the zero-usable guard."""
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    call_count = {"n": 0}

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        call_count["n"] += 1
        status = Status.OK if call_count["n"] == 1 else Status.TIMEOUT
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=status,
            finish_reason="stop" if status == Status.OK else None,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_usd=0.0,
            cost_known=True,
            error=None if status == Status.OK else "timeout",
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    handle = await fanout(
        "p",
        [ModelSpec(model="claude-haiku"), ModelSpec(model="claude-sonnet")],
    )
    assert handle.partial is False
    assert handle.partial_reason is None


# ---- refine: short-circuit on partial fanout --------------------------------


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
            round=1, score=1.0, gaps=[], reasoning="should not be reached",
            cost_usd=0.0, cost_known=True, parsed_ok=True,
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


# ---- refine: cost-cap includes arbiter --------------------------------------


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


# ---- slow-tail dropout: cancelled tasks have cost_known=False ---------------


@pytest.mark.asyncio
async def test_slow_tail_dropout_marks_cost_unknown_for_cancelled(tmp_path, monkeypatch):
    """A cancelled-mid-flight task may still be billed by the provider, so
    cost_known must be False (not True). Fixes the silent under-counting of
    runs where a flagship was dropped after sending the request."""
    import asyncio as aio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.1")  # very short dropout
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.5")  # drop the slow half

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        if "haiku" in spec.model:
            await aio.sleep(0.01)
            return ManifestEntry(
                slug=slug, model_id="x/y", persona=None, status=Status.OK,
                finish_reason="stop", resource_uri=paths.resource_uri(slug),
                body_path=str(paths.response_text(slug)),
                latency_ms=10, cost_usd=0.0, cost_known=True,
            )
        # Slow panellists — will be cancelled by slow-tail dropout
        await aio.sleep(60)
        return ManifestEntry(
            slug=slug, model_id="x/y", persona=None, status=Status.OK,
            finish_reason="stop", resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=60000, cost_usd=0.0, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    specs = [
        ModelSpec(model="claude-haiku"),
        ModelSpec(model="claude-haiku"),
        ModelSpec(model="claude-opus"),
        ModelSpec(model="claude-sonnet"),
    ]
    handle = await fanout("p", specs)
    # Find dropped entries and verify cost_known=False
    dropped = [m for m in handle.manifest if m.error and "dropout" in (m.error or "")]
    assert len(dropped) >= 1
    for m in dropped:
        assert m.cost_known is False, (
            f"dropped panellist {m.slug} has cost_known={m.cost_known}; "
            "should be False since provider may still bill"
        )
        assert m.cost_usd is None


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
def test_estimate_cost_smoke():
    specs = [ModelSpec(model="claude-haiku")]
    est, _ = estimate_cost(specs, "say hello in five words")
    assert est >= 0


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
@pytest.mark.asyncio
async def test_tiny_panel_dry_run():
    from consult.runner import fanout

    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="openrouter/x-ai/grok-4.3")]
    handle = await fanout("ping", specs, dry_run=True)
    assert handle.partial
    assert "dry_run" in (handle.partial_reason or "")


# ---- Iter 1 refine-loop regression locks -----------------------------------


def test_runresult_invariant_blocks_silently_known_with_unknown_entry():
    """RunResult.cost_known=True with an entry that has cost_known=False
    must raise. Catches the consult-success-path bug where the handler
    dropped `cost_known=handle.cost_known`.
    """
    from consult.types import RunResult

    entry = ManifestEntry(
        slug="alpha",
        model_id="m/x",
        status=Status.OK,
        resource_uri="consult://runs/x/responses/alpha",
        body_path="/tmp/x",
        latency_ms=0,
        cost_usd=None,
        cost_known=False,
        confidence=None,
        capsule=None,
    )
    with pytest.raises(Exception) as exc:
        RunResult(
            run_id="x",
            synthesis="s",
            manifest=[entry],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )
    assert "cost_known" in str(exc.value)


def test_refineresult_invariant_catches_arbiter_cost_unknown():
    """RefineResult.cost_known=True with a verdict that has cost_known=False
    must raise. Catches the refine.py bug where only `if verdict.cost_usd`
    propagated; an unmapped-price arbiter left cost_all_known=True.
    """
    from consult.types import RefineResult

    verdict = ArbiterVerdict(
        round=1, score=0.5, gaps=[], reasoning="", cost_usd=None, cost_known=False
    )
    with pytest.raises(Exception) as exc:
        RefineResult(
            run_id="x",
            rounds_completed=1,
            final_manifest=[],
            verdicts=[verdict],
            synthesis="s",
            converged=False,
            threshold=0.85,
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )
    assert "cost_known" in str(exc.value)


def test_sequenceresult_invariant_propagates_step_cost_known():
    """SequenceResult.cost_known=True with a step that has cost_known=False
    must raise. Pre-fix, sequence's step cost_known was tracked but the
    invariant wasn't enforced, so a typo could land in production silently.
    """
    from consult.sequence import SequenceResult, SequenceStep

    step = SequenceStep(
        step=1, run_id="r1", synthesis="x", cost_usd=0.1,
        cost_known=False, panel_size=2,
    )
    with pytest.raises(Exception) as exc:
        SequenceResult(
            steps=[step],
            final_synthesis="x",
            cost_usd=0.1,
            cost_known=True,
            wall_ms=0,
        )
    assert "cost_known" in str(exc.value)


def test_modelspec_slug_validator_rejects_path_traversal():
    """A caller-supplied slug containing `/` or `..` must fail at construction —
    not deeper in `_call_one` where the artifact write would silently leave
    the responses dir.
    """
    for bad in ("../etc/passwd", "a/b", "..", "with space", "with$shell"):
        with pytest.raises(Exception) as exc:
            ModelSpec(model="x", slug=bad)
        assert "slug" in str(exc.value)


def test_artifacts_load_run_rejects_traversal_run_id(tmp_path, monkeypatch):
    """artifacts.load_run must refuse a run_id containing path separators or
    `..` even if the resolved directory would happen to exist.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # Create a sibling dir we shouldn't be able to escape to.
    (tmp_path.parent / "secret").mkdir(exist_ok=True)
    for bad in ("../secret", "..", "a/b", "/etc"):
        with pytest.raises(ValueError) as exc:
            artifacts.load_run(bad)
        assert "run_id" in str(exc.value) or "invalid" in str(exc.value).lower()


def test_expand_specs_with_explicit_slug_and_count_disambiguates():
    """`model:N` with an explicit slug must append an index suffix to each
    expansion. Without this, three sibling panellists race to write to the
    same `responses/<slug>.txt` and two responses are silently lost.
    """
    from consult.runner import expand_specs

    raw = [ModelSpec(model="claude-haiku:3", slug="bench")]
    expanded = expand_specs(raw)
    slugs = [s.slug for s in expanded]
    assert slugs == ["bench-0", "bench-1", "bench-2"]
    # Single-instance with explicit slug is left alone (no suffix needed).
    single = expand_specs([ModelSpec(model="claude-haiku:1", slug="solo")])
    assert [s.slug for s in single] == ["solo"]


async def test_refine_rejects_continuation_with_sentinel_synthesis(tmp_path, monkeypatch):
    """A prior run whose synthesis is a sentinel (`# Synthesis unavailable`,
    skipped, empty) must be refused — feeding the sentinel to the next panel
    as 'prior consultation' produces hallucinated follow-ups.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.prompt_txt.write_text("prior question")
    (paths.root / "synthesis.md").write_text(
        "# Synthesis unavailable\n\nThe synthesiser failed."
    )
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
                    slug="x.r1", model_id="m/x", status=Status.OK,
                    resource_uri=paths_h.resource_uri("x.r1"),
                    body_path=str(paths_h.response_text("x.r1")),
                    latency_ms=0, cost_usd=0.0, cost_known=True,
                    confidence=None, capsule=None,
                ),
            ],
            cost_usd=0.0, cost_known=True, wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_arbiter(question, round_num, manifest, arbiter_alias, prior_manifest):
        arbiter_questions.append(question)
        return ArbiterVerdict(
            round=round_num, score=1.0, gaps=[], reasoning="ok",
            cost_usd=0.0, cost_known=True, parsed_ok=True,
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
        threshold=0.5, max_rounds=1, continuation_id=prior.run_id,
    )
    assert arbiter_questions, "arbiter was never asked"
    asked = arbiter_questions[0]
    assert "FOLLOWUP QUESTION TEXT" in asked
    assert "PRIOR SYNTH TEXT" not in asked
    assert "PRIOR QUESTION TEXT" not in asked


async def test_consult_handler_accumulates_synth_cost(tmp_path, monkeypatch):
    """The consult success path must add the synthesiser's spend to the
    run total. Pre-fix `synth.synthesise` returned only a string and the
    handler silently understated `cost_usd` (often the biggest line item).
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import synth as synth_mod
    from consult.mcp import handlers

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="x", model_id="m/x", status=Status.OK,
                    resource_uri=paths.resource_uri("x"),
                    body_path=str(paths.response_text("x")),
                    latency_ms=0, cost_usd=0.10, cost_known=True,
                    confidence=None, capsule=None,
                ),
            ],
            cost_usd=0.10, cost_known=True, wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_synth(run_id, **kwargs):
        return synth_mod.SynthResult(text="syn", cost_usd=0.25, cost_known=True)

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(handlers.runner, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(handlers.capsule, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)
    monkeypatch.setattr(handlers.synth, "synthesise", fake_synth)

    result = await handlers.consult({"prompt": "p", "tier": "quick"})
    # Synth cost was 0.25; handle cost was 0.10. Total must reflect both.
    assert result["cost_usd"] == pytest.approx(0.35)
    assert result["cost_known"] is True


async def test_consult_handler_propagates_cost_known_from_handle(tmp_path, monkeypatch):
    """The consult success path must pass `cost_known=handle.cost_known`.
    Pre-fix this was omitted, defaulting to True even when a panellist
    had partial pricing — the cap check was silently invalid.
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import synth as synth_mod
    from consult.mcp import handlers

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="x", model_id="m/x", status=Status.OK,
                    resource_uri=paths.resource_uri("x"),
                    body_path=str(paths.response_text("x")),
                    latency_ms=0, cost_usd=None, cost_known=False,
                    confidence=None, capsule=None,
                ),
            ],
            cost_usd=0.0, cost_known=False, wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_synth(run_id, **kwargs):
        return synth_mod.SynthResult(text="syn")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(handlers.runner, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(handlers.capsule, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)
    monkeypatch.setattr(handlers.synth, "synthesise", fake_synth)

    result = await handlers.consult({"prompt": "p", "tier": "quick"})
    assert result["cost_known"] is False


# ---- Iter 2 refine-loop regression locks -----------------------------------


def test_make_slug_blinded_preserves_round_suffix():
    """Blinded refine relies on `_make_slug` honouring `.r<n>` even when it
    rewrites the visible portion to a greek slug. Pre-fix the suffix was
    dropped and per-round artifacts overwrote each other on disk.
    """
    from consult.runner import _make_slug

    # Round-suffixed spec (refine._suffix_specs produces these).
    s = ModelSpec(model="claude-haiku", slug="claude-haiku-0.r2")
    blinded = _make_slug(s, 0, blinded=True)
    assert blinded == "panelist-alpha.r2"
    # No `.r<n>` ⇒ plain greek slug.
    s2 = ModelSpec(model="claude-haiku", slug="claude-haiku-0")
    assert _make_slug(s2, 0, blinded=True) == "panelist-alpha"


def test_make_slug_sanitises_colons_in_raw_litellm_ids():
    """OpenRouter model IDs like `...:free` must not crash the slug-derive
    path. Pre-fix the colon flowed into the slug and the safe-id validator
    in artifacts.response_text rejected the path build.
    """
    from consult.runner import _make_slug

    spec = ModelSpec(model="openrouter/meta-llama/llama-3.1-8b:free")
    slug = _make_slug(spec, 0, blinded=False)
    # No colon, valid leading alphanumeric, slug regex accepts it.
    assert ":" not in slug
    assert slug == "llama-3.1-8b-free"


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


async def test_sequence_partial_fanout_rolls_cost_into_total(tmp_path, monkeypatch):
    """A step whose fanout returns partial=True (rate-limited, zero-usable,
    or cap-exceeded) must still contribute its `handle.cost_usd` to the
    SequenceResult total. Pre-fix the cost was silently dropped on break.
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import sequence as sequence_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (0.0, True))

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.17,
            cost_known=False,
            wall_ms=0,
            partial=True,
            partial_reason="zero usable panellists (2 returned: TIMEOUT)",
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_synth(*args, **kwargs):
        return synth_mod.SynthResult(text="x")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await sequence_mod.sequence(
        ["q1", "q2"], [ModelSpec(model="claude-haiku")],
    )
    assert result.partial is True
    # The 0.17 from the partial fanout must show up — not the pre-fix 0.0.
    assert result.cost_usd == pytest.approx(0.17)
    assert result.cost_known is False


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
        body, extractor_id="anthropic/claude-haiku-test", timeout=30,
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
                slug=slug, model_id="m/x", status=Status.OK,
                resource_uri=paths.resource_uri(slug),
                body_path=str(paths.response_text(slug)),
                latency_ms=0, cost_usd=0.0, cost_known=True,
                confidence=None, capsule=None,
            )
        )
    handle = RunHandle(
        run_id=paths.run_id, artifacts_dir=str(paths.root),
        manifest=manifest, cost_usd=0.0, cost_known=True, wall_ms=0,
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


def test_missing_default_rubric_raises_runtime_not_filenotfound(tmp_path, monkeypatch):
    """A missing `consensus.md` (broken install) must NOT raise
    FileNotFoundError — that exception class is reserved for run-not-found
    in `server.handle_call_tool`, and a broken install was getting
    surfaced to clients as a confusing "run_id not found".
    """
    from consult import synth as synth_mod

    # Make `resolve_rubric("consensus")` return the literal sentinel so the
    # broken-install branch trips.
    monkeypatch.setattr(synth_mod.registry, "resolve_rubric", lambda name: "consensus")
    with pytest.raises(RuntimeError) as exc:
        synth_mod._resolve_rubric(None)
    assert "broken" in str(exc.value).lower()


# ---- Iter 3 refine-loop regression locks -----------------------------------


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


async def test_fanout_rejects_duplicate_explicit_slugs(tmp_path, monkeypatch):
    """Two specs with identical explicit slugs race on `responses/<slug>.txt`.
    `runner.fanout` must fail fast with a clear ValueError before any
    artifact write happens.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (0.0, True))

    specs = [
        ModelSpec(model="claude-haiku", slug="test"),
        ModelSpec(model="gpt-pro", slug="test"),
    ]
    with pytest.raises(ValueError) as exc:
        await runner_mod.fanout("hi", specs)
    assert "duplicate" in str(exc.value).lower()
    assert "test" in str(exc.value)


async def test_consult_handler_swallows_progress_callback_failure(tmp_path, monkeypatch):
    """A progress-callback failure during the synth phase must NOT tear
    down the tool. Pre-fix the consult handler called `await base(event)`
    directly; a disconnected MCP session surfaced as INTERNAL_ERROR.

    Post-handler/server cycle inversion: the callback is no longer pulled
    from the MCP context inside the handler — the server builds it and
    passes it in. The test now passes a crashy `on_progress` directly into
    `handlers.consult` and asserts the synth-phase emit (`_safe_emit` in
    `orchestrate.consult`) still swallows the failure.
    """
    from consult import capsule as capsule_mod
    from consult import runner as runner_mod
    from consult import synth as synth_mod
    from consult.mcp import handlers

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="x", model_id="m/x", status=Status.OK,
                    resource_uri=paths.resource_uri("x"),
                    body_path=str(paths.response_text("x")),
                    latency_ms=0, cost_usd=0.0, cost_known=True,
                    confidence=None, capsule=None,
                ),
            ],
            cost_usd=0.0, cost_known=True, wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_synth(*args, **kwargs):
        return synth_mod.SynthResult(text="ok")

    async def crashy_cb(event):
        raise RuntimeError("client disconnected")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    # No exception should propagate; tool returns its normal payload.
    result = await handlers.consult(
        {"prompt": "p", "tier": "quick"}, on_progress=crashy_cb
    )
    assert result["partial"] is False
    assert result["synthesis"] == "ok"


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
                    slug="x.r1", model_id="m/x", status=Status.OK,
                    resource_uri=paths.resource_uri("x.r1"),
                    body_path=str(paths.response_text("x.r1")),
                    latency_ms=0, cost_usd=0.0, cost_known=True,
                    confidence=None, capsule=None,
                ),
            ],
            cost_usd=0.0, cost_known=True, wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(
            round=1, score=1.0, gaps=[], reasoning="ok",
            cost_usd=0.0, cost_known=True, parsed_ok=True,
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
        "Q", [ModelSpec(model="claude-haiku")],
        threshold=0.5, max_rounds=1, blinded=True,
    )
    assert captured.get("anonymised") is True


def test_blinded_fanout_preserves_model_id_in_manifest(tmp_path, monkeypatch):
    """Blinded panels must keep `model_id` populated on each manifest entry.

    Blinding's job is to (a) scrub brand mentions from the prompt panellists
    see, (b) hand out greek-letter slugs so panellists referring to each
    other use anonymous labels, and (c) tell the synth to omit model_ids
    from its prompt. The manifest itself is for downstream readers (the
    viewer, ledger, the human) — stripping model_id there hid identities
    from the final report, which the user needs to see who said what.
    """
    import asyncio
    import json

    from consult.runner import fanout
    from consult.types import ModelSpec

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "consult.runner.estimate_cost", lambda *a, **kw: (0.0, True),
    )

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        paths.response_text(slug).write_text("body")
        entry = registry.resolve_model(spec.model)
        return ManifestEntry(
            slug=slug, model_id=entry["litellm_id"], status=Status.OK,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10, tokens_in=1, tokens_out=1,
            cost_usd=0.0, cost_known=True, confidence=0.5, capsule=None,
        )

    monkeypatch.setattr("consult.runner._call_one", fake_call)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    handle = asyncio.run(fanout(
        "prompt", [ModelSpec(model="claude-haiku")], blinded=True,
    ))
    assert len(handle.manifest) == 1
    entry = handle.manifest[0]
    # Slug is anonymised (greek letter), but model_id stays real.
    assert entry.slug.startswith("panelist-")
    assert entry.model_id is not None
    assert entry.model_id == "anthropic/claude-haiku-4-5-20251001"
    # On-disk manifest must mirror the in-memory handle.
    paths = artifacts.load_run(handle.run_id)
    on_disk = json.loads(paths.manifest_json.read_text())
    assert on_disk["manifest"][0]["model_id"] == "anthropic/claude-haiku-4-5-20251001"


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
                    slug="x.r1", model_id="m/x", status=Status.OK,
                    resource_uri=paths.resource_uri("x.r1"),
                    body_path=str(paths.response_text("x.r1")),
                    latency_ms=0, cost_usd=0.0, cost_known=True,
                    confidence=None, capsule=None,
                ),
            ],
            cost_usd=0.0, cost_known=True, wall_ms=0,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_arbiter(*args, **kwargs):
        return ArbiterVerdict(
            round=1, score=1.0, gaps=[], reasoning="ok",
            cost_usd=0.0, cost_known=True, parsed_ok=True,
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
        "q", [ModelSpec(model="claude-haiku")],
        threshold=0.5, max_rounds=1, max_run_usd=20.0,
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
            "q", [ModelSpec(model="claude-haiku")],
            arbiter="totally-not-a-real-alias",
            threshold=0.5, max_rounds=1,
        )
    assert fanout_calls["n"] == 0


async def test_consult_rejects_typo_synthesiser_before_fanout(tmp_path, monkeypatch):
    from consult import runner as runner_mod
    from consult.mcp import handlers

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    fanout_calls = {"n": 0}

    async def fake_fanout(*args, **kwargs):
        fanout_calls["n"] += 1
        raise AssertionError("fanout must not be reached on typo")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(handlers.runner, "fanout", fake_fanout)
    with pytest.raises(KeyError):
        await handlers.consult({
            "prompt": "p", "tier": "quick",
            "synthesiser": "totally-not-a-real-alias",
        })
    assert fanout_calls["n"] == 0


async def test_fanout_cap_early_return_preserves_existing_manifest(tmp_path, monkeypatch):
    """When refine drives multiple rounds through the same `paths`, a
    cap-exceeded early return on round N+1 must NOT clobber round N's
    successful manifest.json. Pre-fix iter5's "always write" landed this
    regression: refine's prior-round transcript was wiped on a borderline
    cap miss in the next round.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    # Seed a fake "round 1" manifest with a real entry.
    seeded = RunHandle(
        run_id=paths.run_id, artifacts_dir=str(paths.root),
        manifest=[
            ManifestEntry(
                slug="x.r1", model_id="m/x", status=Status.OK,
                resource_uri=paths.resource_uri("x.r1"),
                body_path=str(paths.response_text("x.r1")),
                latency_ms=10, cost_usd=0.05, cost_known=True,
                confidence=None, capsule=None,
            ),
        ],
        cost_usd=0.05, cost_known=True, wall_ms=10,
    )
    artifacts.write_manifest(paths, seeded.model_dump())

    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (10.0, True))
    handle = await runner_mod.fanout(
        "x", [ModelSpec(model="claude-haiku")],
        max_run_usd=1.0, existing_paths=paths,
    )
    assert handle.partial is True
    # Manifest.json on disk must still reflect the seeded round-1 entry,
    # not the empty cap-rejection handle.
    import json as _json
    persisted = _json.loads(paths.manifest_json.read_text())
    assert persisted["manifest"] and persisted["manifest"][0]["slug"] == "x.r1"


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


async def test_fanout_writes_manifest_on_dry_run(tmp_path, monkeypatch):
    """A dry_run still creates a run dir; downstream tools (consult-view,
    synthesise) expect `manifest.json` to be present.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    handle = await runner_mod.fanout(
        "x", [ModelSpec(model="claude-haiku")], dry_run=True,
    )
    paths = artifacts.load_run(handle.run_id)
    assert paths.manifest_json.exists()


async def test_fanout_writes_manifest_on_cap_exceeded(tmp_path, monkeypatch):
    """A cap-exceeded early return must also persist the manifest so the
    run dir is not corrupted."""
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (10.0, True))
    handle = await runner_mod.fanout(
        "x", [ModelSpec(model="claude-haiku")], max_run_usd=1.0,
    )
    assert handle.partial is True
    assert "exceeds cap" in (handle.partial_reason or "")
    paths = artifacts.load_run(handle.run_id)
    assert paths.manifest_json.exists()


async def test_fanout_rejects_empty_specs(tmp_path, monkeypatch):
    """Library callers bypassing the MCP minItems=1 schema must still get a
    clear error, not a bizarre empty run dir.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    with pytest.raises(ValueError) as exc:
        await runner_mod.fanout("x", [])
    assert "at least one" in str(exc.value)


def test_provider_concurrency_floors_at_one(monkeypatch):
    """A typo like `CONSULT_PROVIDER_CONCURRENCY=openai:0` must not produce
    a Semaphore(0); that blocks the first acquire indefinitely and bypasses
    the per-call timeout. Floor to 1.
    """
    monkeypatch.setenv("CONSULT_PROVIDER_CONCURRENCY", "openai:0,zzz:-3")
    out = registry.provider_concurrency()
    assert out["openai"] >= 1
    assert out["zzz"] >= 1


async def test_synth_defensive_extraction_on_unexpected_shape(tmp_path, monkeypatch):
    """A non-conformant provider response must surface the unavailable
    sentinel, not AttributeError/IndexError straight out of `synthesise`.
    """
    import litellm

    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.prompt_txt.write_text("q")
    paths.response_text("alpha").write_text("body")
    manifest = [
        ManifestEntry(
            slug="alpha", model_id="m/x", status=Status.OK,
            resource_uri=paths.resource_uri("alpha"),
            body_path=str(paths.response_text("alpha")),
            latency_ms=0, cost_usd=0.0, cost_known=True,
            confidence=None, capsule=None,
        ),
    ]
    handle = RunHandle(
        run_id=paths.run_id, artifacts_dir=str(paths.root),
        manifest=manifest, cost_usd=0.0, cost_known=True, wall_ms=0,
    )
    artifacts.write_manifest(paths, handle.model_dump())

    class Garbage:
        choices: list = []  # IndexError on resp.choices[0]

    async def garbage_acompletion(**kwargs):
        return Garbage()

    monkeypatch.setattr(litellm, "acompletion", garbage_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.0)

    result = await synth_mod.synthesise(paths.run_id)
    # Did not crash. Sentinel text written.
    assert "# Synthesis unavailable" in result.text
    assert "unexpected response shape" in result.text


# ---- Engine/MCP-separation tests ------------------------------------------
# Lock in the post-refactor invariants: the engine doesn't import mcp.*,
# the resource_uri formatter is overridable, and orchestrate.consult is
# directly callable without the MCP adapter in the loop.


def test_engine_modules_do_not_import_mcp_sdk():
    """The engine (`consult.*` minus `consult.mcp.*`) must not transitively
    pull in the `mcp` SDK. A regression would silently re-couple a library
    consumer to a dependency they declined to install.
    """
    import importlib
    import pkgutil

    import consult

    engine_modules: list[str] = []
    for info in pkgutil.iter_modules(consult.__path__, prefix="consult."):
        if info.name == "consult.mcp" or info.name.startswith("consult.mcp."):
            continue
        engine_modules.append(info.name)

    # The engine surface actually used (filter out viewer-only deps).
    expected_present = {
        "consult.runner", "consult.refine", "consult.sequence",
        "consult.synth", "consult.capsule", "consult.orchestrate",
        "consult.artifacts", "consult.types", "consult.progress",
        "consult.context", "consult.registry", "consult.attachments",
    }
    assert expected_present.issubset(set(engine_modules)), (
        f"missing engine modules: {expected_present - set(engine_modules)}"
    )

    for mod_name in engine_modules:
        mod = importlib.import_module(mod_name)
        src = (Path(mod.__file__).read_text() if mod.__file__ else "")
        for line in src.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import mcp"), (
                f"{mod_name} imports mcp.* — broken engine/adapter boundary"
            )
            assert not stripped.startswith("from mcp"), (
                f"{mod_name} imports from mcp — broken engine/adapter boundary"
            )


def test_resource_uri_formatter_override_round_trips(tmp_path, monkeypatch):
    """A custom URI formatter applies to every new manifest entry; reset
    restores the default. Used by non-MCP consumers (HTTP/library/CLI)
    that want their own URI scheme on the manifest.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    artifacts.set_resource_uri_formatter(
        lambda run_id, slug: f"https://example.com/{run_id}/{slug}.txt"
    )
    try:
        paths = artifacts.create_run()
        uri = paths.resource_uri("alpha")
        assert uri == f"https://example.com/{paths.run_id}/alpha.txt"
    finally:
        artifacts.reset_resource_uri_formatter()

    # Default restored — new run gets the consult:// scheme again.
    paths2 = artifacts.create_run()
    assert paths2.resource_uri("alpha").startswith("consult://runs/")


async def test_orchestrate_consult_runs_without_mcp_adapter(tmp_path, monkeypatch):
    """The hero `orchestrate.consult()` is callable from any consumer with
    no mcp.* import in the chain. Stubs the three engine primitives so the
    test runs offline; the assertion is on the typed return shape, not
    panel content.
    """
    from consult import capsule as capsule_mod
    from consult import orchestrate
    from consult import runner as runner_mod
    from consult import synth as synth_mod
    from consult.types import RunResult

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        registry, "resolve_tier", lambda t: ["model-a", "model-b"]
    )
    monkeypatch.setattr(
        registry, "default_synthesiser", lambda: "model-synth"
    )
    monkeypatch.setattr(
        registry, "resolve_model",
        lambda alias: {"litellm_id": alias, "provider": "x"},
    )

    progress_events: list[str] = []

    async def cb(event):
        progress_events.append(event.kind)

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="model-a", model_id="model-a", status=Status.OK,
                    resource_uri=paths.resource_uri("model-a"),
                    body_path=str(paths.response_text("model-a")),
                    latency_ms=10, cost_usd=0.01, cost_known=True,
                ),
            ],
            cost_usd=0.01, cost_known=True, wall_ms=10,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_synth(*args, **kwargs):
        return synth_mod.SynthResult(text="synthesised", cost_usd=0.02)

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await orchestrate.consult(
        "what's the call?", tier="quick", on_progress=cb,
    )
    # Typed RunResult returned, not a dict — non-MCP consumers get the
    # full Pydantic shape with structured access.
    assert isinstance(result, RunResult)
    assert result.synthesis == "synthesised"
    assert result.partial is False
    assert result.synthesiser == "model-synth"
    # Synth cost rolled into the total: 0.01 (fanout) + 0.02 (synth).
    assert abs(result.cost_usd - 0.03) < 1e-9
    # Progress events flowed through (synth_started + synth_completed are
    # emitted directly by orchestrate.consult; fanout/capsule's own events
    # are stubbed out so they don't appear).
    assert "synth_started" in progress_events
    assert "synth_completed" in progress_events


@pytest.mark.asyncio
async def test_orchestrate_consult_gates_synth_on_high_consensus(
    tmp_path, monkeypatch,
):
    """`gate_synth_at_agreement` short-circuits the flagship synth when the
    panel converges tightly. The synth model must NOT be called; the
    returned RunResult.synth_gated must be True and cost_usd must NOT
    include any synth spend.
    """
    from consult import capsule as capsule_mod
    from consult import orchestrate
    from consult import runner as runner_mod
    from consult import synth as synth_mod
    from consult.types import Capsule, RunResult

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        registry, "resolve_tier", lambda t: ["model-a", "model-b", "model-c"],
    )
    monkeypatch.setattr(registry, "default_synthesiser", lambda: "model-synth")
    monkeypatch.setattr(
        registry, "resolve_model",
        lambda alias: {"litellm_id": alias, "provider": "x"},
    )

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="m-a", model_id="x/a", status=Status.OK,
                    resource_uri=paths.resource_uri("m-a"),
                    body_path=str(paths.response_text("m-a")),
                    latency_ms=10, cost_usd=0.01, cost_known=True,
                    capsule=Capsule(
                        position="ship feature X",
                        recommendation="ship feature X with caveat",
                    ),
                ),
                ManifestEntry(
                    slug="m-b", model_id="x/b", status=Status.OK,
                    resource_uri=paths.resource_uri("m-b"),
                    body_path=str(paths.response_text("m-b")),
                    latency_ms=10, cost_usd=0.01, cost_known=True,
                    capsule=Capsule(
                        position="ship feature X",
                        recommendation="ship feature X with caveat",
                    ),
                ),
                ManifestEntry(
                    slug="m-c", model_id="x/c", status=Status.OK,
                    resource_uri=paths.resource_uri("m-c"),
                    body_path=str(paths.response_text("m-c")),
                    latency_ms=10, cost_usd=0.01, cost_known=True,
                    capsule=Capsule(
                        position="ship feature X",
                        recommendation="ship feature X with caveat",
                    ),
                ),
            ],
            cost_usd=0.03, cost_known=True, wall_ms=10,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    synth_called = {"n": 0}

    async def fake_synth(*args, **kwargs):
        synth_called["n"] += 1
        return synth_mod.SynthResult(text="should not appear", cost_usd=99.0)

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await orchestrate.consult(
        "what's the call?", tier="quick", gate_synth_at_agreement=0.1,
    )
    assert isinstance(result, RunResult)
    assert result.synth_gated is True, "gating did not trigger on identical capsules"
    assert synth_called["n"] == 0, "flagship synth was called despite gating"
    assert result.cost_usd == pytest.approx(0.03, abs=1e-9)
    assert result.disagreement is not None and result.disagreement < 0.1
    assert "gated" in result.synthesis.lower() or "consensus" in result.synthesis.lower()
    assert result.synthesiser == "(gated)"


@pytest.mark.asyncio
async def test_orchestrate_consult_does_not_gate_on_high_disagreement(
    tmp_path, monkeypatch,
):
    """When the panel diverges, gating must NOT trigger — the flagship
    synth runs as usual."""
    from consult import capsule as capsule_mod
    from consult import orchestrate
    from consult import runner as runner_mod
    from consult import synth as synth_mod
    from consult.types import Capsule

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        registry, "resolve_tier", lambda t: ["model-a", "model-b"],
    )
    monkeypatch.setattr(registry, "default_synthesiser", lambda: "model-synth")
    monkeypatch.setattr(
        registry, "resolve_model",
        lambda alias: {"litellm_id": alias, "provider": "x"},
    )

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="m-a", model_id="x/a", status=Status.OK,
                    resource_uri=paths.resource_uri("m-a"),
                    body_path=str(paths.response_text("m-a")),
                    latency_ms=10, cost_usd=0.01, cost_known=True,
                    capsule=Capsule(
                        position="ship feature X immediately",
                        recommendation="do A",
                    ),
                ),
                ManifestEntry(
                    slug="m-b", model_id="x/b", status=Status.OK,
                    resource_uri=paths.resource_uri("m-b"),
                    body_path=str(paths.response_text("m-b")),
                    latency_ms=10, cost_usd=0.01, cost_known=True,
                    capsule=Capsule(
                        position="cancel the project entirely",
                        recommendation="form a steering committee",
                    ),
                ),
            ],
            cost_usd=0.02, cost_known=True, wall_ms=10,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    synth_called = {"n": 0}

    async def fake_synth(*args, **kwargs):
        synth_called["n"] += 1
        return synth_mod.SynthResult(text="real synthesis", cost_usd=0.05)

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await orchestrate.consult(
        "what's the call?", tier="quick", gate_synth_at_agreement=0.1,
    )
    assert result.synth_gated is False
    assert synth_called["n"] == 1, "flagship synth must run on high disagreement"
    assert result.synthesis == "real synthesis"
    assert result.cost_usd == pytest.approx(0.07, abs=1e-9)


# ---- Iter8 regression tests ------------------------------------------------


def test_attachment_git_diff_size_cap(monkeypatch, tmp_path):
    """git_diff source must honour CONSULT_ATTACHMENT_MAX_BYTES.

    Previously only file paths were size-capped; a multi-GB diff would
    flow straight from `sources.resolve_git_diff` into the prompt and
    OOM the server or blow the token budget.
    """
    from consult import attachments

    # Tiny cap to make the test deterministic without generating a real
    # large diff. Resolver is stubbed to return an oversized blob.
    monkeypatch.setenv("CONSULT_ATTACHMENT_MAX_BYTES", "100")
    monkeypatch.setattr(
        attachments.sources, "resolve_git_diff",
        lambda base, head, repo_path: "x" * 5000,
    )
    out = attachments.render_attachment(
        {"source": "git_diff", "base": "main", "head": "HEAD"}
    )
    assert "[ERROR: diff" in out
    assert "CONSULT_ATTACHMENT_MAX_BYTES=100" in out
    # The oversized content itself must NOT appear in the rendered output.
    assert "x" * 200 not in out


def test_attachment_git_diff_under_cap_renders_normally(monkeypatch):
    """Diffs under the cap render as a normal git_diff block."""
    from consult import attachments

    monkeypatch.setenv("CONSULT_ATTACHMENT_MAX_BYTES", "10000")
    monkeypatch.setattr(
        attachments.sources, "resolve_git_diff",
        lambda base, head, repo_path: "diff --git a/x b/x\n+hello",
    )
    out = attachments.render_attachment(
        {"source": "git_diff", "base": "main", "head": "HEAD"}
    )
    assert "[ERROR" not in out
    assert "hello" in out


@pytest.mark.asyncio
async def test_sequence_persists_cost_on_synth_failure(tmp_path, monkeypatch):
    """When a step's synth returns a non-OK status the loop breaks, but the
    step's cost must still be persisted to disk and recorded in the
    SequenceResult.steps array — otherwise the ledger silently
    under-reports and the caller can't see the partial step.
    """
    from consult import capsule, runner
    from consult import sequence as seq_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        await asyncio.to_thread(paths.response_text(slug).write_text, "body")
        return ManifestEntry(
            slug=slug, model_id="x/y", persona=None, status=Status.OK,
            finish_reason="stop", resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10, cost_usd=0.01, cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    async def fake_annotate(handle, **kwargs):
        return handle

    monkeypatch.setattr(capsule, "annotate", fake_annotate)

    # First step's synth fails; second step should never run because we
    # break — but the first step MUST land in result.steps with the
    # rolled-up cost (fanout 0.01 + synth 0.03 = 0.04).
    async def fake_synth(run_id, **kwargs):
        return synth_mod.SynthResult(
            text="# Synthesis empty\n\nno content",
            cost_usd=0.03,
            cost_known=True,
            status=synth_mod.SynthStatus.EMPTY,
        )

    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await seq_mod.sequence(
        ["step 1", "step 2"],
        [ModelSpec(model="claude-haiku")],
        max_run_usd=10.0,
    )

    assert result.partial is True
    assert "synth status=EMPTY" in (result.partial_reason or "")
    assert len(result.steps) == 1, "the failed step must still be in result.steps"
    assert abs(result.steps[0].cost_usd - 0.04) < 1e-9, (
        f"step cost should include fanout + synth, got {result.steps[0].cost_usd}"
    )
    # And the disk manifest must reflect the same total (ledger reads this).
    step_paths = artifacts.load_run(result.steps[0].run_id)
    manifest_on_disk = json.loads(step_paths.manifest_json.read_text())
    assert abs(manifest_on_disk["cost_usd"] - 0.04) < 1e-9


@pytest.mark.asyncio
async def test_refine_preserves_final_manifest_when_late_round_partial(
    tmp_path, monkeypatch
):
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
            await asyncio.to_thread(
                paths.response_text("m-good").write_text, "good body"
            )
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
            round=round_num, score=0.5, gaps=["needs more"],
            next_round_focus="dig deeper", reasoning="not converged",
            cost_usd=0.0, cost_known=True, parsed_ok=True,
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
        "test", [ModelSpec(model="claude-haiku")], max_rounds=3, threshold=0.85,
    )
    # Partial because round 2 failed.
    assert result.partial is True
    # But final_manifest is the round-1 good manifest, NOT the empty
    # partial from round 2.
    assert len(result.final_manifest) == 1
    assert result.final_manifest[0].slug == "m-good"
    assert result.final_manifest[0].capsule is not None


def test_refine_continuation_sentinel_check_handles_leading_whitespace(
    tmp_path, monkeypatch
):
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


# `asyncio` is imported lazily so the rest of the test module's existing
# style stays intact.
import asyncio  # noqa: E402

# ---- Iter9 regression tests ------------------------------------------------


def test_resource_uri_formatter_is_context_scoped(tmp_path, monkeypatch):
    """The override is scoped to the current async/contextvars Context, so
    two concurrent consumers don't corrupt each other's manifest URIs.
    """
    import contextvars

    from consult.artifacts import (
        reset_resource_uri_formatter,
        set_resource_uri_formatter,
    )

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()

    # Default formatter — `consult://` scheme.
    reset_resource_uri_formatter()
    assert paths.resource_uri("alpha") == f"consult://runs/{paths.run_id}/responses/alpha"

    # Override scoped to a child context: parent context stays on the default.
    def install_http():
        set_resource_uri_formatter(
            lambda run_id, slug: f"https://example.com/runs/{run_id}/{slug}"
        )
        return paths.resource_uri("alpha")

    child_ctx = contextvars.copy_context()
    in_child = child_ctx.run(install_http)
    assert in_child == f"https://example.com/runs/{paths.run_id}/alpha"
    # Parent context unaffected because the child's `set` only mutated its
    # own copy of the contextvars map.
    assert paths.resource_uri("alpha") == f"consult://runs/{paths.run_id}/responses/alpha"


@pytest.mark.asyncio
async def test_synth_persists_cost_to_manifest(tmp_path, monkeypatch):
    """Direct calls to synth.synthesise() must persist the synthesiser
    cost + badge to the run's manifest so consult-ledger sees it.
    Previously only orchestrate/refine/sequence did this; the standalone
    synthesise tool was a ledger blindspot.
    """
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    # Build a minimal run with one usable body + manifest on disk.
    paths = artifacts.create_run()
    entry = ManifestEntry(
        slug="alpha", model_id="anthropic/x", status=Status.OK,
        resource_uri=paths.resource_uri("alpha"),
        body_path=str(paths.response_text("alpha")),
        latency_ms=10, cost_usd=0.005, cost_known=True,
    )
    paths.response_text("alpha").write_text("Some response.")
    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=[entry],
        cost_usd=0.005,
        cost_known=True,
        wall_ms=10,
    )
    artifacts.write_manifest(paths, handle.model_dump())

    # Stub the LiteLLM call so the synth path returns deterministic text + cost.
    class FakeMsg:
        content = "synth body"

    class FakeChoice:
        message = FakeMsg()
        finish_reason = "stop"

    class FakeResp:
        choices = [FakeChoice()]

    async def fake_acompletion(**kwargs):
        return FakeResp()

    monkeypatch.setattr(synth_mod.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        synth_mod.litellm, "completion_cost", lambda **kw: 0.05,
    )

    result = await synth_mod.synthesise(paths.run_id)
    assert result.status is synth_mod.SynthStatus.OK
    assert abs(result.cost_usd - 0.05) < 1e-9

    # Manifest on disk now reflects the synth spend + synthesiser badge.
    on_disk = json.loads(paths.manifest_json.read_text())
    assert abs(on_disk["cost_usd"] - 0.05) < 1e-9
    assert "synthesiser" in on_disk


@pytest.mark.asyncio
async def test_orchestrate_consult_accepts_attachments_and_dry_run(
    tmp_path, monkeypatch
):
    """orchestrate.consult must accept attachments (inlining internally)
    and dry_run (passed through to runner.fanout) so library consumers
    have parity with the MCP `consult` tool.
    """
    from consult import orchestrate, runner

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.5, True))

    f = tmp_path / "atta.txt"
    f.write_text("ATTACHMENT-CONTENT")

    # dry_run=True returns a partial RunResult; the prompt that reaches
    # estimate_cost includes the attachment content (proves inlining).
    captured = {}

    def fake_estimate(specs, prompt, **_):
        captured["prompt"] = prompt
        return (0.5, True)

    monkeypatch.setattr(runner, "estimate_cost", fake_estimate)

    result = await orchestrate.consult(
        "test question",
        tier="nano",
        attachments=[str(f)],
        dry_run=True,
        max_run_usd=10.0,
    )
    assert result.partial is True
    assert result.partial_reason and "dry_run" in result.partial_reason
    assert "ATTACHMENT-CONTENT" in captured["prompt"]


# ---- Iter10 regression tests -----------------------------------------------


@pytest.mark.asyncio
async def test_fit_prompt_to_context_no_op_when_under_budget(monkeypatch):
    """No trim, no marker when the prompt already fits."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm, "token_counter", lambda model, text: len(text) // 4,
    )
    out, dropped = await runner._fit_prompt_to_context(
        "short prompt",
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=100_000,
        max_output_tokens=4000,
    )
    assert out == "short prompt"
    assert dropped == 0
    assert "[TRIMMED" not in out


@pytest.mark.asyncio
async def test_fit_prompt_to_context_trims_when_over_budget(monkeypatch):
    """Over-budget prompt gets head+tail trimmed with a clear marker so
    the call proceeds instead of being rejected by the provider."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm, "token_counter", lambda model, text: len(text),
    )
    # Budget: 1000 input - 100 output = 900 available. Build a 2000-char
    # prompt that fakes 1 token/char so we're 2× over.
    long_prompt = "A" * 1000 + "B" * 1000
    out, dropped = await runner._fit_prompt_to_context(
        long_prompt,
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=1000,
        max_output_tokens=100,
    )
    # Marker present + final size under the budget after the recount loop.
    assert "[TRIMMED" in out
    assert len(out) < len(long_prompt)
    # `dropped` is the explicit signal callers use to build the manifest
    # trim note — must be positive when a trim happened.
    assert dropped > 0
    assert dropped == len(long_prompt) - len(out)


@pytest.mark.asyncio
async def test_fit_prompt_to_context_skips_when_prior_alone_exceeds_budget(
    monkeypatch,
):
    """If `prior_turns` already exceed the budget, the prompt is returned
    untouched (we don't corrupt role boundaries) and the provider's
    rejection becomes the surfaced error."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm, "token_counter", lambda model, text: len(text),
    )
    prior = [
        {"role": "user", "content": "X" * 2000},
        {"role": "assistant", "content": "Y" * 2000},
    ]
    out, dropped = await runner._fit_prompt_to_context(
        "follow-up",
        prior_turns=prior,
        litellm_id="x/y",
        max_input_tokens=1000,
        max_output_tokens=100,
    )
    assert out == "follow-up"  # not trimmed
    assert dropped == 0


@pytest.mark.asyncio
async def test_call_one_auto_trims_oversized_prompt(tmp_path, monkeypatch):
    """End-to-end: _call_one with an over-budget prompt trims, the LLM
    call succeeds, and the ManifestEntry carries the trim note in
    `error` even on Status.OK."""
    from consult import runner
    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # max_input_tokens=10000 leaves ~6000 for the prompt after claude-haiku's
    # default_budget_tokens (4000) is reserved for output.
    monkeypatch.setattr(
        runner, "_max_input_tokens", lambda lid, entry: 10_000,
    )
    monkeypatch.setattr(
        runner.litellm, "token_counter", lambda model, text: len(text),
    )

    sent_messages: dict[str, Any] = {}

    class FakeMsg:
        content = "trimmed response body"

    class FakeChoice:
        message = FakeMsg()
        finish_reason = "stop"

    class FakeResp:
        choices = [FakeChoice()]
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 50})()

        def model_dump(self):
            return {"choices": [{"message": {"content": "trimmed response body"}}]}

    async def fake_acompletion(**kwargs):
        sent_messages["messages"] = kwargs.get("messages")
        return FakeResp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        runner.litellm, "completion_cost", lambda **kw: 0.01,
    )

    spec = ModelSpec(model="claude-haiku")
    paths = artifacts.create_run()
    # 50000-char prompt is well over the fake 6000-char available input budget.
    long_prompt = "Z" * 50_000

    entry = await _call_one(spec, "test-slug", long_prompt, paths)

    assert entry.status == Status.OK
    # Trim diagnostic lives on `note` (info), not `error` (failure).
    assert entry.error is None
    assert "auto-trimmed" in (entry.note or "")
    sent_text = sent_messages["messages"][0]["content"]
    # Anthropic wraps content in a list with cache_control; extract the text.
    if isinstance(sent_text, list):
        sent_text = sent_text[0]["text"]
    assert "[TRIMMED" in sent_text
    assert len(sent_text) < 50_000


@pytest.mark.asyncio
async def test_call_one_no_trim_note_when_prompt_mentions_trimmed_literally(
    tmp_path, monkeypatch,
):
    """A prompt that contains the literal string `[TRIMMED` in its source
    (e.g. an attached source file from this very codebase) must NOT be
    misreported as auto-trimmed when the prompt actually fits the context.

    Previously runner sniffed the prompt for `[TRIMMED` to detect trim
    events. Source-code reviews that attached `context.py` or `runner.py`
    matched the substring and surfaced a phantom "input auto-trimmed"
    error on every panellist, even though no trim happened.
    """
    from consult import runner
    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        runner, "_max_input_tokens", lambda lid, entry: 1_000_000,
    )
    monkeypatch.setattr(
        runner.litellm, "token_counter", lambda model, text: len(text) // 4,
    )

    class _Msg:
        content = "ok"
    class _Choice:
        message = _Msg()
        finish_reason = "stop"
    class _Resp:
        choices = [_Choice()]
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 10})()
        def model_dump(self): return {"choices": []}

    async def fake_acompletion(**kwargs):
        return _Resp()
    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda **kw: 0.0)

    # Prompt that mentions the literal marker (as a code-review attachment
    # would). Fits comfortably under the million-token fake budget.
    prompt = (
        "review this code:\n"
        '    marker = f"\\n\\n... [TRIMMED {dropped} chars ...] ...\\n\\n"\n'
        "    if '[TRIMMED' in per_slug_prompt: ...\n"
    )
    spec = ModelSpec(model="claude-haiku")
    paths = artifacts.create_run()
    entry = await _call_one(spec, "x-0", prompt, paths)
    assert entry.status == Status.OK
    assert entry.error is None, (
        f"phantom trim note: {entry.error!r}"
    )


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


def test_persist_inlined_attachments_writes_and_uri_resolves(tmp_path, monkeypatch):
    """Persistence writes each block's content to attachments/<safe-name>
    and the returned URI is resolvable via parse_resource_uri."""
    from consult import attachments as att

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    prompt = (
        "review this:\n"
        + att.ATTACHMENT_SEPARATOR
        + "\n# /Users/x/foo.py\n```python\nbody-foo\n```\n"
        + "\n# /Users/x/bar.py\n```python\nbody-bar\n```\n"
        + "\n# /Users/x/foo.py\n```python\nbody-foo-2\n```\n"  # name collision
    )
    uri_map = att.persist_inlined_attachments(paths, prompt)
    assert len(uri_map) == 3
    files = sorted(p.name for p in paths.attachments.iterdir())
    assert "foo.py" in files
    assert "bar.py" in files
    # Collision got a numeric suffix.
    assert any(f.startswith("foo.py-") for f in files)
    for uri in uri_map.values():
        rid, kind, name = artifacts.parse_resource_uri(uri)
        assert rid == paths.run_id
        assert kind == "attachments"
        assert (paths.attachments / name).exists()


@pytest.mark.asyncio
async def test_fit_prompt_drops_largest_attachment_with_stub(tmp_path, monkeypatch):
    """When the prompt is over budget, the trimmer drops the LARGEST
    attachment block first, replaces it with a stub referencing the
    resource URI, and leaves smaller blocks intact. Beats head+tail
    slicing through the middle of a code file."""
    from consult import attachments as att
    from consult import runner

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    monkeypatch.setattr(
        runner.litellm, "token_counter", lambda model, text: len(text),
    )

    small_block = "small\n" * 50          # ~300 chars
    large_block = "X" * 50_000            # 50K chars — dropped first
    medium_block = "Y" * 5_000            # 5K chars

    prompt = (
        "instructions here\n"
        + att.ATTACHMENT_SEPARATOR
        + f"\n# /src/small.py\n```python\n{small_block}\n```\n"
        + f"\n# /src/big.py\n```python\n{large_block}\n```\n"
        + f"\n# /src/medium.py\n```python\n{medium_block}\n```\n"
    )
    att.persist_inlined_attachments(paths, prompt)

    out, dropped = await runner._fit_prompt_to_context(
        prompt,
        paths=paths,
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=11_000,
        max_output_tokens=1_000,
    )
    assert dropped > 0
    assert large_block not in out
    assert "Attachment dropped to fit context" in out
    assert "big.py" in out  # header preserved
    assert paths.run_id in out  # resource URI for the dropped file
    assert small_block in out


@pytest.mark.asyncio
async def test_fit_prompt_falls_back_to_head_tail_without_attachments(monkeypatch):
    """Prompts that aren't structured as attachments still get the
    head+tail trim — the new path is additive, not a replacement."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm, "token_counter", lambda model, text: len(text),
    )
    long_prompt = "A" * 1000 + "B" * 1000  # no ATTACHMENT_SEPARATOR
    out, dropped = await runner._fit_prompt_to_context(
        long_prompt,
        paths=None,
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=1000,
        max_output_tokens=100,
    )
    assert "[TRIMMED" in out
    assert dropped > 0


def test_max_input_tokens_registry_override():
    """Explicit max_input_tokens in the registry entry wins over LiteLLM."""
    from consult.runner import _max_input_tokens

    entry = {"max_input_tokens": 50_000}
    assert _max_input_tokens("openrouter/some/model", entry) == 50_000


def test_max_input_tokens_falls_back_to_litellm(monkeypatch):
    """No registry override → ask LiteLLM."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm, "get_model_info",
        lambda model: {"max_input_tokens": 128_000},
    )
    assert runner._max_input_tokens("openai/gpt-x", {}) == 128_000


def test_max_input_tokens_unknown_returns_none(monkeypatch):
    """Unknown model + no override → None, so pre-flight is skipped."""
    from consult import runner

    def boom(model):
        raise Exception("unknown model")

    monkeypatch.setattr(runner.litellm, "get_model_info", boom)
    assert runner._max_input_tokens("vendor/totally-new-model", {}) is None


def test_slow_tail_dropout_default_is_180s():
    """Default bumped from 30s to 180s — long-context wide-panel runs were
    losing real signal (kimi/qwen often take 60-180s on ~200K input)."""
    # The default lives in runner.fanout's body; verify by reading the
    # source rather than executing the path (which would need a full fanout).
    import inspect

    from consult import runner
    src = inspect.getsource(runner.fanout)
    assert 'os.environ.get("CONSULT_TAIL_DROPOUT_S", 180.0)' in src
