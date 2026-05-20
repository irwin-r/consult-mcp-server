"""Smoke tests. The unit slice runs offline (no API keys); the live slice
hits real providers only when relevant API keys are present.

Run: `pytest -v`
"""

from __future__ import annotations

import json
import os
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
    rid, slug = artifacts.parse_resource_uri(uri)
    assert rid == paths.run_id
    assert slug == "alpha"
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
    """A valid continuation_id loads the prior run's synthesis.md and
    prepends it as 'Prior consultation summary' before the follow-up.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()
    (prior.root / "synthesis.md").write_text("ANSWER: pick DuckDB.")

    result = _apply_continuation("Now what about Polars for ETL?", prior.run_id)
    assert "Prior consultation summary" in result
    assert "ANSWER: pick DuckDB." in result
    assert "Follow-up question" in result
    assert "Now what about Polars for ETL?" in result


def test_refine_continuation_none_or_empty_is_passthrough(tmp_path, monkeypatch):
    """No continuation_id (or empty string) leaves the prompt untouched."""
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    assert _apply_continuation("hello", None) == "hello"
    assert _apply_continuation("hello", "") == "hello"


def test_refine_continuation_unknown_id_raises(tmp_path, monkeypatch):
    """An unknown continuation_id must raise — silently dropping the prior
    context would leave the caller thinking the new round had it.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="continuation_id not found"):
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
    with pytest.raises(ValueError, match="max_rounds"):
        await refine_mod.refine("q", [ModelSpec(model="claude-haiku")], max_rounds=5)


def test_arbiter_json_extractor_tolerates_fences():
    from consult.jsonparse import extract_json

    fenced = '```json\n{"score": 0.7, "gaps": ["x"], "next_round_focus": "", "reasoning": ""}\n```'
    data = extract_json(fenced)
    assert data["score"] == 0.7
    assert data["gaps"] == ["x"]


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
    """fanout must call on_progress once per panellist with a typed
    `PanellistCompleted` event carrying monotonically increasing `done`.
    The "A" half of the A + D progress design.
    """
    from consult import runner
    from consult.progress import PanellistCompleted, ProgressEvent
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda specs, prompt: (0.0, True))

    async def fake_call(spec, slug, per_prompt, paths):
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

    # 3 panellists ⇒ 3 PanellistCompleted events; each has total=3,
    # done ∈ {1, 2, 3} (gather order is non-deterministic).
    assert len(events) == 3
    assert all(isinstance(e, PanellistCompleted) for e in events)
    assert {e.done for e in events} == {1, 2, 3}
    assert all(e.total == 3 for e in events)
    assert all(e.status == "OK" for e in events)


def test_append_progress_log_writes_jsonl(tmp_path):
    """The "D" half: a tailable JSONL log in the run dir. Each line is one
    event.model_dump() with a `ts` prepended so programmatic consumers can
    parse by `kind` without scraping free-text.
    """
    from consult.progress import CapsuleExtracted, PanellistCompleted
    from consult.runner import _append_progress_log

    _append_progress_log(tmp_path, PanellistCompleted(
        done=1, total=2, slug="haiku", status="OK", latency_ms=42,
    ))
    _append_progress_log(tmp_path, CapsuleExtracted(done=1, total=2, slug="haiku"))

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
        PanellistCompleted,
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


def test_error_envelope_shape_round_trips():
    """The structured-error envelope must round-trip through JSON with the
    exact shape agents pattern-match against. Locks in the wire contract.
    """
    from consult.errors import ConsultError, ErrorCode, ErrorEnvelope

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
    """
    from consult import server as server_mod

    async def bad_handler(args):
        raise ValueError("max_rounds must be between 1 and 3")

    monkeypatch.setattr(server_mod, "_handle_refine", bad_handler)

    result = await server_mod.handle_call_tool("refine", {"prompt": "x", "models": []})
    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input"
    assert "max_rounds" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_key_error_as_unknown_model(monkeypatch):
    """`registry.resolve_model` raises KeyError on a missing alias —
    `synthesise(by_model="bogus")` would propagate that through. Must
    surface as `unknown_model`, not `invalid_input`.
    """
    from consult import server as server_mod

    async def bad_handler(args):
        raise KeyError("Unknown model: bogus-alias")

    monkeypatch.setattr(server_mod, "_handle_synth", bad_handler)
    result = await server_mod.handle_call_tool("synthesise", {"run_id": "x"})
    payload = json.loads(result[0].text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unknown_model"


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_file_not_found_as_run_not_found(monkeypatch):
    """`artifacts.load_run` raises FileNotFoundError on missing run_id."""
    from consult import server as server_mod

    async def bad_handler(args):
        raise FileNotFoundError("Run not found: 20260520-foo")

    monkeypatch.setattr(server_mod, "_handle_synth", bad_handler)
    result = await server_mod.handle_call_tool("synthesise", {"run_id": "20260520-foo"})
    payload = json.loads(result[0].text)
    assert payload["error"]["code"] == "run_not_found"


@pytest.mark.asyncio
async def test_handle_call_tool_unknown_tool_returns_envelope():
    """Asking for a tool that doesn't exist returns an invalid_input
    envelope rather than raising a ValueError out the top.
    """
    from consult import server as server_mod

    result = await server_mod.handle_call_tool("not-a-tool", {})
    payload = json.loads(result[0].text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input"
    assert "not-a-tool" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_unhandled_exception_becomes_internal_error(monkeypatch):
    """Any unanticipated exception type from a handler must become an
    `internal_error` envelope rather than tearing out the MCP dispatch.
    """
    from consult import server as server_mod

    async def bad_handler(args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(server_mod, "_handle_panel", bad_handler)
    result = await server_mod.handle_call_tool("panel", {"prompt": "x", "models": []})
    payload = json.loads(result[0].text)
    assert payload["error"]["code"] == "internal_error"
    assert "kaboom" in payload["error"]["message"]


def test_progress_event_round_trips_through_json():
    """JSONL log line → dict → discriminated-union dispatch. Pydantic's
    `discriminator='kind'` on `ProgressEvent` enables programmatic consumers
    to parse one line and get a typed object back.
    """
    from pydantic import TypeAdapter

    from consult.progress import CapsuleExtracted, PanellistCompleted, ProgressEvent

    adapter = TypeAdapter(ProgressEvent)
    p = PanellistCompleted(done=1, total=2, slug="x", status="OK", latency_ms=10)
    parsed = adapter.validate_json(p.model_dump_json())
    assert isinstance(parsed, PanellistCompleted)
    assert parsed.slug == "x"

    c = CapsuleExtracted(done=1, total=2, slug="x")
    parsed = adapter.validate_json(c.model_dump_json())
    assert isinstance(parsed, CapsuleExtracted)


@pytest.mark.asyncio
async def test_fanout_progress_callback_failure_does_not_abort_run(tmp_path, monkeypatch):
    """A raising on_progress callback must not tear down the fanout —
    progress is best-effort, the run completes regardless.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda specs, prompt: (0.0, True))

    async def fake_call(spec, slug, per_prompt, paths):
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
    monkeypatch.setattr(runner, "estimate_cost", lambda specs, prompt: (0.99, True))
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

    monkeypatch.setattr(runner, "estimate_cost", lambda specs, prompt: (0.50, False))
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
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda specs, prompt: (0.0, True))

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
        return f"SYNTH_{synth_counter['i']}"

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
        runner_mod, "estimate_cost", lambda specs, prompt: (0.0, False)
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
        return "X"

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
    rid, slug = artifacts.parse_resource_uri("consult://runs/r1/responses/alpha.r2")
    assert rid == "r1"
    assert slug == "alpha.r2"


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


def test_synth_build_input_anonymised_and_filters_failures():
    """Privacy-relevant for blinded mode + correctness for the synthesiser input.

    ERROR/EMPTY entries must be excluded from the synthesiser input, and
    anonymised mode must not leak model IDs.
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

    blinded = _build_input(manifest, bodies, rubric=rubric, anonymised=True)
    assert "anthropic/claude-opus-4-7" not in blinded
    assert "openai/gpt-5.5" not in blinded
    assert "panelist-alpha" in blinded
    assert "panelist-gamma" in blinded
    assert "panelist-beta" not in blinded  # filtered
    assert "alpha body" in blinded
    assert "rubric 2" in blinded  # only OK + TRUNCATED counted

    unblinded = _build_input(manifest, bodies, rubric=rubric, anonymised=False)
    assert "anthropic/claude-opus-4-7" in unblinded
    assert "gemini/gemini-3.1-pro-preview" in unblinded


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


# ---- Live tests (gated on API keys) ----------------------------------------


HAVE_KEYS = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
)


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
