"""Chained multi-step sequence flow.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from consult import artifacts
from consult.types import ArbiterVerdict, Capsule, ManifestEntry, ModelSpec, RunHandle, Status


def test_sequence_step_prompt_threads_prior_synth():
    """Step N>1 must include step N-1's synthesis as 'prior synthesis' context."""
    from consult.sequence import _step_prompt

    p1 = _step_prompt(1, 3, None, "What is X?")
    assert p1 == "What is X?"  # first step: no prior context

    p2 = _step_prompt(2, 3, ["X is foo."], "Given X is foo, what about Y?")
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
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (0.0, False))

    async def fake_fanout(prompt, specs, **kwargs):
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
                    cost_known=False,
                )
            ],
            cost_usd=0.01,
            cost_known=False,
            wall_ms=1,
            partial=False,
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
        ManifestEntry,
        ModelSpec,
        RefineResult,
        RunHandle,
        RunResult,
    )

    cases: list[tuple[type, dict]] = [
        (ModelSpec, {"model": "x"}),
        (Capsule, {}),
        (
            ManifestEntry,
            {
                "slug": "s",
                "status": "OK",
                "resource_uri": "consult://x",
                "body_path": "/tmp/x",
            },
        ),
        (
            RunHandle,
            {
                "run_id": "r",
                "artifacts_dir": "/tmp",
                "manifest": [],
                "cost_usd": 0.0,
                "wall_ms": 0,
            },
        ),
        (
            RunResult,
            {
                "run_id": "r",
                "synthesis": "",
                "manifest": [],
                "cost_usd": 0.0,
                "wall_ms": 0,
            },
        ),
        (ArbiterVerdict, {"round": 1, "score": 0.5}),
        (
            RefineResult,
            {
                "run_id": "r",
                "rounds_completed": 0,
                "final_manifest": [],
                "verdicts": [],
                "synthesis": "",
                "converged": False,
                "threshold": 0.85,
                "cost_usd": 0.0,
                "wall_ms": 0,
            },
        ),
        (
            SequenceStep,
            {
                "step": 1,
                "run_id": "r",
                "synthesis": "",
                "cost_usd": 0.0,
                "panel_size": 0,
            },
        ),
        (
            SequenceResult,
            {
                "final_synthesis": "",
                "cost_usd": 0.0,
                "wall_ms": 0,
            },
        ),
        (
            LedgerRunEntry,
            {
                "run_id": "r",
                "cost_usd": 0.0,
                "cost_known": True,
                "panel_size": 0,
            },
        ),
        (
            DailyLedger,
            {
                "date": "2026-01-01",
                "total_usd": 0.0,
                "total_known": True,
            },
        ),
        (
            PanellistCompleted,
            {
                "done": 0,
                "total": 0,
                "slug": "s",
                "status": "OK",
                "latency_ms": 0,
            },
        ),
    ]
    for cls, kwargs in cases:
        # Sanity: the baseline kwargs construct successfully
        cls(**kwargs)
        # An unknown kwarg must be rejected, not silently dropped
        with pytest.raises(pydantic.ValidationError):
            cls(**kwargs, definitely_not_a_field="boom")


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
            steps=[],
            final_synthesis="",
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
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
        ["q1", "q2"],
        [ModelSpec(model="claude-haiku")],
    )
    assert result.partial is True
    # The 0.17 from the partial fanout must show up — not the pre-fix 0.0.
    assert result.cost_usd == pytest.approx(0.17)
    assert result.cost_known is False


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
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_usd=0.01,
            cost_known=True,
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
