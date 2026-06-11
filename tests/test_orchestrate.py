"""The consult() hero flow.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import pytest

from consult import artifacts, registry
from consult.types import Capsule, ManifestEntry, RunHandle, Status


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
                    slug="x",
                    model_id="m/x",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("x"),
                    body_path=str(paths.response_text("x")),
                    latency_ms=0,
                    cost_usd=0.10,
                    cost_known=True,
                    confidence=None,
                    capsule=None,
                ),
            ],
            cost_usd=0.10,
            cost_known=True,
            wall_ms=0,
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
                    slug="x",
                    model_id="m/x",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("x"),
                    body_path=str(paths.response_text("x")),
                    latency_ms=0,
                    cost_usd=None,
                    cost_known=False,
                    confidence=None,
                    capsule=None,
                ),
            ],
            cost_usd=0.0,
            cost_known=False,
            wall_ms=0,
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
                    slug="x",
                    model_id="m/x",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("x"),
                    body_path=str(paths.response_text("x")),
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

    async def fake_synth(*args, **kwargs):
        return synth_mod.SynthResult(text="ok")

    async def crashy_cb(event):
        raise RuntimeError("client disconnected")

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    # No exception should propagate; tool returns its normal payload.
    result = await handlers.consult({"prompt": "p", "tier": "quick"}, on_progress=crashy_cb)
    assert result["partial"] is False
    assert result["synthesis"] == "ok"


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
        await handlers.consult(
            {
                "prompt": "p",
                "tier": "quick",
                "synthesiser": "totally-not-a-real-alias",
            }
        )
    assert fanout_calls["n"] == 0


@pytest.mark.asyncio
async def test_orchestrate_consult_gates_synth_on_high_consensus(
    tmp_path,
    monkeypatch,
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
    from consult.types import RunResult

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        registry,
        "resolve_tier",
        lambda t: ["model-a", "model-b", "model-c"],
    )
    monkeypatch.setattr(registry, "default_synthesiser", lambda: "model-synth")
    monkeypatch.setattr(
        registry,
        "resolve_model",
        lambda alias: {"litellm_id": alias, "provider": "x"},
    )

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="m-a",
                    model_id="x/a",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("m-a"),
                    body_path=str(paths.response_text("m-a")),
                    latency_ms=10,
                    cost_usd=0.01,
                    cost_known=True,
                    capsule=Capsule(
                        position="ship feature X",
                        recommendation="ship feature X with caveat",
                    ),
                ),
                ManifestEntry(
                    slug="m-b",
                    model_id="x/b",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("m-b"),
                    body_path=str(paths.response_text("m-b")),
                    latency_ms=10,
                    cost_usd=0.01,
                    cost_known=True,
                    capsule=Capsule(
                        position="ship feature X",
                        recommendation="ship feature X with caveat",
                    ),
                ),
                ManifestEntry(
                    slug="m-c",
                    model_id="x/c",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("m-c"),
                    body_path=str(paths.response_text("m-c")),
                    latency_ms=10,
                    cost_usd=0.01,
                    cost_known=True,
                    capsule=Capsule(
                        position="ship feature X",
                        recommendation="ship feature X with caveat",
                    ),
                ),
            ],
            cost_usd=0.03,
            cost_known=True,
            wall_ms=10,
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
        "what's the call?",
        tier="quick",
        gate_synth_at_agreement=0.1,
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
    tmp_path,
    monkeypatch,
):
    """When the panel diverges, gating must NOT trigger — the flagship
    synth runs as usual."""
    from consult import capsule as capsule_mod
    from consult import orchestrate
    from consult import runner as runner_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        registry,
        "resolve_tier",
        lambda t: ["model-a", "model-b"],
    )
    monkeypatch.setattr(registry, "default_synthesiser", lambda: "model-synth")
    monkeypatch.setattr(
        registry,
        "resolve_model",
        lambda alias: {"litellm_id": alias, "provider": "x"},
    )

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="m-a",
                    model_id="x/a",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("m-a"),
                    body_path=str(paths.response_text("m-a")),
                    latency_ms=10,
                    cost_usd=0.01,
                    cost_known=True,
                    capsule=Capsule(
                        position="ship feature X immediately",
                        recommendation="do A",
                    ),
                ),
                ManifestEntry(
                    slug="m-b",
                    model_id="x/b",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("m-b"),
                    body_path=str(paths.response_text("m-b")),
                    latency_ms=10,
                    cost_usd=0.01,
                    cost_known=True,
                    capsule=Capsule(
                        position="cancel the project entirely",
                        recommendation="form a steering committee",
                    ),
                ),
            ],
            cost_usd=0.02,
            cost_known=True,
            wall_ms=10,
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
        "what's the call?",
        tier="quick",
        gate_synth_at_agreement=0.1,
    )
    assert result.synth_gated is False
    assert synth_called["n"] == 1, "flagship synth must run on high disagreement"
    assert result.synthesis == "real synthesis"
    assert result.cost_usd == pytest.approx(0.07, abs=1e-9)


@pytest.mark.asyncio
async def test_orchestrate_consult_accepts_attachments_and_dry_run(tmp_path, monkeypatch):
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


@pytest.mark.asyncio
async def test_consult_marks_synth_failure_as_partial(tmp_path, monkeypatch):
    """Regression: a non-OK synth status must surface as partial=True with a
    reason, not a clean success whose `synthesis` is a sentinel string.
    `sequence` already did this; `consult`/`orchestrate` did not.
    """
    from consult import capsule as capsule_mod
    from consult import orchestrate as orchestrate_mod
    from consult import runner as runner_mod
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

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

    async def fake_synth(run_id, **kwargs):
        return synth_mod.SynthResult(
            text="# Synthesis unavailable",
            status=synth_mod.SynthStatus.FAILED,
        )

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await orchestrate_mod.consult("q", tier="quick")

    assert result.partial is True
    assert result.partial_reason is not None
    assert "synthesis status=" in result.partial_reason
    assert result.synthesis == "# Synthesis unavailable"
