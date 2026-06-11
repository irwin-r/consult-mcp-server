"""CostMeter semantics plus the two cap gates that consume it."""

from __future__ import annotations

import pytest

from consult import artifacts, orchestrate
from consult import refine as refine_mod
from consult.cost import CostMeter
from consult.types import ArbiterVerdict, Capsule, ManifestEntry, ModelSpec, RunHandle, Status


def test_cost_meter_accumulates_known_spend():
    m = CostMeter()
    m.add(0.1)
    m.add(0.2, cost_known=True)
    assert m.total == pytest.approx(0.3)
    assert m.known is True


def test_cost_meter_none_means_unknown_not_free():
    m = CostMeter()
    m.add(0.1)
    m.add(None, cost_known=False)
    assert m.total == pytest.approx(0.1)  # lower bound, nothing added
    assert m.known is False


def test_cost_meter_known_flag_propagates_with_real_cost():
    m = CostMeter()
    m.add(0.5, cost_known=False)  # partial figure: count it, mark unknown
    assert m.total == pytest.approx(0.5)
    assert m.known is False


def test_cost_meter_mark_unknown_without_spend():
    m = CostMeter()
    m.mark_unknown()
    assert m.total == 0.0
    assert m.known is False


def _entry(slug: str, cost: float = 0.1, tokens_out: int = 500) -> ManifestEntry:
    return ManifestEntry(
        slug=slug,
        model_id=f"x/{slug}",
        status=Status.OK,
        resource_uri=f"consult://x/{slug}",
        body_path=f"/x/{slug}",
        latency_ms=10,
        tokens_out=tokens_out,
        cost_usd=cost,
        cost_known=True,
        capsule=Capsule(position=f"{slug} position"),
    )


def _consult_mocks(monkeypatch, tmp_path, *, panel_cost: float, synth_estimate):
    """Wire consult() with a fake panel and a fixed synth estimate.

    Returns the dict that records whether the flagship synth ran.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        manifest = [_entry("m-a", cost=panel_cost / 2), _entry("m-b", cost=panel_cost / 2)]
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=manifest,
            cost_usd=panel_cost,
            cost_known=True,
            wall_ms=5,
        )

    synth_ran = {"called": False}

    async def fake_synth(run_id, **kwargs):
        from consult import synth as synth_mod

        synth_ran["called"] = True
        return synth_mod.SynthResult(text="real synthesis", cost_usd=0.05, cost_known=True)

    monkeypatch.setattr("consult.orchestrate.runner.fanout", fake_fanout)
    monkeypatch.setattr("consult.orchestrate.synth.synthesise", fake_synth)
    monkeypatch.setattr("consult.orchestrate._estimate_synth_cost", lambda alias, manifest: synth_estimate)
    return synth_ran


async def test_consult_skips_synth_when_estimate_breaches_cap(monkeypatch, tmp_path):
    """The fanout gate only covered the panel; the flagship synth could
    push the run past max_run_usd unchecked. With spend at $0.50 and a
    $10 synth estimate against a $1 cap, consult must skip the flagship,
    return the deterministic aggregate, and mark the result partial."""
    synth_ran = _consult_mocks(monkeypatch, tmp_path, panel_cost=0.5, synth_estimate=(10.0, True))

    result = await orchestrate.consult("q?", tier="nano", extract_capsules=False, max_run_usd=1.0)

    assert synth_ran["called"] is False
    assert result.partial is True
    assert result.partial_reason is not None and "cost cap" in result.partial_reason
    assert result.synthesiser == "(cap-skipped)"
    assert "cost cap reached" in result.synthesis
    assert "m-a" in result.synthesis  # positions still surfaced
    assert result.cost_usd == pytest.approx(0.5)


async def test_consult_runs_synth_when_estimate_fits_cap(monkeypatch, tmp_path):
    synth_ran = _consult_mocks(monkeypatch, tmp_path, panel_cost=0.5, synth_estimate=(0.1, True))

    result = await orchestrate.consult("q?", tier="nano", extract_capsules=False, max_run_usd=1.0)

    assert synth_ran["called"] is True
    assert result.partial is False
    assert result.synthesis == "real synthesis"
    assert result.cost_usd == pytest.approx(0.55)


async def test_consult_unknown_estimate_proceeds_unless_cap_spent(monkeypatch, tmp_path):
    """Unpriced synthesiser mirrors fanout's warn-don't-block posture —
    except when spend has already reached the cap."""
    synth_ran = _consult_mocks(monkeypatch, tmp_path, panel_cost=0.5, synth_estimate=(0.0, False))
    result = await orchestrate.consult("q?", tier="nano", extract_capsules=False, max_run_usd=1.0)
    assert synth_ran["called"] is True
    assert result.partial is False

    synth_ran2 = _consult_mocks(monkeypatch, tmp_path, panel_cost=1.2, synth_estimate=(0.0, False))
    result2 = await orchestrate.consult("q?", tier="nano", extract_capsules=False, max_run_usd=1.0)
    assert synth_ran2["called"] is False
    assert result2.partial is True


async def test_refine_round2_gate_prices_per_slug_history(monkeypatch, tmp_path):
    """The round-2+ cost gate must estimate each panellist against its
    accumulated history plus the refinement prompt, not the bare prompt —
    histories grow per slug and the old shared-text estimate ignored them."""
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    estimate_calls: list[tuple[list[str], str]] = []

    async def fake_aestimate(specs, cost_input, **kwargs):
        estimate_calls.append(([s.model for s in specs], cost_input))
        return (0.001, True)

    async def fake_fanout(prompt, specs, **kwargs):
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        manifest = []
        for spec in specs:
            body_path = paths.root / "responses" / f"{spec.slug}.txt"
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_text(f"{spec.slug} body r1")
            manifest.append(
                ManifestEntry(
                    slug=spec.slug,
                    model_id=f"x/{spec.model}",
                    status=Status.OK,
                    resource_uri=f"consult://x/{spec.slug}",
                    body_path=str(body_path),
                    latency_ms=10,
                    cost_usd=0.001,
                    cost_known=True,
                    capsule=Capsule(position=f"{spec.slug} position"),
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
        arbiter_n["n"] += 1
        score = 0.2 if arbiter_n["n"] == 1 else 0.95
        return ArbiterVerdict(round=arbiter_n["n"], score=score, parsed_ok=True)

    async def fake_synth(*args, **kwargs):
        from consult import synth as synth_mod

        return synth_mod.SynthResult(text="final synth")

    monkeypatch.setattr("consult.refine.runner.fanout", fake_fanout)
    monkeypatch.setattr("consult.refine.capsule.annotate", fake_annotate)
    monkeypatch.setattr("consult.refine._ask_arbiter", fake_arbiter)
    monkeypatch.setattr("consult.refine.synth.synthesise", fake_synth)
    monkeypatch.setattr("consult.refine.runner.aestimate_cost", fake_aestimate)

    await refine_mod.refine(
        "Original question?",
        [ModelSpec(model="claude-haiku"), ModelSpec(model="gpt-mini")],
        threshold=0.85,
        max_rounds=2,
    )

    # Gate call layout: round 1 = 2 per-spec + 1 arbiter; round 2 = same.
    assert len(estimate_calls) == 6
    round2_per_spec = estimate_calls[3:5]
    for models, cost_input in round2_per_spec:
        assert len(models) == 1
        # Round 2's estimate input carries the panellist's round-1 history
        # (its body) ahead of the refinement prompt.
        assert "body r1" in cost_input
        assert "round 2" in cost_input
    # Round 1's per-spec inputs had no history.
    for _models, cost_input in estimate_calls[0:2]:
        assert "body r1" not in cost_input
