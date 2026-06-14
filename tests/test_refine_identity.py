"""Stable panellist identity across refine rounds.

refine pins each spec a stable unique slug at the original index so a
panellist keeps its conversation history even when the round's positional
order shifts. These tests pin that behaviour for the blinded path (where
fanout renames panellists to panelist-<greek>.r<n>) and the slug-assignment
helper itself.
"""

from __future__ import annotations

import copy

from consult import artifacts
from consult import refine as refine_mod
from consult.types import (
    ArbiterVerdict,
    Capsule,
    ManifestEntry,
    ModelSpec,
    RunHandle,
    Status,
)


async def test_refine_blinded_rounds_receive_history(monkeypatch, tmp_path):
    """Blinded refine: fanout names panellists panelist-<greek>.r<n>, so the
    per-slug history map must be keyed by those slugs. The old code keyed
    by refine's own slugs and blinded rounds 2+ never saw their history."""
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    fanout_calls: list[dict] = []
    round_counter = {"n": 0}

    async def fake_fanout(prompt, specs, **kwargs):
        from consult import runner as runner_mod

        round_counter["n"] += 1
        blinded = kwargs.get("blinded", False)
        slugs = runner_mod._make_slugs(specs, blinded)
        fanout_calls.append(
            {
                "round": round_counter["n"],
                "slugs": list(slugs),
                "prior_turns_by_slug": copy.deepcopy(kwargs.get("prior_turns_by_slug")),
            }
        )
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        manifest = []
        for slug in slugs:
            body_path = paths.root / "responses" / f"{slug}.txt"
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_text(f"{slug} body r{round_counter['n']}")
            manifest.append(
                ManifestEntry(
                    slug=slug,
                    model_id=None,
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
            cost_usd=0.001 * len(slugs),
            cost_known=True,
            wall_ms=10,
            blinded=blinded,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    arbiter_n = {"n": 0}

    async def fake_arbiter(*args, **kwargs):
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
        "Blinded question?",
        [ModelSpec(model="claude-haiku"), ModelSpec(model="gpt-mini")],
        threshold=0.85,
        max_rounds=2,
        blinded=True,
    )
    assert result.rounds_completed == 2

    r2 = fanout_calls[1]
    assert r2["slugs"] == ["panelist-alpha.r2", "panelist-beta.r2"]
    by_slug = r2["prior_turns_by_slug"]
    assert by_slug is not None and set(by_slug) == {"panelist-alpha.r2", "panelist-beta.r2"}
    for history in by_slug.values():
        assert len(history) == 2
        assert history[1]["role"] == "assistant"


def test_assign_stable_slugs_respects_user_slug_and_index():
    specs = refine_mod._assign_stable_slugs(
        [
            ModelSpec(model="claude-haiku"),
            ModelSpec(model="claude-haiku"),
            ModelSpec(model="gpt-mini", slug="custom"),
        ]
    )
    assert [s.slug for s in specs] == ["claude-haiku-0", "claude-haiku-1", "custom-2"]
