"""Stable panellist identity across refine rounds.

The old per-round slug enumeration re-indexed whatever subset the strategy
returned, so dropping a panellist shifted every later panellist's identity:
the elimination strategy then removed the wrong model from round 3 onwards,
and shifted panellists silently lost their conversation history. These
tests pin the stable-slug behaviour, deliberately eliminating a MIDDLE
panellist (the existing tests only ever dropped the last one, where the
shift cannot bite).
"""

from __future__ import annotations

import copy

from consult import artifacts
from consult import refine as refine_mod
from consult.strategies import EliminationStrategy
from consult.types import (
    ArbiterVerdict,
    Capsule,
    ManifestEntry,
    ModelSpec,
    RunHandle,
    Status,
)


def _ok_entry(slug: str, position: str) -> ManifestEntry:
    return ManifestEntry(
        slug=slug,
        model_id=f"x/{slug}",
        status=Status.OK,
        resource_uri=f"consult://x/{slug}",
        body_path=f"/x/{slug}",
        capsule=Capsule(position=position, recommendation=position),
    )


def _slugged(models: list[str]) -> list[ModelSpec]:
    """Specs the way refine hands them to the strategy: stable unique slugs
    pinned to the original index."""
    return [ModelSpec(model=m, slug=f"{m}-{i}") for i, m in enumerate(models)]


def test_elimination_targets_right_spec_after_panel_shrinks():
    """Round 1 eliminates the MIDDLE panellist (b). Round 2 then runs
    [a, c, d] and its outlier is d. The old positional mapping resolved
    d's round-2 position (2) against the FULL spec list and eliminated c
    instead; slug matching must target d."""
    strat = EliminationStrategy()
    specs = _slugged(["a", "b", "c", "d"])

    prior_r1 = [
        _ok_entry("a-0.r1", "approach X solid"),
        _ok_entry("b-1.r1", "wildly different approach Y entirely"),
        _ok_entry("c-2.r1", "approach X solid"),
        _ok_entry("d-3.r1", "approach X solid"),
    ]
    out_r2 = strat.before_round(round_num=2, base_specs=specs, prior_manifest=prior_r1)
    assert [s.model for s in out_r2] == ["a", "c", "d"]

    # Round 2 manifest: stable slugs, so c keeps -2 and d keeps -3 even
    # though their positions in the round are now 1 and 2.
    prior_r2 = [
        _ok_entry("a-0.r2", "approach X solid"),
        _ok_entry("c-2.r2", "approach X solid"),
        _ok_entry("d-3.r2", "now utterly divergent take Z instead"),
    ]
    out_r3 = strat.before_round(round_num=3, base_specs=specs, prior_manifest=prior_r2)
    assert [s.model for s in out_r3] == ["a", "c"]  # d gone, c kept
    assert "b" not in [s.model for s in out_r3]  # earlier eliminee stays out


def test_no_elimination_paths_never_readmit_prior_eliminees():
    """Every no-new-elimination branch must return the surviving panel,
    not the full original — returning base_specs re-admitted eliminees."""
    strat = EliminationStrategy()
    specs = _slugged(["a", "b", "c"])
    prior_r1 = [
        _ok_entry("a-0.r1", "approach X solid"),
        _ok_entry("b-1.r1", "wildly different approach Y entirely"),
        _ok_entry("c-2.r1", "approach X solid"),
    ]
    out_r2 = strat.before_round(round_num=2, base_specs=specs, prior_manifest=prior_r1)
    assert [s.model for s in out_r2] == ["a", "c"]

    # Round 2 had only two usable capsules → no outlier computable → the
    # panel must stay [a, c], not bounce back to [a, b, c].
    prior_r2 = [
        _ok_entry("a-0.r2", "approach X solid"),
        _ok_entry("c-2.r2", "different enough to disagree with a"),
    ]
    out_r3 = strat.before_round(round_num=3, base_specs=specs, prior_manifest=prior_r2)
    assert [s.model for s in out_r3] == ["a", "c"]

    # Unmappable worst slug (e.g. blinded greek manifest) → same rule.
    prior_r2_blinded = [
        _ok_entry("panelist-alpha.r2", "approach X solid"),
        _ok_entry("panelist-beta.r2", "approach X solid"),
        _ok_entry("panelist-gamma.r2", "wildly different approach Y entirely"),
    ]
    out_r3b = strat.before_round(round_num=3, base_specs=specs, prior_manifest=prior_r2_blinded)
    assert [s.model for s in out_r3b] == ["a", "c"]


async def test_refine_elimination_keeps_history_for_shifted_panellists(monkeypatch, tmp_path):
    """End-to-end through refine(): eliminate the middle panellist after
    round 1, then assert (a) round 3 eliminates the genuine round-2
    outlier and (b) survivors whose round position shifted still receive
    their accumulated per-slug history. Under the old positional
    identity, c and d lost their history the moment b was dropped."""
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    # position scripts per model per round: round 1 outlier is gpt-mini
    # (index 1), round 2 outlier is grok (index 3).
    positions = {
        1: {
            "claude-haiku": "approach X solid",
            "gpt-mini": "wildly different approach Y entirely",
            "gemini-flash": "approach X solid",
            "grok": "approach X solid",
        },
        2: {
            "claude-haiku": "approach X solid",
            "gemini-flash": "approach X solid",
            "grok": "now utterly divergent take Z instead",
        },
        3: {
            "claude-haiku": "approach X solid",
            "gemini-flash": "approach X solid",
        },
    }

    fanout_calls: list[dict] = []
    round_counter = {"n": 0}

    async def fake_fanout(prompt, specs, **kwargs):
        round_counter["n"] += 1
        rnd = round_counter["n"]
        fanout_calls.append(
            {
                "round": rnd,
                "slugs": [s.slug for s in specs],
                "models": [s.model for s in specs],
                "prior_turns_by_slug": copy.deepcopy(kwargs.get("prior_turns_by_slug")),
            }
        )
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        manifest = []
        for spec in specs:
            body_path = paths.root / "responses" / f"{spec.slug}.txt"
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_text(f"{spec.model} body r{rnd}")
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
                    capsule=Capsule(
                        position=positions[rnd][spec.model],
                        recommendation=positions[rnd][spec.model],
                    ),
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
        score = 0.2 if arbiter_n["n"] < 3 else 0.95
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
        [
            ModelSpec(model="claude-haiku"),
            ModelSpec(model="gpt-mini"),
            ModelSpec(model="gemini-flash"),
            ModelSpec(model="grok"),
        ],
        threshold=0.85,
        max_rounds=3,
        strategy="elimination",
    )
    assert result.rounds_completed == 3

    r1, r2, r3 = fanout_calls
    # Round 2 drops the middle panellist; survivors keep their original
    # index in the slug (gemini-flash stays -2, grok stays -3).
    assert r2["models"] == ["claude-haiku", "gemini-flash", "grok"]
    assert r2["slugs"] == ["claude-haiku-0.r2", "gemini-flash-2.r2", "grok-3.r2"]

    # Round 3 eliminates grok — the actual round-2 outlier. The old
    # positional mapping would have eliminated gemini-flash here.
    assert r3["models"] == ["claude-haiku", "gemini-flash"]
    assert r3["slugs"] == ["claude-haiku-0.r3", "gemini-flash-2.r3"]

    # History: every surviving panellist carries BOTH prior rounds even
    # though its round position shifted. Keys are the round-3 slugs.
    by_slug = r3["prior_turns_by_slug"]
    assert set(by_slug) == {"claude-haiku-0.r3", "gemini-flash-2.r3"}
    for slug, history in by_slug.items():
        assert len(history) == 4, f"{slug} lost history: {history}"
        assert history[1]["content"].endswith("body r1")
        assert history[3]["content"].endswith("body r2")


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
