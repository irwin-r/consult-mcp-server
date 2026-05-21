"""Tests for `consult.strategies` — pluggable refine round-to-round."""

from __future__ import annotations

import pytest

from consult.strategies import (
    DefaultStrategy,
    EliminationStrategy,
    list_strategies,
    strategy_for,
)
from consult.types import (
    Capsule,
    ManifestEntry,
    ModelSpec,
    Status,
)


def _ok_entry(slug: str, model_id: str, position: str) -> ManifestEntry:
    return ManifestEntry(
        slug=slug,
        model_id=model_id,
        status=Status.OK,
        resource_uri=f"consult://x/{slug}",
        body_path=f"/x/{slug}",
        capsule=Capsule(position=position, recommendation=position),
    )


def test_strategy_for_unknown_raises():
    with pytest.raises(ValueError, match="Unknown refine strategy"):
        strategy_for("not-real")


def test_list_strategies_includes_default_and_elimination():
    names = list_strategies()
    assert "default" in names
    assert "elimination" in names


def test_default_strategy_passes_specs_through():
    strat = DefaultStrategy()
    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="gpt-mini")]
    out_r1 = strat.before_round(round_num=1, base_specs=specs, prior_manifest=None)
    assert [s.model for s in out_r1] == ["claude-haiku", "gpt-mini"]
    out_r2 = strat.before_round(
        round_num=2,
        base_specs=specs,
        prior_manifest=[_ok_entry("a-0.r1", "x/a", "anything")],
    )
    assert [s.model for s in out_r2] == ["claude-haiku", "gpt-mini"]


def test_elimination_strategy_round_one_returns_full_panel():
    """Elimination is round-2-onwards. Round 1 must always be the full panel —
    we need round-1 capsules before we can identify an outlier."""
    strat = EliminationStrategy()
    specs = [ModelSpec(model="a"), ModelSpec(model="b"), ModelSpec(model="c")]
    out = strat.before_round(round_num=1, base_specs=specs, prior_manifest=None)
    assert [s.model for s in out] == ["a", "b", "c"]


def test_elimination_drops_outlier_by_round_two():
    """Three panellists; two agree closely, one diverges. Elimination
    should drop the outlier from round 2."""
    strat = EliminationStrategy()
    specs = [
        ModelSpec(model="a"),
        ModelSpec(model="b"),
        ModelSpec(model="c"),
    ]
    prior_manifest = [
        _ok_entry("a-0.r1", "x/a", "approach X with caveat"),
        _ok_entry("b-1.r1", "x/b", "approach X with caveat plus"),
        _ok_entry("c-2.r1", "x/c", "totally different approach Y dont do anything"),
    ]
    out = strat.before_round(round_num=2, base_specs=specs, prior_manifest=prior_manifest)
    out_models = [s.model for s in out]
    # The outlier (c) should be eliminated. a and b stay.
    assert "c" not in out_models
    assert "a" in out_models
    assert "b" in out_models


def test_elimination_floors_at_two_panellists():
    """Don't eliminate below 2 panellists — would defeat the consensus
    purpose. Once we have 2 or fewer, elimination is a no-op."""
    strat = EliminationStrategy()
    specs = [ModelSpec(model="a"), ModelSpec(model="b")]
    prior = [
        _ok_entry("a-0.r1", "x/a", "X"),
        _ok_entry("b-1.r1", "x/b", "Y differs entirely from anything else here"),
    ]
    out = strat.before_round(round_num=2, base_specs=specs, prior_manifest=prior)
    # With only 2 to start, the outlier compute returns None (we need
    # 3+ usable to call one an outlier). Full panel passes through.
    assert len(out) == 2


def test_elimination_is_monotone_across_rounds():
    """A panellist eliminated in round 2 stays eliminated in round 3 —
    the assumption is that structural outliers don't suddenly align."""
    strat = EliminationStrategy()
    specs = [
        ModelSpec(model="a"),
        ModelSpec(model="b"),
        ModelSpec(model="c"),
        ModelSpec(model="d"),
    ]
    prior_r1 = [
        _ok_entry("a-0.r1", "x/a", "approach X tight"),
        _ok_entry("b-1.r1", "x/b", "approach X tighter"),
        _ok_entry("c-2.r1", "x/c", "approach X tightly"),
        _ok_entry("d-3.r1", "x/d", "abandon the whole project entirely"),
    ]
    out_r2 = strat.before_round(round_num=2, base_specs=specs, prior_manifest=prior_r1)
    assert "d" not in [s.model for s in out_r2]

    # Round 3 with different prior_manifest (round 2's). 'd' is gone.
    prior_r2 = [
        _ok_entry("a-0.r2", "x/a", "X tight"),
        _ok_entry("b-1.r2", "x/b", "X tighter"),
        _ok_entry("c-2.r2", "x/c", "X tightly with extra"),
    ]
    out_r3 = strat.before_round(round_num=3, base_specs=specs, prior_manifest=prior_r2)
    assert "d" not in [s.model for s in out_r3]


def test_elimination_falls_back_when_panel_has_no_usable_capsules():
    """When round-N-1 produced an all-failed manifest, there's no signal
    to identify an outlier — strategy must fall back to the full panel."""
    strat = EliminationStrategy()
    specs = [ModelSpec(model="a"), ModelSpec(model="b"), ModelSpec(model="c")]
    failed_prior = [
        ManifestEntry(
            slug=f"{m}-{i}.r1", model_id=f"x/{m}", status=Status.ERROR,
            resource_uri=f"consult://x/{m}", body_path=f"/x/{m}",
            error="boom",
        )
        for i, m in enumerate(("a", "b", "c"))
    ]
    out = strat.before_round(round_num=2, base_specs=specs, prior_manifest=failed_prior)
    # No usable capsules ⇒ no outlier found ⇒ full panel
    assert len(out) == 3
