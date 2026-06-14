"""Tests for `consult.strategies` — pluggable refine round-to-round."""

from __future__ import annotations

import pytest

from consult.strategies import (
    DefaultStrategy,
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


def test_list_strategies_is_default_only():
    """Only the default strategy ships now; elimination was retired (issue #59).
    The hook stays, so this guards against an accidental re-add to the surface.
    """
    names = list_strategies()
    assert names == ["default"]


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
