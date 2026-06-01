"""Unit tests for the shared run-slug grammar (`consult.slugs`)."""

from __future__ import annotations

from consult import slugs


def test_strip_round():
    assert slugs.strip_round("claude-haiku-0.r2") == "claude-haiku-0"
    assert slugs.strip_round("claude-haiku-0") == "claude-haiku-0"
    assert slugs.strip_round("panelist-alpha.r10") == "panelist-alpha"


def test_round_suffix():
    assert slugs.round_suffix("claude-haiku-0.r2") == ".r2"
    assert slugs.round_suffix("claude-haiku-0") == ""


def test_round_number():
    assert slugs.round_number("x.r3") == 3
    assert slugs.round_number("x.r10") == 10
    assert slugs.round_number("x") is None


def test_panel_index():
    assert slugs.panel_index("claude-haiku-2.r1") == 2
    assert slugs.panel_index("claude-haiku-2") == 2
    assert slugs.panel_index("claude-haiku") is None  # no trailing -<int>
    assert slugs.panel_index("foo-bar") is None  # trailing part isn't a digit


def test_round_suffix_only_matches_trailing():
    # A ".r" that isn't a trailing `.r<digits>` must not be treated as a round
    # suffix — the improvement over the old rfind(".r") heuristic.
    assert slugs.round_suffix("dr.who-1") == ""
    assert slugs.strip_round("dr.who-1") == "dr.who-1"
