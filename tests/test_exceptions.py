"""The typed exception taxonomy must actually be raised by the engine.

These tests lock the contract documented in `consult/__init__.py`: a
library caller catching `ConsultError` (or a specific subclass) really
catches the engine's deliberate failures, while legacy `except KeyError`
and `except ValueError` call sites keep working via multiple inheritance.
"""

from __future__ import annotations

import pytest

from consult import ConsultError, PathTrustError, UnknownModelError, artifacts, registry, sources


def test_resolve_model_raises_unknown_model_error():
    with pytest.raises(UnknownModelError):
        registry.resolve_model("definitely-not-a-model")
    # Legacy call sites catch KeyError; library callers catch ConsultError.
    with pytest.raises(KeyError):
        registry.resolve_model("definitely-not-a-model")
    with pytest.raises(ConsultError):
        registry.resolve_model("definitely-not-a-model")


def test_resolve_tier_raises_unknown_model_error():
    with pytest.raises(UnknownModelError):
        registry.resolve_tier("definitely-not-a-tier")


def test_containment_failure_raises_path_trust_error(tmp_path, monkeypatch):
    outside = tmp_path / "outside.txt"
    outside.write_text("data")
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", str(trusted))
    with pytest.raises(PathTrustError):
        sources.validate_under_trusted_roots(outside)
    # The MCP adapter maps ValueError → INVALID_INPUT; the subclass keeps
    # that path working without a special case.
    with pytest.raises(ValueError):
        sources.validate_under_trusted_roots(outside)


def test_missing_path_is_plain_value_error(tmp_path):
    """Missing/unreadable is an input problem, not a trust breach — it must
    NOT be a PathTrustError."""
    with pytest.raises(ValueError) as exc_info:
        sources.validate_under_trusted_roots(tmp_path / "nope.txt")
    assert not isinstance(exc_info.value, PathTrustError)


def test_load_run_escape_raises_path_trust_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSULT_RUNS_DIR", str(tmp_path / "runs"))
    # A run-id that passes the charset check but resolves outside the runs
    # root via a symlink.
    runs = artifacts.runs_root()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (runs / "sneaky").symlink_to(target)
    with pytest.raises(PathTrustError):
        artifacts.load_run("sneaky")
