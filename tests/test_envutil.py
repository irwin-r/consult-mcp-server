"""Tolerant numeric env parsing — garbage falls back instead of crashing."""

from __future__ import annotations

import pytest

from consult import registry
from consult.envutil import env_float, env_int


def test_env_float_parses_and_defaults(monkeypatch):
    monkeypatch.setenv("X_FLOAT", "2.5")
    assert env_float("X_FLOAT", 1.0) == 2.5
    monkeypatch.delenv("X_FLOAT")
    assert env_float("X_FLOAT", 1.0) == 1.0
    monkeypatch.setenv("X_FLOAT", "   ")
    assert env_float("X_FLOAT", 1.0) == 1.0


def test_env_float_garbage_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("X_FLOAT", "5,00")
    with caplog.at_level("WARNING"):
        assert env_float("X_FLOAT", 3.0) == 3.0
    assert "X_FLOAT" in caplog.text


def test_env_int_garbage_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("X_INT", "ten")
    with caplog.at_level("WARNING"):
        assert env_int("X_INT", 7) == 7
    assert "X_INT" in caplog.text


def test_max_run_usd_typo_no_longer_crashes_mid_run(monkeypatch):
    """CONSULT_MAX_RUN_USD previously hit a bare float() inside the fanout
    path, so a typo crashed the run as INTERNAL_ERROR."""
    monkeypatch.setenv("CONSULT_MAX_RUN_USD", "$5")
    assert registry.default_max_run_usd() == pytest.approx(5.0)  # packaged default
