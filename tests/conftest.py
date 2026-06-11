"""Shared fixtures for the whole suite.

Every test runs with CONSULT_RUNS_DIR pointed at a per-test tmp dir.
Without this, any test that touches the artifact layer writes real run
directories into the developer's runs dir (1,200+ had accumulated by
2026-06-11) and `consult-ledger` counts each as a $0 run, drowning the
day's real spend in noise. Tests that exercise the env-fallback chain in
`artifacts.runs_root` keep working: `monkeypatch.delenv` inside a test
removes this fixture's value for that test's duration.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSULT_RUNS_DIR", str(tmp_path / "consult-runs"))
