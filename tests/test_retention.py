"""Tests for run-artifact retention (`consult-gc` / `artifacts.prune_runs`)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from consult import artifacts


def _make_run(root: Path, name: str, age_days: float) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    mtime = time.time() - age_days * 86400.0
    os.utime(d, (mtime, mtime))
    return d


def test_prune_runs_by_count(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    for i in range(5):
        _make_run(tmp_path, f"20260101-00000{i}-1000", age_days=i)  # run i is i days old
    deleted = artifacts.prune_runs(max_count=2)  # keep the newest 2
    assert len(deleted) == 3
    assert sum(1 for p in tmp_path.iterdir() if p.is_dir()) == 2


def test_prune_runs_by_age(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    _make_run(tmp_path, "20260101-000000-1000", age_days=40)
    _make_run(tmp_path, "20260102-000000-1000", age_days=10)
    _make_run(tmp_path, "20260103-000000-1000", age_days=1)
    deleted = artifacts.prune_runs(max_age_days=30)
    assert deleted == ["20260101-000000-1000"]
    assert (tmp_path / "20260102-000000-1000").exists()
    assert (tmp_path / "20260103-000000-1000").exists()


def test_prune_runs_dry_run_deletes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    for i in range(3):
        _make_run(tmp_path, f"20260101-00000{i}-1000", age_days=i)
    deleted = artifacts.prune_runs(max_count=1, dry_run=True)
    assert len(deleted) == 2
    assert sum(1 for p in tmp_path.iterdir() if p.is_dir()) == 3  # nothing removed


def test_prune_runs_noop_without_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    _make_run(tmp_path, "20260101-000000-1000", age_days=99)
    assert artifacts.prune_runs() == []
    assert sum(1 for p in tmp_path.iterdir() if p.is_dir()) == 1


def test_prune_runs_ignores_non_run_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    _make_run(tmp_path, "20260101-000000-1000", age_days=99)
    # A stray non-run-id directory must be left alone even under an age bound.
    stray = tmp_path / ".cache"
    stray.mkdir()
    os.utime(stray, (time.time() - 99 * 86400.0,) * 2)
    deleted = artifacts.prune_runs(max_age_days=1)
    assert deleted == ["20260101-000000-1000"]
    assert stray.exists()


def test_prune_runs_rejects_nonpositive_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            artifacts.prune_runs(max_count=bad)
        with pytest.raises(ValueError):
            artifacts.prune_runs(max_age_days=bad)


def test_prune_runs_skips_symlinked_run_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    _make_run(tmp_path, "20260101-000000-1000", age_days=99)  # real run, doomed by age
    target = tmp_path / "_target"  # non-run-id name, skipped on its own
    target.mkdir()
    link = tmp_path / "20260102-000000-2000"  # run-id-named symlink
    link.symlink_to(target, target_is_directory=True)
    deleted = artifacts.prune_runs(max_age_days=1)
    assert deleted == ["20260101-000000-1000"]
    assert link.is_symlink() and target.exists()  # symlink + its target untouched


def test_env_helpers_warn_on_malformed(monkeypatch, caplog):
    import logging

    from consult import retention

    monkeypatch.setenv("CONSULT_RUNS_MAX", "abc")
    monkeypatch.setenv("CONSULT_RUNS_RETENTION_DAYS", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="consult.retention"):
        assert retention._env_int("CONSULT_RUNS_MAX") is None
        assert retention._env_float("CONSULT_RUNS_RETENTION_DAYS") is None
    assert "CONSULT_RUNS_MAX" in caplog.text
    assert "CONSULT_RUNS_RETENTION_DAYS" in caplog.text
