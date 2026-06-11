"""Run-dir layout, manifests, pruning, URIs.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import pytest

from consult import artifacts
from consult.types import ManifestEntry, ModelSpec, Status


def test_artifacts_create_and_uri():
    paths = artifacts.create_run()
    assert paths.root.exists()
    assert paths.responses.exists()
    uri = paths.resource_uri("alpha")
    rid, kind, name = artifacts.parse_resource_uri(uri)
    assert rid == paths.run_id
    assert kind == "responses"
    assert name == "alpha"
    # cleanup
    import shutil

    shutil.rmtree(paths.root)


def test_artifacts_load_run_rejects_traversal_run_id(tmp_path, monkeypatch):
    """artifacts.load_run must refuse a run_id containing path separators or
    `..` even if the resolved directory would happen to exist.
    """
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # Create a sibling dir we shouldn't be able to escape to.
    (tmp_path.parent / "secret").mkdir(exist_ok=True)
    for bad in ("../secret", "..", "a/b", "/etc"):
        with pytest.raises(ValueError) as exc:
            artifacts.load_run(bad)
        assert "run_id" in str(exc.value) or "invalid" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_fanout_dropout_recovers_panellist_completed_during_notify(tmp_path, monkeypatch):
    """Regression for the dropout cancel-race: a panellist that finished
    `_call_one` (so `record_completed` ran) but is still in its final
    on_progress await when the dropout fires must NOT be mis-reported as
    TIMEOUT — its real OK entry is recovered from completed_entries. Only
    reachable with a non-None on_progress (the suspension point), which is why
    the other dropout tests (on_progress=None) miss it.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.progress import PanellistCompleted
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.05")
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.25")  # k=1, trigger=3 for n=4

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        paths.response_text(slug).write_text("ok")
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=1,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    # Block the racer's COMPLETED notify so its task stays in-flight (past
    # record_completed) until the 0.05s dropout cancels it mid-await — the
    # exact race window. CancelledError is a BaseException, so _safe_notify's
    # `except Exception` doesn't swallow it; it propagates and cancels the task.
    async def on_progress(event):
        if isinstance(event, PanellistCompleted) and event.slug == "racer-3":
            await _asyncio.sleep(10)

    specs = [
        ModelSpec(model="claude-haiku", slug="a-0"),
        ModelSpec(model="claude-haiku", slug="b-1"),
        ModelSpec(model="claude-haiku", slug="c-2"),
        ModelSpec(model="claude-haiku", slug="racer-3"),
    ]
    handle = await fanout("anything", specs, on_progress=on_progress)

    assert [m.slug for m in handle.manifest] == ["a-0", "b-1", "c-2", "racer-3"]
    racer = handle.manifest[3]
    assert racer.status is Status.OK, f"racer mis-reported as {racer.status} (cancel-race regression)"
    assert sum(1 for m in handle.manifest if m.status is Status.TIMEOUT) == 0
