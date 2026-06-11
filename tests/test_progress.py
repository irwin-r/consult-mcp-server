"""Typed progress events and bucketing.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import json

import pytest

from consult import artifacts
from consult.types import ManifestEntry, ModelSpec, Status


@pytest.mark.asyncio
async def test_fanout_emits_progress_callbacks(tmp_path, monkeypatch):
    """fanout emits a layered progress stream:
    - one `PhaseStarted(phase="fanout")` before any panellist begins
    - one `PanellistStarted` per panellist (before its network call)
    - one `PanellistCompleted` per panellist (after the call returns)
    Heartbeat ticks are disabled (interval=0) so the assertions stay
    deterministic; a separate test covers the heartbeat path.
    """
    from consult import runner
    from consult.progress import (
        PanellistCompleted,
        PanellistStarted,
        PhaseStarted,
        ProgressEvent,
    )
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    events: list[ProgressEvent] = []

    async def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    specs = [
        ModelSpec(model="claude-haiku"),
        ModelSpec(model="claude-sonnet"),
        ModelSpec(model="claude-opus"),
    ]
    handle = await fanout("p", specs, on_progress=on_progress)
    assert handle.partial is False

    # First event must be the phase boundary so the parent sees fanout
    # begin before any panellist completion fires.
    assert isinstance(events[0], PhaseStarted)
    assert events[0].phase == "fanout"
    assert events[0].total == 3

    started = [e for e in events if isinstance(e, PanellistStarted)]
    completed = [e for e in events if isinstance(e, PanellistCompleted)]

    assert len(started) == 3
    assert {e.started_count for e in started} == {1, 2, 3}
    assert all(e.total == 3 for e in started)

    assert len(completed) == 3
    assert {e.done for e in completed} == {1, 2, 3}
    assert all(e.total == 3 for e in completed)
    assert all(e.status == "OK" for e in completed)


def test_append_progress_log_writes_jsonl(tmp_path):
    """The "D" half: a tailable JSONL log in the run dir. Each line is one
    event.model_dump() with a `ts` prepended so programmatic consumers can
    parse by `kind` without scraping free-text.
    """
    from consult.progress import CapsuleExtracted, PanellistCompleted, append_progress_log

    append_progress_log(
        tmp_path,
        PanellistCompleted(
            done=1,
            total=2,
            slug="haiku",
            status="OK",
            latency_ms=42,
        ),
    )
    append_progress_log(tmp_path, CapsuleExtracted(done=1, total=2, slug="haiku"))

    lines = (tmp_path / "_progress.log").read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["kind"] == "panellist_completed"
    assert parsed[0]["slug"] == "haiku"
    assert parsed[0]["status"] == "OK"
    assert parsed[0]["latency_ms"] == 42
    assert "ts" in parsed[0]
    assert parsed[1]["kind"] == "capsule_extracted"
    assert parsed[1]["slug"] == "haiku"


def test_progress_event_message_for_every_kind():
    """Wire-format message must cover every event kind. New events must
    extend `event_message` — this test fails fast if a kind is added
    without updating the helper.
    """
    from consult.progress import (
        ArbiterScored,
        CapsuleExtracted,
        Heartbeat,
        PanellistCompleted,
        PanellistStarted,
        PhaseStarted,
        SequenceStepCompleted,
        SequenceStepStarted,
        SynthCompleted,
        SynthStarted,
        event_message,
    )

    assert "OK" in event_message(
        PanellistCompleted(
            done=1,
            total=2,
            slug="x",
            status="OK",
            latency_ms=1,
        )
    )
    assert "capsule" in event_message(CapsuleExtracted(done=1, total=2, slug="x"))
    assert "r2 arbiter" in event_message(
        ArbiterScored(
            done=1,
            total=2,
            round=2,
            score=0.5,
        )
    )
    assert event_message(SynthStarted(done=1, total=2)) == "synthesising"
    assert event_message(SynthCompleted(done=2, total=2)) == "synthesis complete"
    assert "step 3" in event_message(SequenceStepStarted(done=1, total=5, step=3))
    assert "step 3" in event_message(SequenceStepCompleted(done=2, total=5, step=3))

    started_msg = event_message(
        PanellistStarted(
            done=0,
            total=3,
            slug="alpha",
            started_count=1,
        )
    )
    assert "alpha" in started_msg
    assert "1/3" in started_msg

    assert event_message(PhaseStarted(done=0, total=3, phase="fanout")) == "phase: fanout"
    assert event_message(PhaseStarted(done=3, total=6, phase="capsules")) == "phase: capsules"

    hb_msg = event_message(
        Heartbeat(
            done=1,
            total=3,
            elapsed_ms=12_500,
            cost_so_far_usd=0.0234,
            cost_known=True,
            pending_count=2,
            pending_slugs=["gpt-pro", "claude-opus"],
        )
    )
    assert "12s" in hb_msg or "13s" in hb_msg
    assert "$0.0234" in hb_msg
    assert "2 pending" in hb_msg
    assert "gpt-pro" in hb_msg

    # Cost-unknown variant uses ≥ prefix to mark the total as a lower bound.
    hb_unknown = event_message(
        Heartbeat(
            done=1,
            total=3,
            elapsed_ms=1000,
            cost_so_far_usd=0.5,
            cost_known=False,
            pending_count=0,
            pending_slugs=[],
        )
    )
    assert "≥$0.5000" in hb_unknown


@pytest.mark.asyncio
async def test_fanout_progress_callback_failure_does_not_abort_run(tmp_path, monkeypatch):
    """A raising on_progress callback must not tear down the fanout —
    progress is best-effort, the run completes regardless.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
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

    async def bad_progress(event):
        raise RuntimeError("client went away")

    handle = await fanout(
        "p",
        [ModelSpec(model="claude-haiku")],
        on_progress=bad_progress,
    )
    assert handle.partial is False
    assert len(handle.manifest) == 1
    assert handle.manifest[0].status is Status.OK


@pytest.mark.asyncio
async def test_fanout_emits_heartbeat_while_panellists_in_flight(tmp_path, monkeypatch):
    """With a short heartbeat interval and an artificially slow panellist,
    at least one `Heartbeat` event must fire before the panellist completes
    — proving the "still working" liveness pulse works.

    The heartbeat snapshot shows the in-flight slug as pending and elapsed
    time greater than the interval.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.progress import Heartbeat, ProgressEvent
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0.05")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        await _asyncio.sleep(0.2)
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=200,
            cost_usd=0.01,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    events: list[ProgressEvent] = []

    async def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    handle = await fanout(
        "p",
        [ModelSpec(model="claude-haiku", slug="alpha")],
        on_progress=on_progress,
    )
    assert handle.partial is False

    heartbeats = [e for e in events if isinstance(e, Heartbeat)]
    assert heartbeats, f"expected at least one Heartbeat, got: {[e.kind for e in events]}"
    hb = heartbeats[0]
    assert hb.elapsed_ms >= 50  # at least one interval
    assert hb.pending_count == 1
    assert hb.pending_slugs == ["alpha"]
    # Cost-so-far is 0 before any panellist returns.
    assert hb.cost_so_far_usd == 0.0


@pytest.mark.asyncio
async def test_fanout_heartbeat_disabled_when_interval_zero(tmp_path, monkeypatch):
    """`CONSULT_HEARTBEAT_INTERVAL_S=0` disables the heartbeat task so tests
    (and clients that don't want the pulse) get a clean event stream.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.progress import Heartbeat, ProgressEvent
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        await _asyncio.sleep(0.1)
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=100,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    events: list[ProgressEvent] = []

    async def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    await fanout(
        "p",
        [ModelSpec(model="claude-haiku")],
        on_progress=on_progress,
    )
    assert not any(isinstance(e, Heartbeat) for e in events)


def test_progress_panellist_partial_event_message():
    """The new PanellistPartial event has a stable wire message."""
    from consult.progress import PanellistPartial, event_message

    ev = PanellistPartial(done=2, total=8, slug="alpha", chars_so_far=1500, elapsed_ms=4200)
    msg = event_message(ev)
    assert "alpha" in msg
    assert "1500" in msg
    assert "4200" in msg
