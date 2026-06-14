"""Routing `mode=responses` models through litellm's Responses API.

gpt-pro / gpt-codex are `mode=responses`; they 404 on chat completions, so
`_call_one` routes them through `_aresponses_as_completion`, which adapts the
ResponsesAPIResponse into the chat-completions shape the rest of the pipeline
consumes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from consult import artifacts
from consult.types import ModelSpec, Status


@pytest.mark.asyncio
async def test_aresponses_adapter_wraps_to_completion_shape(monkeypatch):
    from consult import runner

    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_text="hello from responses",
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            model_dump=lambda: {"id": "resp_1"},
        )

    monkeypatch.setattr(runner.litellm, "aresponses", fake_aresponses)

    resp = await runner._aresponses_as_completion(
        timeout=30,
        model="openai/gpt-5.5-pro",
        messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
        max_completion_tokens=2000,
        reasoning_effort="medium",
    )

    # Adapted to chat-completions shape.
    assert resp.choices[0].message.content == "hello from responses"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 11
    assert resp.usage.completion_tokens == 7
    assert resp.model_dump() == {"id": "resp_1"}

    # Chat params mapped to Responses params; chat-only keys not forwarded.
    assert captured["input"] == "hi"
    assert captured["instructions"] == "be terse"
    assert captured["max_output_tokens"] == 2000
    assert captured["reasoning"] == {"effort": "medium"}
    assert "messages" not in captured and "max_completion_tokens" not in captured


@pytest.mark.asyncio
async def test_aresponses_adapter_maps_incomplete_to_length(monkeypatch):
    from consult import runner

    async def fake_aresponses(**kwargs):
        return SimpleNamespace(
            output_text="partial",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            usage=SimpleNamespace(input_tokens=5, output_tokens=2000),
            model_dump=lambda: {},
        )

    monkeypatch.setattr(runner.litellm, "aresponses", fake_aresponses)

    resp = await runner._aresponses_as_completion(
        timeout=30,
        model="openai/gpt-5.5-pro",
        messages=[{"role": "user", "content": "x"}],
        max_completion_tokens=2000,
    )
    assert resp.choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_fanout_routes_responses_model_through_aresponses(tmp_path, monkeypatch):
    """A mode=responses panellist (gpt-pro) must hit aresponses, never the chat
    endpoint, and price via cost_per_token."""
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    async def fake_acompletion(**kwargs):
        raise AssertionError("a mode=responses model must not call chat acompletion")

    async def fake_aresponses(**kwargs):
        return SimpleNamespace(
            output_text="responses answer",
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
            model_dump=lambda: {"id": "r"},
        )

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "aresponses", fake_aresponses)
    monkeypatch.setattr(runner.litellm, "cost_per_token", lambda **kw: (0.001, 0.002))

    handle = await fanout("question?", [ModelSpec(model="gpt-pro")])

    assert len(handle.manifest) == 1
    entry = handle.manifest[0]
    assert entry.status is Status.OK
    assert entry.cost_usd == pytest.approx(0.003)
    assert entry.cost_known is True


def test_collect_stream_annotations_recovers_from_delta_and_message():
    """stream_chunk_builder drops url_citation annotations; the collector must
    recover them from the streaming delta (or a non-delta message) across
    chunks so a streamed web panellist keeps its Sources footer (issue #66).
    """
    from consult.runner.transport import _collect_stream_annotations

    ann = [{"url_citation": {"url": "https://example.com", "title": "Example"}}]
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", annotations=None))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="", annotations=ann))]),
    ]
    assert _collect_stream_annotations(chunks) == ann


def test_collect_stream_annotations_is_total_on_malformed_chunks():
    """A malformed or annotation-less chunk is skipped, never raised — the
    recovery is best-effort enrichment that must not fail a panellist.
    """
    from consult.runner.transport import _collect_stream_annotations

    chunks = [
        SimpleNamespace(choices=[]),  # empty choices → IndexError, skipped
        SimpleNamespace(),  # no choices attr → AttributeError, skipped
        SimpleNamespace(choices=[SimpleNamespace(delta=None, message=None)]),  # no holders
    ]
    assert _collect_stream_annotations(chunks) == []
