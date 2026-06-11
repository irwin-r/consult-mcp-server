"""SEP-1686 task-mode protocol surface: tasks/get, tasks/result, tasks/cancel.

Task mode previously registered only tasks/get — a client could watch a
task reach `completed` but had no protocol path to the stored result and
no way to cancel. These tests drive the request handlers directly with
real SDK request objects so the wire contract stays locked.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import (
    CancelTaskRequest,
    CancelTaskRequestParams,
    GetTaskPayloadRequest,
    GetTaskPayloadRequestParams,
    GetTaskRequest,
    GetTaskRequestParams,
    TextContent,
)

from consult import task_store
from consult.mcp import server as mcp_server


@pytest.fixture(autouse=True)
def _clean_registry():
    task_store._reset_for_tests()
    yield
    task_store._reset_for_tests()


def _payload_req(task_id: str) -> GetTaskPayloadRequest:
    return GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(taskId=task_id))


def _cancel_req(task_id: str) -> CancelTaskRequest:
    return CancelTaskRequest(params=CancelTaskRequestParams(taskId=task_id))


async def test_tasks_result_returns_dict_result_in_call_tool_shape():
    rec = task_store.create()
    result = {"run_id": "r1", "synthesis": "answer", "cost_usd": 0.1}
    task_store.complete(rec.task_id, result)

    resp = await mcp_server._handle_get_task_payload(_payload_req(rec.task_id))
    dumped = resp.root.model_dump()
    assert dumped["structuredContent"] == result
    assert dumped["isError"] is False
    assert dumped["content"][0]["type"] == "text"
    assert '"run_id": "r1"' in dumped["content"][0]["text"]


async def test_tasks_result_passes_text_content_through():
    """The synthesise tool stores list[TextContent]; tasks/result must
    return it as content, not wrap it again."""
    rec = task_store.create()
    task_store.complete(rec.task_id, [TextContent(type="text", text="# Synthesis\nbody")])

    resp = await mcp_server._handle_get_task_payload(_payload_req(rec.task_id))
    dumped = resp.root.model_dump()
    assert dumped["structuredContent"] is None
    assert dumped["content"][0]["text"].startswith("# Synthesis")


async def test_tasks_result_unknown_id_errors():
    with pytest.raises(McpError, match="Unknown taskId"):
        await mcp_server._handle_get_task_payload(_payload_req("task-nope"))


async def test_tasks_result_while_working_errors():
    rec = task_store.create()
    with pytest.raises(McpError, match="still working"):
        await mcp_server._handle_get_task_payload(_payload_req(rec.task_id))


async def test_tasks_result_failed_task_returns_error_envelope():
    """A failed task's tasks/result carries the same ErrorEnvelope shape a
    foreground failure would have returned, so client error handling is
    mode-agnostic."""
    rec = task_store.create()
    task_store.fail(rec.task_id, "provider exploded")

    resp = await mcp_server._handle_get_task_payload(_payload_req(rec.task_id))
    dumped = resp.root.model_dump()
    assert dumped["structuredContent"]["ok"] is False
    assert dumped["structuredContent"]["error"]["code"] == "internal_error"
    assert "provider exploded" in dumped["structuredContent"]["error"]["message"]


async def test_tasks_result_cancelled_task_errors():
    rec = task_store.create()
    task_store.cancel(rec.task_id)
    with pytest.raises(McpError, match="cancelled"):
        await mcp_server._handle_get_task_payload(_payload_req(rec.task_id))


async def test_tasks_cancel_cancels_background_work():
    rec = task_store.create()
    started = asyncio.Event()

    async def _slow():
        started.set()
        await asyncio.sleep(60)

    bg = asyncio.create_task(_slow())
    task_store.attach_bg_task(rec.task_id, bg)
    await started.wait()

    resp = await mcp_server._handle_cancel_task(_cancel_req(rec.task_id))
    assert resp.root.status == "cancelled"
    with pytest.raises(asyncio.CancelledError):
        await bg

    # tasks/get agrees afterwards.
    get_resp = await mcp_server._handle_get_task(
        GetTaskRequest(params=GetTaskRequestParams(taskId=rec.task_id))
    )
    assert get_resp.root.status == "cancelled"


async def test_tasks_cancel_terminal_task_errors():
    rec = task_store.create()
    task_store.complete(rec.task_id, {"ok": True})
    with pytest.raises(McpError, match="already completed"):
        await mcp_server._handle_cancel_task(_cancel_req(rec.task_id))


def test_eviction_bounds_no_ttl_records(monkeypatch):
    """Terminal records without a ttl must still be evicted once the
    registry exceeds its cap — they hold full tool results and previously
    accumulated forever."""
    monkeypatch.setattr(task_store, "_MAX_RECORDS", 5)
    oldest = task_store.create()
    task_store.complete(oldest.task_id, {"n": 0})
    oldest.last_updated_at = 1.0  # force deterministic oldest-first order
    for n in range(1, 5):
        rec = task_store.create()
        task_store.complete(rec.task_id, {"n": n})
        rec.last_updated_at = float(n + 1)
    worker = task_store.create()  # never terminal, never evicted

    assert len(task_store.list_all()) == 6
    newest = task_store.create()  # triggers eviction of the oldest terminal

    ids = {r.task_id for r in task_store.list_all()}
    assert oldest.task_id not in ids
    assert worker.task_id in ids
    assert newest.task_id in ids


def test_complete_does_not_resurrect_cancelled_task():
    """(issue #33) A cancel can land while the background coroutine is past
    its last await; the late complete() must not flip the record back."""
    rec = task_store.create()
    task_store.cancel(rec.task_id)
    task_store.complete(rec.task_id, {"late": True})
    assert rec.status == task_store.STATUS_CANCELLED
    assert rec.result is None


def test_fail_does_not_overwrite_completed_task():
    """(issue #33) Same terminal guard for the failure callback."""
    rec = task_store.create()
    task_store.complete(rec.task_id, {"ok": True})
    task_store.fail(rec.task_id, "late failure")
    assert rec.status == task_store.STATUS_COMPLETED
    assert rec.error is None
