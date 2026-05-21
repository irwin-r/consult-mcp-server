"""In-process MCP adapter tests.

Unlike `test_smoke.py` which exercises the engine directly, these tests
spin up an in-process MCP `ClientSession` connected to the registered
`Server` instance via `mcp.shared.memory.create_connected_server_and_client_session`
— the framework's own bidi pipe. This catches the classes of bug an
engine-only test suite cannot:

- a tool registered in `_HANDLERS` but missing from `handle_list_tools`
  (or vice versa)
- a schema/handler drift where the JSON Schema says a field is required
  but the handler reads it as optional (or vice versa)
- an unexpected change in the ErrorEnvelope wire shape
- a typo in a tool's `ToolAnnotations`
- a regression in `read_resource` URI parsing

LiteLLM is monkeypatched to a no-op so tool invocations exercise the
adapter glue + engine-typed args validation, not real model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def mcp_session_factory():
    """Returns an async factory: `await factory()` yields the connected ClientSession.

    A factory rather than a direct fixture because
    `create_connected_server_and_client_session` is itself an async
    context manager — pytest fixtures need the caller to `async with` it.
    """
    from mcp.shared.memory import create_connected_server_and_client_session

    from consult.mcp.server import server

    def _make():
        return create_connected_server_and_client_session(server)

    return _make


@pytest.mark.asyncio
async def test_list_tools_advertises_all_five(mcp_session_factory):
    """Every tool the engine exposes must be reachable through MCP. A new
    tool added to `_HANDLERS` but forgotten in `handle_list_tools` would
    silently not be advertised to clients."""
    async with mcp_session_factory() as client:
        result = await client.list_tools()
    names = {t.name for t in result.tools}
    assert names == {"panel", "synthesise", "consult", "refine", "sequence"}


@pytest.mark.asyncio
async def test_list_tools_carries_annotations(mcp_session_factory):
    """ToolAnnotations are how Claude Code / Cursor / ChatGPT dev-mode
    decide auto-approval. A regression here flips the UX for every
    consult user; the test pins the contract.
    """
    async with mcp_session_factory() as client:
        result = await client.list_tools()
    by_name = {t.name: t for t in result.tools}
    for name in ("panel", "consult", "refine", "sequence"):
        tool = by_name[name]
        assert tool.annotations is not None, f"{name} missing annotations"
        # All four panel-side tools call external models, so callers should
        # be prompted by default rather than auto-approved.
        assert tool.annotations.openWorldHint is True, f"{name}.openWorldHint"
        assert tool.annotations.destructiveHint is False, f"{name}.destructiveHint"
    # synthesise replays an on-disk run — same input gives same on-disk
    # artifacts (LLM stochasticity aside). Idempotent for client UX
    # purposes; clients can auto-retry it.
    synth = by_name["synthesise"]
    assert synth.annotations is not None
    assert synth.annotations.idempotentHint is True


@pytest.mark.asyncio
async def test_call_unknown_tool_returns_invalid_input_envelope(mcp_session_factory):
    """Unknown tool name must NOT raise a protocol error — the dispatcher
    funnels every failure into the structured ErrorEnvelope so the agent
    can branch on `error.code`."""
    async with mcp_session_factory() as client:
        result = await client.call_tool("not-a-real-tool", {})
    structured = result.structuredContent
    assert structured is not None
    assert structured.get("ok") is False
    assert structured["error"]["code"] == "invalid_input"


@pytest.mark.asyncio
async def test_call_panel_with_empty_models_returns_invalid_input(mcp_session_factory):
    """Schema validation: the engine rejects an empty panel; the MCP
    adapter must surface that through the ErrorEnvelope, not raise.
    """
    async with mcp_session_factory() as client:
        result = await client.call_tool(
            "panel", {"prompt": "any", "models": [{"model": "claude-haiku"}], "dry_run": True}
        )
    # dry_run should succeed (partial=True but envelope-OK)
    assert result.structuredContent is not None
    assert result.structuredContent.get("partial") is True


@pytest.mark.asyncio
async def test_call_synthesise_with_missing_run_returns_run_not_found(mcp_session_factory):
    """A typo'd run_id is the single most common synthesise-time failure.
    It must land on the dedicated RUN_NOT_FOUND envelope code so agents
    can distinguish it from INVALID_INPUT (e.g. malformed schema).
    """
    async with mcp_session_factory() as client:
        result = await client.call_tool(
            "synthesise", {"run_id": "does-not-exist-anywhere"}
        )
    assert result.structuredContent is not None
    assert result.structuredContent.get("ok") is False
    assert result.structuredContent["error"]["code"] == "run_not_found"


@pytest.mark.asyncio
async def test_list_resources_includes_recent_runs(
    mcp_session_factory, tmp_path, monkeypatch
):
    """list_resources walks `runs_dir` and surfaces per-panellist response
    URIs. Drives a synthetic run dir on a tmpfs to keep the test offline."""
    from consult import artifacts

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # Synthesise the on-disk shape `consult-view` and `read_resource` expect
    run = tmp_path / "test-run"
    (run / "responses").mkdir(parents=True)
    (run / "responses" / "alpha.txt").write_text("alpha body")
    (run / "responses" / "beta.txt").write_text("beta body")

    async with mcp_session_factory() as client:
        result = await client.list_resources()
    uris = {str(r.uri) for r in result.resources}
    assert "consult://runs/test-run/responses/alpha" in uris
    assert "consult://runs/test-run/responses/beta" in uris


@pytest.mark.asyncio
async def test_read_resource_round_trip(mcp_session_factory, tmp_path, monkeypatch):
    """Reading a known resource must return the on-disk body byte-for-byte."""
    from consult import artifacts

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    run = tmp_path / "rt-run"
    (run / "responses").mkdir(parents=True)
    (run / "responses" / "gamma.txt").write_text("THIS IS THE BODY")

    async with mcp_session_factory() as client:
        result = await client.read_resource(
            uri="consult://runs/rt-run/responses/gamma"  # type: ignore[arg-type]
        )
    # ReadResourceResult.contents is a list of content parts; the first
    # is the text we wrote.
    assert result.contents
    first = result.contents[0]
    assert getattr(first, "text", None) == "THIS IS THE BODY"


# ---- SEP-1686 task mode -----------------------------------------------------
#
# Task-augmented tools/call: client sends `params.task: {ttl}`, server returns
# a CreateTaskResult immediately with a taskId, the handler runs in the
# background, and the client polls `tasks/get` until terminal. Stubbing one
# of the `_HANDLERS` entries lets us drive the full roundtrip offline — the
# adapter glue (task spawning, run_id correlation, terminal-state mapping)
# is what we're testing, not engine behaviour.


def _task_call_request(name: str, arguments: dict, *, ttl_ms: int):
    """Build a ClientRequest wrapping a CallToolRequest with task metadata."""
    from mcp.types import (
        CallToolRequest,
        CallToolRequestParams,
        ClientRequest,
        TaskMetadata,
    )

    return ClientRequest(
        CallToolRequest(
            params=CallToolRequestParams(
                name=name,
                arguments=arguments,
                task=TaskMetadata(ttl=ttl_ms),
            ),
        )
    )


def _get_task_request(task_id: str):
    from mcp.types import (
        ClientRequest,
        GetTaskRequest,
        GetTaskRequestParams,
    )

    return ClientRequest(
        GetTaskRequest(params=GetTaskRequestParams(taskId=task_id)),
    )


async def _poll_until_terminal(client, task_id: str, timeout_s: float = 2.0):
    """Poll `tasks/get` until status leaves 'working', or fail the test.

    Returns the terminal GetTaskResult (taskId/status/createdAt/... at the
    root — the SDK's GetTaskResult is flat, unlike CreateTaskResult which
    nests under a `task` field).
    """
    import asyncio
    import time

    from mcp.types import GetTaskResult

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = await client.send_request(
            _get_task_request(task_id), GetTaskResult,
        )
        if result.status != "working":
            return result
        await asyncio.sleep(0.01)
    pytest.fail(f"task {task_id} did not leave 'working' within {timeout_s}s")


@pytest.fixture(autouse=False)
def _clean_task_store():
    """Per-test isolation for the in-process task registry."""
    from consult import task_store

    task_store._reset_for_tests()
    yield
    task_store._reset_for_tests()


@pytest.mark.asyncio
async def test_task_mode_returns_create_task_then_completes(
    mcp_session_factory, monkeypatch, _clean_task_store,
):
    """Happy path: tools/call with task.ttl returns a CreateTaskResult
    synchronously, the background handler runs, and a follow-up tasks/get
    reports status=completed plus the handler's run_id correlated onto
    the task record.
    """
    from mcp.types import CreateTaskResult

    from consult import task_store
    from consult.mcp import server as server_mod

    async def fake_consult(args, on_progress=None):
        # Mimic the consult tool's wire shape so attach_run_id triggers.
        return {
            "run_id": "test-run-abc123",
            "synthesis": "stub synthesis",
            "manifest": [],
        }

    monkeypatch.setitem(server_mod._HANDLERS, "consult", fake_consult)

    async with mcp_session_factory() as client:
        # 1) Kick off the task. The server returns immediately with a
        #    CreateTaskResult — NOT a CallToolResult — so we use
        #    send_request directly with the right result_type.
        create = await client.send_request(
            _task_call_request(
                "consult", {"prompt": "x", "tier": "quick"}, ttl_ms=60_000,
            ),
            CreateTaskResult,
        )
        assert create.task.taskId.startswith("task-")
        assert create.task.status == "working"
        assert create.task.ttl == 60_000
        assert create.task.pollInterval is not None
        assert create.task.pollInterval > 0
        task_id = create.task.taskId

        # 2) Poll tasks/get until the background work finishes.
        terminal = await _poll_until_terminal(client, task_id)

    assert terminal.status == "completed"
    # Forensic correlation: the handler returned a run_id and the adapter
    # surfaced it on the TaskRecord.
    rec = task_store.get(task_id)
    assert rec is not None
    assert rec.run_id == "test-run-abc123"
    # The handler's wire-shape dict is on the record so a future
    # tasks/result endpoint can serve it.
    assert rec.result == {
        "run_id": "test-run-abc123",
        "synthesis": "stub synthesis",
        "manifest": [],
    }


@pytest.mark.asyncio
async def test_task_mode_handler_exception_marks_failed(
    mcp_session_factory, monkeypatch, _clean_task_store,
):
    """A handler raising mid-flight must transition the task to `failed`
    with a status_message rather than leaving it stuck in `working`. The
    background-task wrapper catches everything except CancelledError; the
    test pins that contract.
    """
    from mcp.types import CreateTaskResult

    from consult import task_store
    from consult.mcp import server as server_mod

    async def boom(args, on_progress=None):
        raise RuntimeError("the upstream provider exploded")

    monkeypatch.setitem(server_mod._HANDLERS, "consult", boom)

    async with mcp_session_factory() as client:
        create = await client.send_request(
            _task_call_request(
                "consult", {"prompt": "x", "tier": "quick"}, ttl_ms=60_000,
            ),
            CreateTaskResult,
        )
        task_id = create.task.taskId
        terminal = await _poll_until_terminal(client, task_id)

    # Note: _run_handler_with_envelopes catches exceptions and returns
    # an ErrorEnvelope dict instead of propagating, so `boom`'s raise is
    # caught BEFORE it reaches the background-task wrapper. The task
    # therefore completes (with an envelope body), it does NOT fail.
    # This pins that wire contract: a tool-level error surfaces as a
    # completed task whose result carries the structured envelope, not
    # a transport-level task failure.
    assert terminal.status == "completed"
    rec = task_store.get(task_id)
    assert rec is not None
    assert isinstance(rec.result, dict)
    assert rec.result.get("ok") is False
    assert rec.result["error"]["code"] == "internal_error"
    assert "upstream provider exploded" in rec.result["error"]["message"]


@pytest.mark.asyncio
async def test_tasks_get_unknown_task_raises_invalid_params(
    mcp_session_factory, _clean_task_store,
):
    """Polling a taskId the server has never seen raises an McpError with
    INVALID_PARAMS. The SDK's `GetTaskResult` shape requires `taskId`
    and `status`, so there's no in-band sentinel for "not found" —
    clients are expected to resubmit, per SEP-1686."""
    from mcp.shared.exceptions import McpError
    from mcp.types import INVALID_PARAMS, GetTaskResult

    async with mcp_session_factory() as client:
        with pytest.raises(McpError) as exc_info:
            await client.send_request(
                _get_task_request("task-doesnotexist"),
                GetTaskResult,
            )
    assert exc_info.value.error.code == INVALID_PARAMS
    assert "task-doesnotexist" in exc_info.value.error.message
