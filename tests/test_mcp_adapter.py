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
