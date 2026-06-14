"""MCP adapter boundary: dispatch, envelopes, schemas, resources, task mode.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from consult import artifacts, registry
from consult.types import ManifestEntry, ModelSpec, RunHandle, Status


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_value_error_in_envelope(monkeypatch):
    """A handler raising ValueError must surface as `invalid_input` envelope,
    not as a raw exception bubbling out of the MCP dispatch.

    The envelope is returned as a `dict` so the MCP SDK populates
    `structuredContent` on the wire — agents can branch on `error.code`
    without re-parsing the text body.
    """
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise ValueError("max_rounds must be between 1 and 3")

    monkeypatch.setitem(server_mod._HANDLERS, "refine", bad_handler)

    payload = await server_mod.handle_call_tool("refine", {"prompt": "x", "models": []})
    assert isinstance(payload, dict)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input"
    assert "max_rounds" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_key_error_as_unknown_model(monkeypatch):
    """`registry.resolve_model` raises KeyError on a missing alias —
    `synthesise(by_model="bogus")` would propagate that through. Must
    surface as `unknown_model`, not `invalid_input`.
    """
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise KeyError("Unknown model: bogus-alias")

    monkeypatch.setitem(server_mod._HANDLERS, "synthesise", bad_handler)
    payload = await server_mod.handle_call_tool("synthesise", {"run_id": "x"})
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unknown_model"


@pytest.mark.asyncio
async def test_handle_call_tool_wraps_file_not_found_as_run_not_found(monkeypatch):
    """`artifacts.load_run` raises FileNotFoundError on missing run_id."""
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise FileNotFoundError("Run not found: 20260520-foo")

    monkeypatch.setitem(server_mod._HANDLERS, "synthesise", bad_handler)
    payload = await server_mod.handle_call_tool("synthesise", {"run_id": "20260520-foo"})
    assert payload["error"]["code"] == "run_not_found"


@pytest.mark.asyncio
async def test_handle_call_tool_unknown_tool_returns_envelope():
    """Asking for a tool that doesn't exist returns an invalid_input
    envelope rather than raising a ValueError out the top.
    """
    from consult.mcp import server as server_mod

    payload = await server_mod.handle_call_tool("not-a-tool", {})
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input"
    assert "not-a-tool" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_unhandled_exception_becomes_internal_error(monkeypatch):
    """Any unanticipated exception type from a handler must become an
    `internal_error` envelope rather than tearing out the MCP dispatch.
    """
    from consult.mcp import server as server_mod

    async def bad_handler(args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(server_mod._HANDLERS, "panel", bad_handler)
    payload = await server_mod.handle_call_tool("panel", {"prompt": "x", "models": []})
    assert payload["error"]["code"] == "internal_error"
    assert "kaboom" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_handle_call_tool_success_path_returns_dict(monkeypatch):
    """The success path must return a `dict` so the MCP SDK populates
    `structuredContent` on the response. Returning `list[TextContent]` would
    leave clients with only the JSON-text-blob fallback.
    """
    from consult.mcp import server as server_mod

    sentinel = {"run_id": "20260520-stub", "synthesis": "ok", "manifest": []}

    async def fake_handler(args, **_kwargs):
        return sentinel

    monkeypatch.setitem(server_mod._HANDLERS, "consult", fake_handler)
    result = await server_mod.handle_call_tool("consult", {"prompt": "x"})
    assert result is sentinel
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_handle_list_tools_advertises_default_surface():
    """Pin the default four-tool surface plus each tool's required input
    fields. Catches schema drift (e.g. dropping `prompt` from `panel`'s
    required list) that the type system can't see. `sequence` is demoted
    off the default surface (issue #59); see the enabled-flag test below."""
    from consult.mcp import server as server_mod

    tools = await server_mod.handle_list_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"panel", "synthesise", "consult", "refine"}
    assert "prompt" in by_name["panel"].inputSchema["required"]
    assert "models" in by_name["panel"].inputSchema["required"]
    assert "prompt" in by_name["consult"].inputSchema["required"]
    assert "prompt" in by_name["refine"].inputSchema["required"]
    assert "run_id" in by_name["synthesise"].inputSchema["required"]


@pytest.mark.asyncio
async def test_handle_list_tools_advertises_sequence_when_enabled(monkeypatch):
    """CONSULT_ENABLE_SEQUENCE re-advertises the demoted sequence tool (issue
    #59). The handler stays registered regardless; this only gates the
    advertised surface."""
    from consult.mcp import server as server_mod

    monkeypatch.setenv("CONSULT_ENABLE_SEQUENCE", "1")
    tools = await server_mod.handle_list_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"panel", "synthesise", "consult", "refine", "sequence"}
    assert "prompts" in by_name["sequence"].inputSchema["required"]


@pytest.mark.asyncio
async def test_handle_list_resources_surfaces_run_bodies(tmp_path, monkeypatch):
    """list_resources advertises each run's panellist bodies so MCP clients
    can discover them without prior knowledge of the URI grammar."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    for i in range(2):
        run_dir = tmp_path / f"20260520-fake-{i}"
        (run_dir / "responses").mkdir(parents=True)
        for slug in ("alpha", "beta"):
            (run_dir / "responses" / f"{slug}.txt").write_text("body")

    resources = await server_mod.handle_list_resources()
    uris = {str(r.uri) for r in resources}
    assert len(uris) == 4
    assert any(u.endswith("/responses/alpha") for u in uris)
    assert any(u.endswith("/responses/beta") for u in uris)
    assert all(r.mimeType == "text/plain" for r in resources)


@pytest.mark.asyncio
async def test_handle_list_resources_empty_dir_returns_empty(tmp_path, monkeypatch):
    """No runs on disk → no resources advertised. Don't crash."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    assert await server_mod.handle_list_resources() == []


@pytest.mark.asyncio
async def test_handle_read_resource_returns_body_text(tmp_path, monkeypatch):
    """Happy path: read a body via its `consult://...` URI."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.response_text("alpha").write_text("the body")

    body = await server_mod.handle_read_resource(f"consult://runs/{paths.run_id}/responses/alpha")
    assert body == "the body"


@pytest.mark.asyncio
async def test_handle_read_resource_missing_body_raises(tmp_path, monkeypatch):
    """Reading a slug whose body wasn't written must raise FileNotFoundError —
    the top-level dispatcher then maps to a RUN_NOT_FOUND envelope rather
    than returning a misleading empty body."""
    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()

    with pytest.raises(FileNotFoundError):
        await server_mod.handle_read_resource(f"consult://runs/{paths.run_id}/responses/missing")


@pytest.mark.asyncio
async def test_handle_call_tool_dispatch_routes_each_tool_name(monkeypatch):
    """The dispatch chain in handle_call_tool must route each tool name to
    its corresponding _handle_*. A typo (e.g. `"Panel"` vs `"panel"`) here
    silently breaks one tool with no compile-time signal."""
    from consult.mcp import server as server_mod

    for tool_name in ("panel", "consult", "refine", "sequence", "synthesise"):
        called: list[str] = []

        async def fake(args, _name=tool_name, _called=called, **_kwargs):
            _called.append(_name)
            return {"routed": _name}

        monkeypatch.setitem(server_mod._HANDLERS, tool_name, fake)
        await server_mod.handle_call_tool(tool_name, {})
        assert called == [tool_name]


def test_parse_resource_uri_rejects_malformed():
    """Permissive parsing would be a path-traversal hazard."""
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("http://example.com/runs/abc/responses/x")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc/responses/x/extra")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc/capsules/x")
    # Happy path still works
    rid, kind, name = artifacts.parse_resource_uri("consult://runs/r1/responses/alpha.r2")
    assert rid == "r1"
    assert kind == "responses"
    assert name == "alpha.r2"
    # attachments/ kind also parses
    rid, kind, name = artifacts.parse_resource_uri(
        "consult://runs/r1/attachments/main.py",
    )
    assert (rid, kind, name) == ("r1", "attachments", "main.py")


@pytest.mark.asyncio
async def test_runner_fanout_dispatches_prior_turns_by_slug(monkeypatch, tmp_path):
    """`prior_turns_by_slug` lets a caller (refine round 2+) give each
    panellist its own conversation history. The runner must pick the
    per-slug entry when present and fall back to the global `prior_turns`
    otherwise."""
    from consult import registry, runner
    from consult.types import ModelSpec

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(registry, "provider_concurrency", lambda: {})
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    captured: dict[str, list] = {}

    async def fake_call_one(spec, slug, per_slug_prompt, paths, provider_sems, **kw):
        captured[slug] = kw.get("prior_turns") or []
        from consult.types import ManifestEntry, Status

        return ManifestEntry(
            slug=slug,
            model_id=spec.model,
            status=Status.OK,
            resource_uri=f"consult://x/{slug}",
            body_path=str(paths.response_text(slug)),
            latency_ms=1,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call_one)
    # Disable token-counter and cost-estimation so the test stays offline
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    global_pt = [{"role": "user", "content": "global"}]
    per_slug_pt = {
        "claude-haiku": [
            {"role": "user", "content": "haiku-specific"},
            {"role": "assistant", "content": "prior haiku answer"},
        ],
    }
    await runner.fanout(
        "the prompt",
        [
            ModelSpec(model="claude-haiku"),
            ModelSpec(model="gpt-mini"),
        ],
        prior_turns=global_pt,
        prior_turns_by_slug=per_slug_pt,
    )
    # claude-haiku used the per-slug override
    assert captured["claude-haiku"] == per_slug_pt["claude-haiku"]
    # gpt-mini fell back to the global prior_turns
    assert captured["gpt-mini"] == global_pt


def test_missing_default_rubric_raises_runtime_not_filenotfound(tmp_path, monkeypatch):
    """A missing `consensus.md` (broken install) must NOT raise
    FileNotFoundError — that exception class is reserved for run-not-found
    in `server.handle_call_tool`, and a broken install was getting
    surfaced to clients as a confusing "run_id not found".
    """
    from consult import synth as synth_mod

    # Make `resolve_rubric("consensus")` return the literal sentinel so the
    # broken-install branch trips.
    monkeypatch.setattr(synth_mod.registry, "resolve_rubric", lambda name: "consensus")
    with pytest.raises(RuntimeError) as exc:
        synth_mod._resolve_rubric(None)
    assert "broken" in str(exc.value).lower()


def test_engine_modules_do_not_import_mcp_sdk():
    """The engine (`consult.*` minus `consult.mcp.*`) must not transitively
    pull in the `mcp` SDK. A regression would silently re-couple a library
    consumer to a dependency they declined to install.
    """
    import importlib
    import pkgutil

    import consult

    engine_modules: list[str] = []
    for info in pkgutil.iter_modules(consult.__path__, prefix="consult."):
        if info.name == "consult.mcp" or info.name.startswith("consult.mcp."):
            continue
        engine_modules.append(info.name)

    # The engine surface actually used (filter out viewer-only deps).
    expected_present = {
        "consult.runner",
        "consult.refine",
        "consult.sequence",
        "consult.synth",
        "consult.capsule",
        "consult.orchestrate",
        "consult.artifacts",
        "consult.types",
        "consult.progress",
        "consult.context",
        "consult.registry",
        "consult.attachments",
    }
    assert expected_present.issubset(set(engine_modules)), (
        f"missing engine modules: {expected_present - set(engine_modules)}"
    )

    for mod_name in engine_modules:
        mod = importlib.import_module(mod_name)
        src = Path(mod.__file__).read_text() if mod.__file__ else ""
        for line in src.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import mcp"), (
                f"{mod_name} imports mcp.* — broken engine/adapter boundary"
            )
            assert not stripped.startswith("from mcp"), (
                f"{mod_name} imports from mcp — broken engine/adapter boundary"
            )


async def test_orchestrate_consult_runs_without_mcp_adapter(tmp_path, monkeypatch):
    """The hero `orchestrate.consult()` is callable from any consumer with
    no mcp.* import in the chain. Stubs the three engine primitives so the
    test runs offline; the assertion is on the typed return shape, not
    panel content.
    """
    from consult import capsule as capsule_mod
    from consult import orchestrate
    from consult import runner as runner_mod
    from consult import synth as synth_mod
    from consult.types import RunResult

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(registry, "resolve_tier", lambda t: ["model-a", "model-b"])
    monkeypatch.setattr(registry, "default_synthesiser", lambda: "model-synth")
    monkeypatch.setattr(
        registry,
        "resolve_model",
        lambda alias: {"litellm_id": alias, "provider": "x"},
    )

    progress_events: list[str] = []

    async def cb(event):
        progress_events.append(event.kind)

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[
                ManifestEntry(
                    slug="model-a",
                    model_id="model-a",
                    status=Status.OK,
                    resource_uri=paths.resource_uri("model-a"),
                    body_path=str(paths.response_text("model-a")),
                    latency_ms=10,
                    cost_usd=0.01,
                    cost_known=True,
                ),
            ],
            cost_usd=0.01,
            cost_known=True,
            wall_ms=10,
        )

    async def fake_annotate(handle, **kwargs):
        return handle

    async def fake_synth(*args, **kwargs):
        return synth_mod.SynthResult(text="synthesised", cost_usd=0.02)

    monkeypatch.setattr(runner_mod, "fanout", fake_fanout)
    monkeypatch.setattr(capsule_mod, "annotate", fake_annotate)
    monkeypatch.setattr(synth_mod, "synthesise", fake_synth)

    result = await orchestrate.consult(
        "what's the call?",
        tier="quick",
        on_progress=cb,
    )
    # Typed RunResult returned, not a dict — non-MCP consumers get the
    # full Pydantic shape with structured access.
    assert isinstance(result, RunResult)
    assert result.synthesis == "synthesised"
    assert result.partial is False
    assert result.synthesiser == "model-synth"
    # Synth cost rolled into the total: 0.01 (fanout) + 0.02 (synth).
    assert abs(result.cost_usd - 0.03) < 1e-9
    # Progress events flowed through (synth_started + synth_completed are
    # emitted directly by orchestrate.consult; fanout/capsule's own events
    # are stubbed out so they don't appear).
    assert "synth_started" in progress_events
    assert "synth_completed" in progress_events


def test_resource_uri_formatter_is_context_scoped(tmp_path, monkeypatch):
    """The override is scoped to the current async/contextvars Context, so
    two concurrent consumers don't corrupt each other's manifest URIs.
    """
    import contextvars

    from consult.artifacts import (
        reset_resource_uri_formatter,
        set_resource_uri_formatter,
    )

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()

    # Default formatter — `consult://` scheme.
    reset_resource_uri_formatter()
    assert paths.resource_uri("alpha") == f"consult://runs/{paths.run_id}/responses/alpha"

    # Override scoped to a child context: parent context stays on the default.
    def install_http():
        set_resource_uri_formatter(lambda run_id, slug: f"https://example.com/runs/{run_id}/{slug}")
        return paths.resource_uri("alpha")

    child_ctx = contextvars.copy_context()
    in_child = child_ctx.run(install_http)
    assert in_child == f"https://example.com/runs/{paths.run_id}/alpha"
    # Parent context unaffected because the child's `set` only mutated its
    # own copy of the contextvars map.
    assert paths.resource_uri("alpha") == f"consult://runs/{paths.run_id}/responses/alpha"


@pytest.mark.asyncio
async def test_fanout_cancel_drains_child_tasks(tmp_path, monkeypatch):
    """Cancelling the fanout task must cancel and drain the dropout path's
    child tasks — none left running (and billing). Exercises the gatherer's
    `except BaseException` drain end-to-end.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    cancelled = {"n": 0}
    first_started = _asyncio.Event()

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        first_started.set()
        try:
            await _asyncio.sleep(30)  # block until the parent cancel reaches us
        except _asyncio.CancelledError:
            cancelled["n"] += 1
            raise
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

    specs = [ModelSpec(model="claude-haiku", slug=f"m-{i}") for i in range(4)]  # dropout path
    task = _asyncio.create_task(fanout("anything", specs))
    await first_started.wait()
    await _asyncio.sleep(0.02)  # let all four enter their sleep
    task.cancel()
    with pytest.raises(_asyncio.CancelledError):
        await task

    assert cancelled["n"] == 4  # every child cancelled and drained, no orphans


def test_schemas_declare_wire_level_bounds():
    """(issue #32) maxItems on every caller-supplied array: the engine's
    panel cap fires after parsing; these stop oversized payloads at schema
    validation."""
    from consult.mcp import schemas

    assert schemas.PANEL_SCHEMA["properties"]["models"]["maxItems"] == 64
    assert schemas.REFINE_SCHEMA["properties"]["models"]["maxItems"] == 64
    assert schemas.SEQUENCE_SCHEMA["properties"]["models"]["maxItems"] == 64
    assert schemas.SEQUENCE_SCHEMA["properties"]["prompts"]["maxItems"] == 25
    for schema in (schemas.PANEL_SCHEMA, schemas.REFINE_SCHEMA, schemas.SEQUENCE_SCHEMA):
        assert schema["properties"]["attachments"]["maxItems"] == 32
    assert schemas.consult_schema()["properties"]["attachments"]["maxItems"] == 32
    per_step = schemas.SEQUENCE_SCHEMA["properties"]["prompts"]["items"]["anyOf"][1]
    assert per_step["properties"]["attachments"]["maxItems"] == 32
