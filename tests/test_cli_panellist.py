"""Tests for the CLI-as-panellist transport (`consult/cli_executor.py`
plus the dispatch branch in `runner._call_one`).

The fixtures mock `asyncio.create_subprocess_exec` so the tests run
offline; we don't actually spawn `gemini` / `codex` / `claude`. The
mocked subprocess returns a configurable stdout/exitcode pair.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeProc:
    """Minimal stand-in for `asyncio.subprocess.Process`."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        raise_communicate: BaseException | None = None,
    ):
        self.stdout_buf = stdout
        self.stderr_buf = stderr
        self.returncode = returncode
        self.raise_communicate = raise_communicate
        self.killed = False
        self.waited = False

    async def communicate(self, input: bytes | None = None):
        if self.raise_communicate is not None:
            raise self.raise_communicate
        return self.stdout_buf, self.stderr_buf

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        return self.returncode


@pytest.mark.asyncio
async def test_call_cli_returns_litellm_shape_on_success(monkeypatch):
    from consult import cli_executor

    captured_argv = {}
    captured_stdin = {}

    async def fake_exec(*argv, **kw):
        captured_argv["argv"] = list(argv)
        return _FakeProc(stdout=b"The model says hello.")

    # Capture what we sent on stdin via the proc's communicate
    original_communicate = _FakeProc.communicate

    async def patched_communicate(self, input=None):
        captured_stdin["input"] = input
        return await original_communicate(self, input)

    monkeypatch.setattr(_FakeProc, "communicate", patched_communicate)
    monkeypatch.setattr(
        "consult.cli_executor.asyncio.create_subprocess_exec", fake_exec,
    )

    resp = await cli_executor.call_cli(
        ["gemini", "--yolo"], "please summarise X", timeout=30.0,
    )
    # argv must be the registry-configured command, no shell interpolation
    assert captured_argv["argv"] == ["gemini", "--yolo"]
    # The prompt goes to stdin, not argv (no escaping concerns)
    assert captured_stdin["input"] == b"please summarise X"
    # LiteLLM-shape response
    assert resp.choices[0].message.content == "The model says hello."
    assert resp.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_call_cli_nonzero_exit_raises_with_stderr(monkeypatch):
    from consult import cli_executor

    async def fake_exec(*argv, **kw):
        return _FakeProc(stdout=b"", stderr=b"auth required", returncode=2)

    monkeypatch.setattr(
        "consult.cli_executor.asyncio.create_subprocess_exec", fake_exec,
    )

    with pytest.raises(RuntimeError, match="exited 2"):
        await cli_executor.call_cli(["gemini"], "x", timeout=30.0)


@pytest.mark.asyncio
async def test_call_cli_timeout_kills_the_child(monkeypatch):
    import asyncio
    from consult import cli_executor

    timed_out_proc = _FakeProc(
        raise_communicate=None,  # we won't reach the body
    )
    # Patch communicate to actually hang forever
    async def hang(self, input=None):
        await asyncio.sleep(10)

    monkeypatch.setattr(_FakeProc, "communicate", hang)

    async def fake_exec(*argv, **kw):
        return timed_out_proc

    monkeypatch.setattr(
        "consult.cli_executor.asyncio.create_subprocess_exec", fake_exec,
    )

    with pytest.raises(asyncio.TimeoutError):
        await cli_executor.call_cli(["gemini"], "x", timeout=0.05)
    # SIGKILL was sent to the child
    assert timed_out_proc.killed


@pytest.mark.asyncio
async def test_runner_dispatches_cli_provider(monkeypatch, tmp_path):
    """`runner.fanout` must route a `provider="cli"` registry entry through
    `cli_executor.call_cli`, not LiteLLM, and produce a normal-shaped
    ManifestEntry with cost_usd=0."""
    from consult import artifacts, registry, runner
    from consult.types import ModelSpec, Status

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    # Register a synthetic CLI panellist via monkeypatch (don't pollute
    # the real models.json). resolve_model returns a dict with the CLI
    # fields when the alias matches.
    def fake_resolve(alias):
        if alias == "gemini-cli":
            return {
                "alias": "gemini-cli",
                "provider": "cli",
                "cli_command": ["fake-gemini", "--yolo"],
                "default_timeout_s": 30,
                "family": "gemini",
            }
        raise KeyError(alias)

    monkeypatch.setattr(registry, "resolve_model", fake_resolve)
    monkeypatch.setattr(registry, "models_config", lambda: {"models": {}, "tiers": {}, "defaults": {}})
    # Disable per-provider concurrency caps for the test
    monkeypatch.setattr(registry, "provider_concurrency", lambda: {})
    # Heartbeat off so the test runs deterministically fast
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    captured_argv = {}

    async def fake_call_cli(cli_command, prompt, *, timeout, extra_env=None):
        captured_argv["argv"] = cli_command
        captured_argv["prompt"] = prompt
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="cli body output"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None),
        )

    monkeypatch.setattr("consult.cli_executor.call_cli", fake_call_cli)

    # No litellm cost lookup should ever fire — stub it to a sentinel so
    # the test fails loudly if dispatch leaked to LiteLLM.
    def must_not_be_called(**_):
        raise AssertionError("LiteLLM cost lookup invoked for a CLI panellist")

    monkeypatch.setattr("consult.runner.litellm.completion_cost", must_not_be_called)

    handle = await runner.fanout(
        "summarise X please",
        [ModelSpec(model="gemini-cli")],
        capsule_kind="decision",
    )
    # The CLI executor was called with the registered argv and the
    # per-slug prompt (which includes the footer).
    assert captured_argv["argv"] == ["fake-gemini", "--yolo"]
    assert "summarise X please" in captured_argv["prompt"]
    # The manifest has a normal-shape OK entry with cost=0
    assert len(handle.manifest) == 1
    entry = handle.manifest[0]
    assert entry.status == Status.OK
    assert entry.cost_usd == 0.0
    assert entry.cost_known is True
    # The body file got the CLI's stdout
    body_path = tmp_path / handle.run_id / "responses" / f"{entry.slug}.txt"
    assert body_path.read_text() == "cli body output"
