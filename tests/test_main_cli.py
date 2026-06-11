"""The consult-mcp entry point's argument handling (previously 0% covered)."""

from __future__ import annotations

import pytest

from consult import __version__
from consult.mcp import __main__ as main_mod


def test_version_flag_prints_and_exits(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["consult-mcp", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        main_mod.cli()
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_check_flag_runs_quick_check_and_exits_with_its_code(monkeypatch):
    monkeypatch.setattr("sys.argv", ["consult-mcp", "--check"])
    monkeypatch.setattr("consult.doctor.quick_check", lambda: 3)
    with pytest.raises(SystemExit) as exc_info:
        main_mod.cli()
    assert exc_info.value.code == 3


def test_no_args_starts_the_stdio_server(monkeypatch):
    """The default path must configure LiteLLM then hand off to the server
    main loop — locked with stubs so no transport is touched."""
    calls: list[str] = []
    monkeypatch.setattr("sys.argv", ["consult-mcp"])
    monkeypatch.setattr(main_mod, "configure_litellm", lambda: calls.append("configure"))
    monkeypatch.setattr(main_mod.asyncio, "run", lambda coro: (coro.close(), calls.append("run")))
    main_mod.cli()
    assert calls == ["configure", "run"]
