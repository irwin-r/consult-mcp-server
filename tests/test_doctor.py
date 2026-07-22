"""Offline doctor checks — the install-debug front door had 15% coverage."""

from __future__ import annotations

import json

import pytest

from consult import doctor


@pytest.fixture()
def isolated_env(monkeypatch, tmp_path):
    """Point the runs root at tmp and clear every provider key."""
    monkeypatch.setenv("CONSULT_RUNS_DIR", str(tmp_path / "runs"))
    for key in doctor._PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    return tmp_path


def test_check_paths_ok(isolated_env):
    lines, fails = doctor._check_paths()
    assert fails == 0
    assert any("runs root" in line for line in lines)
    assert any("0o700" in line for line in lines)


def test_check_keys_no_keys_is_blocking(isolated_env):
    lines, fails = doctor._check_keys()
    assert fails == 1
    assert any("no provider keys present" in line for line in lines)


def test_check_keys_one_key_passes(isolated_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    lines, fails = doctor._check_keys()
    assert fails == 0
    assert any("ANTHROPIC_API_KEY present" in line for line in lines)


def test_check_registry_reports_models_and_synthesiser(isolated_env):
    lines, fails = doctor._check_registry()
    assert fails == 0
    assert any("registry loaded" in line for line in lines)
    assert any("default synthesiser" in line for line in lines)


def test_check_trusted_roots_unset_warns_about_uncontained_attachments(isolated_env):
    lines, fails = doctor._check_trusted_roots()
    assert fails == 0
    joined = "\n".join(lines)
    # The warning must describe the real behaviour: git_diff confined to
    # CWD, file attachments NOT containment-checked.
    assert "git_diff repos restricted to CWD" in joined
    assert "NOT containment-checked" in joined


def test_check_trusted_roots_set_lists_roots(isolated_env, monkeypatch, tmp_path):
    root = tmp_path / "trusted"
    root.mkdir()
    missing = tmp_path / "missing"
    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", f"{root}:{missing}")
    lines, fails = doctor._check_trusted_roots()
    assert fails == 0
    joined = "\n".join(lines)
    assert str(root) in joined
    assert "does not exist" in joined  # the missing root gets flagged


def test_claude_desktop_config_shape(isolated_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    cfg = doctor._claude_desktop_config()
    server = cfg["mcpServers"]["consult"]
    assert server["env"]["OPENAI_API_KEY"] == "sk-test-openai"
    # Command must be absolute (Claude Desktop has no shell PATH) or the
    # documented uvx fallback shape.
    assert server["command"]
    if "args" in server:
        assert server["args"][-1] == "consult-mcp"
    payload = json.dumps(cfg)  # must be JSON-serialisable as printed
    assert "consult" in payload


def test_quick_check_fails_without_keys(isolated_env, capsys):
    rc = doctor.quick_check()
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out


def test_quick_check_passes_with_a_key(isolated_env, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    rc = doctor.quick_check()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: ready to serve" in out


def test_check_pricing_flags_first_party_but_not_aggregator(monkeypatch):
    """The canary fails (non-zero) on an unpriced first-party model, since the
    cap and ledger then run blind, but only warns on an unpriced aggregator
    model, which is expected (issue #57)."""
    fake_models = {
        "models": {
            "claude-opus": {"litellm_id": "anthropic/claude-x", "privacy_tier": "first_party"},
            "grok": {"litellm_id": "openrouter/x-ai/grok-x", "privacy_tier": "aggregator"},
            "gemini-pro": {"litellm_id": "gemini/gemini-x", "privacy_tier": "first_party"},
        }
    }
    monkeypatch.setattr(doctor.registry, "models_config", lambda: fake_models)
    # Only the gemini id is priced; the anthropic (first-party) and grok
    # (aggregator) ids are not.
    monkeypatch.setattr(doctor, "_has_pricing", lambda lid: lid == "gemini/gemini-x")

    lines, fails = doctor._check_pricing()
    text = "\n".join(lines)
    # One unpriced first-party model => fail count 1.
    assert fails == 1
    assert "first-party model claude-opus" in text
    assert "aggregator model grok" in text
    # The aggregator miss must not count toward the fail total.
    assert "first-party model grok" not in text


def test_check_pricing_all_priced_passes(monkeypatch):
    fake_models = {
        "models": {
            "claude-opus": {"litellm_id": "anthropic/claude-x", "privacy_tier": "first_party"},
        }
    }
    monkeypatch.setattr(doctor.registry, "models_config", lambda: fake_models)
    monkeypatch.setattr(doctor, "_has_pricing", lambda lid: True)

    lines, fails = doctor._check_pricing()
    assert fails == 0
    assert any("priced: 1/1" in line for line in lines)


async def test_ping_output_limit_error_counts_as_success(isolated_env, monkeypatch):
    """A reasoning model can burn the 1-token grant thinking and return an
    output-limit error. The provider authenticated and routed, which is all
    the ping asks — this must not read as a failed key.
    """
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "k")

    async def boom(**kwargs):
        raise Exception(
            "BadRequestError: Could not finish the message because max_tokens "
            "or model output limit was reached."
        )

    monkeypatch.setattr(litellm, "acompletion", boom)
    ok, msg = await doctor._ping_provider("OPENAI_API_KEY", "gpt-nano")
    assert ok
    assert "output-capped" in msg


async def test_ping_auth_error_still_fails(isolated_env, monkeypatch):
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "k")

    async def boom(**kwargs):
        raise Exception("AuthenticationError: invalid api key")

    monkeypatch.setattr(litellm, "acompletion", boom)
    ok, msg = await doctor._ping_provider("OPENAI_API_KEY", "gpt-nano")
    assert not ok
