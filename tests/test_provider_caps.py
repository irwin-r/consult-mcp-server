"""Per-provider capability flags.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_capsule_extractor_omits_temperature_for_gemini(monkeypatch):
    """Gemini-3 emits a warning + can loop when temperature < 1.0. The
    extractor must omit the param entirely for any gemini-routed model
    (direct or via openrouter) while keeping it for everyone else.
    """
    import litellm

    from consult import capsule as capsule_mod

    captured: list[dict[str, Any]] = []

    async def fake_completion(**kwargs):
        captured.append(kwargs)

        # Return a shape capsule._extract_one can parse cleanly
        class _Resp:
            def __init__(self):
                self.choices = [
                    type("Msg", (), {"message": type("M", (), {"content": '{"position": "x"}'})()})()
                ]

            def model_dump(self):
                return {}

        return _Resp()

    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0)

    # Gemini direct
    await capsule_mod._extract_one("hello body", "gemini/gemini-3.1-pro", 30)
    # Gemini via openrouter
    await capsule_mod._extract_one("hello body", "openrouter/google/gemini-3-pro", 30)
    # Anthropic — should still get temperature=0.0
    await capsule_mod._extract_one("hello body", "anthropic/claude-haiku-4-5", 30)

    assert len(captured) == 3
    assert "temperature" not in captured[0]  # gemini direct
    assert "temperature" not in captured[1]  # gemini via openrouter
    assert captured[2]["temperature"] == 0.0  # anthropic keeps it


def test_provider_caps_temperature_blocks_gemini_and_opus():
    """Both the legacy gemini substring case and the new claude-opus-4-7
    case must be blocked. Regressions here cause every refine arbiter
    call to fail with a BadRequestError (live-found during the dogfood
    pass that produced this whole sweep)."""
    from consult import provider_caps

    provider_caps.reset_cache()
    assert provider_caps.supports_temperature("gpt-4") is True
    assert provider_caps.supports_temperature("anthropic/claude-sonnet-4-6") is True
    assert provider_caps.supports_temperature("anthropic/claude-opus-4-7") is False
    assert provider_caps.supports_temperature("openrouter/google/gemini-3.1-pro-preview") is False
    assert provider_caps.supports_temperature("gemini/gemini-3.1-pro-preview") is False


def test_provider_caps_env_override_extends_deny_list(monkeypatch):
    """Operators add to the deny list via env without code edits."""
    from consult import provider_caps

    monkeypatch.setenv("CONSULT_NO_TEMPERATURE", "weird-future-model")
    provider_caps.reset_cache()
    try:
        assert provider_caps.supports_temperature("vendor/weird-future-model-v2") is False
        # Built-ins still apply.
        assert provider_caps.supports_temperature("anthropic/claude-opus-4-7") is False
    finally:
        provider_caps.reset_cache()


def test_provider_caps_apply_temperature_skips_blocked_models():
    """apply_temperature is the single place call sites set the kwarg."""
    from consult import provider_caps

    provider_caps.reset_cache()
    kwargs: dict[str, Any] = {"model": "x"}
    provider_caps.apply_temperature(kwargs, "anthropic/claude-opus-4-7", 0.0)
    assert "temperature" not in kwargs

    kwargs2: dict[str, Any] = {"model": "x"}
    provider_caps.apply_temperature(kwargs2, "anthropic/claude-sonnet-4-6", 0.0)
    assert kwargs2["temperature"] == 0.0
