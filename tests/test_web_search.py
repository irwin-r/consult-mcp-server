"""`supports_web` registry annotations + provider search plumbing (issue #92 PR 1).

Covers the registry helper (including user-overlay merges), the transport's
per-entry parameter dispatch, the Responses-adapter tools passthrough, and
the end-to-end kwarg injection through `_call_one`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from consult import artifacts, registry
from consult.runner.transport import apply_web_search
from consult.types import ModelSpec, Status


def test_packaged_web_capable_models():
    """The packaged annotations: sonar (native), gemini (grounding via
    web_search_options), gpt-pro (Responses web_search tool)."""
    assert registry.web_capable_models() == ["gemini-flash", "gemini-pro", "gpt-pro", "sonar-pro"]


def test_web_capable_models_honours_overlay(monkeypatch):
    monkeypatch.setattr(
        registry,
        "models_config",
        lambda: {
            "models": {
                "my-web": {"litellm_id": "x/y", "supports_web": True},
                "plain": {"litellm_id": "x/z"},
                "off": {"litellm_id": "x/w", "supports_web": False},
            }
        },
    )
    assert registry.web_capable_models() == ["my-web"]


def test_overlay_merge_carries_supports_web(user_config_dir):
    """A user overlay can annotate a packaged model without restating it."""
    (user_config_dir / "models.json").write_text(
        json.dumps({"models": {"claude-haiku": {"supports_web": True}}})
    )
    cfg = registry._load_json("models.json")
    entry = cfg["models"]["claude-haiku"]
    assert entry["supports_web"] is True
    assert entry["litellm_id"].startswith("anthropic/")  # merge, not replace


@pytest.fixture()
def user_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_USER_CONFIG", tmp_path)
    return tmp_path


# ---- apply_web_search dispatch -----------------------------------------------


def test_apply_web_search_is_noop_without_flag():
    kwargs: dict = {}
    apply_web_search(kwargs, {"litellm_id": "anthropic/claude-opus-4-8"})
    assert kwargs == {}


def test_apply_web_search_noop_for_sonar():
    """Sonar searches unconditionally; sending extra params buys nothing."""
    kwargs: dict = {}
    apply_web_search(kwargs, {"litellm_id": "openrouter/perplexity/sonar-pro", "supports_web": True})
    assert kwargs == {}


def test_apply_web_search_uses_tools_for_responses_mode():
    kwargs: dict = {}
    apply_web_search(kwargs, {"litellm_id": "openai/gpt-5.5-pro", "mode": "responses", "supports_web": True})
    assert kwargs == {"tools": [{"type": "web_search"}]}


def test_apply_web_search_uses_options_for_chat_models():
    kwargs: dict = {}
    apply_web_search(kwargs, {"litellm_id": "gemini/gemini-3.1-pro-preview", "supports_web": True})
    assert kwargs == {"web_search_options": {}}


def test_apply_web_search_does_not_clobber_existing_values():
    kwargs: dict = {"web_search_options": {"search_context_size": "high"}}
    apply_web_search(kwargs, {"litellm_id": "gemini/gemini-3.1-pro-preview", "supports_web": True})
    assert kwargs["web_search_options"] == {"search_context_size": "high"}


# ---- Responses adapter passthrough -------------------------------------------


@pytest.mark.asyncio
async def test_aresponses_adapter_forwards_tools(monkeypatch):
    from consult import runner

    captured: dict = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_text="ok",
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            model_dump=lambda: {},
        )

    monkeypatch.setattr(runner.litellm, "aresponses", fake_aresponses)

    await runner._aresponses_as_completion(
        timeout=30,
        model="openai/gpt-5.5-pro",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=2000,
        tools=[{"type": "web_search"}],
    )
    assert captured["tools"] == [{"type": "web_search"}]


# ---- End-to-end kwarg injection through _call_one ----------------------------


@pytest.mark.asyncio
async def test_fanout_web_search_reaches_the_chat_call(tmp_path, monkeypatch):
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        )

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda completion_response: 0.001)

    handle = await fanout("question?", [ModelSpec(model="gemini-pro")], web_search=True)

    assert handle.manifest[0].status is Status.OK
    assert captured["web_search_options"] == {}


@pytest.mark.asyncio
async def test_fanout_web_search_skips_non_web_models(tmp_path, monkeypatch):
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        )

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda completion_response: 0.001)

    handle = await fanout("question?", [ModelSpec(model="claude-haiku")], web_search=True)

    assert handle.manifest[0].status is Status.OK
    assert "web_search_options" not in captured
    assert "tools" not in captured


@pytest.mark.asyncio
async def test_timeout_floor_raises_per_spec_timeouts(tmp_path, monkeypatch):
    """Patience plumbing: the floor must reach the transport call so a deep
    model can run for hours instead of its interactive registry default."""
    import importlib

    from consult import runner
    from consult.runner.fanout import fanout

    # The facade rebinds the name `fanout` to the function, shadowing the
    # submodule as a package attribute — importlib reaches the module itself.
    fanout_mod = importlib.import_module("consult.runner.fanout")

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    captured: dict = {}

    async def fake_retry(*, timeout, **kwargs):
        captured["timeout"] = timeout
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        )

    monkeypatch.setattr(fanout_mod, "_acompletion_with_retry", fake_retry)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda completion_response: 0.001)

    # claude-haiku's registry timeout is 120s; the floor must win.
    handle = await fanout("q", [ModelSpec(model="claude-haiku")], timeout_floor_s=7200.0)

    assert handle.manifest[0].status is Status.OK
    assert captured["timeout"] == 7200.0
