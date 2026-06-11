"""ContextBundle, brand scrubbing, trim budgets.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from consult import artifacts
from consult.types import ModelSpec, Status


def test_context_scrub_brands_masks_known_providers():
    """Brand-name scrubber masks model + provider names so blinded mode
    doesn't leak identity through the prompt itself.
    """
    from consult import context as ctx

    src = "Compare claude-opus, gpt-pro, and gemini-pro for code review on Anthropic."
    out = ctx.scrub_brands(src)
    for brand in ("claude", "gpt", "gemini", "Anthropic"):
        assert brand.lower() not in out.lower(), (brand, out)
    assert out.count("[MODEL]") >= 4


def test_context_scrub_brands_handles_provider_prefixed_ids():
    """Raw LiteLLM IDs like `x-ai/grok-4.3` survive a simple word-boundary
    regex; the provider-prefix pass must catch them.
    """
    from consult import context as ctx

    src = "I asked openrouter/x-ai/grok-4.3 and meta-llama/llama-4-maverick."
    out = ctx.scrub_brands(src)
    assert "x-ai" not in out.lower()
    assert "grok" not in out.lower()
    assert "meta-llama" not in out.lower()
    assert "llama" not in out.lower()


def test_context_scrub_brands_is_idempotent():
    """Running the scrubber twice produces the same result — the
    replacement token `[MODEL]` doesn't itself match the regex."""
    from consult import context as ctx

    src = "claude vs gpt for coding"
    once = ctx.scrub_brands(src)
    twice = ctx.scrub_brands(once)
    assert once == twice


def test_context_build_keeps_raw_when_not_blinded():
    """When blinded=False, prompt_scrubbed equals prompt — scrubbing
    only fires when downstream stages will actually use it."""
    from consult import context as ctx

    src = "claude vs gpt — which is better?"
    bundle = ctx.build(src, blinded=False)
    assert bundle.prompt == src
    assert bundle.prompt_scrubbed == src
    assert bundle.blinded is False


def test_context_build_scrubs_when_blinded():
    from consult import context as ctx

    src = "claude vs gpt — which is better?"
    bundle = ctx.build(src, blinded=True)
    assert bundle.prompt == src  # raw preserved
    assert "[MODEL]" in bundle.prompt_scrubbed
    assert "claude" not in bundle.prompt_scrubbed.lower()
    assert bundle.blinded is True


def test_context_load_or_none_returns_none_for_legacy_run(tmp_path, monkeypatch):
    """Legacy run dirs (no context.json) load as None so callers can fall
    back to pre-Phase-1 behaviour rather than crashing."""
    from consult import context as ctx

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()  # no context.write
    assert ctx.load_or_none(paths) is None


def test_context_trim_text_head_and_tail():
    """Trimming preserves head + tail with a marker indicating the cut."""
    from consult import context as ctx

    src = "A" * 1000 + "BBBB" + "Z" * 1000  # distinct middle marker
    out = ctx.trim_text(src, max_chars=400)
    assert len(out) < len(src)
    assert "TRIMMED" in out
    # Head from the front; tail from the back
    assert out.startswith("AAAA")
    assert out.endswith("ZZZZ")


def test_context_trim_text_under_budget_is_passthrough():
    from consult import context as ctx

    src = "small"
    assert ctx.trim_text(src, max_chars=100) == src


def test_context_bundle_v1_loads_with_default_kind(tmp_path, monkeypatch):
    """Legacy bundles (schema_version=1) had no `capsule_kind` field. The
    Pydantic default makes them load as `capsule_kind="decision"` without
    raising."""
    from consult import context as ctx

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    (paths.root / "context.json").write_text(
        '{"schema_version": 1, "prompt": "p", "prompt_scrubbed": "p", "blinded": false}'
    )
    loaded = ctx.load_or_none(paths)
    assert loaded is not None
    assert loaded.capsule_kind == "decision"


def test_context_brand_regex_includes_registry_models():
    """The brand regex is derived from `registry.models_config()` so adding
    a model to models.json extends scrub coverage automatically. `sonnet`,
    `codex`, and other tier suffixes are picked up via alias parsing."""
    from consult import context as ctx

    text = "Compare claude-sonnet against gpt-codex for refactoring."
    out = ctx.scrub_brands(text)
    assert "claude" not in out.lower()
    assert "sonnet" not in out.lower()
    assert "gpt" not in out.lower()
    assert "codex" not in out.lower()


def test_runner_writes_context_bundle_at_run_init(tmp_path, monkeypatch):
    """`runner.fanout` writes context.json alongside prompt.txt — every
    fresh run has a bundle downstream stages can load.
    """

    from consult import context as ctx
    from consult.runner import fanout
    from consult.types import ModelSpec

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # dry_run avoids any API call but still triggers run-init.
    handle = asyncio.run(
        fanout(
            "test prompt with claude reference",
            [ModelSpec(model="claude-haiku")],
            dry_run=True,
            blinded=True,
        )
    )
    paths = artifacts.load_run(handle.run_id)
    bundle = ctx.load_or_none(paths)
    assert bundle is not None
    assert bundle.prompt == "test prompt with claude reference"
    assert bundle.blinded is True
    # Blinded mode scrubbed the brand from prompt_scrubbed
    assert "claude" not in bundle.prompt_scrubbed.lower()
    assert "[MODEL]" in bundle.prompt_scrubbed


@pytest.mark.asyncio
async def test_fit_prompt_to_context_no_op_when_under_budget(monkeypatch):
    """No trim, no marker when the prompt already fits."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm,
        "token_counter",
        lambda model, text: len(text) // 4,
    )
    out, dropped = await runner._fit_prompt_to_context(
        "short prompt",
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=100_000,
        max_output_tokens=4000,
    )
    assert out == "short prompt"
    assert dropped == 0
    assert "[TRIMMED" not in out


@pytest.mark.asyncio
async def test_fit_prompt_to_context_trims_when_over_budget(monkeypatch):
    """Over-budget prompt gets head+tail trimmed with a clear marker so
    the call proceeds instead of being rejected by the provider."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm,
        "token_counter",
        lambda model, text: len(text),
    )
    # Budget: 1000 input - 100 output = 900 available. Build a 2000-char
    # prompt that fakes 1 token/char so we're 2× over.
    long_prompt = "A" * 1000 + "B" * 1000
    out, dropped = await runner._fit_prompt_to_context(
        long_prompt,
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=1000,
        max_output_tokens=100,
    )
    # Marker present + final size under the budget after the recount loop.
    assert "[TRIMMED" in out
    assert len(out) < len(long_prompt)
    # `dropped` is the explicit signal callers use to build the manifest
    # trim note — must be positive when a trim happened.
    assert dropped > 0
    assert dropped == len(long_prompt) - len(out)


@pytest.mark.asyncio
async def test_fit_prompt_to_context_skips_when_prior_alone_exceeds_budget(
    monkeypatch,
):
    """If `prior_turns` already exceed the budget, the prompt is returned
    untouched (we don't corrupt role boundaries) and the provider's
    rejection becomes the surfaced error."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm,
        "token_counter",
        lambda model, text: len(text),
    )
    prior = [
        {"role": "user", "content": "X" * 2000},
        {"role": "assistant", "content": "Y" * 2000},
    ]
    out, dropped = await runner._fit_prompt_to_context(
        "follow-up",
        prior_turns=prior,
        litellm_id="x/y",
        max_input_tokens=1000,
        max_output_tokens=100,
    )
    assert out == "follow-up"  # not trimmed
    assert dropped == 0


@pytest.mark.asyncio
async def test_call_one_auto_trims_oversized_prompt(tmp_path, monkeypatch):
    """End-to-end: _call_one with an over-budget prompt trims, the LLM
    call succeeds, and the ManifestEntry carries the trim note in
    `error` even on Status.OK."""
    from consult import runner
    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # max_input_tokens=10000 leaves ~6000 for the prompt after claude-haiku's
    # default_budget_tokens (4000) is reserved for output.
    monkeypatch.setattr(
        runner,
        "_max_input_tokens",
        lambda lid, entry: 10_000,
    )
    monkeypatch.setattr(
        runner.litellm,
        "token_counter",
        lambda model, text: len(text),
    )

    sent_messages: dict[str, Any] = {}

    class FakeMsg:
        content = "trimmed response body"

    class FakeChoice:
        message = FakeMsg()
        finish_reason = "stop"

    class FakeResp:
        choices = [FakeChoice()]
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 50})()

        def model_dump(self):
            return {"choices": [{"message": {"content": "trimmed response body"}}]}

    async def fake_acompletion(**kwargs):
        sent_messages["messages"] = kwargs.get("messages")
        return FakeResp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        runner.litellm,
        "completion_cost",
        lambda **kw: 0.01,
    )

    spec = ModelSpec(model="claude-haiku")
    paths = artifacts.create_run()
    # 50000-char prompt is well over the fake 6000-char available input budget.
    long_prompt = "Z" * 50_000

    entry = await _call_one(spec, "test-slug", long_prompt, paths)

    assert entry.status == Status.OK
    # Trim diagnostic lives on `note` (info), not `error` (failure).
    assert entry.error is None
    assert "auto-trimmed" in (entry.note or "")
    sent_text = sent_messages["messages"][0]["content"]
    # Anthropic wraps content in a list with cache_control; extract the text.
    if isinstance(sent_text, list):
        sent_text = sent_text[0]["text"]
    assert "[TRIMMED" in sent_text
    assert len(sent_text) < 50_000


@pytest.mark.asyncio
async def test_call_one_no_trim_note_when_prompt_mentions_trimmed_literally(
    tmp_path,
    monkeypatch,
):
    """A prompt that contains the literal string `[TRIMMED` in its source
    (e.g. an attached source file from this very codebase) must NOT be
    misreported as auto-trimmed when the prompt actually fits the context.

    Previously runner sniffed the prompt for `[TRIMMED` to detect trim
    events. Source-code reviews that attached `context.py` or `runner.py`
    matched the substring and surfaced a phantom "input auto-trimmed"
    error on every panellist, even though no trim happened.
    """
    from consult import runner
    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        runner,
        "_max_input_tokens",
        lambda lid, entry: 1_000_000,
    )
    monkeypatch.setattr(
        runner.litellm,
        "token_counter",
        lambda model, text: len(text) // 4,
    )

    class _Msg:
        content = "ok"

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 10})()

        def model_dump(self):
            return {"choices": []}

    async def fake_acompletion(**kwargs):
        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda **kw: 0.0)

    # Prompt that mentions the literal marker (as a code-review attachment
    # would). Fits comfortably under the million-token fake budget.
    prompt = (
        "review this code:\n"
        '    marker = f"\\n\\n... [TRIMMED {dropped} chars ...] ...\\n\\n"\n'
        "    if '[TRIMMED' in per_slug_prompt: ...\n"
    )
    spec = ModelSpec(model="claude-haiku")
    paths = artifacts.create_run()
    entry = await _call_one(spec, "x-0", prompt, paths)
    assert entry.status == Status.OK
    assert entry.error is None, f"phantom trim note: {entry.error!r}"
