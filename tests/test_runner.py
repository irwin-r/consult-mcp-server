"""Fan-out runner: specs, slugs, retries, fitting, dropout, fanout itself.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from smoke_helpers import HAVE_KEYS

from consult import artifacts, registry
from consult.runner import _build_per_slug_prompt, _make_slug, estimate_cost
from consult.types import ManifestEntry, ModelSpec, RunHandle, Status


def test_expand_specs_rejects_zero_count():
    """`model:0` is almost certainly a typo and must fail loudly rather
    than silently dropping the spec from the panel.
    """
    from consult.runner import expand_specs

    with pytest.raises(ValueError, match="must be ≥1"):
        expand_specs([ModelSpec(model="claude-haiku:0")])


def test_expand_specs_allows_count_at_cap():
    """A single `model:N` at exactly the cap expands fully — the boundary
    is inclusive, so the default-64 cap admits a 64-instance panel.
    """
    from consult.runner import _DEFAULT_MAX_PANEL_SIZE, expand_specs

    expanded = expand_specs([ModelSpec(model=f"claude-haiku:{_DEFAULT_MAX_PANEL_SIZE}")])
    assert len(expanded) == _DEFAULT_MAX_PANEL_SIZE


def test_expand_specs_rejects_count_over_cap():
    """A single `model:N` one past the cap fails loudly, before the
    oversized list is ever built (the real OOM-prevention path).
    """
    from consult.runner import _DEFAULT_MAX_PANEL_SIZE, expand_specs

    over = _DEFAULT_MAX_PANEL_SIZE + 1
    with pytest.raises(ValueError, match="exceeds the .* cap"):
        expand_specs([ModelSpec(model=f"claude-haiku:{over}")])


def test_expand_specs_rejects_multi_spec_inflation(monkeypatch):
    """Several specs each under the per-spec cap can still sum past it.
    The total-size check must fire, not just the per-spec one.
    """
    from consult.runner import expand_specs

    monkeypatch.setenv("CONSULT_MAX_PANEL_SIZE", "4")
    # Two specs of :3 each pass the per-spec check but total 6 > 4.
    specs = [ModelSpec(model="claude-haiku:3"), ModelSpec(model="gpt-pro:3")]
    with pytest.raises(ValueError, match="more than 4 panellists"):
        expand_specs(specs)


def test_expand_specs_rejects_too_many_bare_specs(monkeypatch):
    """A flood of distinct bare specs (no `:N` at all) is still capped —
    this is the MCP `models` array with no maxItems reaching the engine.
    """
    from consult.runner import expand_specs

    monkeypatch.setenv("CONSULT_MAX_PANEL_SIZE", "4")
    specs = [ModelSpec(model=f"m{i}") for i in range(5)]  # 5 bare specs
    with pytest.raises(ValueError, match="more than 4 panellists"):
        expand_specs(specs)


def test_expand_specs_allows_aggregate_at_cap(monkeypatch):
    """Specs summing to exactly the cap are admitted — the aggregate check
    rejects only strictly over the cap, matching the single-spec boundary.
    """
    from consult.runner import expand_specs

    monkeypatch.setenv("CONSULT_MAX_PANEL_SIZE", "4")
    out = expand_specs([ModelSpec(model="claude-haiku:2"), ModelSpec(model="gpt-pro:2")])
    assert len(out) == 4


def test_expand_specs_zero_padded_count_is_magnitude_not_length():
    """A zero-padded but legal count keeps expanding; the digit guard reads
    magnitude, not raw string length, so ':0000005' is five, not too long.
    """
    from consult.runner import expand_specs

    assert len(expand_specs([ModelSpec(model="claude-haiku:0000005")])) == 5
    # All-zeros is still count 0, reported as the ≥1 typo, not "too large".
    with pytest.raises(ValueError, match="must be ≥1"):
        expand_specs([ModelSpec(model="claude-haiku:00000000000")])


def test_expand_specs_rejects_implausible_digit_count():
    """A pathological 20-digit count is rejected by the digit guard before
    int() touches it (CVE-2020-10735 quadratic str->int defence).
    """
    from consult.runner import expand_specs

    with pytest.raises(ValueError, match="implausibly large"):
        expand_specs([ModelSpec(model="claude-haiku:99999999999999999999")])


def test_expand_specs_env_override(monkeypatch):
    """CONSULT_MAX_PANEL_SIZE raises (or lowers) the cap at call time."""
    from consult.runner import expand_specs

    monkeypatch.setenv("CONSULT_MAX_PANEL_SIZE", "2")
    assert len(expand_specs([ModelSpec(model="claude-haiku:2")])) == 2
    with pytest.raises(ValueError, match="exceeds the 2-panellist cap"):
        expand_specs([ModelSpec(model="claude-haiku:3")])


def test_expand_specs_invalid_env_falls_back(monkeypatch):
    """An unparseable or non-positive override is ignored (with a warning)
    rather than silently disabling the guard."""
    from consult.runner import _DEFAULT_MAX_PANEL_SIZE, _max_panel_size

    for bad in ("abc", "0", "-1", ""):
        monkeypatch.setenv("CONSULT_MAX_PANEL_SIZE", bad)
        assert _max_panel_size() == _DEFAULT_MAX_PANEL_SIZE


def test_slug_and_prompt_assembly():
    spec = ModelSpec(model="claude-haiku", stance="security")
    assert _make_slug(spec, 0, blinded=False).startswith("claude-haiku")
    assert _make_slug(spec, 2, blinded=True) == "panelist-gamma"


def test_footer_injected_on_every_prompt():
    """Capsule confidence extraction depends on the footer being present."""
    with_stance = _build_per_slug_prompt("How to ship X?", "You are an SRE.")
    assert with_stance.startswith("You are an SRE.")
    assert "How to ship X?" in with_stance
    assert "CONFIDENCE:" in with_stance
    assert "KEY_REASON:" in with_stance

    no_stance = _build_per_slug_prompt("How to ship X?", "")
    assert no_stance.startswith("How to ship X?")
    assert "CONFIDENCE:" in no_stance
    assert "KEY_REASON:" in no_stance


@pytest.mark.asyncio
async def test_fanout_dry_run_returns_partial():
    """Dry run must never make a billable call and must explain itself."""
    from consult.runner import fanout

    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="claude-sonnet")]
    handle = await fanout("any prompt", specs, dry_run=True)
    assert handle.partial is True
    assert handle.partial_reason and "dry_run" in handle.partial_reason
    assert handle.manifest == []
    assert handle.cost_usd == 0.0


@pytest.mark.asyncio
async def test_fanout_slow_tail_dropout_cancels_stragglers(tmp_path, monkeypatch):
    """Once `N - k` panellists return, slow stragglers get cancelled and
    surface as Status.TIMEOUT with a "slow-tail dropout" error. The full
    panel is returned (no panellist silently missing) so cost math and the
    progress total still see a complete N.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.05")
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.25")  # k=1 for n=4 → trigger=3

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        if "slow" in slug:
            await _asyncio.sleep(5.0)
        paths.response_text(slug).write_text("ok")
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

    specs = [
        ModelSpec(model="claude-haiku", slug="fast-0"),
        ModelSpec(model="claude-haiku", slug="fast-1"),
        ModelSpec(model="claude-haiku", slug="fast-2"),
        ModelSpec(model="claude-haiku", slug="slow-3"),
    ]
    handle = await fanout("anything", specs)
    assert len(handle.manifest) == 4
    by_status = [m.status for m in handle.manifest]
    assert by_status.count(Status.OK) == 3
    assert by_status.count(Status.TIMEOUT) == 1
    dropped = next(m for m in handle.manifest if m.status is Status.TIMEOUT)
    assert "slow-tail dropout" in (dropped.error or "")


@pytest.mark.asyncio
async def test_fanout_no_dropout_below_threshold_panel_size(tmp_path, monkeypatch):
    """Panels smaller than 4 panellists never trigger slow-tail dropout —
    there's no statistically useful "rest of the panel" signal at N<4.
    A 3-spec panel with a slow panellist still completes all three.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.05")
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.5")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        if "slow" in slug:
            await _asyncio.sleep(0.3)
        paths.response_text(slug).write_text("ok")
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

    specs = [
        ModelSpec(model="claude-haiku", slug="fast-0"),
        ModelSpec(model="claude-haiku", slug="fast-1"),
        ModelSpec(model="claude-haiku", slug="slow-2"),
    ]
    handle = await fanout("anything", specs)
    assert len(handle.manifest) == 3
    assert all(m.status is Status.OK for m in handle.manifest)


@pytest.mark.asyncio
async def test_acompletion_with_retry_recovers_after_rate_limit(monkeypatch):
    """One rate-limit followed by a success: the retry loop sleeps with
    jittered backoff and returns the second response. Without retry, a
    single 429 from a shared OpenAI key wastes the panel's full call cost.
    """
    from consult import runner

    class _StubRateLimit(BaseException):
        pass

    monkeypatch.setattr(runner, "_rate_limit_class", lambda: _StubRateLimit)
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _StubRateLimit("rate-limited")

        class _Resp:
            class _Choice:
                class _Msg:
                    content = "ok"

                message = _Msg()
                finish_reason = "stop"

            choices = [_Choice()]

        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    resp = await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 2
    assert resp.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_acompletion_with_retry_exhausts_then_raises(monkeypatch):
    """Persistent rate-limit across all attempts must re-raise the last
    RateLimitError, not swallow it — the caller's `Status.RATE_LIMITED`
    classification depends on the exception propagating out.
    """
    from consult import runner

    class _StubRateLimit(BaseException):
        pass

    monkeypatch.setattr(runner, "_rate_limit_class", lambda: _StubRateLimit)
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    async def always_rate_limit(**kwargs):
        raise _StubRateLimit("nope")

    monkeypatch.setattr(runner.litellm, "acompletion", always_rate_limit)
    with pytest.raises(_StubRateLimit):
        await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])


@pytest.mark.asyncio
async def test_acompletion_with_retry_does_not_retry_non_rate_limit(monkeypatch):
    """Auth/content-filter/bad-request errors must NOT trigger retry —
    they aren't transient and retrying just burns spend.
    """
    from consult import runner

    class _StubRateLimit(BaseException):
        pass

    monkeypatch.setattr(runner, "_rate_limit_class", lambda: _StubRateLimit)
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    attempts = 0

    async def raise_auth_error(**kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("invalid api key")

    monkeypatch.setattr(runner.litellm, "acompletion", raise_auth_error)
    with pytest.raises(RuntimeError, match="invalid api key"):
        await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 1


@pytest.mark.asyncio
async def test_acompletion_with_retry_recovers_from_bare_api_error(monkeypatch):
    """Bare `litellm.APIError` (no subclass) is the OpenRouter "Unable to
    get json response" failure mode — upstream returned all-whitespace.
    Retry must recover from it; previously it surfaced as a one-shot ERROR.
    """
    from consult import runner

    bare_api = runner._bare_api_error_class()
    assert bare_api is not None, "litellm.exceptions.APIError must resolve"
    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # Construct a bare APIError the way litellm raises it.
            raise bare_api(
                status_code=500,
                message="Unable to get json response",
                llm_provider="openrouter",
                model=kwargs.get("model", "m"),
            )

        class _Resp:
            class _Choice:
                class _Msg:
                    content = "ok"

                message = _Msg()
                finish_reason = "stop"

            choices = [_Choice()]

        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    resp = await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 2
    assert resp.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_acompletion_with_retry_does_not_retry_api_error_subclass(monkeypatch):
    """A subclass of APIError (e.g. AuthenticationError) must NOT trigger
    retry — only bare APIError is treated as transient. Subclasses are
    terminal failures we shouldn't burn spend on.
    """
    from consult import runner

    monkeypatch.setenv("CONSULT_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CONSULT_RETRY_BASE_DELAY", "0.01")

    from litellm import exceptions as lex

    auth_cls = getattr(lex, "AuthenticationError", None)
    if auth_cls is None:
        import pytest as _pytest

        _pytest.skip("litellm.AuthenticationError not available")

    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        raise auth_cls(
            message="bad key",
            llm_provider="openai",
            model=kwargs.get("model", "m"),
        )

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    with pytest.raises(auth_cls):
        await runner._acompletion_with_retry(timeout=10.0, model="m", messages=[])
    assert attempts == 1, "AuthenticationError must be terminal, not retried"


@pytest.mark.asyncio
async def test_fanout_caps_per_provider_concurrency(tmp_path, monkeypatch):
    """With CONSULT_PROVIDER_CONCURRENCY=anthropic:1, only one Anthropic
    panellist may be in-flight at a time even if the panel has 5 of them.
    Locks the FRICTION-driven rate-limit mitigation: without the cap, every
    panellist hit the provider in lockstep.
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_PROVIDER_CONCURRENCY", "anthropic:1")
    # Drop the registry cache so the env var takes effect on this call.
    registry.models_config.cache_clear()

    inflight = 0
    peak = 0
    real_acompletion = runner.litellm.acompletion

    async def fake_acompletion(**kwargs):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await _asyncio.sleep(0.05)
        finally:
            inflight -= 1

        # Build a minimal LiteLLM-like response object
        class _Msg:
            content = "ok\n\nCONFIDENCE: 0.7\nKEY_REASON: x"
            tool_calls = None

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]
            usage = None

            def model_dump(self):
                return {"_stub": True}

        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda **_: 0.0)

    specs = [ModelSpec(model="claude-haiku") for _ in range(5)]
    handle = await fanout("anything", specs)
    assert handle.partial is False
    assert len(handle.manifest) == 5
    assert peak == 1, f"semaphore cap=1 violated: peak in-flight = {peak}"

    # Restore registry cache so other tests aren't affected
    registry.models_config.cache_clear()
    monkeypatch.setattr(runner.litellm, "acompletion", real_acompletion)


@pytest.mark.asyncio
async def test_call_one_unknown_alias_returns_error_entry(tmp_path, monkeypatch):
    """An unknown alias must surface as a per-spec Status.ERROR rather than
    crashing the panel. Regression guard: KeyError out of `resolve_model`
    previously propagated through `asyncio.gather` and aborted every sibling.
    """
    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    spec = ModelSpec(model="definitely-not-a-real-alias")
    entry = await _call_one(spec, "bogus-0", "prompt", paths)
    assert entry.status is Status.ERROR
    assert entry.error and "definitely-not-a-real-alias" in entry.error
    assert entry.model_id is None
    assert entry.cost_known is True  # no call was billable


def test_estimate_cost_skips_unknown_alias_without_raising():
    """Unknown aliases mark cost_known=False but must not raise — `fanout`
    relies on this so a typo doesn't abort the run before any panel work.
    """
    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="bogus-xyz")]
    total, all_known = estimate_cost(specs, "hello")
    assert total >= 0.0
    assert all_known is False


@pytest.mark.asyncio
async def test_fanout_cost_cap_returns_partial(monkeypatch):
    """Setting max_run_usd to 0 must abort before any model call."""
    from consult import runner
    from consult.runner import fanout

    # Force a non-zero estimate so the cap path is exercised
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.99, True))
    specs = [ModelSpec(model="claude-haiku")]
    handle = await fanout("p", specs, max_run_usd=0.01)
    assert handle.partial is True
    assert handle.partial_reason and "exceeds cap" in handle.partial_reason
    assert "known-priced" not in handle.partial_reason  # all_known=True path
    assert handle.manifest == []


@pytest.mark.asyncio
async def test_fanout_cost_cap_message_discloses_partial_pricing(monkeypatch):
    """When estimate_cost returns all_known=False, the cap message must say
    so — otherwise the displayed estimate (only the known-priced portion)
    looks misleadingly low. Mirrors the dry_run branch.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.50, False))
    specs = [ModelSpec(model="claude-haiku")]
    handle = await fanout("p", specs, max_run_usd=0.01)
    assert handle.partial is True
    assert handle.partial_reason
    assert "known-priced portion only" in handle.partial_reason
    assert "exceeds cap" in handle.partial_reason
    assert handle.cost_known is False


def test_litellm_logger_does_not_propagate_to_root():
    """LiteLLM's logger must not propagate, otherwise callers that enable
    a root handler (basicConfig at INFO etc.) see every line twice — once
    via LiteLLM's own coloured handler, once via root. The disable lives
    in consult/runner.py at module level so import is enough to set it.
    """
    import logging

    # Importing the package runs runner.py at module load (via consult.server
    # → consult.runner), which disables propagation. Confirm the effect.
    import consult.runner  # noqa: F401

    assert logging.getLogger("LiteLLM").propagate is False


@pytest.mark.asyncio
async def test_stream_acompletion_raises_on_builder_failure(monkeypatch):
    """SECURITY/CORRECTNESS: when `stream_chunk_builder` fails, the
    streaming variant must raise rather than return a malformed partial
    chunk that downstream `classify()` and `completion_cost()` would
    mishandle."""
    import litellm

    from consult.runner import _stream_acompletion

    class _FakeAsyncStream:
        def __init__(self, chunks):
            self.chunks = chunks
            self._i = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._i >= len(self.chunks):
                raise StopAsyncIteration
            c = self.chunks[self._i]
            self._i += 1
            return c

    async def fake_acompletion(**kwargs):
        return _FakeAsyncStream([{"raw": "chunk1"}, {"raw": "chunk2"}])

    def fake_builder(chunks, messages=None):
        raise ValueError("builder unsupported chunk shape")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "stream_chunk_builder", fake_builder)

    with pytest.raises(RuntimeError, match="stream_chunk_builder failed"):
        await _stream_acompletion(
            timeout=10.0,
            on_partial=None,
            start=0.0,
            model="x/y",
            messages=[],
            max_tokens=100,
        )


@pytest.mark.asyncio
async def test_fanout_stream_env_var_enables_streaming(tmp_path, monkeypatch):
    """The CONSULT_STREAM env var flips fanout's `stream` default on so
    callers can opt into streaming without changing their tool call."""
    from consult.runner import fanout
    from consult.types import ManifestEntry, ModelSpec, Status

    captured: dict[str, bool] = {}

    async def fake_call(
        spec,
        slug,
        per_prompt,
        paths,
        provider_sems=None,
        *,
        stream=False,
        on_partial=None,
        prior_turns=None,
        **_,
    ):
        captured["stream"] = stream
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug,
            status=Status.OK,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_known=True,
        )

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr("consult.runner._call_one", fake_call)
    monkeypatch.setenv("CONSULT_STREAM", "1")

    await fanout("hi", [ModelSpec(model="claude-haiku")])
    assert captured.get("stream") is True


@pytest.mark.asyncio
async def test_fanout_stream_default_off(tmp_path, monkeypatch):
    """Without CONSULT_STREAM or explicit stream=True, streaming stays off."""
    from consult.runner import fanout
    from consult.types import ManifestEntry, ModelSpec, Status

    captured: dict[str, bool] = {}

    async def fake_call(
        spec,
        slug,
        per_prompt,
        paths,
        provider_sems=None,
        *,
        stream=False,
        on_partial=None,
        prior_turns=None,
        **_,
    ):
        captured["stream"] = stream
        paths.response_text(slug).write_text("body")
        return ManifestEntry(
            slug=slug,
            status=Status.OK,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_known=True,
        )

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr("consult.runner._call_one", fake_call)
    monkeypatch.delenv("CONSULT_STREAM", raising=False)

    await fanout("hi", [ModelSpec(model="claude-haiku")])
    assert captured.get("stream") is False


def test_build_messages_prepends_prior_turns_for_anthropic():
    """prior_turns must be prepended verbatim; the final user turn is
    Anthropic-cache-tagged. This is what gives continuation runs proper
    role boundaries instead of one giant user blob."""
    from consult.runner import build_messages

    prior_turns = [
        {"role": "user", "content": "What database?"},
        {"role": "assistant", "content": "DuckDB."},
    ]
    msgs = build_messages("Now for ETL?", "anthropic", prior_turns)
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "What database?"}
    assert msgs[1] == {"role": "assistant", "content": "DuckDB."}
    # Final user turn — Anthropic uses the structured content list with cache_control.
    assert msgs[2]["role"] == "user"
    assert isinstance(msgs[2]["content"], list)
    assert msgs[2]["content"][0]["type"] == "text"
    assert msgs[2]["content"][0]["text"] == "Now for ETL?"
    assert msgs[2]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_build_messages_prepends_prior_turns_for_non_anthropic():
    """Same shape, OpenAI-style flat string content on the final user turn."""
    from consult.runner import build_messages

    prior_turns = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    msgs = build_messages("Q2", "openai", prior_turns)
    assert msgs == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]


def test_build_messages_no_prior_turns_is_unchanged():
    """The default (no prior_turns) path must not regress the existing single-turn shape."""
    from consult.runner import build_messages

    msgs = build_messages("hi", "openai")
    assert msgs == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_fanout_threads_prior_turns_into_call_one(tmp_path, monkeypatch):
    """fanout(prior_turns=...) must reach `_call_one`. Critical for refine's
    continuation flow: without this, the prior consultation context is
    silently dropped from the panellist's messages."""
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    seen: list = []

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **kwargs):
        seen.append(kwargs.get("prior_turns"))
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    prior_turns = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    await fanout(
        "follow-up",
        [ModelSpec(model="claude-haiku")],
        prior_turns=prior_turns,
    )
    assert seen == [prior_turns]


@pytest.mark.asyncio
async def test_fanout_returns_partial_when_zero_usable_panellists(tmp_path, monkeypatch):
    """Every panellist times out → fanout must return partial=True with a
    clear reason. Without this, refine.py would hand an empty manifest to
    the arbiter and burn its cost on a verdict with no signal — the bug
    that surfaced during the dogfood pass producing this PR.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        return ManifestEntry(
            slug=slug,
            model_id=None,
            persona=None,
            status=Status.TIMEOUT,
            finish_reason=None,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=100,
            cost_usd=None,
            cost_known=False,
            error="timeout after 1s",
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    handle = await fanout(
        "p",
        [ModelSpec(model="claude-haiku"), ModelSpec(model="claude-sonnet")],
    )
    assert handle.partial is True
    assert handle.partial_reason and "zero usable panellists" in handle.partial_reason
    # Manifest still surfaces the failure entries for diagnosis
    assert len(handle.manifest) == 2
    assert all(m.status == Status.TIMEOUT for m in handle.manifest)


@pytest.mark.asyncio
async def test_fanout_succeeds_when_any_panellist_usable(tmp_path, monkeypatch):
    """One OK + one TIMEOUT must NOT trip the zero-usable guard."""
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    call_count = {"n": 0}

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        call_count["n"] += 1
        status = Status.OK if call_count["n"] == 1 else Status.TIMEOUT
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=status,
            finish_reason="stop" if status == Status.OK else None,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            cost_usd=0.0,
            cost_known=True,
            error=None if status == Status.OK else "timeout",
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    handle = await fanout(
        "p",
        [ModelSpec(model="claude-haiku"), ModelSpec(model="claude-sonnet")],
    )
    assert handle.partial is False
    assert handle.partial_reason is None


@pytest.mark.asyncio
async def test_slow_tail_dropout_marks_cost_unknown_for_cancelled(tmp_path, monkeypatch):
    """A cancelled-mid-flight task may still be billed by the provider, so
    cost_known must be False (not True). Fixes the silent under-counting of
    runs where a flagship was dropped after sending the request."""
    import asyncio as aio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.1")  # very short dropout
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.5")  # drop the slow half

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        if "haiku" in spec.model:
            await aio.sleep(0.01)
            return ManifestEntry(
                slug=slug,
                model_id="x/y",
                persona=None,
                status=Status.OK,
                finish_reason="stop",
                resource_uri=paths.resource_uri(slug),
                body_path=str(paths.response_text(slug)),
                latency_ms=10,
                cost_usd=0.0,
                cost_known=True,
            )
        # Slow panellists — will be cancelled by slow-tail dropout
        await aio.sleep(60)
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            persona=None,
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=60000,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    specs = [
        ModelSpec(model="claude-haiku"),
        ModelSpec(model="claude-haiku"),
        ModelSpec(model="claude-opus"),
        ModelSpec(model="claude-sonnet"),
    ]
    handle = await fanout("p", specs)
    # Find dropped entries and verify cost_known=False
    dropped = [m for m in handle.manifest if m.error and "dropout" in (m.error or "")]
    assert len(dropped) >= 1
    for m in dropped:
        assert m.cost_known is False, (
            f"dropped panellist {m.slug} has cost_known={m.cost_known}; "
            "should be False since provider may still bill"
        )
        assert m.cost_usd is None


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
def test_estimate_cost_smoke():
    specs = [ModelSpec(model="claude-haiku")]
    est, _ = estimate_cost(specs, "say hello in five words")
    assert est >= 0


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
@pytest.mark.asyncio
async def test_tiny_panel_dry_run():
    from consult.runner import fanout

    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="openrouter/x-ai/grok-4.3")]
    handle = await fanout("ping", specs, dry_run=True)
    assert handle.partial
    assert "dry_run" in (handle.partial_reason or "")


def test_modelspec_slug_validator_rejects_path_traversal():
    """A caller-supplied slug containing `/` or `..` must fail at construction —
    not deeper in `_call_one` where the artifact write would silently leave
    the responses dir.
    """
    for bad in ("../etc/passwd", "a/b", "..", "with space", "with$shell"):
        with pytest.raises(Exception) as exc:
            ModelSpec(model="x", slug=bad)
        assert "slug" in str(exc.value)


def test_expand_specs_with_explicit_slug_and_count_disambiguates():
    """`model:N` with an explicit slug must append an index suffix to each
    expansion. Without this, three sibling panellists race to write to the
    same `responses/<slug>.txt` and two responses are silently lost.
    """
    from consult.runner import expand_specs

    raw = [ModelSpec(model="claude-haiku:3", slug="bench")]
    expanded = expand_specs(raw)
    slugs = [s.slug for s in expanded]
    assert slugs == ["bench-0", "bench-1", "bench-2"]
    # Single-instance with explicit slug is left alone (no suffix needed).
    single = expand_specs([ModelSpec(model="claude-haiku:1", slug="solo")])
    assert [s.slug for s in single] == ["solo"]


def test_make_slug_sanitises_colons_in_raw_litellm_ids():
    """OpenRouter model IDs like `...:free` must not crash the slug-derive
    path. Pre-fix the colon flowed into the slug and the safe-id validator
    in artifacts.response_text rejected the path build.
    """
    from consult.runner import _make_slug

    spec = ModelSpec(model="openrouter/meta-llama/llama-3.1-8b:free")
    slug = _make_slug(spec, 0, blinded=False)
    # No colon, valid leading alphanumeric, slug regex accepts it.
    assert ":" not in slug
    assert slug == "llama-3.1-8b-free"


async def test_fanout_rejects_duplicate_explicit_slugs(tmp_path, monkeypatch):
    """Two specs with identical explicit slugs race on `responses/<slug>.txt`.
    `runner.fanout` must fail fast with a clear ValueError before any
    artifact write happens.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (0.0, True))

    specs = [
        ModelSpec(model="claude-haiku", slug="test"),
        ModelSpec(model="gpt-pro", slug="test"),
    ]
    with pytest.raises(ValueError) as exc:
        await runner_mod.fanout("hi", specs)
    assert "duplicate" in str(exc.value).lower()
    assert "test" in str(exc.value)


def test_blinded_fanout_preserves_model_id_in_manifest(tmp_path, monkeypatch):
    """Blinded panels must keep `model_id` populated on each manifest entry.

    Blinding's job is to (a) scrub brand mentions from the prompt panellists
    see, (b) hand out greek-letter slugs so panellists referring to each
    other use anonymous labels, and (c) tell the synth to omit model_ids
    from its prompt. The manifest itself is for downstream readers (the
    viewer, ledger, the human) — stripping model_id there hid identities
    from the final report, which the user needs to see who said what.
    """

    from consult.runner import fanout
    from consult.types import ModelSpec

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "consult.runner.estimate_cost",
        lambda *a, **kw: (0.0, True),
    )

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        paths.response_text(slug).write_text("body")
        entry = registry.resolve_model(spec.model)
        return ManifestEntry(
            slug=slug,
            model_id=entry["litellm_id"],
            status=Status.OK,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=10,
            tokens_in=1,
            tokens_out=1,
            cost_usd=0.0,
            cost_known=True,
            confidence=0.5,
            capsule=None,
        )

    monkeypatch.setattr("consult.runner._call_one", fake_call)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0")

    handle = asyncio.run(
        fanout(
            "prompt",
            [ModelSpec(model="claude-haiku")],
            blinded=True,
        )
    )
    assert len(handle.manifest) == 1
    entry = handle.manifest[0]
    # Slug is anonymised (greek letter), but model_id stays real.
    assert entry.slug.startswith("panelist-")
    assert entry.model_id is not None
    assert entry.model_id == "anthropic/claude-haiku-4-5-20251001"
    # On-disk manifest must mirror the in-memory handle.
    paths = artifacts.load_run(handle.run_id)
    on_disk = json.loads(paths.manifest_json.read_text())
    assert on_disk["manifest"][0]["model_id"] == "anthropic/claude-haiku-4-5-20251001"


async def test_fanout_cap_early_return_preserves_existing_manifest(tmp_path, monkeypatch):
    """When refine drives multiple rounds through the same `paths`, a
    cap-exceeded early return on round N+1 must NOT clobber round N's
    successful manifest.json. Pre-fix iter5's "always write" landed this
    regression: refine's prior-round transcript was wiped on a borderline
    cap miss in the next round.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    # Seed a fake "round 1" manifest with a real entry.
    seeded = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=[
            ManifestEntry(
                slug="x.r1",
                model_id="m/x",
                status=Status.OK,
                resource_uri=paths.resource_uri("x.r1"),
                body_path=str(paths.response_text("x.r1")),
                latency_ms=10,
                cost_usd=0.05,
                cost_known=True,
                confidence=None,
                capsule=None,
            ),
        ],
        cost_usd=0.05,
        cost_known=True,
        wall_ms=10,
    )
    artifacts.write_manifest(paths, seeded.model_dump())

    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (10.0, True))
    handle = await runner_mod.fanout(
        "x",
        [ModelSpec(model="claude-haiku")],
        max_run_usd=1.0,
        existing_paths=paths,
    )
    assert handle.partial is True
    # Manifest.json on disk must still reflect the seeded round-1 entry,
    # not the empty cap-rejection handle.
    import json as _json

    persisted = _json.loads(paths.manifest_json.read_text())
    assert persisted["manifest"] and persisted["manifest"][0]["slug"] == "x.r1"


async def test_fanout_writes_manifest_on_dry_run(tmp_path, monkeypatch):
    """A dry_run still creates a run dir; downstream tools (consult-view,
    synthesise) expect `manifest.json` to be present.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    handle = await runner_mod.fanout(
        "x",
        [ModelSpec(model="claude-haiku")],
        dry_run=True,
    )
    paths = artifacts.load_run(handle.run_id)
    assert paths.manifest_json.exists()


async def test_fanout_writes_manifest_on_cap_exceeded(tmp_path, monkeypatch):
    """A cap-exceeded early return must also persist the manifest so the
    run dir is not corrupted."""
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner_mod, "estimate_cost", lambda *a, **kw: (10.0, True))
    handle = await runner_mod.fanout(
        "x",
        [ModelSpec(model="claude-haiku")],
        max_run_usd=1.0,
    )
    assert handle.partial is True
    assert "exceeds cap" in (handle.partial_reason or "")
    paths = artifacts.load_run(handle.run_id)
    assert paths.manifest_json.exists()


async def test_fanout_rejects_empty_specs(tmp_path, monkeypatch):
    """Library callers bypassing the MCP minItems=1 schema must still get a
    clear error, not a bizarre empty run dir.
    """
    from consult import runner as runner_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    with pytest.raises(ValueError) as exc:
        await runner_mod.fanout("x", [])
    assert "at least one" in str(exc.value)


def test_provider_concurrency_floors_at_one(monkeypatch):
    """A typo like `CONSULT_PROVIDER_CONCURRENCY=openai:0` must not produce
    a Semaphore(0); that blocks the first acquire indefinitely and bypasses
    the per-call timeout. Floor to 1.
    """
    monkeypatch.setenv("CONSULT_PROVIDER_CONCURRENCY", "openai:0,zzz:-3")
    out = registry.provider_concurrency()
    assert out["openai"] >= 1
    assert out["zzz"] >= 1


def test_max_input_tokens_falls_back_to_litellm(monkeypatch):
    """No registry override → ask LiteLLM."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm,
        "get_model_info",
        lambda model: {"max_input_tokens": 128_000},
    )
    assert runner._max_input_tokens("openai/gpt-x", {}) == 128_000


def test_max_input_tokens_unknown_returns_none(monkeypatch):
    """Unknown model + no override → None, so pre-flight is skipped."""
    from consult import runner

    def boom(model):
        raise Exception("unknown model")

    monkeypatch.setattr(runner.litellm, "get_model_info", boom)
    assert runner._max_input_tokens("vendor/totally-new-model", {}) is None


def test_slow_tail_dropout_default_is_180s():
    """Default bumped from 30s to 180s — long-context wide-panel runs were
    losing real signal (kimi/qwen often take 60-180s on ~200K input)."""
    # The default lives in runner.fanout's body; verify by reading the
    # source rather than executing the path (which would need a full fanout).
    import inspect

    from consult import runner

    src = inspect.getsource(runner.fanout)
    assert 'env_float("CONSULT_TAIL_DROPOUT_S", 180.0)' in src


@pytest.mark.asyncio
async def test_fanout_warns_when_cap_set_but_pricing_unknown(tmp_path, monkeypatch, caplog):
    """A cap with unknown panellist pricing can't be enforced (the estimate
    covers only known-priced models). fanout proceeds but warns, so the silent
    bypass is at least visible in the logs.
    """
    import logging

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    # Low estimate, NOT all known → under any cap, but unenforceable.
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, False))

    async def fake_acompletion(**kwargs):
        class _Msg:
            content = "ok\n\nCONFIDENCE: 0.7"
            tool_calls = None

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]
            usage = None

            def model_dump(self):
                return {"_stub": True}

        return _Resp()

    monkeypatch.setattr(runner.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(runner.litellm, "completion_cost", lambda **_: 0.0)

    specs = [ModelSpec(model="claude-haiku")]
    with caplog.at_level(logging.WARNING, logger="consult.runner"):
        handle = await fanout("p", specs, max_run_usd=5.0)

    assert handle.partial is False
    assert len(handle.manifest) == 1
    assert "cap cannot be fully enforced" in caplog.text


@pytest.mark.asyncio
async def test_fanout_dropout_preserves_manifest_order(tmp_path, monkeypatch):
    """The assembled manifest stays in input-spec order even when a straggler
    in a middle position is dropped. Characterization guard for the fanout
    tail-dropout assembly (pins ordering before the gatherer extraction).
    """
    import asyncio as _asyncio

    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.05")
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.25")  # k=1, trigger=3 for n=4

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        if "slow" in slug:
            await _asyncio.sleep(5.0)
        paths.response_text(slug).write_text("ok")
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

    specs = [
        ModelSpec(model="claude-haiku", slug="a-0"),
        ModelSpec(model="claude-haiku", slug="slow-1"),  # middle straggler, dropped
        ModelSpec(model="claude-haiku", slug="c-2"),
        ModelSpec(model="claude-haiku", slug="d-3"),
    ]
    handle = await fanout("anything", specs)

    assert [m.slug for m in handle.manifest] == ["a-0", "slow-1", "c-2", "d-3"]
    dropped = handle.manifest[1]
    assert dropped.status is Status.TIMEOUT
    assert dropped.cost_known is False


async def test_dropout_cancel_recovers_completed_entry_with_on_progress(tmp_path, monkeypatch):
    """(issue #43) The race-recovery branch in _gather_with_tail_dropout:
    a panellist can finish _call_one and record its entry, then get
    cancelled while blocked in its post-completion progress await (only
    reachable with on_progress set). The gatherer must keep the real OK
    entry instead of writing a synthetic TIMEOUT, and re-emit the
    completion event the cancel ate."""
    import asyncio as _asyncio

    from consult import runner
    from consult.progress import PanellistCompleted, ProgressEvent

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")
    monkeypatch.setenv("CONSULT_TAIL_DROPOUT_S", "0.2")
    monkeypatch.setenv("CONSULT_TAIL_K_FRAC", "0.25")
    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (0.0, True))

    async def fake_call(spec, slug, per_prompt, paths, provider_sems=None, **_):
        return ManifestEntry(
            slug=slug,
            model_id="x/y",
            status=Status.OK,
            finish_reason="stop",
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=1,
            cost_usd=0.0,
            cost_known=True,
        )

    monkeypatch.setattr(runner, "_call_one", fake_call)

    hang_once = {"armed": True}
    events: list[ProgressEvent] = []

    async def on_progress(event):
        events.append(event)
        # Hang exactly once, inside m-3's post-completion notify: its entry
        # is already in completed_entries, so the dropout cancel lands in
        # this await and the recovery branch must keep the real result.
        if isinstance(event, PanellistCompleted) and event.slug == "m-3" and hang_once["armed"]:
            hang_once["armed"] = False
            await _asyncio.sleep(60)  # cancelled by the dropout path

    specs = [ModelSpec(model="claude-haiku", slug=f"m-{i}") for i in range(4)]
    handle = await runner.fanout("q", specs, on_progress=on_progress)

    entry = next(m for m in handle.manifest if m.slug == "m-3")
    assert entry.status == Status.OK  # the real result, not a synthetic TIMEOUT
    assert entry.error is None
    assert handle.partial is False
    completions = [e for e in events if isinstance(e, PanellistCompleted) and e.slug == "m-3"]
    assert len(completions) == 2  # the eaten emit plus the recovery re-emit
    assert completions[-1].status == "OK"


# --- issue #55: reasoning-aware per-model output budgets ---------------------
#
# The per-kind cap alone bound below what reasoning models burn before any
# text lands (gpt-pro filled 2000 AND 4000-token budgets with pure reasoning
# on the 2026-06-11 probes). The granted budget is now max(kind cap,
# model default_budget_tokens), and estimate_cost prices the same ceiling.


@pytest.mark.asyncio
async def test_call_one_budget_uses_model_floor_for_reasoning_models(tmp_path, monkeypatch):
    """A model whose default_budget_tokens exceeds the kind cap gets its own
    budget: kimi (12000) on a decision panel must not be capped at 2000."""
    import litellm

    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        class _Resp:
            choices = [
                type(
                    "C",
                    (),
                    {
                        "message": type("M", (), {"content": "fine answer"})(),
                        "finish_reason": "stop",
                    },
                )()
            ]
            usage = None

            def model_dump(self):
                return {}

        return _Resp()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0)

    entry = await _call_one(ModelSpec(model="kimi"), "kimi", "q", paths, capsule_kind="decision")
    assert entry.status is Status.OK
    assert captured["max_completion_tokens"] == 12000  # model floor, not the 2000 kind cap


@pytest.mark.asyncio
async def test_call_one_budget_keeps_kind_floor_for_small_models(tmp_path, monkeypatch):
    """The kind cap still wins when it is the larger side: claude-haiku
    (default_budget_tokens 4000) on a review panel keeps the 16000 review
    floor — the FRICTION #16 fix must not regress."""
    import litellm

    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        class _Resp:
            choices = [
                type(
                    "C",
                    (),
                    {
                        "message": type("M", (), {"content": "LGTM"})(),
                        "finish_reason": "stop",
                    },
                )()
            ]
            usage = None

            def model_dump(self):
                return {}

        return _Resp()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0)

    await _call_one(ModelSpec(model="claude-haiku"), "haiku", "q", paths, capsule_kind="review")
    assert captured["max_completion_tokens"] == 16000


def test_estimate_cost_prices_the_model_floor(monkeypatch):
    """estimate_cost must price the same ceiling _call_one grants — a
    12000-budget model on a decision panel is estimated at 12000 output
    tokens, keeping the max_run_usd gate conservative and consistent."""
    import litellm

    seen: dict = {}

    monkeypatch.setattr(litellm, "token_counter", lambda model, text: 100)

    def fake_cost_per_token(model, prompt_tokens, completion_tokens):
        seen[model] = completion_tokens
        return (0.0001, 0.0002)

    monkeypatch.setattr(litellm, "cost_per_token", fake_cost_per_token)
    total, all_known = estimate_cost([ModelSpec(model="kimi")], "q", capsule_kind="decision")
    assert all_known is True
    assert seen["openrouter/moonshotai/kimi-k2.6"] == 12000


def test_estimate_drivers_sorts_and_skips_unpriced(monkeypatch):
    """estimate_drivers returns known-priced specs, highest estimate first,
    omitting models whose pricing lookup fails."""
    import litellm

    from consult.runner import estimate_drivers

    monkeypatch.setattr(litellm, "token_counter", lambda model, text: 100)

    def fake_cost_per_token(model, prompt_tokens, completion_tokens):
        if "kimi" in model:
            raise ValueError("no price")
        if "haiku" in model:
            return (0.001, 0.002)
        return (0.01, 0.05)  # claude-opus: the big driver

    monkeypatch.setattr(litellm, "cost_per_token", fake_cost_per_token)
    drivers = estimate_drivers(
        [ModelSpec(model="claude-haiku"), ModelSpec(model="kimi"), ModelSpec(model="claude-opus")],
        "q",
    )
    assert [m for m, _ in drivers] == ["claude-opus", "claude-haiku"]
    assert drivers[0][1] == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_fanout_cost_cap_message_names_estimate_drivers(monkeypatch):
    """The over-cap rejection names the panellists driving the estimate so
    the caller can act (raise the cap or drop the named models) instead of
    staring at a bare number."""
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (3.10, True))
    monkeypatch.setattr(
        runner,
        "estimate_drivers",
        lambda *a, **kw: [("gpt-pro", 1.44), ("claude-opus", 0.61)],
    )
    handle = await fanout("p", [ModelSpec(model="claude-haiku")], max_run_usd=1.0)
    assert handle.partial is True
    assert handle.partial_reason is not None
    assert "top estimate drivers: gpt-pro ~$1.44, claude-opus ~$0.61" in handle.partial_reason


@pytest.mark.asyncio
async def test_fanout_cost_cap_survives_driver_enrichment_failure(monkeypatch):
    """Driver naming is best-effort message enrichment: when it raises, the
    clean rejection must still come back."""
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(runner, "estimate_cost", lambda *a, **kw: (3.10, True))

    def boom(*a, **kw):
        raise RuntimeError("enrichment broke")

    monkeypatch.setattr(runner, "estimate_drivers", boom)
    handle = await fanout("p", [ModelSpec(model="claude-haiku")], max_run_usd=1.0)
    assert handle.partial is True
    assert handle.partial_reason is not None
    assert "exceeds cap" in handle.partial_reason
    assert "drivers" not in handle.partial_reason
