"""LiteLLM transport: retry policy, streaming, the Responses-API adapter,
provider concurrency gates, and message construction.

`_acompletion_with_retry` resolves the retry-class helpers through the
package facade at call time so `monkeypatch.setattr(runner,
"_rate_limit_class", ...)` keeps working exactly as it did when runner
was a single module.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

import litellm

from consult import runner as _facade

from .. import registry
from ..envutil import env_float, env_int
from ..redact import install_redaction_filter, redact_secrets, scrub_exception_attrs

logger = logging.getLogger(__name__)


_LITELLM_CONFIGURED = False


def configure_litellm() -> None:
    """Apply consult's LiteLLM tweaks. Idempotent.

    Called from `__main__.cli`, from `fanout()` on first use, and from test
    fixtures. Three tweaks:
    - `drop_params=True`: silently drop unsupported kwargs (e.g.
      `reasoning_effort` on a non-reasoning model) instead of failing the
      whole panel.
    - `suppress_debug_info=True`: kill the ANSI "Give Feedback / Get Help"
      footer LiteLLM prints to stderr on every caught exception.
    - `LiteLLM` logger `propagate=False`: LiteLLM attaches its own coloured
      handler AND lets messages propagate to root, so any caller that
      configures the root logger at INFO sees every line twice.

    Previously these ran as module-level side effects at import time, which
    meant any importer (incl. test helpers and `consult-view`) silently
    picked them up. Making it explicit keeps the side effect at the entry
    points that actually need it.
    """
    global _LITELLM_CONFIGURED
    if _LITELLM_CONFIGURED:
        return
    litellm.drop_params = True
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").propagate = False
    # LITELLM_LOG=DEBUG makes LiteLLM's logger print request kwargs, headers
    # included; their handler renders it before any consult code sees the
    # text, so the redaction has to live on the logger itself (issue #40).
    install_redaction_filter("LiteLLM")
    _LITELLM_CONFIGURED = True


# CONTRACT: capsule.py:_CONFIDENCE and the capsule extractor prompt depend on


def _rate_limit_class() -> type[BaseException]:
    """Lazily resolve `litellm.exceptions.RateLimitError`. Returns a sentinel
    class that nothing isinstance-matches if the symbol is missing — i.e.
    the retry loop becomes a no-op rather than crashing on a broken import.
    """
    try:
        from litellm import exceptions as lex

        cls = getattr(lex, "RateLimitError", None)
        if cls is not None:
            return cls
    except Exception:  # pragma: no cover — litellm always present in prod
        pass

    class _NeverRaised(BaseException):
        pass

    return _NeverRaised


def _transient_error_classes() -> tuple[type[BaseException], ...]:
    """Resolve the set of LiteLLM exception subclasses we treat as retriable
    but distinct from RateLimitError. Returns an empty tuple (matches
    nothing via isinstance) when litellm is unavailable or the classes
    have moved.

    Specifically: connection drops and transient upstream 5xx. Does NOT
    include bare `APIError` — that's the root of the openai/litellm
    exception tree and would also match `AuthenticationError`,
    `BadRequestError`, etc. (terminal failures we don't want to retry).
    The bare-`APIError` case (e.g. OpenRouter's "Unable to get json
    response" all-whitespace body) is caught separately by exact-type
    match in `_acompletion_with_retry`.
    """
    try:
        from litellm import exceptions as lex
    except Exception:  # pragma: no cover — litellm always present in prod
        return ()
    classes: list[type[BaseException]] = []
    for name in (
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
    ):
        cls = getattr(lex, name, None)
        if cls is not None:
            classes.append(cls)
    return tuple(classes)


def _bare_api_error_class() -> type[BaseException] | None:
    """Resolve the root `litellm.exceptions.APIError` class for exact-type
    matching. Used to retry the bare-`APIError` failure mode (e.g.
    OpenRouter returning whitespace) without sweeping in auth/bad-request
    subclasses.
    """
    try:
        from litellm import exceptions as lex
    except Exception:  # pragma: no cover
        return None
    return getattr(lex, "APIError", None)


_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 2.0

# Process-level provider concurrency gate. Keyed by the running event loop so
# nothing breaks if a test harness or REPL drives the API from a fresh loop
# (asyncio.Semaphore is bound to the loop that created it; a stale semaphore
# from a torn-down loop would silently no-op). Lazily populated on first
# `_get_provider_sems()` call in a given loop.
#
# WeakKeyDictionary: test harnesses and notebook kernels routinely create and
# tear down event loops. With a regular dict, each new loop would add a
# permanent entry (loop ref + semaphores) that's never collected — a slow
# memory leak proportional to test/run count. WeakKeyDictionary drops the
# entry as soon as the loop is garbage-collected.
import weakref  # noqa: E402

_provider_sems_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)


def _get_provider_sems() -> dict[str, asyncio.Semaphore]:
    """Return the process-level provider semaphore dict for the running loop.

    Cross-call rate limiting: if a single MCP server process drives two
    concurrent fanouts (panel + refine in flight, or two clients), they
    share these semaphores — otherwise each fanout would hit the OpenAI
    bucket independently (the FRICTION-logged failure mode that the
    semaphores were originally meant to solve).
    """
    loop = asyncio.get_running_loop()
    sems = _provider_sems_by_loop.get(loop)
    if sems is None:
        caps = registry.provider_concurrency()
        sems = {p: asyncio.Semaphore(n) for p, n in caps.items()}
        _provider_sems_by_loop[loop] = sems
    return sems


async def _acompletion_with_retry(*, timeout: float, **kwargs: Any) -> Any:
    """`litellm.acompletion` with bounded, jittered retry on transient errors.

    Retries on `RateLimitError` and the transient-API family (`APIError`,
    `APIConnectionError`, `InternalServerError`, `ServiceUnavailableError`).
    The OpenRouter "Unable to get json response" failure (upstream returned
    all-whitespace) surfaces as `APIError` and was previously a one-shot
    ERROR — recovers cleanly on a second attempt. Auth, content-filter,
    bad-request, and other terminal errors propagate immediately (retrying
    them just burns spend).

    Total wall-clock (calls + sleeps) is bounded by `timeout`: each
    attempt's `asyncio.wait_for` uses the *remaining* budget, so the last
    retry can't push the run past the per-spec ceiling.

    Configurable via env: `CONSULT_RETRY_MAX_ATTEMPTS` (default 3, set to 1
    to disable), `CONSULT_RETRY_BASE_DELAY` (default 2.0s). Backoff is
    `base * 2^attempt * (0.5 + random())` — exponential with ±50% jitter
    so panels of N concurrently-failing siblings don't retry in lockstep.
    """
    # Resolved via the facade so test monkeypatching of
    # `consult.runner._rate_limit_class` (and siblings) is honoured.
    rate_cls = _facade._rate_limit_class()
    transient_classes = _facade._transient_error_classes()
    bare_api_cls = _facade._bare_api_error_class()
    retriable: tuple[type[BaseException], ...] = (rate_cls, *transient_classes)
    max_attempts = max(1, env_int("CONSULT_RETRY_MAX_ATTEMPTS", _RETRY_MAX_ATTEMPTS))
    base_delay = env_float("CONSULT_RETRY_BASE_DELAY", _RETRY_BASE_DELAY_S)
    model_label = kwargs.get("model", "?")

    start = time.monotonic()
    for attempt in range(max_attempts):
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError(f"retry budget exhausted before attempt {attempt + 1}")
        last_exc: BaseException
        try:
            return await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=remaining)
        except retriable as e:
            # Rate-limit or known transient subclass.
            last_exc = scrub_exception_attrs(e)
        except Exception as e:
            # Bare `APIError` (not a subclass) is the OpenRouter
            # "Unable to get json response" failure mode — retriable.
            # Any subclass (auth/bad-request/content-policy) is terminal.
            # Scrub the OBJECT before it escapes: its message/body attrs
            # can carry the Authorization header, and downstream consumers
            # (OTel record_exception, host logging) format the object, not
            # consult's redacted strings (issue #39).
            scrub_exception_attrs(e)
            if bare_api_cls is None or type(e) is not bare_api_cls:
                raise
            last_exc = e
        if attempt == max_attempts - 1:
            raise last_exc
        delay = base_delay * (2**attempt) * (0.5 + random.random())
        remaining_after = timeout - (time.monotonic() - start)
        # Leave a 0.5s margin so the next attempt has time to start.
        sleep_for = min(delay, remaining_after - 0.5)
        if sleep_for <= 0:
            raise last_exc  # already scrubbed at capture
        kind = "rate-limited" if isinstance(last_exc, rate_cls) else "transient API error"
        logger.warning(
            "%s on %s attempt %d/%d (%s); retry in %.2fs",
            kind,
            model_label,
            attempt + 1,
            max_attempts,
            type(last_exc).__name__,
            sleep_for,
        )
        await asyncio.sleep(sleep_for)
    # Unreachable — the loop either returns or raises above.


_ERROR_MAX_CHARS = 4096


def _format_error_message(exc: BaseException) -> str:
    """Coerce a provider exception into a compact manifest-friendly string.

    LiteLLM happily includes the upstream's raw response in the exception
    message, which for OpenRouter's "Unable to get json response" failure
    means 500+ blank lines of whitespace get embedded. The manifest gets
    enormous and the feed renders a giant empty error block.

    Strategy: keep the first non-empty line (the diagnostic), then collapse
    long runs of consecutive whitespace-only lines into a single `[...]`
    marker, redact secret-shaped tokens (see `consult.redact`), and
    hard-cap at `_ERROR_MAX_CHARS`.
    """
    raw = str(exc)
    if not raw:
        return type(exc).__name__
    lines = raw.split("\n")
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            continue
        if blank_run >= 3:
            out.append(f"[... {blank_run} blank lines elided ...]")
        elif blank_run > 0:
            out.extend([""] * blank_run)
        blank_run = 0
        out.append(line)
    collapsed = "\n".join(out).strip()
    collapsed = redact_secrets(collapsed)
    if len(collapsed) > _ERROR_MAX_CHARS:
        collapsed = collapsed[: _ERROR_MAX_CHARS - 1] + "…"
    return collapsed or type(exc).__name__


def build_messages(
    prompt: str,
    provider: str,
    prior_turns: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the messages array for one panellist call.

    Public so `synth.py` can share the exact same message-construction logic
    (including the Anthropic `cache_control` breakpoint placement) without
    importing a private symbol. Drift between fanout and synth on this
    structure has been a recurring source of silent failures.

    `prior_turns`, when set, is a sequence of `{role, content}` turns
    prepended verbatim before the final user prompt — used by `refine`
    when continuing a prior run so the model sees the previous
    consultation as a proper user/assistant exchange rather than a single
    block of stitched-together text. Role boundaries help the model
    distinguish "what was previously asked & answered" from "what we're
    asking now"; prepending also gives Anthropic + OpenAI prompt caches
    a stable prefix to key on across follow-ups from the same parent run.

    Anthropic-only: the final user prompt is wrapped with a
    `cache_control: ephemeral` breakpoint. Within a single fanout,
    repeat panellists share the bulk of this prefix (base prompt +
    footer; only stance varies); the breakpoint lets the second-onwards
    calls reuse the cached input.
    """
    turns: list[dict[str, Any]] = list(prior_turns) if prior_turns else []
    if provider == "anthropic":
        turns.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
            }
        )
    else:
        turns.append({"role": "user", "content": prompt})
    return turns


_STREAM_PARTIAL_INTERVAL_S_DEFAULT = 1.0


def _stream_partial_interval_s() -> float:
    """Read at call time so test monkeypatching of the env var works."""
    return env_float("CONSULT_STREAM_PARTIAL_INTERVAL_S", _STREAM_PARTIAL_INTERVAL_S_DEFAULT)


async def _stream_acompletion(
    *,
    timeout: float,
    on_partial: Callable[[int, int], Awaitable[None]] | None,
    start: float,
    **kwargs: Any,
) -> Any:
    """Streaming variant of `_acompletion_with_retry`.

    Streams chunks from LiteLLM, accumulates them, and uses
    `litellm.stream_chunk_builder` to reconstruct a single response object
    compatible with `litellm.completion_cost` and `classify()`. Emits
    progress callbacks throttled to ~`_STREAM_PARTIAL_INTERVAL_S`.

    Retry is NOT layered on top of streaming — a rate-limit mid-stream is
    rare and recovering it well requires re-emitting partials, which adds
    complexity without much practical benefit. Falls back to the synthetic
    response object on the happy path.
    """
    stream = cast(Any, await litellm.acompletion(stream=True, **kwargs))
    chunks: list[Any] = []
    body = ""
    last_emit = time.monotonic()
    partial_interval = _stream_partial_interval_s()

    async def _read():
        nonlocal body, last_emit
        async for chunk in stream:
            chunks.append(chunk)
            try:
                delta = chunk.choices[0].delta.content if chunk.choices else None
            except (AttributeError, IndexError):
                delta = None
            if delta:
                body += delta
            now = time.monotonic()
            if on_partial is not None and (now - last_emit) >= partial_interval:
                last_emit = now
                try:
                    await on_partial(len(body), int((time.time() - start) * 1000))
                except Exception as e:  # noqa: BLE001 — best-effort
                    # `warning` (not `debug`) so a bug in the callback shows
                    # up in default log configs; the call site still doesn't
                    # abort the panel.
                    logger.warning("partial callback failed: %s", e)

    await asyncio.wait_for(_read(), timeout=timeout)

    # Reconstruct a single ModelResponse so the rest of `_call_one` can
    # treat the streamed call identically to the non-streamed path. If
    # the builder fails (unsupported chunk shape, partial stream, etc),
    # RAISE rather than returning `chunks[-1]` — `chunks[-1]` is a raw
    # streaming chunk that `classify()` and `litellm.completion_cost()`
    # would mishandle, silently dropping the body. The exception will
    # propagate to `_call_one`'s outer try/except, which surfaces the
    # panellist as Status.ERROR with a clear error string.
    try:
        return litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))
    except Exception as e:
        raise RuntimeError(
            f"stream_chunk_builder failed after {len(chunks)} chunks: {type(e).__name__}: {e}"
        ) from e


async def _aresponses_as_completion(
    *, timeout: float, model: str, messages: list[dict[str, Any]], max_completion_tokens: int, **extra: Any
) -> Any:
    """Call litellm's Responses API for a `mode=responses` model and wrap the
    result in the chat-completions shape the rest of `_call_one` consumes.

    OpenAI Responses-API models (e.g. gpt-5.5-pro, gpt-5.3-codex) 404 on the
    chat-completions endpoint, so they route here instead of `acompletion`.
    The result is adapted into a SimpleNamespace with the chat-completions
    attributes the pipeline reads. System messages become `instructions`; the
    rest become `input`. Non-streaming only — the streaming path stays on chat
    completions.
    """
    instructions = "\n\n".join(m["content"] for m in messages if m.get("role") == "system") or None
    convo = [m for m in messages if m.get("role") != "system"]
    if len(convo) == 1:
        rinput: Any = convo[0]["content"]
    else:
        # Multi-turn (refine continuation): the Responses API takes a string or
        # an input-item list; a role-tagged string is the simplest faithful
        # rendering of the prior turns.
        rinput = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)

    kwargs: dict[str, Any] = {"model": model, "input": rinput, "max_output_tokens": max_completion_tokens}
    if instructions:
        kwargs["instructions"] = instructions
    if extra.get("reasoning_effort"):
        kwargs["reasoning"] = {"effort": extra["reasoning_effort"]}
    # Other chat-shaped params in `extra` (e.g. temperature) are intentionally
    # dropped: the Responses API rejects several of them.

    rresp = await asyncio.wait_for(litellm.aresponses(**kwargs), timeout=timeout)

    text = getattr(rresp, "output_text", None) or ""
    status_str = getattr(rresp, "status", None)
    finish = "stop"
    if status_str and status_str != "completed":
        details = getattr(rresp, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details else None
        finish = "length" if reason == "max_output_tokens" else (status_str or "stop")
    usage = getattr(rresp, "usage", None)
    tokens_in = getattr(usage, "input_tokens", None) if usage else None
    tokens_out = getattr(usage, "output_tokens", None) if usage else None

    def _dump() -> dict[str, Any]:
        try:
            return rresp.model_dump()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return {"_responses_repr": repr(rresp)}

    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
        model_dump=_dump,
    )
