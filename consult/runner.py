"""Async fan-out runner. Calls LiteLLM in parallel, classifies responses,
writes artifacts, and assembles a RunHandle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm

from . import artifacts, context, registry
from .progress import (
    Heartbeat,
    PanellistCompleted,
    PanellistPartial,
    PanellistStarted,
    PhaseStarted,
    ProgressEvent,
)
from .status import classify
from .types import ManifestEntry, ModelSpec, RunHandle, Status

logger = logging.getLogger(__name__)

# Async progress callback. Receives a typed `ProgressEvent`; the server-side
# adapter (server._progress_callback) converts to the MCP wire shape. Wrapped
# at each call site in a try/except so a notification failure never aborts
# the real work (best-effort observability, not a hard contract).
ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


async def _write_text_async(path: Path, content: str) -> None:
    """Off-loop write helper. Wraps `Path.write_text` in `asyncio.to_thread`
    so per-panellist artifact writes inside `_call_one` don't block the
    event loop while N parallel panellists finish around the same time.
    Sync-in-async writes on a 10-spec panel previously stacked ~30 blocking
    syscalls that could push past per-call timeouts.
    """
    await asyncio.to_thread(path.write_text, content)


def _append_progress_log(run_root: Path, event: ProgressEvent) -> None:
    """Append a JSONL line to `<run>/_progress.log` for client-less tailing.

    Always on — gives mid-run observability via `tail -f` even when the MCP
    client didn't ask for `notifications/progress`. Each line is the event's
    `model_dump()` with a `ts` field prepended; a write error here is logged
    at debug and swallowed.
    """
    payload: dict[str, Any] = {"ts": datetime.now(UTC).isoformat(), **event.model_dump()}
    try:
        with (run_root / "_progress.log").open("a") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError as e:  # pragma: no cover — log-write failure is benign
        logger.debug("progress log write failed: %s", e)

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
    _LITELLM_CONFIGURED = True

# CONTRACT: capsule.py:_CONFIDENCE and the capsule extractor prompt depend on
# these exact line prefixes (`CONFIDENCE:` and `KEY_REASON:`). Don't rename
# either without updating both.
_FOOTER = """\
---
End your response with EXACTLY these two lines (after your main answer):
CONFIDENCE: <number between 0.0 and 1.0 reflecting how sure you are>
KEY_REASON: <one sentence — the single most important reason for your view>"""


def _build_per_slug_prompt(base_prompt: str, stance_prompt: str) -> str:
    head = f"{stance_prompt}\n\n" if stance_prompt else ""
    return f"{head}{base_prompt}\n\n{_FOOTER}"


_MODEL_COUNT_SUFFIX = re.compile(r"^(.+):(\d+)$")

# Raw LiteLLM IDs can legitimately contain characters the slug regex
# rejects — e.g. OpenRouter's `:free` suffix. Sanitise model-derived slugs
# so a valid model alias never crashes the path-build downstream. The
# user-supplied slug path is unchanged: that goes through ModelSpec's
# field_validator which fails fast at the input boundary.
_SLUG_BAD_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
# refine `_suffix_specs` writes `.r<n>` suffixes onto slugs; recognise
# the same shape here so blinded mode can preserve per-round identity.
_ROUND_SUFFIX_RE = re.compile(r"\.r\d+$")


def _sanitise_derived_slug(base: str) -> str:
    """Coerce a model-derived slug fragment into the safe-id character set.

    Multiple bad characters in a row collapse to a single `-` and any
    leading/trailing `-`/`.` are stripped so the result is also a legal
    leading character (the slug regex anchors on an alphanumeric).
    """
    out = _SLUG_BAD_CHARS_RE.sub("-", base)
    return out.strip("-.")


def expand_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    """Expand `model:N` syntax into N copies of the spec.

    A trailing `:<positive int>` on `spec.model` requests N instances of the
    same model in the panel — useful for stochastic averaging (run the same
    prompt N times and compare), or to grow a panel without adding new
    aliases. The slug-disambiguation in `_make_slug` already appends an
    index suffix when the same base name repeats, so no extra work needed
    downstream.

    Idempotent: passing already-expanded specs (no `:N` suffix on any
    `model`) returns them unchanged. Bare model strings with internal
    colons that aren't followed by digits (rare, but possible for raw
    LiteLLM IDs) are left alone — the regex anchors to the end.

    Raises `ValueError` if N < 1 (zero copies is almost certainly a typo).
    """
    out: list[ModelSpec] = []
    for spec in specs:
        m = _MODEL_COUNT_SUFFIX.match(spec.model)
        if not m:
            out.append(spec)
            continue
        base, count_str = m.group(1), m.group(2)
        count = int(count_str)
        if count < 1:
            raise ValueError(
                f"model:count must be ≥1 (got {spec.model!r})"
            )
        # When an explicit slug is set and count > 1, suffix each copy with
        # its index. Without this all N copies share the same slug, race to
        # write to the same `responses/<slug>.txt`, and N-1 responses are
        # silently overwritten. The bare-model branch is safe because
        # `_make_slug` already disambiguates by index.
        for i in range(count):
            slug = f"{spec.slug}-{i}" if (spec.slug and count > 1) else spec.slug
            out.append(ModelSpec(model=base, stance=spec.stance, slug=slug))
    return out


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


_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 2.0

# Process-level provider concurrency gate. Keyed by the running event loop so
# nothing breaks if a test harness or REPL drives the API from a fresh loop
# (asyncio.Semaphore is bound to the loop that created it; a stale semaphore
# from a torn-down loop would silently no-op). Lazily populated on first
# `_get_provider_sems()` call in a given loop.
_provider_sems_by_loop: dict[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]] = {}


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
    """`litellm.acompletion` with bounded, jittered retry on RateLimitError.

    Only `RateLimitError` retries — auth, content-filter, bad-request, and
    other terminal errors propagate immediately (retrying them just burns
    spend). The total wall-clock (calls + sleeps) is bounded by `timeout`:
    each attempt's `asyncio.wait_for` uses the *remaining* budget, so the
    last retry can't push the run past the per-spec ceiling.

    Configurable via env: `CONSULT_RETRY_MAX_ATTEMPTS` (default 3, set to 1
    to disable), `CONSULT_RETRY_BASE_DELAY` (default 2.0s). Backoff is
    `base * 2^attempt * (0.5 + random())` — exponential with ±50% jitter
    so panels of N concurrently-rate-limited siblings don't retry in lockstep.
    """
    rate_cls = _rate_limit_class()
    max_attempts = max(
        1, int(os.environ.get("CONSULT_RETRY_MAX_ATTEMPTS", _RETRY_MAX_ATTEMPTS))
    )
    base_delay = float(
        os.environ.get("CONSULT_RETRY_BASE_DELAY", _RETRY_BASE_DELAY_S)
    )
    model_label = kwargs.get("model", "?")

    start = time.monotonic()
    for attempt in range(max_attempts):
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError(
                f"retry budget exhausted before attempt {attempt + 1}"
            )
        try:
            return await asyncio.wait_for(
                litellm.acompletion(**kwargs), timeout=remaining
            )
        except rate_cls as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) * (0.5 + random.random())
            remaining_after = timeout - (time.monotonic() - start)
            # Leave a 0.5s margin so the next attempt has time to start.
            sleep_for = min(delay, remaining_after - 0.5)
            if sleep_for <= 0:
                raise
            logger.warning(
                "rate-limited on %s attempt %d/%d (%s); retry in %.2fs",
                model_label, attempt + 1, max_attempts, type(e).__name__, sleep_for,
            )
            await asyncio.sleep(sleep_for)
    # Unreachable — the loop either returns or raises above.
    raise TimeoutError(f"retry budget exhausted ({timeout}s)")


def _concat_turn_text(turns: list[dict[str, Any]]) -> str:
    """Flatten a list of `{role, content}` turns into a single text blob
    for token counting. `content` may be a string or a list of content
    parts (Anthropic-style blocks); we only count text. Other block kinds
    (images, tool_use) are skipped — none flow through `prior_turns` today.
    """
    parts: list[str] = []
    for turn in turns:
        content = turn.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _build_messages(
    prompt: str,
    provider: str,
    prior_turns: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the messages array for one panellist call.

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
        turns.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
            ],
        })
    else:
        turns.append({"role": "user", "content": prompt})
    return turns


def _make_slug(spec: ModelSpec, idx: int, blinded: bool) -> str:
    if blinded:
        # alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu
        greek = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
        ]
        base = f"panelist-{greek[idx]}" if idx < len(greek) else f"panelist-{idx}"
        # Preserve refine's `.r<n>` round suffix even when blinded so the
        # round-N artifact doesn't overwrite round-(N-1)'s. Without this,
        # blinded refine writes every round to the same `panelist-alpha.txt`
        # file and the per-round transcript is destroyed.
        if spec.slug:
            m = _ROUND_SUFFIX_RE.search(spec.slug)
            if m:
                base = f"{base}{m.group(0)}"
        return base
    if spec.slug:
        return spec.slug
    # Derive a stable, readable slug from the alias/id. Raw LiteLLM IDs may
    # contain characters the safe-id regex rejects (e.g. `:free` on
    # OpenRouter); sanitise so a valid model never crashes the path build.
    base = _sanitise_derived_slug(spec.model.split("/")[-1].lower())
    return f"{base}-{idx}" if idx > 0 else base


_STREAM_PARTIAL_INTERVAL_S_DEFAULT = 1.0


def _stream_partial_interval_s() -> float:
    """Read at call time so test monkeypatching of the env var works."""
    return float(
        os.environ.get(
            "CONSULT_STREAM_PARTIAL_INTERVAL_S", _STREAM_PARTIAL_INTERVAL_S_DEFAULT
        )
    )


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
    stream = await litellm.acompletion(stream=True, **kwargs)
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
            f"stream_chunk_builder failed after {len(chunks)} chunks: "
            f"{type(e).__name__}: {e}"
        ) from e


async def _call_one(
    spec: ModelSpec,
    slug: str,
    per_slug_prompt: str,
    paths: artifacts.RunPaths,
    provider_sems: dict[str, asyncio.Semaphore] | None = None,
    *,
    stream: bool = False,
    on_partial: Callable[[int, int], Awaitable[None]] | None = None,
    prior_turns: list[dict[str, Any]] | None = None,
) -> ManifestEntry:
    # An unknown alias must fail this single panellist, not the whole panel.
    # `asyncio.gather` without return_exceptions=True would otherwise cancel
    # every sibling call when the KeyError propagates out.
    try:
        entry = registry.resolve_model(spec.model)
    except KeyError as e:
        await _write_text_async(paths.response_text(slug), "")
        # Unknown alias — no provider call was made, so the cost is genuinely
        # zero (NOT unknown). cost_usd=0.0 + cost_known=True is the correct
        # encoding; using cost_usd=None would now fail the cost-invariant
        # validator on ManifestEntry.
        return ManifestEntry(
            slug=slug,
            model_id=None,
            persona=spec.stance if spec.stance else None,
            status=Status.ERROR,
            finish_reason=None,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=0,
            tokens_in=None,
            tokens_out=None,
            cost_usd=0.0,
            cost_known=True,
            error=str(e),
            confidence=None,
            capsule=None,
        )
    litellm_id = entry["litellm_id"]
    budget = entry.get("default_budget_tokens", 8000)
    timeout = entry.get("default_timeout_s", 180)
    provider = entry.get("provider", "")

    extra: dict[str, Any] = {}
    if "reasoning_effort" in entry:
        extra["reasoning_effort"] = entry["reasoning_effort"]

    await _write_text_async(paths.prompt_for(slug), per_slug_prompt)
    start = time.time()
    status: Status
    finish: str | None = None
    body = ""
    cost: float | None = None
    cost_known: bool = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None

    # Per-provider concurrency gate. The FRICTION log shows OpenAI panellists
    # rate-limiting concurrently on every panel run — the shared key's
    # per-minute bucket is exhausted by N parallel calls. The semaphore caps
    # the in-flight count per provider so e.g. only 2 OpenAI calls run at
    # once, the rest queue up. `provider_sems["default"]` covers raw LiteLLM
    # IDs whose provider isn't enumerated in models.json. `nullcontext()` is
    # async-compatible since Python 3.10 — same shape as a Semaphore, no-op.
    sem: asyncio.Semaphore | None = None
    if provider_sems:
        sem = provider_sems.get(provider) or provider_sems.get("default")

    try:
        async with sem if sem is not None else nullcontext():
            messages = _build_messages(per_slug_prompt, provider, prior_turns)
            if stream:
                resp = await _stream_acompletion(
                    timeout=timeout,
                    on_partial=on_partial,
                    start=start,
                    model=litellm_id,
                    messages=messages,
                    max_tokens=budget,
                    **extra,
                )
            else:
                resp = await _acompletion_with_retry(
                    timeout=timeout,
                    model=litellm_id,
                    messages=messages,
                    max_tokens=budget,
                    **extra,
                )
        # Persist raw response — use model_dump for Pydantic, fall back to dict
        try:
            raw = resp.model_dump()  # type: ignore[attr-defined]
        except AttributeError:
            raw = dict(resp) if hasattr(resp, "__iter__") else {"_repr": repr(resp)}
        await _write_text_async(
            paths.response_raw(slug), json.dumps(raw, indent=2, default=str)
        )

        status, finish, body = classify(resp)
        usage = getattr(resp, "usage", None)
        if usage:
            tokens_in = getattr(usage, "prompt_tokens", None)
            tokens_out = getattr(usage, "completion_tokens", None)
        try:
            cost = litellm.completion_cost(completion_response=resp)
            cost_known = cost is not None
        except Exception as ce:  # noqa: BLE001
            logger.warning("cost lookup failed for %s: %s", litellm_id, ce)
            cost = None
            cost_known = False

    except TimeoutError:
        status, finish, body = Status.TIMEOUT, None, ""
        error = f"timeout after {timeout}s"
        # Conservative: a TimeoutError from `asyncio.wait_for` means we
        # gave up waiting, NOT that the HTTP request never landed. The
        # provider may have processed and billed us; cost_known=False
        # surfaces that uncertainty in the ledger as a lower bound rather
        # than silently understating spend. Mirrors slow-tail dropout.
        cost = None
        cost_known = False
    except Exception as e:  # noqa: BLE001 — LiteLLM raises many concrete types
        status, finish, body = classify(None, exception=e)
        error = str(e)[:4096] or f"{type(e).__name__}"
        # Same logic: most provider exceptions imply no billable call, but
        # we can't be sure for every case (e.g. 502 mid-stream may have
        # billed). cost_known=False is the conservative encoding.
        cost = None
        cost_known = False

    await _write_text_async(paths.response_text(slug), body)
    latency_ms = int((time.time() - start) * 1000)

    # Per-panellist completion line. Visible in MCP server logs
    # (~/.claude/logs/mcp-logs-consult/) — gives mid-run observability so
    # a slow panellist's status is knowable without waiting for the full
    # manifest. Use info-level so production stays quiet by default.
    logger.info("panellist %s: %s in %dms", slug, status.value, latency_ms)

    # Per-run progress log — always on, tailable as JSONL even if the MCP
    # client didn't subscribe to notifications/progress. The typed event
    # makes programmatic consumers easy; (done, total) here are placeholders
    # since `_call_one` doesn't know the panel size — the wrapper in `fanout`
    # constructs the real progress event for the callback path.
    _append_progress_log(
        paths.root,
        PanellistCompleted(
            done=0,
            total=0,
            slug=slug,
            status=status.value,
            latency_ms=latency_ms,
        ),
    )

    persona_label = spec.stance if spec.stance else None

    # Status.ERROR/TIMEOUT require a non-empty error per the model invariant.
    # Defensive: if classify() returns ERROR with no exception path taken (e.g.
    # malformed empty response), synthesise a placeholder so construction
    # doesn't blow up — the underlying classifier already logged the shape.
    if status in (Status.ERROR, Status.TIMEOUT) and not error:
        error = f"{status.value}: no provider exception captured"

    return ManifestEntry(
        slug=slug,
        model_id=litellm_id,
        persona=persona_label,
        status=status,
        finish_reason=finish,
        resource_uri=paths.resource_uri(slug),
        body_path=str(paths.response_text(slug)),
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        cost_known=cost_known,
        error=error,
        confidence=None,  # populated by capsule extractor
        capsule=None,
    )


def estimate_cost(specs: list[ModelSpec], prompt: str) -> tuple[float, bool]:
    """Returns (total_estimate, all_known).

    Uses LiteLLM's per-token price tables via `cost_per_token()`. Models that
    don't have pricing data set `all_known=False`; caller must treat unknown
    costs conservatively (a panel with even one unknown-cost spec cannot be
    validated against `max_run_usd`).

    Unknown-alias specs are treated as cost-unknown (rather than raising) so
    a single typo can't abort the panel here; `_call_one` surfaces the alias
    as a per-spec Status.ERROR.
    """
    total = 0.0
    all_known = True
    for spec in specs:
        try:
            entry = registry.resolve_model(spec.model)
        except KeyError:
            all_known = False
            continue
        litellm_id = entry["litellm_id"]
        try:
            tin = litellm.token_counter(model=litellm_id, text=prompt)
            tout = entry.get("default_budget_tokens", 8000)
            in_per_tok, out_per_tok = litellm.cost_per_token(
                model=litellm_id, prompt_tokens=tin, completion_tokens=tout
            )
            if in_per_tok is None or out_per_tok is None:
                all_known = False
                continue
            total += in_per_tok + out_per_tok
        except Exception as e:  # noqa: BLE001
            logger.warning("estimate_cost: no price for %s (%s)", litellm_id, e)
            all_known = False
            continue
    return total, all_known


async def fanout(
    prompt: str,
    specs: list[ModelSpec],
    *,
    blinded: bool = False,
    dry_run: bool = False,
    max_run_usd: float | None = None,
    existing_paths: artifacts.RunPaths | None = None,
    on_progress: ProgressCallback | None = None,
    stream: bool = False,
    capsule_kind: str = "decision",
    prior_turns: list[dict[str, Any]] | None = None,
) -> RunHandle:
    """Parallel fan-out. Creates a fresh run by default. Pass `existing_paths`
    to write into an existing run dir (used by `refine` to keep all rounds
    under one run_id with round-suffixed slugs).

    `prior_turns`, when set, is a sequence of `{role, content}` dicts
    prepended to the messages array for every panellist call. Used by
    `refine` with a `continuation_id` to expose the prior consultation as
    a proper user/assistant exchange. The text is included in cost
    estimation so the cap check stays accurate.

    If `on_progress` is set, it's called once per panellist as it completes
    with `(done, total, message)`. Failures inside the callback are logged
    and swallowed — progress is best-effort, not load-bearing.
    """
    # Resolve `model:N` sugar BEFORE estimate_cost so the cap reflects the
    # real panel size, not the pre-expansion request count.
    specs = expand_specs(specs)
    # Env-var override for streaming — lets a user enable streaming across
    # every fanout (incl. the ones nested inside consult/refine/sequence)
    # without changing the tool-call surface.
    if not stream and os.environ.get("CONSULT_STREAM", "0") == "1":
        stream = True
    # Apply LiteLLM tweaks lazily — covers callers that bypass __main__.cli
    # (tests, library use, the `consult-view`/`consult-ledger` CLIs that
    # import this module).
    configure_litellm()
    if existing_paths is None:
        paths = artifacts.create_run()
        paths.prompt_txt.write_text(prompt)
        # Snapshot the registry so replays are stable
        paths.registry_snapshot.write_text(json.dumps(registry.models_config(), indent=2))
        # Single immutable per-run context bundle. Downstream stages
        # (synth, capsule, arbiter) load this rather than re-receiving
        # the prompt — keeps the blinding scrub centralised and avoids
        # silent prompt-prompt skew across stages. `capsule_kind` is
        # persisted here so a `continuation_id` can inherit the prior
        # run's shape without the caller having to specify it again.
        context.write(
            paths,
            context.build(prompt, blinded=blinded, capsule_kind=capsule_kind),
        )
    else:
        paths = existing_paths

    # Estimate cost up front; if dry_run, return immediately with empty
    # manifest. When `prior_turns` is set (continuation), include their
    # text in the token count so the cap check sees the real input size.
    cost_input = prompt
    if prior_turns:
        cost_input = _concat_turn_text(prior_turns) + "\n" + prompt
    estimate, all_known = estimate_cost(specs, cost_input)
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()
    if estimate > cap:
        # When some prices are unknown, `estimate` is only the known-priced
        # portion; the actual run could cost more. Surface that so the cap
        # message isn't misleading low. Mirrors the dry_run branch below.
        suffix = "" if all_known else " (known-priced portion only; some unknown)"
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=(
                f"estimated cost ${estimate:.2f}{suffix} exceeds cap ${cap:.2f}"
            ),
            blinded=blinded,
        )
    if dry_run:
        suffix = "" if all_known else " (some prices unknown — actual cost may differ)"
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=f"dry_run: estimated cost ${estimate:.4f}{suffix}",
            blinded=blinded,
        )

    # Build slugs + prompts
    slugs = [_make_slug(s, i, blinded) for i, s in enumerate(specs)]
    per_prompts = [
        _build_per_slug_prompt(prompt, registry.resolve_stance(s.stance)) for s in specs
    ]

    # Process-level provider semaphores (shared across concurrent fanouts
    # in the same event loop). See `_get_provider_sems` for the rationale.
    provider_sems = _get_provider_sems()

    start = time.time()
    total = len(specs)
    done = 0
    started = 0
    # The heartbeat reads these to summarise live state. Mutation is
    # confined to `_run_one` and the slow-tail dropout block — both run in
    # the same event loop so atomicity between awaits is enough; no lock.
    completed_entries: list[ManifestEntry] = []
    pending_slugs_set: set[str] = set(slugs)

    async def _safe_notify(event: ProgressEvent) -> None:
        """Wrapper that swallows callback exceptions. Progress is best-effort:
        a notification failure (closed session, slow client, raising user
        callback) must never tear down the real work.
        """
        if on_progress is None:
            return
        try:
            await on_progress(event)
        except Exception as e:  # noqa: BLE001
            logger.debug("on_progress callback failed (%s): %s", event.kind, e)

    # PhaseStarted("fanout"): emitted before any panellist begins, so the
    # parent sees fanout starting rather than receiving silence until the
    # first panellist completes.
    phase_event = PhaseStarted(done=0, total=total, phase="fanout")
    _append_progress_log(paths.root, phase_event)
    await _safe_notify(phase_event)

    # Heartbeat task: periodic liveness pulse. Set CONSULT_HEARTBEAT_INTERVAL_S=0
    # to disable (used in tests that mock _call_one to instant returns).
    hb_interval = float(os.environ.get("CONSULT_HEARTBEAT_INTERVAL_S", 5.0))
    heartbeat_task: asyncio.Task[None] | None = None
    if hb_interval > 0:
        async def _heartbeat_loop() -> None:
            while True:
                await asyncio.sleep(hb_interval)
                elapsed_ms = int((time.time() - start) * 1000)
                cost_so_far = sum((e.cost_usd or 0.0) for e in completed_entries)
                cost_known = all(e.cost_known for e in completed_entries)
                pending = sorted(pending_slugs_set)
                event = Heartbeat(
                    done=done,
                    total=total,
                    elapsed_ms=elapsed_ms,
                    cost_so_far_usd=cost_so_far,
                    cost_known=cost_known,
                    pending_count=len(pending),
                    pending_slugs=pending,
                )
                _append_progress_log(paths.root, event)
                await _safe_notify(event)
        heartbeat_task = asyncio.create_task(_heartbeat_loop())

    async def _run_one(spec: ModelSpec, slug: str, per_prompt: str) -> ManifestEntry:
        nonlocal done, started
        # PanellistStarted: fires BEFORE the LiteLLM call so the parent
        # sees which slugs are in flight, not just which have completed.
        started += 1
        started_event = PanellistStarted(
            done=done, total=total, slug=slug, started_count=started,
        )
        _append_progress_log(paths.root, started_event)
        await _safe_notify(started_event)

        # When streaming is enabled, wire each panellist's mid-stream chunk
        # callback to emit `PanellistPartial` events. Throttled to
        # ~1 chunk/sec by `_STREAM_PARTIAL_INTERVAL_S` so the progress
        # channel doesn't drown in micro-updates.
        on_partial: Callable[[int, int], Awaitable[None]] | None = None
        if stream and on_progress is not None:
            async def _emit_partial(chars: int, elapsed_ms: int) -> None:
                await _safe_notify(PanellistPartial(
                    done=done, total=total, slug=slug,
                    chars_so_far=chars, elapsed_ms=elapsed_ms,
                ))
            on_partial = _emit_partial
        entry = await _call_one(
            spec, slug, per_prompt, paths, provider_sems,
            stream=stream, on_partial=on_partial,
            prior_turns=prior_turns,
        )
        done += 1
        completed_entries.append(entry)
        pending_slugs_set.discard(slug)
        await _safe_notify(PanellistCompleted(
            done=done,
            total=total,
            slug=slug,
            status=entry.status.value,
            latency_ms=entry.latency_ms,
        ))
        return entry

    # Slow-tail dropout: once most of the panel has returned, cancel the
    # slowest stragglers rather than waiting for the per-spec timeout. FRICTION
    # observed a 5× spread between fastest and slowest in the same panel; the
    # median wall-time win comes from releasing 1-2 laggards. Disabled for
    # small panels (no useful "rest of the panel" signal) and when
    # CONSULT_TAIL_DROPOUT_S=0. The full per-spec timeout still bounds the
    # worst case if dropout is off.
    tail_dropout_s = float(os.environ.get("CONSULT_TAIL_DROPOUT_S", 30.0))
    tail_k_frac = float(os.environ.get("CONSULT_TAIL_K_FRAC", 0.2))
    enable_dropout = total >= 4 and tail_dropout_s > 0 and 0 < tail_k_frac < 1.0

    try:
        if not enable_dropout:
            coros = [
                _run_one(spec, slug, per_prompt)
                for spec, slug, per_prompt in zip(specs, slugs, per_prompts, strict=True)
            ]
            manifest = list(await asyncio.gather(*coros))
        else:
            task_list: list[asyncio.Task[ManifestEntry]] = []
            task_meta: dict[asyncio.Task[ManifestEntry], tuple[str, ModelSpec]] = {}
            for spec, slug, per_prompt in zip(specs, slugs, per_prompts, strict=True):
                t = asyncio.create_task(_run_one(spec, slug, per_prompt))
                task_list.append(t)
                task_meta[t] = (slug, spec)

            completed_tasks: set[asyncio.Task[ManifestEntry]] = set()
            pending: set[asyncio.Task[ManifestEntry]] = set(task_list)
            k = max(1, math.ceil(total * tail_k_frac))
            trigger = max(1, total - k)

            while len(completed_tasks) < trigger and pending:
                done_set, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                completed_tasks.update(done_set)

            if pending:
                logger.info(
                    "slow-tail dropout: %d/%d complete, waiting up to %.1fs for %d stragglers",
                    len(completed_tasks), total, tail_dropout_s, len(pending),
                )
                done_set, pending = await asyncio.wait(pending, timeout=tail_dropout_s)
                completed_tasks.update(done_set)

            drop_entries: dict[asyncio.Task[ManifestEntry], ManifestEntry] = {}
            for t in pending:
                t.cancel()
            for t in pending:
                slug, spec = task_meta[t]
                try:
                    # A task may complete in the race between asyncio.wait
                    # returning and t.cancel(); in that case _run_one already
                    # ran its progress emission and we keep its result.
                    await t
                    completed_tasks.add(t)
                    continue
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                latency_ms = int((time.time() - start) * 1000)
                if not paths.response_text(slug).exists():
                    paths.response_text(slug).write_text("")
                # Cancelled tasks set cost_known=False: asyncio.cancel()
                # may race with an already-in-flight HTTP request, in which
                # case the provider WILL bill us even though we never see
                # the response. Treating the cost as "known to be zero"
                # would silently understate the run total. The handle's
                # `cost_known` will go False as soon as any dropped entry
                # is present, mirroring the partial-pricing path.
                entry = ManifestEntry(
                    slug=slug,
                    model_id=None,
                    persona=spec.stance if spec.stance else None,
                    status=Status.TIMEOUT,
                    finish_reason=None,
                    resource_uri=paths.resource_uri(slug),
                    body_path=str(paths.response_text(slug)),
                    latency_ms=latency_ms,
                    cost_usd=None,
                    cost_known=False,
                    error=f"slow-tail dropout after {tail_dropout_s}s",
                    confidence=None,
                    capsule=None,
                )
                drop_entries[t] = entry
                done += 1
                completed_entries.append(entry)
                pending_slugs_set.discard(slug)
                await _safe_notify(PanellistCompleted(
                    done=done, total=total, slug=slug,
                    status=Status.TIMEOUT.value, latency_ms=latency_ms,
                ))
                _append_progress_log(paths.root, PanellistCompleted(
                    done=0, total=0, slug=slug,
                    status=Status.TIMEOUT.value, latency_ms=latency_ms,
                ))

            manifest = []
            for t in task_list:
                if t in drop_entries:
                    manifest.append(drop_entries[t])
                else:
                    manifest.append(t.result())
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

    wall_ms = int((time.time() - start) * 1000)

    # If blinded, scrub model_id from the manifest (kept in registry_snapshot for audit)
    if blinded:
        for m in manifest:
            m.model_id = None

    cost_total = sum((m.cost_usd or 0.0) for m in manifest)
    all_known = all(m.cost_known for m in manifest)
    # Zero-usable-panel guard. When every panellist times out, errors, or is
    # rate-limited, callers (refine arbiter, sequence) must not proceed —
    # there is no signal to evaluate and the downstream spend (arbiter,
    # synthesis) would burn for nothing. partial=True here lets the caller
    # short-circuit; a non-empty manifest of failure entries is still
    # returned for diagnosis.
    usable_count = sum(1 for m in manifest if m.status in (Status.OK, Status.TRUNCATED))
    if manifest and usable_count == 0:
        statuses = sorted({m.status.value for m in manifest})
        partial = True
        partial_reason: str | None = (
            f"zero usable panellists ({len(manifest)} returned: {', '.join(statuses)})"
        )
    else:
        partial = False
        partial_reason = None

    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=manifest,
        cost_usd=cost_total,
        cost_known=all_known,
        wall_ms=wall_ms,
        partial=partial,
        partial_reason=partial_reason,
        blinded=blinded,
    )
    artifacts.write_manifest(paths, handle.model_dump())
    return handle
