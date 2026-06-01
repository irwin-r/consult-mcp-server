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
from pathlib import Path
from typing import Any

import litellm

from . import artifacts, context, registry, telemetry
from . import attachments as attachments_mod
from .capsule import MAX_TOKENS_BY_KIND
from .progress import (
    Heartbeat,
    PanellistCompleted,
    PanellistPartial,
    PanellistStarted,
    PhaseStarted,
    ProgressCallback,
    ProgressEvent,
    append_progress_log,
)
from .status import classify
from .types import ManifestEntry, ModelSpec, RunHandle, Status

logger = logging.getLogger(__name__)


async def _write_text_async(path: Path, content: str) -> None:
    """Off-loop write helper. Wraps `Path.write_text` in `asyncio.to_thread`
    so per-panellist artifact writes inside `_call_one` don't block the
    event loop while N parallel panellists finish around the same time.
    Sync-in-async writes on a 10-spec panel previously stacked ~30 blocking
    syscalls that could push past per-call timeouts.
    """
    await asyncio.to_thread(path.write_text, content)


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


def sanitise_derived_slug(base: str) -> str:
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
            raise ValueError(f"model:count must be ≥1 (got {spec.model!r})")
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
    rate_cls = _rate_limit_class()
    transient_classes = _transient_error_classes()
    bare_api_cls = _bare_api_error_class()
    retriable: tuple[type[BaseException], ...] = (rate_cls, *transient_classes)
    max_attempts = max(1, int(os.environ.get("CONSULT_RETRY_MAX_ATTEMPTS", _RETRY_MAX_ATTEMPTS)))
    base_delay = float(os.environ.get("CONSULT_RETRY_BASE_DELAY", _RETRY_BASE_DELAY_S))
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
            last_exc = e
        except Exception as e:
            # Bare `APIError` (not a subclass) is the OpenRouter
            # "Unable to get json response" failure mode — retriable.
            # Any subclass (auth/bad-request/content-policy) is terminal.
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
            raise last_exc
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
    raise TimeoutError(f"retry budget exhausted ({timeout}s)")


_ERROR_MAX_CHARS = 4096

# Secret-shaped tokens that may end up embedded in LiteLLM exception strings.
# LiteLLM frequently includes upstream response bodies / request headers in the
# exception when an HTTP error occurs, and those bodies routinely echo back the
# `Authorization: Bearer sk-…` header (or the provider-specific equivalent).
# Redacting at the manifest/log boundary means a leaked exception message can
# never carry a working key to disk under ~/.consult/runs/ or into a parent
# agent's transcript.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"sk-or-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"or-v1-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)Authorization\s*[:=]\s*Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    re.compile(r"(?i)x-api-key\s*[:=]\s*[A-Za-z0-9_\-\.]{20,}"),
    re.compile(r'(?i)["\']?api[_-]?key["\']?\s*[:=]\s*["\'][A-Za-z0-9_\-\.]{20,}["\']'),
)


def _redact_secrets(text: str) -> str:
    """Replace API-key-shaped tokens with `[REDACTED]`.

    Defence-in-depth: LiteLLM's exception text often embeds the raw HTTP
    response, which on auth-failure paths can carry the request
    `Authorization` header verbatim. Redacting here ensures a leaked
    manifest or `_progress.log` line never carries a working key.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _format_error_message(exc: BaseException) -> str:
    """Coerce a provider exception into a compact manifest-friendly string.

    LiteLLM happily includes the upstream's raw response in the exception
    message, which for OpenRouter's "Unable to get json response" failure
    means 500+ blank lines of whitespace get embedded. The manifest gets
    enormous and the feed renders a giant empty error block.

    Strategy: keep the first non-empty line (the diagnostic), then collapse
    long runs of consecutive whitespace-only lines into a single `[...]`
    marker, redact secret-shaped tokens (see `_SECRET_PATTERNS`), and
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
    collapsed = _redact_secrets(collapsed)
    if len(collapsed) > _ERROR_MAX_CHARS:
        collapsed = collapsed[: _ERROR_MAX_CHARS - 1] + "…"
    return collapsed or type(exc).__name__


def concat_turn_text(turns: list[dict[str, Any]]) -> str:
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


def _make_slug(spec: ModelSpec, idx: int, blinded: bool) -> str:
    """Single-spec slug derivation. Prefer `_make_slugs` for whole panels;
    this exists for backwards-compat and is used by tests that construct
    one entry at a time.
    """
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
    base = sanitise_derived_slug(spec.model.split("/")[-1].lower())
    return f"{base}-{idx}" if idx > 0 else base


def _make_slugs(specs: list[ModelSpec], blinded: bool) -> list[str]:
    """Compute slugs for a whole panel with per-base disambiguation.

    Single-instance models get the bare base name. Only repeats of the
    same base get a `-N` index suffix, so `[claude-opus, gpt-pro]` yields
    `["claude-opus", "gpt-pro"]` instead of the previous global-index
    `["claude-opus", "gpt-pro-1"]` (unearned suffix on the only gpt-pro).

    The refine `.r<n>` round suffix and the blinded greek-letter slug
    flow still go through `_make_slug` per spec — only the non-blinded
    bare-model path needs panel-level awareness.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, spec in enumerate(specs):
        if blinded or spec.slug:
            out.append(_make_slug(spec, i, blinded))
            continue
        base = sanitise_derived_slug(spec.model.split("/")[-1].lower())
        count = seen.get(base, 0)
        out.append(f"{base}-{count}" if count else base)
        seen[base] = count + 1
    return out


_STREAM_PARTIAL_INTERVAL_S_DEFAULT = 1.0


def _stream_partial_interval_s() -> float:
    """Read at call time so test monkeypatching of the env var works."""
    return float(os.environ.get("CONSULT_STREAM_PARTIAL_INTERVAL_S", _STREAM_PARTIAL_INTERVAL_S_DEFAULT))


def _max_input_tokens(litellm_id: str, entry: dict[str, Any]) -> int | None:
    """Return the model's maximum input-token budget, or None if unknown.

    Lookup order:
    1. `entry["max_input_tokens"]` — explicit registry override. Use when
       LiteLLM's table is wrong or stale for a model.
    2. `litellm.get_model_info(litellm_id)["max_input_tokens"]` — LiteLLM
       maintains this for most known models. Returns None when missing.
    3. None — unknown context size; the pre-flight check below skips
       gracefully (provider will reject the call if it's too large, same
       as the prior behaviour).

    Kept narrow on purpose: we don't want a broad provider-capability layer
    here, just enough to surface "your prompt is too big" as a manifest
    Status.ERROR rather than as a raw provider BadRequestError that the
    user has to decode.
    """
    override = entry.get("max_input_tokens")
    if override is not None:
        try:
            return int(override)
        except (TypeError, ValueError):
            logger.warning(
                "registry max_input_tokens for %s is not an int: %r",
                litellm_id,
                override,
            )
    try:
        info = litellm.get_model_info(litellm_id)
    except Exception:  # noqa: BLE001 — get_model_info raises on unknown IDs
        return None
    if isinstance(info, dict):
        v = info.get("max_input_tokens")
        if isinstance(v, int) and v > 0:
            return v
    return None


def _replace_attachments_with_stubs(
    prompt: str,
    paths: artifacts.RunPaths,
    available_chars: int,
) -> str:
    """Drop oversized inlined attachment blocks largest-first, replacing
    each with a short stub that references the persisted resource URI.

    Stops as soon as the prompt fits `available_chars`. Returns the
    prompt unchanged when there are no parseable attachment blocks or
    when no single drop would help (a block smaller than its stub isn't
    worth dropping). The caller still re-counts tokens after this —
    char→token is approximate, so this is a fast pre-filter, not the
    final budget check.

    Why drop whole blocks instead of head+tail-slicing through them:
    half a source file with `[TRIMMED 50000 chars from middle]` in the
    middle is worse than useless for a code reviewer — they can't trust
    any claim about the body. A clean "this file was too big, here's
    where to read it" stub lets the panellist reason about what it
    can't see rather than pretending the partial view is complete.
    """
    blocks = attachments_mod.extract_inlined_blocks(prompt)
    if not blocks:
        return prompt
    used: set[str] = set()
    named = [(b, attachments_mod.safe_attachment_name(b, used)) for b in blocks]

    def _stub(block: attachments_mod.InlinedBlock, name: str) -> str:
        uri = paths.attachment_resource_uri(name)
        line_count = block.content.count("\n") + 1
        return (
            f"{block.header}\n"
            f"[Attachment dropped to fit context: {line_count:,} lines, "
            f"{len(block.content):,} chars. Full source at {uri}]"
        )

    # Largest-first drop priority. We commit each drop only if it
    # actually shrinks the prompt — pathological case: a 60-char block
    # whose stub is 200 chars is not worth dropping.
    drop_priority = sorted(
        range(len(named)),
        key=lambda i: -(named[i][0].end - named[i][0].start),
    )
    dropped: set[int] = set()
    current_chars = len(prompt)
    for idx in drop_priority:
        if current_chars <= available_chars:
            break
        block, name = named[idx]
        block_len = block.end - block.start
        stub_len = len(_stub(block, name))
        if stub_len >= block_len:
            continue  # would grow the prompt
        dropped.add(idx)
        current_chars -= block_len - stub_len

    if not dropped:
        return prompt

    parts: list[str] = []
    last_end = 0
    for i, (block, name) in enumerate(named):
        parts.append(prompt[last_end : block.start])
        parts.append(_stub(block, name) if i in dropped else prompt[block.start : block.end])
        last_end = block.end
    parts.append(prompt[last_end:])
    logger.info(
        "attachment-aware trim: dropped %d/%d blocks (%d chars → ~%d chars)",
        len(dropped),
        len(named),
        len(prompt),
        current_chars,
    )
    return "".join(parts)


async def _fit_prompt_to_context(
    per_slug_prompt: str,
    *,
    paths: artifacts.RunPaths | None = None,
    prior_turns: list[dict[str, Any]] | None,
    litellm_id: str,
    max_input_tokens: int,
    max_output_tokens: int,
) -> tuple[str, int]:
    """Trim `per_slug_prompt` so input+output fits the model's context.

    Returns `(maybe_trimmed_prompt, dropped_chars)`. `dropped_chars == 0`
    means no trim happened (prompt already fit, or budget made trimming
    impossible). Callers use the explicit count to surface a quantified
    trim note in the manifest — sniffing the prompt for a marker substring
    false-positives on source-code attachments that contain the word.

    Trim strategy: head+tail truncates the prompt via `context.trim_text`
    (preserves the stance preface at the head and the CONFIDENCE/KEY_REASON
    footer at the tail) and re-counts; if still over budget after one pass
    (rare — usually means `prior_turns` alone exceed the budget), shrinks
    further via a smaller char target.

    The `prior_turns` text is included in the token count but never
    trimmed — those are the prior consultation's role-separated exchange
    in a `refine` continuation, and trimming them would corrupt the
    user/assistant boundary the model relies on. If they alone exceed the
    budget the caller should re-prompt without continuation.
    """
    from . import context

    prior_text = concat_turn_text(prior_turns) if prior_turns else ""
    target_input = max_input_tokens - max_output_tokens
    if target_input <= 0:
        # Defensive: the registry's `default_budget_tokens` shouldn't ever
        # be larger than the model's whole context, but if it is, return
        # the prompt as-is and let the provider reject — the caller's
        # config is the real bug.
        return per_slug_prompt, 0

    async def _count(text: str) -> int:
        # token_counter is sync + CPU-bound; offload so we don't block the
        # event loop during the pre-flight check.
        try:
            return int(
                await asyncio.to_thread(
                    litellm.token_counter,
                    model=litellm_id,
                    text=text,
                )
            )
        except Exception:  # noqa: BLE001
            return -1  # unknown — caller treats as "skip the check"

    prior_tokens = await _count(prior_text) if prior_text else 0
    if prior_tokens < 0:
        return per_slug_prompt, 0  # token_counter is broken; let provider decide
    available_for_prompt = target_input - prior_tokens
    if available_for_prompt <= 0:
        # prior_turns alone exceed the budget. We don't trim prior_turns
        # (would corrupt role boundaries); log and pass through so the
        # caller sees the provider's rejection with the real reason.
        logger.warning(
            "fit_prompt: prior_turns alone (%d tokens) exceed available input "
            "budget (%d). Returning prompt untrimmed; provider will reject.",
            prior_tokens,
            target_input,
        )
        return per_slug_prompt, 0

    prompt_tokens = await _count(per_slug_prompt)
    if prompt_tokens < 0:
        return per_slug_prompt, 0
    if prompt_tokens <= available_for_prompt:
        return per_slug_prompt, 0  # already fits, no-op

    original_len = len(per_slug_prompt)
    # Attachment-aware shrink first (when we have a run dir to anchor
    # resource URIs to). Drops whole attachment blocks largest-first,
    # replaces each with a stub pointing at the persisted resource.
    # Better signal than head+tail slicing through code files. The 4x
    # char-per-token rule of thumb sets the char target — final budget
    # check is the re-count below.
    if paths is not None:
        avail_chars = available_for_prompt * 4
        shrunk = _replace_attachments_with_stubs(
            per_slug_prompt,
            paths,
            avail_chars,
        )
        if shrunk is not per_slug_prompt:
            per_slug_prompt = shrunk
            prompt_tokens = await _count(per_slug_prompt)
            if prompt_tokens < 0:
                return per_slug_prompt, original_len - len(per_slug_prompt)
            if prompt_tokens <= available_for_prompt:
                return per_slug_prompt, original_len - len(per_slug_prompt)

    # Still over budget — fall back to head+tail trim. token_counter
    # <-> char-count is approximate; aim for 90% of the available
    # budget so a recount comes in under cleanly. The 500-char floor
    # prevents pathological "shrink to nothing" outcomes on tiny-context
    # models. Cap iterations at 3 — usually one pass suffices.
    trimmed = per_slug_prompt
    for _attempt in range(3):
        ratio = (available_for_prompt * 0.9) / prompt_tokens
        target_chars = max(500, int(len(trimmed) * ratio))
        if target_chars >= len(trimmed):
            # Can't shrink further without breaking the floor — return what
            # we have and let the provider reject (or accept) the call.
            break
        trimmed = context.trim_text(
            trimmed,
            target_chars,
            label="panellist prompt",
        )
        prompt_tokens = await _count(trimmed)
        if prompt_tokens < 0 or prompt_tokens <= available_for_prompt:
            break
    dropped = max(0, original_len - len(trimmed))
    return trimmed, dropped


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
            f"stream_chunk_builder failed after {len(chunks)} chunks: {type(e).__name__}: {e}"
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
    capsule_kind: str = "decision",
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
    # CLI panellists don't have a LiteLLM ID — they invoke a subprocess.
    # Use `litellm_id` when present, otherwise fall back to the spec's
    # alias for logging/diagnostics so error messages stay attributable.
    litellm_id = entry.get("litellm_id") or spec.model
    # Output budget is sized by what we're asking for (the capsule_kind),
    # not by which model is answering. A "review" needs ~8K tokens of
    # body to enumerate findings regardless of whether haiku or opus is
    # writing it; previously the per-model default_budget_tokens (4K on
    # haiku) silently truncated reviews on small models.
    budget = MAX_TOKENS_BY_KIND.get(capsule_kind, MAX_TOKENS_BY_KIND["decision"])
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

    # Pre-flight context-budget check + auto-trim. Each provider rejects
    # calls that exceed its context window with a hard 400; without this
    # we'd burn latency (and a per-provider rate-limit slot) to get back
    # a raw BadRequestError that surfaces in the manifest as a wall of
    # provider JSON.
    #
    # When the prompt would overflow, trim per_slug_prompt head+tail
    # (preserving the stance preface and the CONFIDENCE/KEY_REASON
    # footer — both live in the head/tail respectively) so the call
    # succeeds with a slightly-reduced view of the source material.
    # Trimming is logged at warning level and noted in the manifest's
    # `error` field even on success ("trimmed N chars..."), so the
    # caller can see that this panellist saw less than the others.
    #
    # Skipped silently when the model's context size is unknown — falls
    # back to the prior "let the provider reject it" behaviour for
    # unfamiliar models.
    trim_note: str | None = None
    max_in = _max_input_tokens(litellm_id, entry)
    if max_in is not None:
        original_chars = len(per_slug_prompt)
        per_slug_prompt, dropped_chars = await _fit_prompt_to_context(
            per_slug_prompt,
            paths=paths,
            prior_turns=prior_turns,
            litellm_id=litellm_id,
            max_input_tokens=max_in,
            max_output_tokens=budget,
        )
        # `_fit_prompt_to_context` reports dropped chars explicitly — we
        # used to sniff for a `[TRIMMED` substring, which false-positived
        # on source-code attachments that mention the word. Now the
        # manifest note is both accurate (no false alarms when we didn't
        # actually trim) and quantified (drops and budget surfaced).
        if dropped_chars > 0:
            pct = (dropped_chars / original_chars * 100) if original_chars else 0.0
            trim_note = (
                f"input auto-trimmed: dropped ~{dropped_chars:,} chars "
                f"(~{pct:.0f}% of {original_chars:,}c source) to fit "
                f"{max_in:,}-token context (reserved {budget:,} tok for output)"
            )

    # OpenTelemetry span per panellist call. No-op when OTel isn't
    # installed or the user hasn't set OTEL_EXPORTER_OTLP_ENDPOINT.
    # Follows the gen_ai.* semantic conventions so consult shows up in
    # off-the-shelf AI-observability dashboards without a custom mapping.
    # Wraps the whole try/except so cost/tokens/finish_reason can be set
    # after the response comes back, the exception handlers can record
    # errors on the span, and the span is guaranteed to close.
    otel_span_name = f"gen_ai.chat {litellm_id}"
    otel_attrs = {
        "gen_ai.system": provider or "unknown",
        "gen_ai.request.model": litellm_id,
        "gen_ai.operation.name": "chat",
        "app.consult.slug": slug,
        "app.consult.run_id": paths.run_id,
    }
    with telemetry.span(otel_span_name, attributes=otel_attrs) as tspan:
        try:
            async with sem if sem is not None else nullcontext():
                messages = build_messages(per_slug_prompt, provider, prior_turns)
                if provider == "cli":
                    # CLI panellists bypass LiteLLM entirely: spawn the
                    # configured executable, send the prompt on stdin,
                    # capture stdout. Cost is zero (the CLI's own auth
                    # covers usage); per-provider semaphore still applies
                    # if the registry configures one for "cli".
                    from . import cli_executor

                    cli_command = entry.get("cli_command") or []
                    if not cli_command:
                        raise ValueError(
                            f"CLI provider for {spec.model!r} has no cli_command in the registry entry"
                        )
                    resp = await cli_executor.call_cli(
                        cli_command,
                        per_slug_prompt,
                        timeout=timeout,
                        extra_env=entry.get("cli_env"),
                    )
                elif stream:
                    resp = await _stream_acompletion(
                        timeout=timeout,
                        on_partial=on_partial,
                        start=start,
                        model=litellm_id,
                        messages=messages,
                        max_completion_tokens=budget,
                        **extra,
                    )
                else:
                    resp = await _acompletion_with_retry(
                        timeout=timeout,
                        model=litellm_id,
                        messages=messages,
                        max_completion_tokens=budget,
                        **extra,
                    )
            # Persist raw response — use model_dump for Pydantic, fall
            # back to dict
            try:
                raw = resp.model_dump()  # type: ignore[attr-defined]
            except AttributeError:
                raw = dict(resp) if hasattr(resp, "__iter__") else {"_repr": repr(resp)}
            await _write_text_async(paths.response_raw(slug), json.dumps(raw, indent=2, default=str))

            status, finish, body = classify(resp)
            usage = getattr(resp, "usage", None)
            if usage:
                tokens_in = getattr(usage, "prompt_tokens", None)
                tokens_out = getattr(usage, "completion_tokens", None)
            if provider == "cli":
                # CLI panellists are free at the per-call level. Skip
                # LiteLLM's cost lookup (it would error on the synthetic
                # response object built by `cli_executor.call_cli`).
                cost = 0.0
                cost_known = True
            else:
                try:
                    cost = litellm.completion_cost(completion_response=resp)
                    cost_known = cost is not None
                except Exception as ce:  # noqa: BLE001
                    logger.warning("cost lookup failed for %s: %s", litellm_id, ce)
                    cost = None
                    cost_known = False
            # Populate the OTel span with per-call telemetry (tokens /
            # cost / finish_reason). No-op when the span is None.
            if tokens_in is not None:
                telemetry.set_attribute(tspan, "gen_ai.usage.input_tokens", tokens_in)
            if tokens_out is not None:
                telemetry.set_attribute(tspan, "gen_ai.usage.output_tokens", tokens_out)
            if cost is not None:
                telemetry.set_attribute(tspan, "app.consult.cost_usd", cost)
            telemetry.set_attribute(tspan, "app.consult.cost_known", cost_known)
            if finish:
                telemetry.set_attribute(tspan, "gen_ai.response.finish_reasons", [finish])
            telemetry.set_attribute(tspan, "app.consult.status", status.value)

        except TimeoutError as te:
            status, finish, body = Status.TIMEOUT, None, ""
            error = f"timeout after {timeout}s"
            # Conservative: a TimeoutError from `asyncio.wait_for` means
            # we gave up waiting, NOT that the HTTP request never landed.
            # The provider may have processed and billed us;
            # cost_known=False surfaces that uncertainty in the ledger
            # as a lower bound rather than silently understating spend.
            # Mirrors slow-tail dropout.
            cost = None
            cost_known = False
            telemetry.record_exception(tspan, te)
            telemetry.set_attribute(tspan, "app.consult.status", status.value)
        except Exception as e:  # noqa: BLE001 — LiteLLM raises many concrete types
            status, finish, body = classify(None, exception=e)
            error = _format_error_message(e)
            # Same logic: most provider exceptions imply no billable
            # call, but we can't be sure for every case (e.g. 502
            # mid-stream may have billed). cost_known=False is the
            # conservative encoding.
            cost = None
            cost_known = False
            telemetry.record_exception(tspan, e)
            telemetry.set_attribute(tspan, "app.consult.status", status.value)

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
    append_progress_log(
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

    # Trim note goes to `note` (info annotation on a successful call),
    # NOT `error` (failure reason). Conflating them made the viewer render
    # the trim message in red error styling on a green-OK card, which read
    # like a contradiction.
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
        note=trim_note,
        confidence=None,  # populated by capsule extractor
        capsule=None,
    )


async def aestimate_cost(
    specs: list[ModelSpec],
    prompt: str,
    *,
    capsule_kind: str = "decision",
) -> tuple[float, bool]:
    """Async wrapper around `estimate_cost`.

    `litellm.token_counter` is blocking and on a cache miss takes 50-200ms
    while it loads the tokenizer; large panels with new aliases can stall
    heartbeat ticks for noticeable real-time. Offloading to a thread keeps
    the event loop responsive. Test monkeypatches still bind to the sync
    `estimate_cost` symbol — this wrapper picks up whatever's currently
    bound there, so test setup is unchanged.
    """
    return await asyncio.to_thread(estimate_cost, specs, prompt, capsule_kind=capsule_kind)


def estimate_cost(
    specs: list[ModelSpec],
    prompt: str,
    *,
    capsule_kind: str = "decision",
) -> tuple[float, bool]:
    """Returns (total_estimate, all_known).

    Uses LiteLLM's per-token price tables via `cost_per_token()`. Models that
    don't have pricing data set `all_known=False`; caller must treat unknown
    costs conservatively (a panel with even one unknown-cost spec cannot be
    validated against `max_run_usd`).

    Unknown-alias specs are treated as cost-unknown (rather than raising) so
    a single typo can't abort the panel here; `_call_one` surfaces the alias
    as a per-spec Status.ERROR.

    Sync by design so tests can monkeypatch it with a plain lambda; the
    async fan-out paths call `aestimate_cost()` to keep the event loop
    free during the blocking `token_counter` lookup.
    """
    total = 0.0
    all_known = True
    for spec in specs:
        try:
            entry = registry.resolve_model(spec.model)
        except KeyError:
            all_known = False
            continue
        # CLI panellists have no per-call dollar cost: the user's CLI
        # auth covers usage. Estimating them as $0 is honest (not
        # cost-unknown — that would inappropriately make the cap-check
        # conservative for what is genuinely free at this layer).
        if entry.get("provider") == "cli":
            continue
        litellm_id = entry["litellm_id"]
        try:
            tin = litellm.token_counter(model=litellm_id, text=prompt)
            # Match _call_one: estimate output by capsule_kind, not per-model
            # default. Keeps the cap-check honest after the dimension flip.
            tout = MAX_TOKENS_BY_KIND.get(capsule_kind, MAX_TOKENS_BY_KIND["decision"])
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
    prior_turns_by_slug: dict[str, list[dict[str, Any]]] | None = None,
    max_concurrency: int | None = None,
) -> RunHandle:
    """Parallel fan-out. Creates a fresh run by default. Pass `existing_paths`
    to write into an existing run dir (used by `refine` to keep all rounds
    under one run_id with round-suffixed slugs).

    `prior_turns`, when set, is a sequence of `{role, content}` dicts
    prepended to the messages array for every panellist call. Used by
    `refine` with a `continuation_id` to expose the prior consultation as
    a proper user/assistant exchange. The text is included in cost
    estimation so the cap check stays accurate.

    `prior_turns_by_slug`, when set, is a per-slug override of `prior_turns`.
    A slug present in the dict uses its dict value; a slug absent falls
    back to `prior_turns`. Used by `refine` round-2+ to give each
    panellist its OWN conversation history (its prior question +
    answer) — round-1 question + answer become a stable prefix that
    Anthropic's prompt cache can reuse across rounds, instead of the
    monolithic refinement prompt that changes every round.

    `max_concurrency` (or `CONSULT_MAX_CONCURRENCY` env var) caps the
    *total* number of panellists in flight at once. The existing
    per-provider semaphores in `_get_provider_sems()` cap concurrency
    *per provider*; on the deep tier (~14 models across ~6 providers)
    they don't bound the aggregate, so all 14 calls launch simultaneously
    and 14 inbound HTTP connections + 14 token-counter cache misses fire
    at once. A global cap (default uncapped, recommended ~5-8 for wide
    panels) smooths this without changing per-provider behaviour.

    If `on_progress` is set, it's called once per panellist as it completes
    with `(done, total, message)`. Failures inside the callback are logged
    and swallowed — progress is best-effort, not load-bearing.
    """
    # Resolve `model:N` sugar BEFORE estimate_cost so the cap reflects the
    # real panel size, not the pre-expansion request count.
    specs = expand_specs(specs)
    # An empty panel would create a bizarre run dir, fan out to zero
    # panellists, and return a manifest=[] handle that downstream tools
    # treat as "all panellists rate-limited". The MCP schema enforces
    # `minItems: 1` for callers going through the wire, but library
    # callers (sequence, refine, custom drivers) need their own gate.
    if not specs:
        raise ValueError("fanout requires at least one model spec")
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
        # Split inlined attachments out to `paths.attachments/<name>` so
        # (a) the per-panellist trim stub can reference a resolvable
        # resource URI and (b) the report and any tool-using downstream
        # model can still read the original source. Best-effort: a parse
        # failure leaves the panellist call unaffected — they still see
        # the inlined blocks in the prompt.
        try:
            attachments_mod.persist_inlined_attachments(paths, prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("persist_inlined_attachments failed: %s", e)
    else:
        paths = existing_paths

    # Estimate cost up front; if dry_run, return immediately with empty
    # manifest. When `prior_turns` is set (continuation), include their
    # text in the token count so the cap check sees the real input size.
    cost_input = prompt
    if prior_turns:
        cost_input = concat_turn_text(prior_turns) + "\n" + prompt
    estimate, all_known = await aestimate_cost(specs, cost_input, capsule_kind=capsule_kind)
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()

    # Don't clobber an existing manifest with the empty-manifest early-return
    # payload. Refine drives multiple rounds through the same `paths`; an
    # over-cap or dry-run rejection on round N+1 would otherwise wipe out
    # round N's successful manifest. New run dirs (existing_paths is None)
    # still get the manifest so downstream tools — `consult-view`,
    # `consult-ledger`, `synth.synthesise` — never face FileNotFoundError on
    # a dry-run / cap-rejected dir.
    def _persist_partial_handle(handle: RunHandle) -> None:
        if existing_paths is None or not paths.manifest_json.exists():
            artifacts.write_manifest(paths, handle.model_dump())

    if estimate > cap:
        # When some prices are unknown, `estimate` is only the known-priced
        # portion; the actual run could cost more. Surface that so the cap
        # message isn't misleading low. Mirrors the dry_run branch below.
        suffix = "" if all_known else " (known-priced portion only; some unknown)"
        handle = RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=(f"estimated cost ${estimate:.2f}{suffix} exceeds cap ${cap:.2f}"),
            blinded=blinded,
        )
        _persist_partial_handle(handle)
        return handle
    if dry_run:
        suffix = "" if all_known else " (some prices unknown — actual cost may differ)"
        handle = RunHandle(
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
        _persist_partial_handle(handle)
        return handle

    # Build slugs + prompts
    slugs = _make_slugs(specs, blinded)
    # Duplicate slugs ⇒ multiple panellists racing to write to the same
    # `responses/<slug>.txt`; the second writer silently overwrites the
    # first. `expand_specs` fixes the model:N case but a caller passing
    # two literal `{slug: "foo"}` specs falls through. Fail fast here
    # before the artifact dance starts.
    if len(set(slugs)) != len(slugs):
        seen: dict[str, int] = {}
        for s in slugs:
            seen[s] = seen.get(s, 0) + 1
        dupes = sorted(slug for slug, n in seen.items() if n > 1)
        raise ValueError(
            f"duplicate panel slugs would race the artifact dir: {dupes}. "
            "Disambiguate by giving each spec a distinct `slug` (or omitting it)."
        )
    per_prompts = [_build_per_slug_prompt(prompt, registry.resolve_stance(s.stance)) for s in specs]

    # Process-level provider semaphores (shared across concurrent fanouts
    # in the same event loop). See `_get_provider_sems` for the rationale.
    provider_sems = _get_provider_sems()

    # Total in-flight cap across the whole fanout. None = uncapped (rely on
    # per-provider sems alone). Resolved at call time so a test that sets
    # the env var per case works.
    if max_concurrency is None:
        env_cap = os.environ.get("CONSULT_MAX_CONCURRENCY", "").strip()
        if env_cap:
            try:
                parsed = int(env_cap)
                if parsed >= 1:
                    max_concurrency = parsed
            except ValueError:
                logger.warning(
                    "CONSULT_MAX_CONCURRENCY=%r is not a positive int; ignoring",
                    env_cap,
                )
    fanout_sem: asyncio.Semaphore | None = asyncio.Semaphore(max_concurrency) if max_concurrency else None

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
    append_progress_log(paths.root, phase_event)
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
                append_progress_log(paths.root, event)
                await _safe_notify(event)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

    async def _run_one(spec: ModelSpec, slug: str, per_prompt: str) -> ManifestEntry:
        nonlocal done, started
        # Acquire the global fanout slot BEFORE emitting PanellistStarted so
        # the started_count reflects "doing real work", not "queued". The
        # per-provider semaphores inside `_call_one` are still acquired below
        # — this gate is additive, never replacing them. `nullcontext()` is
        # async-compatible (Python 3.10+), same shape as Semaphore.
        async with fanout_sem if fanout_sem is not None else nullcontext():
            # PanellistStarted: fires BEFORE the LiteLLM call so the parent
            # sees which slugs are in flight, not just which have completed.
            started += 1
            started_event = PanellistStarted(
                done=done,
                total=total,
                slug=slug,
                started_count=started,
            )
            append_progress_log(paths.root, started_event)
            await _safe_notify(started_event)

            # When streaming is enabled, wire each panellist's mid-stream chunk
            # callback to emit `PanellistPartial` events. Throttled to
            # ~1 chunk/sec by `_STREAM_PARTIAL_INTERVAL_S` so the progress
            # channel doesn't drown in micro-updates.
            on_partial: Callable[[int, int], Awaitable[None]] | None = None
            if stream and on_progress is not None:

                async def _emit_partial(chars: int, elapsed_ms: int) -> None:
                    await _safe_notify(
                        PanellistPartial(
                            done=done,
                            total=total,
                            slug=slug,
                            chars_so_far=chars,
                            elapsed_ms=elapsed_ms,
                        )
                    )

                on_partial = _emit_partial
            # Per-slug history takes precedence when set (refine round-2+
            # passes each panellist its own conversation). Falls back to
            # the global `prior_turns` (set by `_apply_continuation` for
            # cross-run continuations) when the slug isn't in the dict.
            pt = prior_turns_by_slug.get(slug) if prior_turns_by_slug else None
            if pt is None:
                pt = prior_turns
            entry = await _call_one(
                spec,
                slug,
                per_prompt,
                paths,
                provider_sems,
                stream=stream,
                on_partial=on_partial,
                prior_turns=pt,
                capsule_kind=capsule_kind,
            )
            done += 1
            completed_entries.append(entry)
            pending_slugs_set.discard(slug)
            await _safe_notify(
                PanellistCompleted(
                    done=done,
                    total=total,
                    slug=slug,
                    status=entry.status.value,
                    latency_ms=entry.latency_ms,
                )
            )
            return entry

    # Slow-tail dropout: once most of the panel has returned, cancel the
    # slowest stragglers rather than waiting for the per-spec timeout. FRICTION
    # observed a 5× spread between fastest and slowest in the same panel; the
    # median wall-time win comes from releasing 1-2 laggards. Disabled for
    # small panels (no useful "rest of the panel" signal) and when
    # CONSULT_TAIL_DROPOUT_S=0. The full per-spec timeout still bounds the
    # worst case if dropout is off.
    #
    # Default 180s (was 30s): long-context code reviews on wide panels
    # produce genuinely useful capsules from slower models (kimi, qwen,
    # deepseek often take 60-180s on ~200K input). The previous 30s was
    # firing on most wide-panel runs and dropping real signal.
    tail_dropout_s = float(os.environ.get("CONSULT_TAIL_DROPOUT_S", 180.0))
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
                done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                completed_tasks.update(done_set)

            if pending:
                logger.info(
                    "slow-tail dropout: %d/%d complete, waiting up to %.1fs for %d stragglers",
                    len(completed_tasks),
                    total,
                    tail_dropout_s,
                    len(pending),
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
                await _safe_notify(
                    PanellistCompleted(
                        done=done,
                        total=total,
                        slug=slug,
                        status=Status.TIMEOUT.value,
                        latency_ms=latency_ms,
                    )
                )
                append_progress_log(
                    paths.root,
                    PanellistCompleted(
                        done=0,
                        total=0,
                        slug=slug,
                        status=Status.TIMEOUT.value,
                        latency_ms=latency_ms,
                    ),
                )

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

    # `blinded=True` controls what panellists see of each other DURING the
    # run (the brand-scrub in `context.build` and the greek-letter slugs in
    # `_make_slug`); the synth's `anonymised` switch additionally selects
    # the brand-scrubbed prompt for the synthesiser's view of the original
    # question. Panellist labels are blinded to the synth unconditionally
    # via blind labels (Alpha/Beta/...). The manifest itself keeps real
    # model_ids so the final report (viewer, ledger) can surface them to
    # the human reader. Earlier code scrubbed manifest model_ids here,
    # which leaked the blinding past its useful boundary.

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
