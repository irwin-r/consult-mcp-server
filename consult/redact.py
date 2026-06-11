"""Redact secret-shaped tokens from any string that leaves the process.

LiteLLM frequently includes the upstream HTTP response body / request
headers in the exception it raises when a provider call fails, and those
bodies routinely echo back the `Authorization: Bearer sk-…` header (or the
provider-specific equivalent). Without redaction, that exception text can
reach three boundaries that must never carry a working key:

- **disk** under `~/.consult/runs/<id>/` (manifest, `synthesis.md`, arbiter
  verdicts),
- **the tool result** returned to the MCP client (a parent agent's
  transcript), and
- **the log file** at `~/.consult/logs/`.

The patterns and `redact_secrets` used to live in `runner.py`, where
redaction was only applied at the manifest boundary. They moved here so
every boundary can share one redactor. This module imports nothing from the
rest of the package, so any module can import it without a cycle.

A note on tracebacks: the standard `logger.exception(...)` (and
`exc_info=True`) renders the exception's repr via the handler's formatter,
which we don't own and can't redact. So the log-boundary callers format the
traceback themselves with `redact_traceback` and log the redacted string as
a plain message. That loses the structured `exc_info` tuple some log
aggregators key on, but it keeps the stack frames in the message body and
guarantees the key never renders. The threat model makes that trade worth
it.
"""

from __future__ import annotations

import contextlib
import logging
import re
import traceback

# Secret-shaped tokens that may end up embedded in LiteLLM exception strings.
# The header patterns are case-insensitive because a JSON response body often
# echoes a lowercase `authorization` / `x-api-key` key. The bare-token patterns
# (sk-…, AIza…, sk-or-…) are the load-bearing safety net: they match the key
# itself regardless of the surrounding header syntax (or its absence).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"sk-or-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"or-v1-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)Authorization\s*[:=]\s*Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    re.compile(r"(?i)x-api-key\s*[:=]\s*[A-Za-z0-9_\-\.]{20,}"),
    re.compile(r'(?i)["\']?api[_-]?key["\']?\s*[:=]\s*["\'][A-Za-z0-9_\-\.]{20,}["\']'),
)


def redact_secrets(text: str) -> str:
    """Replace API-key-shaped tokens with `[REDACTED]`.

    Defence-in-depth: LiteLLM's exception text often embeds the raw HTTP
    response, which on auth-failure paths can carry the request
    `Authorization` header verbatim. Redacting here ensures a leaked
    manifest, `synthesis.md`, arbiter verdict, or log line never carries a
    working key.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def redact_exc(exc: BaseException, *, limit: int | None = None) -> str:
    """Format `exc` as `"TypeName: message"`, redacted, optionally truncated.

    Redaction runs *before* truncation. If you truncate first, a key that
    starts near the cut point gets sliced into a fragment shorter than the
    pattern's `{20,}` floor and slips through unmasked. So redact the full
    string, then cap.
    """
    text = redact_secrets(f"{type(exc).__name__}: {exc}")
    if limit is not None and limit > 0 and len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def redact_traceback(exc: BaseException) -> str:
    """Render `exc`'s full traceback (including chained causes) and redact it.

    `traceback.format_exception` follows `__cause__` / `__context__`, so a
    secret riding in a chained exception's message is covered too. Callers
    log the result as a plain message instead of passing `exc_info`, since
    the handler's formatter would re-render the unredacted repr otherwise.
    """
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return redact_secrets(tb)


# Exception attributes that LiteLLM (and the openai exceptions it wraps)
# populate with upstream HTTP material. `message` and `body` regularly echo
# the request's Authorization header on auth-failure paths;
# `litellm_debug_info` carries the provider/request summary LiteLLM builds
# for its "Give Feedback" footer.
_EXC_TEXT_ATTRS = ("message", "body", "litellm_debug_info")


def scrub_exception_attrs(exc: BaseException) -> BaseException:
    """Redact secret-shaped tokens on the exception OBJECT, in place.

    `redact_exc` / `redact_traceback` scrub the strings consult itself
    produces, but the exception object keeps its raw attributes, and any
    later consumer that formats the object — the OTel exporter via
    `record_exception`, a crash reporter, the host application's own
    logging — gets the unscrubbed text. Scrubbing `args` also makes plain
    `str(exc)` safe at the source.

    Best-effort by design: read-only attributes (e.g. an `httpx.Response`
    under `exc.response`, whose `.text` is a property) are left alone and
    remain covered by the string-boundary redaction. Returns `exc` for
    chaining.
    """
    for attr in _EXC_TEXT_ATTRS:
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            redacted: object = redact_secrets(value)
        elif isinstance(value, dict):
            redacted = {k: redact_secrets(v) if isinstance(v, str) else v for k, v in value.items()}
        else:
            continue
        try:
            setattr(exc, attr, redacted)
        except Exception:  # noqa: BLE001 — read-only attribute; string boundaries still apply
            continue
    with contextlib.suppress(Exception):  # exotic args tuples; string boundaries still apply
        exc.args = tuple(redact_secrets(a) if isinstance(a, str) else a for a in exc.args)
    return exc


class SecretRedactingFilter(logging.Filter):
    """Logging filter that redacts key-shaped tokens in every record.

    Covers loggers this package doesn't own. With `LITELLM_LOG=DEBUG`,
    LiteLLM's own logger interpolates request kwargs — headers included —
    into its messages, and consult's redact-at-the-call-site discipline
    can't reach those. Attached to a logger (and its handlers), the
    rendered message is scrubbed before any formatter sees it.

    The record's args are collapsed into the formatted message first:
    redacting `msg` alone would miss a key carried in an arg, and editing
    `msg` while keeping `args` would break the next %-format.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — a broken record must not kill logging
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def install_redaction_filter(*logger_names: str) -> None:
    """Attach `SecretRedactingFilter` to each named logger and its handlers.

    Idempotent per logger/handler. Handler-level attachment matters:
    logger-level filters only apply to records emitted through that exact
    logger, not to records propagating up from children, while the
    handlers see both. Pass `""` for the root logger — with the root's
    handlers filtered, everything that propagates (the whole `consult.*`
    tree included) is covered centrally.
    """
    for name in logger_names:
        target = logging.getLogger(name) if name else logging.getLogger()
        if not any(isinstance(f, SecretRedactingFilter) for f in target.filters):
            target.addFilter(SecretRedactingFilter())
        for handler in target.handlers:
            if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
                handler.addFilter(SecretRedactingFilter())
