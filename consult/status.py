"""Classify a LiteLLM response (or exception) into a Status.

Exception classification uses LiteLLM's typed exception hierarchy where
available (`litellm.exceptions.*`) and falls back to substring matching on
the message only for exception types LiteLLM doesn't model. Substring
matching was the v1 default; it mis-classifies (e.g. socket "connection rate
limit reset" → RATE_LIMITED) so we prefer typed checks first.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .types import Status

logger = logging.getLogger(__name__)


def _classify_exception(exc: BaseException) -> Status:
    # LiteLLM exposes typed exceptions; import lazily so tests don't require it
    try:
        from litellm import exceptions as lex
    except Exception:  # pragma: no cover — litellm always available in production
        lex = None  # type: ignore[assignment]

    if lex is not None:
        # Order matters: Timeout is sometimes a subclass of APIConnectionError
        timeout_classes: tuple[type, ...] = (asyncio.TimeoutError, TimeoutError)
        for name in ("Timeout", "APITimeoutError"):
            cls = getattr(lex, name, None)
            if cls:
                timeout_classes = (*timeout_classes, cls)
        if isinstance(exc, timeout_classes):
            return Status.TIMEOUT
        rate_cls = getattr(lex, "RateLimitError", None)
        if rate_cls and isinstance(exc, rate_cls):
            return Status.RATE_LIMITED
        cpv_cls = getattr(lex, "ContentPolicyViolationError", None)
        if cpv_cls and isinstance(exc, cpv_cls):
            return Status.CONTENT_FILTERED
        # Auth / BadRequest / NotFound / ContextWindow → ERROR
        # (these are configuration bugs, not transient — caller should see them
        # as ERROR and inspect the message)
        for name in (
            "AuthenticationError",
            "BadRequestError",
            "NotFoundError",
            "ContextWindowExceededError",
            "InvalidRequestError",
            "PermissionDeniedError",
        ):
            cls = getattr(lex, name, None)
            if cls and isinstance(exc, cls):
                return Status.ERROR
        # ServiceUnavailableError / InternalServerError / APIConnectionError →
        # treat as transient ERROR (caller may retry)
        for name in (
            "ServiceUnavailableError",
            "InternalServerError",
            "APIConnectionError",
        ):
            cls = getattr(lex, name, None)
            if cls and isinstance(exc, cls):
                return Status.ERROR

    # Plain Python timeouts (asyncio.wait_for) still need to map to TIMEOUT
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return Status.TIMEOUT

    # Fallback: substring matching for genuinely unknown exception types.
    msg = str(exc).lower()
    if "rate" in msg and "limit" in msg:
        return Status.RATE_LIMITED
    if "timeout" in msg or "timed out" in msg:
        return Status.TIMEOUT
    if "content_filter" in msg or "content policy" in msg:
        return Status.CONTENT_FILTERED
    return Status.ERROR


def classify(
    response: Any | None,
    exception: BaseException | None = None,
) -> tuple[Status, str | None, str]:
    """Return (status, finish_reason, body_text).

    Designed to be tolerant of provider quirks. When `content` is empty but
    the model put its text in `reasoning_content`/`reasoning` (OR thinking
    models), the reasoning text is salvaged as the body; bodies that are
    still mostly whitespace after that are classified as EMPTY.
    """
    if exception is not None:
        return _classify_exception(exception), None, ""

    if response is None:
        return Status.EMPTY, None, ""

    # LiteLLM normalises to OpenAI shape. Try the common path first.
    try:
        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", None) or getattr(choice.message, "finish_reason", None)
        body = (choice.message.content or "") if choice.message else ""
    except (AttributeError, IndexError, KeyError):
        return Status.MALFORMED, None, ""

    body_stripped = (body or "").strip()
    if not body_stripped:
        # OR thinking models (kimi k2.6, glm-5.1) can return an empty
        # `content` with the actual text in `reasoning_content` — a
        # completed, billed answer that EMPTY would discard (run
        # 20260612-005818: kimi finished with finish_reason=stop, 52k chars
        # of reasoning_content, zero content). Salvage it; truncated
        # reasoning likewise downgrades to TRUNCATED-with-body, which the
        # capsule extractor can still mine.
        msg = getattr(choice, "message", None)
        salvaged = (
            (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "")
            if msg is not None
            else ""
        )
        if isinstance(salvaged, str) and salvaged.strip():
            body = salvaged
            body_stripped = salvaged.strip()
    if not body_stripped:
        # Distinguish: empty body + length finish → token exhaustion
        if finish in ("length", "MAX_TOKENS", "max_tokens"):
            return Status.TRUNCATED, finish, ""
        if finish in ("content_filter", "safety", "SAFETY"):
            return Status.CONTENT_FILTERED, finish, ""
        return Status.EMPTY, finish, ""

    if finish in ("length", "MAX_TOKENS", "max_tokens"):
        return Status.TRUNCATED, finish, body
    if finish in ("content_filter", "safety", "SAFETY"):
        return Status.CONTENT_FILTERED, finish, body
    if finish in ("refusal",):
        return Status.REFUSED, finish, body

    # finish_reason may be None for some providers — treat as OK if body is non-empty
    return Status.OK, finish, body
