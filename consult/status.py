"""Classify a LiteLLM response (or exception) into a Status."""

from __future__ import annotations

from typing import Any

from .types import Status


def classify(
    response: Any | None,
    exception: BaseException | None = None,
) -> tuple[Status, str | None, str]:
    """Return (status, finish_reason, body_text).

    Designed to be tolerant of provider quirks. Bodies that are mostly whitespace
    (a known OR thinking-model failure mode) are classified as EMPTY.
    """
    if exception is not None:
        msg = str(exception).lower()
        if "rate" in msg and "limit" in msg:
            return Status.RATE_LIMITED, None, ""
        if "timeout" in msg or "timed out" in msg:
            return Status.TIMEOUT, None, ""
        if "content_filter" in msg or "content policy" in msg:
            return Status.CONTENT_FILTERED, None, ""
        return Status.ERROR, None, ""

    if response is None:
        return Status.EMPTY, None, ""

    # LiteLLM normalises to OpenAI shape. Try the common path first.
    try:
        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", None) or getattr(
            choice.message, "finish_reason", None
        )
        body = (choice.message.content or "") if choice.message else ""
    except (AttributeError, IndexError, KeyError):
        return Status.MALFORMED, None, ""

    body_stripped = (body or "").strip()
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
