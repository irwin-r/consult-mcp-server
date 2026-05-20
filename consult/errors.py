"""Structured error envelope for the MCP tool surface.

Every `handle_call_tool` failure returns the same JSON shape so an agent
can pattern-match on `error.code` and recover programmatically instead
of regex-scraping free-text.

Wire shape (returned as a single TextContent JSON-serialised string):

    {"ok": false, "error": {"code": "<ErrorCode>", "message": "...",
                            "run_id": "<id-or-null>"}}

Success paths still return their existing result shapes unchanged — the
envelope only wraps the failure case. (Mixed success/partial returns
already carry `partial`/`partial_reason` and don't need re-wrapping.)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Stable error codes. Agents should branch on these, not on `message`."""

    INVALID_INPUT = "invalid_input"
    """Caller's input failed validation — bad ID, out-of-range param, etc."""

    UNKNOWN_MODEL = "unknown_model"
    """A model alias didn't resolve in the registry."""

    RUN_NOT_FOUND = "run_not_found"
    """A `run_id` (synthesise / continuation_id) doesn't exist on disk."""

    UPSTREAM_ERROR = "upstream_error"
    """A provider call failed in a way we can't categorise per-spec."""

    INTERNAL_ERROR = "internal_error"
    """Catch-all for unexpected exceptions. Implies a bug in this server."""


class ConsultError(BaseModel):
    code: ErrorCode
    message: str
    run_id: str | None = Field(
        None,
        description=(
            "If a run was created before the error occurred (e.g. mid-refine "
            "failure), the partial run_id so the agent can fetch artifacts."
        ),
    )


class ErrorEnvelope(BaseModel):
    """Top-level shape of a failure response. `ok` is constant False on this
    type — the agent branches on the outer key. Successful responses don't
    carry `ok` at all (keeps the existing result shapes unchanged).
    """

    ok: bool = Field(False, description="Always False on this type")
    error: ConsultError


def envelope(code: ErrorCode, message: str, run_id: str | None = None) -> dict:
    """Build the standard failure envelope as a dict.

    Returned by `handle_call_tool`'s try/except so every failure mode the
    agent sees has the same shape. Returning a dict (rather than a JSON
    string) lets the MCP SDK populate `structuredContent` alongside the
    JSON text fallback — clients can branch on `error.code` directly
    without parsing the text body.
    """
    return ErrorEnvelope(
        error=ConsultError(code=code, message=message, run_id=run_id),
    ).model_dump()
