"""Typed exception taxonomy for the consult engine.

Library callers can `except ConsultError` to catch anything the engine
raises deliberately, then branch on the subclass. The MCP adapter maps
these onto the wire-shape error envelope (`consult.mcp.errors`) via their
stdlib bases, so MCP clients see the same discriminated cases without
importing this module.

This module imports nothing from the rest of the package so any engine
module (registry, sources, artifacts) can raise from it without cycles.
"""

from __future__ import annotations


class ConsultError(Exception):
    """Root of the consult exception taxonomy."""


class UnknownModelError(ConsultError, KeyError):
    """A model alias, raw LiteLLM ID, or tier name didn't resolve.

    Raised by `registry.resolve_model` and `registry.resolve_tier`.
    Multiple-inherits `KeyError` so existing `except KeyError` call sites
    (the per-panellist guard in `_call_one`, `estimate_cost`, the MCP
    adapter's UNKNOWN_MODEL mapping) catch it unchanged.
    """


class BudgetExceededError(ConsultError):
    """A run's estimated or accumulated cost exceeded `max_run_usd`.

    The engine degrades rather than raising: an over-cap run returns a result
    with `partial=True` and a cap `partial_reason`, so output already produced
    isn't thrown away. Check `partial` for the budget path. This type stays in
    the taxonomy for callers that wrap the engine and want to re-raise a budget
    breach as a typed error.
    """


class PathTrustError(ConsultError, ValueError):
    """A path escaped its containment boundary.

    Raised by `sources.validate_under_trusted_roots` when an attachment or
    git_diff repo path falls outside `CONSULT_TRUSTED_REPO_ROOTS`, and by
    `artifacts.load_run` when a run_id resolves outside the runs root.
    Multiple-inherits `ValueError` so the MCP adapter's existing
    `except ValueError` path (mapped to `INVALID_INPUT`) catches it
    without a special case.
    """


class ProviderError(ConsultError):
    """An upstream LLM provider returned an unrecoverable error after retries
    were exhausted.

    The engine degrades rather than raising: the failing panellist is recorded
    in the manifest with `status=ERROR` and a redacted error message while the
    rest of the panel proceeds, so inspect `ManifestEntry` for per-panellist
    failures. This type stays in the taxonomy for callers that re-raise a
    provider failure as a typed error.
    """


class CapsuleParseError(ConsultError):
    """The structured-extractor output couldn't be parsed into a `Capsule` /
    `ReviewCapsule` / `ResearchCapsule`, even after the salvage path.

    The engine degrades rather than raising: the panellist gets an empty
    capsule (with a body-derived confidence when available) so one bad
    extraction doesn't sink the panel. This type stays in the taxonomy for
    callers that re-raise an extraction failure as a typed error.
    """
