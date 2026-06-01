"""consult — multi-model panel orchestration.

The engine (`consult.*`) is MCP-free and directly usable as a library:

    from consult import consult, panel, synthesise
    from consult.refine import refine
    from consult.sequence import sequence

    result = await consult("question?", tier="standard")

The MCP adapter lives at `consult.mcp.*` and is an optional install
(`pip install consult-mcp-server[mcp]`) that exposes the same engine
over the MCP stdio protocol via the `consult-mcp` console script.

# Stable public surface

The names re-exported below form the supported library API. Other
submodule symbols may be useful but are not covered by the semver
compatibility contract until 1.0:

  consult          — hero: parallel panel + server-side synthesis
  panel            — alias for runner.fanout (lower-level, manifest-only)
  synthesise       — re-synthesise an existing run

  consult.refine.refine     — iterative arbiter-driven loop
  consult.sequence.sequence — chained multi-step consultation
  (these two aren't re-exported at the top level so that
  `from consult import refine` still resolves to the submodule for
  callers that want to access internals.)

  RunResult, RunHandle, RefineResult, SequenceResult, ManifestEntry,
  Capsule, ReviewCapsule, ResearchCapsule, ArbiterVerdict, ModelSpec,
  Status — Pydantic models that flow across the surface

  ConsultError, ProviderError, BudgetExceededError, PathTrustError,
  CapsuleParseError, UnknownModelError — typed exceptions library
  consumers can catch
"""

__version__ = "0.2.0"  # x-release-please-version

# Top-level engine entry points. `refine` and `sequence` are NOT re-exported
# here on purpose: each is also the name of a submodule (`consult.refine`,
# `consult.sequence`), and shadowing the submodule with a function makes
# `from consult import refine as refine_mod; refine_mod._suffix_specs(…)`
# (test idiom) blow up. Users wanting the function should import from the
# submodule: `from consult.refine import refine`.
#
# Result types are re-exported below — those don't share a name with any
# submodule and are safe.
from .orchestrate import consult
from .runner import fanout as panel
from .sequence import SequenceResult
from .synth import synthesise
from .types import (
    AnyCapsule,
    ArbiterVerdict,
    Capsule,
    Finding,
    ManifestEntry,
    ModelSpec,
    RefineResult,
    ResearchCapsule,
    ReviewCapsule,
    RunHandle,
    RunResult,
    Status,
)


class ConsultError(Exception):
    """Root of the consult exception taxonomy.

    Library callers can `except ConsultError` to catch anything raised
    by the engine, then branch on the subclass. The MCP adapter translates
    these into the wire-shape `ErrorEnvelope` so MCP clients get the same
    discriminated cases without importing this module.
    """


class UnknownModelError(ConsultError, KeyError):
    """Raised when a model alias or raw LiteLLM ID can't be resolved.

    Multiple-inherits `KeyError` so existing `except KeyError` blocks in
    older code continue to catch it.
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
    """Raised when an attachment path or git_diff repo path is rejected by
    the `CONSULT_TRUSTED_REPO_ROOTS` containment check.

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


__all__ = [
    "__version__",
    # Functions
    "consult",
    "panel",
    "synthesise",
    # Pydantic models
    "AnyCapsule",
    "ArbiterVerdict",
    "Capsule",
    "Finding",
    "ManifestEntry",
    "ModelSpec",
    "RefineResult",
    "ResearchCapsule",
    "ReviewCapsule",
    "RunHandle",
    "RunResult",
    "SequenceResult",
    "Status",
    # Exception taxonomy
    "ConsultError",
    "UnknownModelError",
    "BudgetExceededError",
    "PathTrustError",
    "ProviderError",
    "CapsuleParseError",
]
