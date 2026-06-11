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

__version__ = "0.4.0"  # x-release-please-version

# Top-level engine entry points. `refine` and `sequence` are NOT re-exported
# here on purpose: each is also the name of a submodule (`consult.refine`,
# `consult.sequence`), and shadowing the submodule with a function makes
# `from consult import refine as refine_mod; refine_mod._suffix_specs(…)`
# (test idiom) blow up. Users wanting the function should import from the
# submodule: `from consult.refine import refine`.
#
# Result types are re-exported below — those don't share a name with any
# submodule and are safe.
from .exceptions import (
    BudgetExceededError,
    CapsuleParseError,
    ConsultError,
    PathTrustError,
    ProviderError,
    UnknownModelError,
)
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
