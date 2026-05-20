"""Core data types for the consult server."""

from __future__ import annotations

from enum import StrEnum
from math import ceil
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Forbid unknown fields by default. Pydantic v2's default `extra="ignore"`
# silently drops kwargs that don't match a field — the same mechanism that
# caused the original `RefineResult` silent-drop incident where `partial`,
# `partial_reason`, and `wall_ms` were quietly discarded. Forbidding extras
# turns any future field-mismatch into a loud ValidationError at the call
# site. Set per-class (not via a shared base) to keep types.py self-contained
# and avoid surprising inheritance interactions with downstream validators.
_STRICT = ConfigDict(extra="forbid")


class Status(StrEnum):
    OK = "OK"
    TRUNCATED = "TRUNCATED"
    MALFORMED = "MALFORMED"
    EMPTY = "EMPTY"
    REFUSED = "REFUSED"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class ModelSpec(BaseModel):
    """One panellist slot. Either `model` alone, or with a stance."""

    model_config = _STRICT

    model: str = Field(..., description="Registry alias (e.g. 'gpt-pro') or LiteLLM ID")
    stance: str | None = Field(None, description="Stance key from stances.json or a custom prompt")
    slug: str | None = Field(
        None, description="Override slug. Otherwise derived from model + index."
    )


class Capsule(BaseModel):
    """Structured ~200-token extract from a panellist response (decision shape).

    Designed so the parent agent can synthesise from the manifest alone in
    most cases, only reading full bodies when it needs depth.

    `kind="decision"` discriminates this from `ReviewCapsule` and
    `ResearchCapsule` in the `AnyCapsule` discriminated union. Defaulted
    so callers that construct `Capsule()` directly (legacy code, tests)
    don't need to thread the literal through.
    """

    model_config = _STRICT

    kind: Literal["decision"] = "decision"
    position: str = Field("", description="One-line summary of stance/conclusion")
    recommendation: str = Field("", description="What the panellist recommends")
    key_points: list[str] = Field(default_factory=list)
    unique_claims: list[str] = Field(
        default_factory=list, description="Claims only this panellist made"
    )
    caveats: list[str] = Field(default_factory=list)
    agrees_with: list[str] = Field(default_factory=list, description="Slugs this agrees with")
    disagrees_with: list[str] = Field(default_factory=list, description="Slugs this disagrees with")
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class Finding(BaseModel):
    """One line-anchored review finding (used by ReviewCapsule)."""

    model_config = _STRICT

    severity: Literal["blocker", "major", "minor", "nit", "praise"]
    file: str | None = Field(None, description="File path, if the finding is file-specific.")
    line_range: tuple[int, int] | None = Field(
        None,
        description="(start, end) line range, if known. Use start=end for a single line.",
    )
    category: Literal[
        "security", "performance", "correctness", "style", "maintainability", "tests", "docs"
    ]
    summary: str = Field(..., description="≤30 words summarising the finding.")
    suggestion: str = Field("", description="≤30 words on the specific change.")


class ReviewCapsule(BaseModel):
    """Structured extract for code/PR review panels.

    Use `capsule_kind="review"` on `panel`/`consult`/`refine` to ask the
    extractor to produce this shape instead of the decision-shape `Capsule`.
    """

    model_config = _STRICT

    kind: Literal["review"] = "review"
    findings: list[Finding] = Field(default_factory=list)
    overall_verdict: Literal["ship", "changes_requested", "discuss"] = "discuss"
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class ResearchCapsule(BaseModel):
    """Structured extract for research-question panels.

    Use `capsule_kind="research"` on `panel`/`consult`/`refine`. Suits
    workflows where the panel surveys evidence rather than picks a stance.
    """

    model_config = _STRICT

    kind: Literal["research"] = "research"
    claims: list[str] = Field(default_factory=list, description="The panellist's main assertions.")
    evidence: list[str] = Field(default_factory=list, description="What backs each claim.")
    uncertainties: list[str] = Field(
        default_factory=list, description="Where the panellist is genuinely unsure."
    )
    sources_cited: list[str] = Field(
        default_factory=list, description="URLs, papers, or vendor docs named in the body."
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0)


# Discriminated union over capsule variants. Pydantic dispatches based on the
# `kind` field. Legacy capsule dicts (manifest.json from before kind existed)
# are handled by ManifestEntry's pre-validator which injects `kind="decision"`
# when absent.
AnyCapsule = Annotated[
    Capsule | ReviewCapsule | ResearchCapsule,
    Field(discriminator="kind"),
]


class ManifestEntry(BaseModel):
    """Per-panellist row returned to the parent. ~200 tokens.

    Invariants enforced by `_validate_status_payload`:
    - Status in {ERROR, TIMEOUT} ⇒ `error` is set
    - Numeric fields (latency_ms, tokens_*, cost_usd) are non-negative when set
    - `cost_known=False` distinguishes "we couldn't look up the price" from
      a true zero cost (matters for the `max_run_usd` cap math in refine).
    """

    model_config = _STRICT

    slug: str
    model_id: str | None = Field(
        None, description="Real model ID. None when blinded, available after audit."
    )
    persona: str | None = None
    status: Status
    finish_reason: str | None = None
    capsule: AnyCapsule | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    resource_uri: str = Field(..., description="consult://runs/<id>/responses/<slug>")
    body_path: str = Field(..., description="On-disk path for direct access")
    latency_ms: int | None = Field(None, ge=0)
    tokens_in: int | None = Field(None, ge=0)
    tokens_out: int | None = Field(None, ge=0)
    cost_usd: float | None = Field(None, ge=0.0)
    cost_known: bool = True
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_capsule_kind(cls, data):
        """Inject `kind="decision"` into legacy capsule dicts that predate the
        discriminated union. Without this, manifest.json files written before
        M2 fail to validate because the discriminator field is absent.
        """
        if isinstance(data, dict):
            cap = data.get("capsule")
            if isinstance(cap, dict) and "kind" not in cap:
                # Copy to avoid mutating the caller's dict, then add kind.
                data = {**data, "capsule": {**cap, "kind": "decision"}}
        return data

    @model_validator(mode="after")
    def _validate_status_payload(self) -> ManifestEntry:
        if self.status in (Status.ERROR, Status.TIMEOUT) and not self.error:
            raise ValueError(
                f"ManifestEntry with status={self.status.value} must carry an error message"
            )
        return self


_MANIFEST_SCHEMA_VERSION = 2


class RunHandle(BaseModel):
    """Returned by `panel`. Manifest-only — no raw bodies inlined.

    `schema_version=2` indicates a manifest whose `capsule` field may be
    any member of the `Capsule | ReviewCapsule | ResearchCapsule` discriminated
    union. v1 manifests (pre-M2) contained only decision-shape `Capsule`
    entries; clients pinned to v1 should check `schema_version` before
    structurally parsing `capsule.*`.
    """

    model_config = _STRICT

    schema_version: int = Field(_MANIFEST_SCHEMA_VERSION, ge=1)
    run_id: str
    artifacts_dir: str
    manifest: list[ManifestEntry]
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    wall_ms: int = Field(..., ge=0)
    partial: bool = False
    partial_reason: str | None = None
    blinded: bool = False

    @model_validator(mode="after")
    def _validate_partial(self) -> RunHandle:
        if self.partial and not self.partial_reason:
            raise ValueError("RunHandle.partial=True requires partial_reason")
        if not self.partial and self.partial_reason:
            raise ValueError("RunHandle.partial=False must not carry a partial_reason")
        return self

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.manifest:
            counts[m.status.value] = counts.get(m.status.value, 0) + 1
        return counts

    def usable(self, min_ok: int | None = None, min_providers: int | None = None) -> bool:
        """Parametric viability check.

        Defaults: min_ok = max(2, ceil(len(manifest) * 0.6)); min_providers = min(2, panel_size).
        """
        n = len(self.manifest)
        if min_ok is None:
            min_ok = max(2, ceil(n * 0.6))
        if min_providers is None:
            min_providers = min(2, n)
        ok_entries = [m for m in self.manifest if m.status == Status.OK]
        if len(ok_entries) < min_ok:
            return False
        # provider extracted from model_id prefix (litellm format) when available
        providers = {
            (m.model_id or "").split("/")[0] for m in ok_entries if m.model_id
        }
        providers.discard("")
        return len(providers) >= min_providers


class RunResult(BaseModel):
    """Returned by `consult` (the hero tool). Includes the synthesis."""

    model_config = _STRICT

    schema_version: int = Field(_MANIFEST_SCHEMA_VERSION, ge=1)
    run_id: str
    synthesis: str
    manifest: list[ManifestEntry]
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    wall_ms: int = Field(..., ge=0)
    partial: bool = False
    partial_reason: str | None = None
    # Surfaced so the caller can tell why the panel size is `tier_size - 1`
    # when the chosen synthesiser is itself a member of the requested tier
    # (the synthesiser is excluded from the panel to avoid self-inclusion
    # bias). Without this, panel-shrinkage is invisible in the response.
    synthesiser: str | None = None

    @model_validator(mode="after")
    def _validate_partial(self) -> RunResult:
        if self.partial and not self.partial_reason:
            raise ValueError("RunResult.partial=True requires partial_reason")
        if not self.partial and self.partial_reason:
            raise ValueError("RunResult.partial=False must not carry a partial_reason")
        return self


class ArbiterVerdict(BaseModel):
    """The arbiter's per-round assessment of panel sufficiency.

    `score` is a sufficiency rating (1.0 = strong consensus, ready to ship)
    rather than absolute truth. `gaps` and `next_round_focus` feed the next
    round's prompt.

    `parsed_ok=False` means the arbiter call or JSON parse failed; downstream
    callers must not feed `gaps` into a follow-up prompt in that case (the
    "gaps" carry an exception message, not a real arbiter finding).
    """

    model_config = _STRICT

    round: int
    score: float = Field(..., ge=0.0, le=1.0)
    gaps: list[str] = Field(default_factory=list)
    next_round_focus: str = ""
    reasoning: str = ""
    cost_usd: float | None = None
    cost_known: bool = True
    parsed_ok: bool = True
    error: str | None = None


class RefineResult(BaseModel):
    """Returned by `refine`. Carries the final round's manifest, the arbiter
    verdicts for every round, and the final synthesis.

    Per-round transcripts live as MCP resources at
    consult://runs/<id>/responses/<slug>.r<n>
    """

    model_config = _STRICT

    schema_version: int = Field(_MANIFEST_SCHEMA_VERSION, ge=1)
    run_id: str
    rounds_completed: int = Field(..., ge=0)
    final_manifest: list[ManifestEntry]
    verdicts: list[ArbiterVerdict]
    synthesis: str
    converged: bool = Field(..., description="True if score >= threshold")
    threshold: float = Field(..., ge=0.0, le=1.0)
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    wall_ms: int = Field(..., ge=0)
    partial: bool = False
    partial_reason: str | None = None
    # The prior run_id this refine continues from. None for a fresh refine;
    # set when the caller passed `continuation_id` so the lineage is visible
    # in the result without requiring the client to track it separately.
    continuation_of: str | None = None

    @model_validator(mode="after")
    def _validate_partial(self) -> RefineResult:
        if self.partial and not self.partial_reason:
            raise ValueError("RefineResult.partial=True requires partial_reason")
        if not self.partial and self.partial_reason:
            raise ValueError("RefineResult.partial=False must not carry a partial_reason")
        return self
