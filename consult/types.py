"""Core data types for the consult server."""

from __future__ import annotations

from enum import Enum
from math import ceil

from pydantic import BaseModel, Field, model_validator


class Status(str, Enum):
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

    model: str = Field(..., description="Registry alias (e.g. 'gpt-pro') or LiteLLM ID")
    stance: str | None = Field(None, description="Stance key from stances.json or a custom prompt")
    slug: str | None = Field(
        None, description="Override slug. Otherwise derived from model + index."
    )


class Capsule(BaseModel):
    """Structured ~200-token extract from a panellist response.

    Designed so the parent agent can synthesise from the manifest alone in
    most cases, only reading full bodies when it needs depth.
    """

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


class ManifestEntry(BaseModel):
    """Per-panellist row returned to the parent. ~200 tokens.

    Invariants enforced by `_validate_status_payload`:
    - Status in {ERROR, TIMEOUT} ⇒ `error` is set
    - Numeric fields (latency_ms, tokens_*, cost_usd) are non-negative when set
    - `cost_known=False` distinguishes "we couldn't look up the price" from
      a true zero cost (matters for the `max_run_usd` cap math in refine).
    """

    slug: str
    model_id: str | None = Field(
        None, description="Real model ID. None when blinded, available after audit."
    )
    persona: str | None = None
    status: Status
    finish_reason: str | None = None
    capsule: Capsule | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    resource_uri: str = Field(..., description="consult://runs/<id>/responses/<slug>")
    body_path: str = Field(..., description="On-disk path for direct access")
    latency_ms: int | None = Field(None, ge=0)
    tokens_in: int | None = Field(None, ge=0)
    tokens_out: int | None = Field(None, ge=0)
    cost_usd: float | None = Field(None, ge=0.0)
    cost_known: bool = True
    error: str | None = None

    @model_validator(mode="after")
    def _validate_status_payload(self) -> ManifestEntry:
        if self.status in (Status.ERROR, Status.TIMEOUT) and not self.error:
            raise ValueError(
                f"ManifestEntry with status={self.status.value} must carry an error message"
            )
        return self


class RunHandle(BaseModel):
    """Returned by `panel`. Manifest-only — no raw bodies inlined."""

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

    run_id: str
    synthesis: str
    manifest: list[ManifestEntry]
    cost_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    wall_ms: int = Field(..., ge=0)
    partial: bool = False


class ArbiterVerdict(BaseModel):
    """The arbiter's per-round assessment of panel sufficiency.

    `score` is a sufficiency rating (1.0 = strong consensus, ready to ship)
    rather than absolute truth. `gaps` and `next_round_focus` feed the next
    round's prompt.

    `parsed_ok=False` means the arbiter call or JSON parse failed; downstream
    callers must not feed `gaps` into a follow-up prompt in that case (the
    "gaps" carry an exception message, not a real arbiter finding).
    """

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
