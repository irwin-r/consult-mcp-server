"""Core data types for the consult server."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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

    model: str = Field(..., description="Registry alias (e.g. 'gpt-5-pro') or LiteLLM ID")
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
    """Per-panellist row returned to the parent. ~200 tokens."""

    slug: str
    model_id: str | None = Field(
        None, description="Real model ID. None when blinded, available after audit."
    )
    persona: str | None = None
    status: Status
    finish_reason: str | None = None
    capsule: Capsule | None = None
    confidence: float | None = None
    resource_uri: str = Field(..., description="consult://runs/<id>/responses/<slug>")
    body_path: str = Field(..., description="On-disk path for direct access")
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None


class RunHandle(BaseModel):
    """Returned by `panel`. Manifest-only — no raw bodies inlined."""

    run_id: str
    artifacts_dir: str
    manifest: list[ManifestEntry]
    cost_usd: float
    wall_ms: int
    partial: bool = False
    partial_reason: str | None = None
    blinded: bool = False

    def usable(self, min_ok: int | None = None, min_providers: int | None = None) -> bool:
        """Parametric viability check.

        Defaults: min_ok = max(2, ceil(len(manifest) * 0.6)); min_providers = min(2, panel_size).
        """
        from math import ceil

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
    cost_usd: float
    wall_ms: int
    partial: bool = False
