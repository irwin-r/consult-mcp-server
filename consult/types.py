"""Core data types for the consult server."""

from __future__ import annotations

import re
from enum import StrEnum
from math import ceil
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Forbid unknown fields by default. Pydantic v2's default `extra="ignore"`
# silently drops kwargs that don't match a field — the same mechanism that
# caused the original `RefineResult` silent-drop incident where `partial`,
# `partial_reason`, and `wall_ms` were quietly discarded. Inheriting from
# this base makes the protection automatic: a new internal model added
# without remembering to set `model_config = ConfigDict(extra="forbid")`
# would silently revert to Pydantic's `extra="ignore"` default and reintroduce
# the same class of bug.
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Slug regex shared between ModelSpec input validation and artifacts.py
# path construction. Module-level so it doesn't collide with Pydantic's
# private-attribute treatment of class-level underscore names. The leading
# alphanumeric anchor rejects values like `..`, `--foo`, or `.hidden` —
# pure-punctuation slugs could still escape an artifact dir (`..`) or set
# shell-hostile traps for downstream tools.
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


class ModelSpec(StrictModel):
    """One panellist slot. Either `model` alone, or with a stance."""

    model: str = Field(..., description="Registry alias (e.g. 'gpt-pro') or LiteLLM ID")
    stance: str | None = Field(None, description="Stance key from stances.json or a custom prompt")
    slug: str | None = Field(None, description="Override slug. Otherwise derived from model + index.")

    # Slug is interpolated into filesystem paths (`responses/<slug>.txt`)
    # and resource URIs. Reject anything that could escape the artifact
    # directory at construction time so the input boundary is the failure
    # point, not the deep path-build inside `_call_one`. `artifacts.py`
    # also enforces the same regex defensively when paths are built.
    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _SLUG_RE.fullmatch(v):
            raise ValueError(
                f"slug {v!r} must match {_SLUG_RE.pattern} (no path separators or special characters)"
            )
        return v


class Capsule(StrictModel):
    """Structured ~200-token extract from a panellist response (decision shape).

    Designed so the parent agent can synthesise from the manifest alone in
    most cases, only reading full bodies when it needs depth.

    `kind="decision"` discriminates this from `ReviewCapsule` and
    `ResearchCapsule` in the `AnyCapsule` discriminated union. Defaulted
    so callers that construct `Capsule()` directly (legacy code, tests)
    don't need to thread the literal through.
    """

    kind: Literal["decision"] = "decision"
    position: str = Field("", description="One-line summary of stance/conclusion")
    recommendation: str = Field("", description="What the panellist recommends")
    key_points: list[str] = Field(default_factory=list)
    unique_claims: list[str] = Field(default_factory=list, description="Claims only this panellist made")
    caveats: list[str] = Field(default_factory=list)
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class Finding(StrictModel):
    """One line-anchored review finding (used by ReviewCapsule)."""

    severity: Literal["blocker", "major", "minor", "nit", "praise"]
    file: str | None = Field(None, description="File path, if the finding is file-specific.")
    line_range: tuple[int, int] | None = Field(
        None,
        description="(start, end) line range, if known. Use start=end for a single line.",
    )
    category: Literal["security", "performance", "correctness", "style", "maintainability", "tests", "docs"]
    summary: str = Field(..., description="≤30 words summarising the finding.")
    suggestion: str = Field("", description="≤30 words on the specific change.")


class ReviewCapsule(StrictModel):
    """Structured extract for code/PR review panels.

    Use `capsule_kind="review"` on `panel`/`consult`/`refine` to ask the
    extractor to produce this shape instead of the decision-shape `Capsule`.
    """

    kind: Literal["review"] = "review"
    findings: list[Finding] = Field(default_factory=list)
    overall_verdict: Literal["ship", "changes_requested", "discuss"] = "discuss"
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class ResearchCapsule(StrictModel):
    """Structured extract for research-question panels.

    Use `capsule_kind="research"` on `panel`/`consult`/`refine`. Suits
    workflows where the panel surveys evidence rather than picks a stance.
    """

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


class ManifestEntry(StrictModel):
    """Per-panellist row returned to the parent. ~200 tokens.

    Invariants enforced by `_validate_status_payload`:
    - Status in {ERROR, TIMEOUT} ⇒ `error` is set
    - Numeric fields (latency_ms, tokens_*, cost_usd) are non-negative when set
    - `cost_known=False` distinguishes "we couldn't look up the price" from
      a true zero cost (matters for the `max_run_usd` cap math in refine).
    """

    slug: str
    model_id: str | None = Field(None, description="Real model ID. None when blinded, available after audit.")
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
    cost_usd: float | None = Field(0.0, ge=0.0)
    cost_known: bool = True
    error: str | None = None
    # Informational annotation on a successful call (auto-trim, partial
    # streaming recovery, etc.). Distinct from `error` so the viewer can
    # render it as a yellow info chip rather than a red error block — the
    # call succeeded; we're just surfacing that something noteworthy
    # happened along the way.
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_capsule_kind(cls, data):
        """Upgrade legacy capsule dicts so old manifest.json files still load.

        Two upgrades: inject `kind="decision"` for pre-M2 capsules that
        predate the discriminated union, and strip the retired
        `agrees_with` / `disagrees_with` keys (they were never populated —
        the cross-referencing orchestrator they were reserved for was not
        built — but on-disk capsules written while the fields existed
        would otherwise trip `extra="forbid"`).
        """
        if isinstance(data, dict):
            cap = data.get("capsule")
            if isinstance(cap, dict) and (
                "kind" not in cap or "agrees_with" in cap or "disagrees_with" in cap
            ):
                # Copy to avoid mutating the caller's dict.
                cap = {k: v for k, v in cap.items() if k not in ("agrees_with", "disagrees_with")}
                cap.setdefault("kind", "decision")
                data = {**data, "capsule": cap}
        return data

    @model_validator(mode="after")
    def _validate_status_payload(self) -> ManifestEntry:
        if self.status in (Status.ERROR, Status.TIMEOUT) and not self.error:
            raise ValueError(f"ManifestEntry with status={self.status.value} must carry an error message")
        # Cost invariant: cost_usd=None must imply cost_known=False. The
        # opposite (a known cost we couldn't look up) is nonsensical and
        # would silently understate ledger totals — a future code path
        # returning `cost_usd=None, cost_known=True` would be reported as
        # a free call. Catching it at construction prevents the silent
        # misreport from ever landing on disk.
        if self.cost_usd is None and self.cost_known:
            raise ValueError("ManifestEntry with cost_usd=None must have cost_known=False")
        return self


_MANIFEST_SCHEMA_VERSION = 2


class RunHandle(StrictModel):
    """Returned by `panel`. Manifest-only — no raw bodies inlined.

    `schema_version=2` indicates a manifest whose `capsule` field may be
    any member of the `Capsule | ReviewCapsule | ResearchCapsule` discriminated
    union. v1 manifests (pre-M2) contained only decision-shape `Capsule`
    entries; clients pinned to v1 should check `schema_version` before
    structurally parsing `capsule.*`.
    """

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
        # Aggregate cost invariant: a handle that claims `cost_known=True`
        # while any child entry has `cost_known=False` would silently mask
        # the unknown in ledger totals. Mirrors `ManifestEntry`'s own cost
        # invariant so the cap-enforcement story is uniform top-down.
        if self.cost_known and any(not m.cost_known for m in self.manifest):
            raise ValueError("RunHandle.cost_known=True but a manifest entry has cost_known=False")
        return self

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.manifest:
            counts[m.status.value] = counts.get(m.status.value, 0) + 1
        return counts

    def usable(self, min_ok: int | None = None, min_providers: int | None = None) -> bool:
        """Parametric viability check.

        Defaults: min_ok = max(2, ceil(len(manifest) * 0.6)); min_providers = min(2, panel_size).

        Counting decision (issue #37): the denominator stays the full
        panel — a panellist that died is still a slot you paid for and
        wanted signal from, so it drags viability down. The numerator is
        OK entries plus TRUNCATED entries that produced a capsule with
        content: a truncated-but-substantive answer is real signal (synth
        and refine already consume it), while a truncated-empty entry
        counts the same as a hard failure. Pre-annotation TRUNCATED
        entries (capsule=None) stay excluded, the conservative reading.

        Blinded panels strip `model_id` from every entry by design, so the
        provider-diversity check is skipped under `blinded=True` — otherwise
        a fully-OK blinded panel would always fail `usable()`. The diversity
        signal lives in `registry_snapshot.json` for audit; check it there
        if needed.
        """
        n = len(self.manifest)
        if min_ok is None:
            min_ok = max(2, ceil(n * 0.6))
        if min_providers is None:
            min_providers = min(2, n)

        def _truncated_with_signal(m: ManifestEntry) -> bool:
            if m.status is not Status.TRUNCATED or m.capsule is None:
                return False
            cap = m.capsule
            return any(
                getattr(cap, field, None)
                for field in ("position", "recommendation", "key_points", "findings", "claims", "evidence")
            )

        ok_entries = [m for m in self.manifest if m.status == Status.OK or _truncated_with_signal(m)]
        if len(ok_entries) < min_ok:
            return False
        if self.blinded:
            return True
        # provider extracted from model_id prefix (litellm format) when available
        providers = {(m.model_id or "").split("/")[0] for m in ok_entries if m.model_id}
        providers.discard("")
        return len(providers) >= min_providers


class Calibration(StrictModel):
    """Per-run bias-control disclosure (issue #52).

    Surfaces what the engine already does but never showed in-band: the
    synthesiser's view is always blinded and shuffled, disagreement is scored
    on every consult, and the panel has a measurable family / privacy-tier /
    stance spread. A caller can read this to judge how much to trust a single
    recommendation without digging through the run dir, and high disagreement
    is the signal that the synthesis was held to a two-sided account.
    """

    blinded: bool = Field(
        ..., description="Whether panellists saw each other under anonymised labels during the run."
    )
    synth_input_blinded: bool = Field(
        True, description="The synthesiser always sees blind labels (Alpha/Beta/...), never real model ids."
    )
    synth_input_shuffled: bool = Field(
        True, description="Panellist order is shuffled per synth call to remove position bias."
    )
    disagreement: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Panel disagreement score; None when fewer than two usable capsules.",
    )
    panellists: int = Field(..., ge=0)
    usable: int = Field(..., ge=0, description="Count of OK or TRUNCATED panellists.")
    status_counts: dict[str, int] = Field(default_factory=dict)
    spend_by_status: dict[str, float | None] = Field(
        default_factory=dict,
        description="Summed cost_usd per status; None for a status with any unpriced entry.",
    )
    family_diversity: int = Field(0, ge=0, description="Distinct model families among usable panellists.")
    families: dict[str, int] = Field(
        default_factory=dict, description="Model-family spread across usable panellists."
    )
    privacy_tiers: dict[str, int] = Field(
        default_factory=dict, description="privacy_tier spread across usable panellists."
    )
    stance_coverage: list[str] = Field(
        default_factory=list, description="Distinct stances assigned across usable panellists."
    )


class RunResult(StrictModel):
    """Returned by `consult` (the hero tool). Includes the synthesis."""

    schema_version: int = Field(_MANIFEST_SCHEMA_VERSION, ge=1)
    run_id: str
    synthesis: str
    # Optional panel-disagreement score from `voting.panel_disagreement`.
    # 0.0 = perfect consensus, 1.0 = no two panellists agree. `None`
    # when there are fewer than two usable capsules to compare (the
    # caller's "I want to gate on this" code must treat None as
    # "unknown", not as "low disagreement").
    disagreement: float | None = Field(None, ge=0.0, le=1.0)
    # True iff the synthesis was produced by the deterministic
    # aggregator (gating triggered) rather than the flagship synth model.
    # Lets callers tell "we saved $0.40 because consensus was high" from
    # "we paid for flagship synth as normal". Default False preserves the
    # historic shape for callers who don't use gating.
    synth_gated: bool = False
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
    # Per-run bias-control disclosure (issue #52). None on the rare path where
    # it couldn't be built; populated on every normal consult result.
    calibration: Calibration | None = None

    @model_validator(mode="after")
    def _validate_partial(self) -> RunResult:
        if self.partial and not self.partial_reason:
            raise ValueError("RunResult.partial=True requires partial_reason")
        if not self.partial and self.partial_reason:
            raise ValueError("RunResult.partial=False must not carry a partial_reason")
        # Top-level cost_known must propagate from the manifest. A single
        # unknown-priced panellist (or slow-tail dropout) means we can't
        # validate against `max_run_usd`; flipping `cost_known=False` is
        # how the rest of the stack signals that uncertainty to the caller.
        # ManifestEntry already enforces cost_usd=None ⇒ cost_known=False;
        # this validator stops the outer result from undoing that.
        if self.cost_known and any(not m.cost_known for m in self.manifest):
            raise ValueError("RunResult.cost_known=True but a manifest entry has cost_known=False")
        return self


class ArbiterVerdict(StrictModel):
    """The arbiter's per-round assessment of panel sufficiency.

    `score` is a sufficiency rating (1.0 = strong consensus, ready to ship)
    rather than absolute truth. For v2 arbiter prompts it is *derived* — the
    arbiter scores each of the dimensions in `dimensions` on a 1-5 Likert
    scale, the engine normalises each to [0,1] and averages. Legacy (v1)
    arbiter responses that emit only a top-level `score` are still
    supported: in that case `dimensions` is empty.

    `gaps` and `next_round_focus` feed the next round's prompt as the
    open-ended punch-list; `dimension_notes` carries per-dimension
    localised critique (a Self-Refine / G-Eval pattern: localised feedback
    converges faster than a single global score).

    `parsed_ok=False` means the arbiter call or JSON parse failed;
    downstream callers must not feed `gaps` into a follow-up prompt in
    that case (the "gaps" carry an exception message, not a real arbiter
    finding).
    """

    round: int
    score: float = Field(..., ge=0.0, le=1.0)
    # Per-dimension Likert scores, normalised to [0,1] (the arbiter emits
    # them on a 1-5 scale; `_ask_arbiter` rescales). Keys are dimension
    # names (coverage / agreement / depth / calibration / actionability for
    # the default rubric; other rubrics can use other keys). Empty dict =
    # legacy verdict with only a top-level `score`.
    dimensions: dict[str, float] = Field(default_factory=dict)
    # One-sentence localised critique per dimension. Used by
    # `_build_refinement_prompt` to focus the next-round prompt on the
    # weakest dimensions — much sharper than the generic `gaps` list
    # because each note points at a specific panellist or claim.
    dimension_notes: dict[str, str] = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)
    next_round_focus: str = ""
    reasoning: str = ""
    # Default to known-zero (matches the ManifestEntry cost-default convention):
    # constructors that don't carry a real cost yet should look like a free call,
    # not "we don't know". The failure paths in refine.py set cost=None +
    # cost_known=False explicitly.
    cost_usd: float | None = 0.0
    cost_known: bool = True
    parsed_ok: bool = True
    error: str | None = None

    @model_validator(mode="after")
    def _validate_invariants(self) -> ArbiterVerdict:
        # `cost_usd=None` ⇒ `cost_known=False`: matches ManifestEntry's
        # invariant so cost roll-ups in refine don't silently treat
        # arbiter-cost-unknown as zero.
        if self.cost_usd is None and self.cost_known:
            raise ValueError("ArbiterVerdict with cost_usd=None must have cost_known=False")
        # `parsed_ok=False` ⇒ `error is not None`: refine.py drops gaps and
        # aborts the loop on parse failure, but only if an error string
        # carries the reason. Silent parsed_ok=False with no error makes
        # the caller's debugging path much harder.
        if not self.parsed_ok and not self.error:
            raise ValueError("ArbiterVerdict.parsed_ok=False requires an error message")
        # Per-dimension scores must obey the same [0,1] bounds as the
        # overall score. The arbiter's 1-5 input is rescaled in
        # `_ask_arbiter` before construction, so any out-of-range value
        # here is a constructor bug, not arbiter noise.
        for dim, val in self.dimensions.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"ArbiterVerdict.dimensions[{dim!r}]={val} is outside [0,1]")
        return self


class RefineResult(StrictModel):
    """Returned by `refine`. Carries the final round's manifest, the arbiter
    verdicts for every round, and the final synthesis.

    Per-round transcripts live as MCP resources at
    consult://runs/<id>/responses/<slug>.r<n>
    """

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
        if self.cost_known:
            if any(not m.cost_known for m in self.final_manifest):
                raise ValueError(
                    "RefineResult.cost_known=True but a final_manifest entry has cost_known=False"
                )
            # The arbiter call is billed too — if any per-round verdict has
            # cost_known=False (price-table miss or call failure), the run
            # total cannot be validated against the cap either.
            if any(not v.cost_known for v in self.verdicts):
                raise ValueError("RefineResult.cost_known=True but an arbiter verdict has cost_known=False")
        return self
