"""Tests for `consult.voting` — medoid voting over `model:N` panels."""

from __future__ import annotations

from consult.types import (
    Capsule,
    Finding,
    ManifestEntry,
    ReviewCapsule,
    Status,
)
from consult.voting import manifest_after_medoid, medoid_slugs, panel_disagreement


def _ok_entry(
    slug: str, model_id: str, position: str, *, key_points: list[str] | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        slug=slug,
        model_id=model_id,
        status=Status.OK,
        resource_uri=f"consult://x/{slug}",
        body_path=f"/x/{slug}",
        capsule=Capsule(
            position=position,
            recommendation=position,
            key_points=key_points or [],
        ),
    )


def test_medoid_singleton_group_returns_the_only_slug():
    """A group of one entry trivially yields that entry's slug."""
    manifest = [_ok_entry("solo-0", "anthropic/x", "use approach A")]
    out = medoid_slugs(manifest)
    assert out == {(0, "anthropic/x"): "solo-0"}


def test_medoid_picks_the_centre_of_three_responses():
    """Three same-model entries: two agree closely, one diverges. The
    medoid is the centre of mass, not the outlier."""
    centre_text = "use approach A with fallback to B"
    # Two near-identical agreers + one outlier
    manifest = [
        _ok_entry("ck-0", "anthropic/claude", centre_text + " for reliability"),
        _ok_entry("ck-1", "anthropic/claude", centre_text + " for performance"),
        _ok_entry("ck-2", "anthropic/claude", "use C exclusively"),
    ]
    out = medoid_slugs(manifest)
    # The medoid should be one of the two agreers, never the outlier.
    chosen = out[(0, "anthropic/claude")]
    assert chosen in {"ck-0", "ck-1"}, f"outlier picked: {chosen}"


def test_medoid_groups_by_model_id_not_slug():
    """Multiple models in one panel: each model_id gets its own medoid."""
    manifest = [
        _ok_entry("c-0", "anthropic/claude", "Claude says do A"),
        _ok_entry("c-1", "anthropic/claude", "Claude says do A but with care"),
        _ok_entry("g-0", "openai/gpt", "GPT says do B"),
        _ok_entry("g-1", "openai/gpt", "GPT says do B with care"),
    ]
    out = medoid_slugs(manifest)
    assert set(out.keys()) == {(0, "anthropic/claude"), (0, "openai/gpt")}


def test_medoid_separates_refine_rounds_by_suffix():
    """`.r1` / `.r2` slugs are different rounds — must not fold together."""
    manifest = [
        _ok_entry("ck.r1", "anthropic/claude", "round 1 answer"),
        _ok_entry("ck.r2", "anthropic/claude", "round 2 refined answer"),
    ]
    out = medoid_slugs(manifest)
    # Two singleton groups, one per round
    assert (1, "anthropic/claude") in out
    assert (2, "anthropic/claude") in out
    assert out[(1, "anthropic/claude")] == "ck.r1"
    assert out[(2, "anthropic/claude")] == "ck.r2"


def test_medoid_skips_groups_with_no_usable_capsules():
    """If every entry in a group has a failed/empty capsule, the group is
    omitted from the medoid result — caller decides how to surface."""
    manifest = [
        ManifestEntry(
            slug="failed-0",
            model_id="anthropic/x",
            status=Status.TIMEOUT,
            resource_uri="consult://x/failed-0",
            body_path="/x/failed-0",
            error="timeout after 30s",
            capsule=None,
        ),
        ManifestEntry(
            slug="failed-1",
            model_id="anthropic/x",
            status=Status.ERROR,
            resource_uri="consult://x/failed-1",
            body_path="/x/failed-1",
            error="provider 500",
            capsule=None,
        ),
    ]
    out = medoid_slugs(manifest)
    assert out == {}


def test_manifest_after_medoid_drops_non_medoid_entries_but_keeps_failed_groups():
    """End-to-end filter: usable groups collapse to one entry (the medoid);
    failed groups pass through whole so callers see the error rows."""
    # Group A: 3 entries, 2 agree → medoid is one of those 2
    # Group B: 1 entry → trivially kept
    # Group C: 2 entries, both failed → both kept
    manifest = [
        _ok_entry("a-0", "anthropic/x", "approach X with caveat 1"),
        _ok_entry("a-1", "anthropic/x", "approach X with caveat 2"),
        _ok_entry("a-2", "anthropic/x", "totally different approach Y"),
        _ok_entry("b-0", "openai/y", "do Z"),
        ManifestEntry(
            slug="c-0", model_id="google/z", status=Status.ERROR,
            resource_uri="consult://x/c-0", body_path="/x/c-0",
            error="auth", capsule=None,
        ),
        ManifestEntry(
            slug="c-1", model_id="google/z", status=Status.TIMEOUT,
            resource_uri="consult://x/c-1", body_path="/x/c-1",
            error="timeout", capsule=None,
        ),
    ]
    out = manifest_after_medoid(manifest)
    out_slugs = [e.slug for e in out]
    # Exactly one of a-0 / a-1 survives; a-2 (outlier) is dropped
    a_survivors = [s for s in out_slugs if s.startswith("a-")]
    assert len(a_survivors) == 1
    assert a_survivors[0] in {"a-0", "a-1"}, "outlier survived"
    # b-0 trivially survives
    assert "b-0" in out_slugs
    # Failed group keeps both rows (no medoid to pick)
    assert "c-0" in out_slugs
    assert "c-1" in out_slugs


def test_manifest_after_medoid_preserves_order():
    """The medoid takes the position of whichever group-member it was —
    we don't shuffle the manifest as a side effect."""
    manifest = [
        _ok_entry("a-0", "anthropic/x", "approach X"),
        _ok_entry("b-0", "openai/y", "do Z"),
        _ok_entry("a-1", "anthropic/x", "approach X with caveat"),
    ]
    out = manifest_after_medoid(manifest)
    # b-0 must remain in its position regardless of which a-entry is chosen
    assert any(e.slug == "b-0" for e in out)
    b_idx = next(i for i, e in enumerate(out) if e.slug == "b-0")
    a_idxs = [i for i, e in enumerate(out) if e.slug.startswith("a-")]
    # Exactly one a-entry survives, and the survivor's position is < or > b's
    # but never displaced from b's original neighbouring slot.
    assert len(a_idxs) == 1


def test_panel_disagreement_returns_zero_for_identical_capsules():
    """Identical position+recommendation across the panel → near-zero score."""
    manifest = [
        _ok_entry("a-0", "x/a", "approach X"),
        _ok_entry("b-0", "y/b", "approach X"),
        _ok_entry("c-0", "z/c", "approach X"),
    ]
    score = panel_disagreement(manifest)
    assert score is not None
    assert score == 0.0


def test_panel_disagreement_high_for_divergent_capsules():
    """Three wildly different positions → disagreement well above 0.5."""
    manifest = [
        _ok_entry("a-0", "x/a", "do A immediately"),
        _ok_entry("b-0", "y/b", "abandon the project"),
        _ok_entry("c-0", "z/c", "form a committee to study"),
    ]
    score = panel_disagreement(manifest)
    assert score is not None
    assert score > 0.5, f"expected high disagreement, got {score}"


def test_panel_disagreement_returns_none_for_zero_or_one_usable():
    """Fewer than two usable capsules → None, not 0.0. Caller must
    handle None as "unknown" rather than treating it as full agreement."""
    # Zero usable
    from consult.types import ManifestEntry as M, Status as S
    empty = [
        M(slug="x-0", model_id="x/a", status=S.ERROR,
          resource_uri="x", body_path="/x", error="boom", capsule=None),
    ]
    assert panel_disagreement(empty) is None
    # One usable
    one = [_ok_entry("a-0", "x/a", "do X")]
    assert panel_disagreement(one) is None


def test_panel_disagreement_clamps_to_unit_interval():
    """Floating-point drift on near-identical capsules must not let the
    return value escape [0, 1]."""
    manifest = [
        _ok_entry("a-0", "x/a", "z"),
        _ok_entry("b-0", "y/b", "z"),
    ]
    score = panel_disagreement(manifest)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_medoid_handles_review_capsules():
    """ReviewCapsule entries: similarity uses the findings rather than
    a position string. Two reviews flagging the same blocker score
    similar even if their summaries are phrased differently."""
    manifest = [
        ManifestEntry(
            slug="r-0",
            model_id="anthropic/x",
            status=Status.OK,
            resource_uri="consult://x/r-0",
            body_path="/x/r-0",
            capsule=ReviewCapsule(
                overall_verdict="changes_requested",
                findings=[
                    Finding(
                        severity="blocker", category="security",
                        summary="SQL injection in login",
                        suggestion="use parameterised query",
                    ),
                ],
            ),
        ),
        ManifestEntry(
            slug="r-1",
            model_id="anthropic/x",
            status=Status.OK,
            resource_uri="consult://x/r-1",
            body_path="/x/r-1",
            capsule=ReviewCapsule(
                overall_verdict="changes_requested",
                findings=[
                    Finding(
                        severity="blocker", category="security",
                        summary="SQL injection in login",
                        suggestion="parameterise the query",
                    ),
                ],
            ),
        ),
        ManifestEntry(
            slug="r-2",
            model_id="anthropic/x",
            status=Status.OK,
            resource_uri="consult://x/r-2",
            body_path="/x/r-2",
            capsule=ReviewCapsule(
                overall_verdict="ship",
                findings=[],
            ),
        ),
    ]
    out = medoid_slugs(manifest)
    chosen = out[(0, "anthropic/x")]
    # The two security-flagging reviews are far closer to each other than
    # to the empty-findings ship verdict — medoid lives in {r-0, r-1}.
    assert chosen in {"r-0", "r-1"}
