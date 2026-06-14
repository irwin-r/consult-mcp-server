"""The per-run calibration block (issue #52)."""

from __future__ import annotations

from consult import calibration
from consult.types import Capsule, ManifestEntry, Status


def _ok(slug: str, model_id: str, persona: str | None, cost: float) -> ManifestEntry:
    return ManifestEntry(
        slug=slug,
        model_id=model_id,
        persona=persona,
        status=Status.OK,
        resource_uri=f"consult://x/{slug}",
        body_path=f"/x/{slug}",
        cost_usd=cost,
        cost_known=True,
        capsule=Capsule(position="p"),
    )


def test_calibration_summarises_panel_diversity_and_spend():
    manifest = [
        _ok("claude-opus-0", "anthropic/claude-opus-4-7", "security", 0.02),
        _ok("gpt-1", "openai/gpt-5.5", "product", 0.03),
        _ok("grok-2", "openrouter/x-ai/grok-4.3", "contrarian", 0.01),
    ]
    cal = calibration.build(manifest, blinded=True, disagreement=0.6)

    assert cal.blinded is True
    # Blinding/shuffle of the synth input is unconditional.
    assert cal.synth_input_blinded is True
    assert cal.synth_input_shuffled is True
    assert cal.disagreement == 0.6
    assert cal.panellists == 3
    assert cal.usable == 3
    assert cal.status_counts == {"OK": 3}
    assert cal.spend_by_status["OK"] == 0.06
    # Two first-party families + one aggregator.
    assert cal.family_diversity == 3
    assert cal.families == {"claude": 1, "gpt": 1, "grok": 1}
    assert cal.privacy_tiers == {"first_party": 2, "aggregator": 1}
    assert cal.stance_coverage == ["contrarian", "product", "security"]


def test_calibration_marks_status_spend_unknown_when_unpriced():
    """A status with any unpriced entry reports None spend rather than
    silently understating it."""
    manifest = [
        _ok("a", "anthropic/claude-opus-4-7", None, 0.02),
        ManifestEntry(
            slug="b",
            model_id="openrouter/x-ai/grok-4.3",
            persona=None,
            status=Status.TIMEOUT,
            error="timed out",
            resource_uri="consult://x/b",
            body_path="/x/b",
            cost_usd=None,
            cost_known=False,
        ),
    ]
    cal = calibration.build(manifest, blinded=False, disagreement=None)
    assert cal.usable == 1
    assert cal.spend_by_status["OK"] == 0.02
    assert cal.spend_by_status["TIMEOUT"] is None
    assert cal.disagreement is None
    # The timed-out panellist contributes no family/stance (not usable).
    assert cal.families == {"claude": 1}
    assert cal.stance_coverage == ["neutral"]
