"""Pydantic model invariants.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import json

import pytest

from consult import artifacts
from consult.types import ArbiterVerdict, Capsule, ManifestEntry, RunHandle, Status


def test_run_handle_usable_parametric():
    entries = [
        ManifestEntry(
            slug=f"m{i}",
            model_id=f"openrouter/x{i}/y",
            status=Status.OK,
            resource_uri="consult://x",
            body_path="/tmp/x",
        )
        for i in range(4)
    ]
    entries[3].model_id = "openai/gpt-5"
    handle = RunHandle(run_id="t", artifacts_dir="/tmp/t", manifest=entries, cost_usd=0, wall_ms=0)
    assert handle.usable() is True
    # Set all-OR — only one provider
    for e in entries:
        e.model_id = "openrouter/foo/bar"
    assert handle.usable(min_providers=2) is False


def test_manifest_entry_validates_error_requirement():
    """Constructing an ERROR/TIMEOUT entry without an error string must fail."""
    import pydantic

    base = dict(slug="x", status=Status.ERROR, resource_uri="consult://x", body_path="/tmp/x")
    with pytest.raises(pydantic.ValidationError):
        ManifestEntry(**base)
    # With error, it succeeds
    ManifestEntry(**base, error="auth failed")


def test_run_handle_validates_partial_coupling():
    """partial=True ⇔ partial_reason set."""
    import pydantic

    base = dict(run_id="r", artifacts_dir="/tmp/r", manifest=[], cost_usd=0.0, wall_ms=0)
    with pytest.raises(pydantic.ValidationError):
        RunHandle(**base, partial=True)  # no reason
    with pytest.raises(pydantic.ValidationError):
        RunHandle(**base, partial=False, partial_reason="oops")  # reason without partial


def test_viewer_render_run_handles_missing_optional_artifacts(tmp_path, monkeypatch):
    """Empty/missing prompt.txt, synthesis.md, _progress.log, and arbiters
    must all be tolerated — the renderer is for whatever state the run is
    in, not a strict completeness check.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-6"
    root = tmp_path / rid
    (root / "responses").mkdir(parents=True)
    # Bare-minimum manifest, no other artifacts at all
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": rid,
                "artifacts_dir": str(root),
                "manifest": [],
                "cost_usd": 0.0,
                "wall_ms": 0,
                "partial": False,
            }
        )
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert rid in text
    # No panellists, no synth, no arbiters — still produces a complete document.
    assert "<!doctype html>" in text
    assert "</html>" in text


def test_manifest_carries_schema_version():
    """Every RunHandle / RunResult / RefineResult dump includes
    `schema_version` so future capsule-shape additions are detectable by
    clients without out-of-band coordination."""
    from consult.types import RefineResult, RunHandle, RunResult

    rh = RunHandle(
        run_id="x",
        artifacts_dir="/x",
        manifest=[],
        cost_usd=0.0,
        wall_ms=0,
    )
    assert rh.model_dump()["schema_version"] >= 2

    rr = RunResult(
        run_id="x",
        synthesis="s",
        manifest=[],
        cost_usd=0.0,
        wall_ms=0,
    )
    assert rr.model_dump()["schema_version"] >= 2

    rfr = RefineResult(
        run_id="x",
        rounds_completed=0,
        final_manifest=[],
        verdicts=[],
        synthesis="s",
        converged=False,
        threshold=0.85,
        cost_usd=0.0,
        wall_ms=0,
    )
    assert rfr.model_dump()["schema_version"] >= 2


def test_runresult_invariant_blocks_silently_known_with_unknown_entry():
    """RunResult.cost_known=True with an entry that has cost_known=False
    must raise. Catches the consult-success-path bug where the handler
    dropped `cost_known=handle.cost_known`.
    """
    from consult.types import RunResult

    entry = ManifestEntry(
        slug="alpha",
        model_id="m/x",
        status=Status.OK,
        resource_uri="consult://runs/x/responses/alpha",
        body_path="/tmp/x",
        latency_ms=0,
        cost_usd=None,
        cost_known=False,
        confidence=None,
        capsule=None,
    )
    with pytest.raises(Exception) as exc:
        RunResult(
            run_id="x",
            synthesis="s",
            manifest=[entry],
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )
    assert "cost_known" in str(exc.value)


def test_refineresult_invariant_catches_arbiter_cost_unknown():
    """RefineResult.cost_known=True with a verdict that has cost_known=False
    must raise. Catches the refine.py bug where only `if verdict.cost_usd`
    propagated; an unmapped-price arbiter left cost_all_known=True.
    """
    from consult.types import RefineResult

    verdict = ArbiterVerdict(round=1, score=0.5, gaps=[], reasoning="", cost_usd=None, cost_known=False)
    with pytest.raises(Exception) as exc:
        RefineResult(
            run_id="x",
            rounds_completed=1,
            final_manifest=[],
            verdicts=[verdict],
            synthesis="s",
            converged=False,
            threshold=0.85,
            cost_usd=0.0,
            cost_known=True,
            wall_ms=0,
        )
    assert "cost_known" in str(exc.value)


def test_sequenceresult_invariant_propagates_step_cost_known():
    """SequenceResult.cost_known=True with a step that has cost_known=False
    must raise. Pre-fix, sequence's step cost_known was tracked but the
    invariant wasn't enforced, so a typo could land in production silently.
    """
    from consult.sequence import SequenceResult, SequenceStep

    step = SequenceStep(
        step=1,
        run_id="r1",
        synthesis="x",
        cost_usd=0.1,
        cost_known=False,
        panel_size=2,
    )
    with pytest.raises(Exception) as exc:
        SequenceResult(
            steps=[step],
            final_synthesis="x",
            cost_usd=0.1,
            cost_known=True,
            wall_ms=0,
        )
    assert "cost_known" in str(exc.value)


def _usable_entry(slug, provider, status, capsule=None, error=None):
    return ManifestEntry(
        slug=slug,
        model_id=f"{provider}/m",
        status=status,
        resource_uri=f"consult://x/{slug}",
        body_path=f"/x/{slug}",
        capsule=capsule,
        error=error,
    )


def test_usable_counts_truncated_with_capsule_content():
    """(issue #37) A truncated panellist that still produced a substantive
    capsule is usable signal, matching how synth and refine treat it."""
    handle = RunHandle(
        run_id="r",
        artifacts_dir="/x",
        manifest=[
            _usable_entry("a", "anthropic", Status.OK, capsule=Capsule(position="yes")),
            _usable_entry("b", "openai", Status.TRUNCATED, capsule=Capsule(position="partial but real")),
            _usable_entry("c", "google", Status.ERROR, error="boom"),
        ],
        cost_usd=0.0,
        wall_ms=1,
    )
    assert handle.usable() is True


def test_usable_excludes_truncated_empty_capsule():
    """(issue #37) Truncated-with-nothing counts the same as a hard failure."""
    handle = RunHandle(
        run_id="r",
        artifacts_dir="/x",
        manifest=[
            _usable_entry("a", "anthropic", Status.OK, capsule=Capsule(position="yes")),
            _usable_entry("b", "openai", Status.TRUNCATED, capsule=Capsule()),
            _usable_entry("c", "google", Status.ERROR, error="boom"),
        ],
        cost_usd=0.0,
        wall_ms=1,
    )
    assert handle.usable() is False


def test_usable_excludes_truncated_before_annotation():
    """(issue #37) Pre-annotation TRUNCATED entries (capsule=None) stay
    excluded — the conservative reading until the extractor has run."""
    handle = RunHandle(
        run_id="r",
        artifacts_dir="/x",
        manifest=[
            _usable_entry("a", "anthropic", Status.OK, capsule=Capsule(position="yes")),
            _usable_entry("b", "openai", Status.TRUNCATED),
        ],
        cost_usd=0.0,
        wall_ms=1,
    )
    assert handle.usable() is False
