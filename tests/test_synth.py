"""Flagship synthesis, blinding, sentinels.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import json

import pytest

from consult import artifacts
from consult.types import ManifestEntry, RunHandle, Status


def test_synth_deblind_replaces_word_boundary_only():
    """`_deblind` must be case-sensitive + word-boundary-anchored.

    The synth's prose may legitimately mention "alpha release" or
    "alphanumeric" — replacing those with a slug would corrupt the
    synthesis. Only the exact `Alpha` / `Beta` tokens the synth was
    told to use should be rewritten.
    """
    from consult.synth import _deblind

    mapping = {"Alpha": "claude-opus", "Beta": "gpt-pro"}
    src = (
        "Alpha argued for X, while Beta supported Y. "
        "Both noted that the alpha release of the library "
        "had alphanumeric IDs. Gamma was not mentioned."
    )
    out = _deblind(src, mapping)
    assert "claude-opus argued for X" in out
    assert "gpt-pro supported Y" in out
    # Casual prose preserved
    assert "alpha release" in out
    assert "alphanumeric" in out
    # Unmapped label left alone
    assert "Gamma was not mentioned" in out


def test_synth_deblind_empty_map_is_noop():
    """Zero-panel runs (or callers that skip blinding) must pass through
    unchanged. Building a regex of an empty alternation would explode."""
    from consult.synth import _deblind

    src = "Synthesis with Alpha mentioned."
    assert _deblind(src, {}) == src


def test_synth_build_input_always_blinds_and_filters_failures():
    """Synth never sees real slug or model_id — bias mitigation per the
    "When Identity Skews Debate" paper. Tests:

    - ERROR/EMPTY entries are filtered out of the input
    - The synth input contains blind labels (Alpha, Beta, ...) not slugs
    - Model IDs never appear in the synth input
    - The returned `label_to_slug` mapping covers every usable panellist
    """
    from consult.synth import _build_input

    manifest = [
        {
            "slug": "panelist-alpha",
            "model_id": "anthropic/claude-opus-4-7",
            "persona": "contrarian",
            "confidence": 0.8,
            "status": "OK",
        },
        {
            "slug": "panelist-beta",
            "model_id": "openai/gpt-5.5",
            "persona": None,
            "confidence": None,
            "status": "EMPTY",  # must be filtered out
        },
        {
            "slug": "panelist-gamma",
            "model_id": "gemini/gemini-3.1-pro-preview",
            "persona": None,
            "confidence": 0.6,
            "status": "TRUNCATED",  # truncated-with-body stays in
        },
    ]
    bodies = {
        "panelist-alpha": "alpha body",
        "panelist-beta": "",
        "panelist-gamma": "gamma body",
    }
    rubric = "rubric {n}"

    text, label_to_slug = _build_input(manifest, bodies, rubric=rubric)
    # Real model identifiers never reach the synth
    assert "anthropic/claude-opus-4-7" not in text
    assert "openai/gpt-5.5" not in text
    assert "gemini/gemini-3.1-pro-preview" not in text
    # Real slugs are hidden too — replaced by blind labels in the label
    # row. (Bodies stay verbatim; if a panellist happened to mention
    # its own slug in the body, that's the panellist's own leak —
    # `consult` doesn't try to scrub bodies.)
    assert "[panelist-alpha" not in text
    assert "[panelist-gamma" not in text
    # Filtered: EMPTY status entry was dropped before labels were assigned
    assert "panelist-beta" not in text  # filtered
    # The mapping covers exactly the 2 usable entries (OK + TRUNCATED)
    assert set(label_to_slug.values()) == {"panelist-alpha", "panelist-gamma"}
    assert len(label_to_slug) == 2
    # Each label appears in the input text somewhere
    for label in label_to_slug:
        assert label in text
    # Bodies survive the relabelling
    assert "alpha body" in text
    assert "gamma body" in text
    assert "rubric 2" in text  # only OK + TRUNCATED counted


def test_context_trim_synth_input_trims_largest_body_first():
    """When overall budget is tight, the largest body is trimmed first
    so we recover the most slack with the least per-body signal loss."""
    from consult import context as ctx

    big = "X" * 50_000
    small = "Y" * 1_000
    bodies = {"big": big, "small": small}
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt="prompt", bodies=bodies, overall_budget=15_000
    )
    assert len(new_bodies["small"]) == 1_000  # untouched
    assert len(new_bodies["big"]) < 50_000  # trimmed
    assert "TRIMMED" in new_bodies["big"]
    assert new_prompt == "prompt"  # prompt is the last resort


def test_context_trim_synth_input_passes_through_when_under_budget():
    from consult import context as ctx

    bodies = {"a": "x" * 1000, "b": "y" * 1000}
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt="prompt", bodies=bodies, overall_budget=100_000
    )
    assert new_prompt == "prompt"
    assert new_bodies == bodies


def test_context_trim_synth_input_trims_prompt_when_bodies_at_floor():
    """If every body is at the 5000-char floor and we're still over budget,
    the original prompt gets trimmed too — preserving the prompt is preferred
    but not at the cost of failing the synth call."""
    from consult import context as ctx

    bodies = {"a": "x" * 8000, "b": "y" * 8000}
    huge_prompt = "P" * 200_000
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt=huge_prompt, bodies=bodies, overall_budget=20_000
    )
    assert new_prompt is not None
    assert len(new_prompt) < len(huge_prompt)
    assert "TRIMMED" in new_prompt


@pytest.mark.asyncio
async def test_synth_writes_synth_input_with_original_prompt(tmp_path, monkeypatch):
    """End-to-end fidelity check: after a fanout+synth, the persisted
    synth_input.txt contains the original prompt and all panellist bodies —
    the synthesiser can fact-check claims against the source."""
    import consult.synth as synth_mod
    from consult import context as ctx
    from consult.types import ManifestEntry, Status

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    src = "Review this PR diff: <SIGNATURE-TOKEN-FOR-TEST>"
    ctx.write(paths, ctx.build(src, blinded=False))
    paths.prompt_txt.write_text(src)

    # Synthetic 2-panellist manifest + bodies on disk
    manifest = [
        ManifestEntry(
            slug="alpha",
            model_id="anthropic/claude-opus-4-7",
            status=Status.OK,
            resource_uri=paths.resource_uri("alpha"),
            body_path=str(paths.response_text("alpha")),
            latency_ms=100,
            cost_known=True,
        ),
        ManifestEntry(
            slug="beta",
            model_id="openai/gpt-5.5-pro",
            status=Status.OK,
            resource_uri=paths.resource_uri("beta"),
            body_path=str(paths.response_text("beta")),
            latency_ms=120,
            cost_known=True,
        ),
    ]
    paths.response_text("alpha").write_text("Found a race condition at line 42.")
    paths.response_text("beta").write_text("Found a SQL injection at line 117.")
    artifacts.write_manifest(
        paths,
        {
            "run_id": paths.run_id,
            "artifacts_dir": str(paths.root),
            "manifest": [m.model_dump(mode="json") for m in manifest],
            "cost_usd": 0.0,
            "cost_known": True,
            "wall_ms": 0,
            "partial": False,
            "blinded": False,
        },
    )

    # Stub out the actual LiteLLM call — we only care about what gets written
    # to synth_input.txt.
    async def fake_acompletion(**kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="STUBBED SYNTHESIS"),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setattr("consult.synth.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("consult.synth.litellm.completion_cost", lambda completion_response: 0.0)

    await synth_mod.synthesise(paths.run_id)

    synth_input = (paths.root / "synth_input.txt").read_text()
    # The original prompt is now in synth_input — synthesiser can fact-check
    assert "SIGNATURE-TOKEN-FOR-TEST" in synth_input
    # Both panellist bodies present
    assert "race condition at line 42" in synth_input
    assert "SQL injection at line 117" in synth_input


def test_synth_build_input_prepends_original_prompt_section():
    """Unit-level: _build_input puts the original prompt before the rubric/bodies."""
    from consult.synth import _build_input

    manifest = [
        {"slug": "alpha", "model_id": "x", "persona": None, "confidence": None, "status": "OK"},
    ]
    bodies = {"alpha": "BODY-TEXT"}
    out, _label_map = _build_input(
        manifest,
        bodies,
        rubric="rubric for {n} responses",
        original_prompt="PROMPT-TEXT",
    )
    assert "Original question / source" in out
    assert "PROMPT-TEXT" in out
    assert "BODY-TEXT" in out
    assert out.index("PROMPT-TEXT") < out.index("BODY-TEXT")


def test_synth_build_input_without_original_prompt_is_legacy_shape():
    """Omitting original_prompt yields the pre-Phase-1 layout (rubric +
    responses only), so legacy runs render identically."""
    from consult.synth import _build_input

    manifest = [
        {"slug": "alpha", "model_id": "x", "persona": None, "confidence": None, "status": "OK"},
    ]
    bodies = {"alpha": "BODY-TEXT"}
    out, _label_map = _build_input(
        manifest,
        bodies,
        rubric="rubric for {n} responses",
    )
    assert "Original question / source" not in out
    assert "BODY-TEXT" in out


def test_synth_build_input_rubric_with_literal_braces_does_not_crash():
    """Regression: `_build_input` previously used `.format(n=...)`, which
    crashes when a user-supplied rubric contains literal `{}` (e.g. a JSON
    example). Switched to `.replace("{n}", ...)`."""
    from consult.synth import _build_input

    rubric_with_braces = 'You have {n} responses.\n\nExpected JSON shape: { "verdict": "ship" }'
    manifest = [
        {"slug": "alpha", "model_id": "x", "persona": None, "confidence": None, "status": "OK"},
    ]
    out, _label_map = _build_input(
        manifest,
        {"alpha": "body"},
        rubric=rubric_with_braces,
    )
    assert "1 responses" in out
    assert '{ "verdict": "ship" }' in out


def test_context_trim_synth_input_proportional_hard_trim_on_large_panel():
    """When N panellists × per-body floor exceeds the budget, the hard-trim
    pass shrinks bodies proportionally so the overall input fits."""
    from consult import context as ctx

    # 20 bodies × 10000 chars = 200000; budget 50000. Floor (5000) × 20 =
    # 100000, still over budget. Hard-trim kicks in.
    bodies = {f"slug-{i}": "X" * 10_000 for i in range(20)}
    new_prompt, new_bodies = ctx.trim_synth_input(
        original_prompt=None,
        bodies=bodies,
        overall_budget=50_000,
    )
    total = sum(len(b) for b in new_bodies.values())
    # Allow a small overhead per body for trim markers
    assert total <= 50_000 + 30 * 200, (total, "should fit within budget + marker overhead")


async def test_synth_defensive_extraction_on_unexpected_shape(tmp_path, monkeypatch):
    """A non-conformant provider response must surface the unavailable
    sentinel, not AttributeError/IndexError straight out of `synthesise`.
    """
    import litellm

    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    paths.prompt_txt.write_text("q")
    paths.response_text("alpha").write_text("body")
    manifest = [
        ManifestEntry(
            slug="alpha",
            model_id="m/x",
            status=Status.OK,
            resource_uri=paths.resource_uri("alpha"),
            body_path=str(paths.response_text("alpha")),
            latency_ms=0,
            cost_usd=0.0,
            cost_known=True,
            confidence=None,
            capsule=None,
        ),
    ]
    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=manifest,
        cost_usd=0.0,
        cost_known=True,
        wall_ms=0,
    )
    artifacts.write_manifest(paths, handle.model_dump())

    class Garbage:
        choices: list = []  # IndexError on resp.choices[0]

    async def garbage_acompletion(**kwargs):
        return Garbage()

    monkeypatch.setattr(litellm, "acompletion", garbage_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.0)

    result = await synth_mod.synthesise(paths.run_id)
    # Did not crash. Sentinel text written.
    assert "# Synthesis unavailable" in result.text
    assert "unexpected response shape" in result.text


@pytest.mark.asyncio
async def test_synth_persists_cost_to_manifest(tmp_path, monkeypatch):
    """Direct calls to synth.synthesise() must persist the synthesiser
    cost + badge to the run's manifest so consult-ledger sees it.
    Previously only orchestrate/refine/sequence did this; the standalone
    synthesise tool was a ledger blindspot.
    """
    from consult import synth as synth_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    # Build a minimal run with one usable body + manifest on disk.
    paths = artifacts.create_run()
    entry = ManifestEntry(
        slug="alpha",
        model_id="anthropic/x",
        status=Status.OK,
        resource_uri=paths.resource_uri("alpha"),
        body_path=str(paths.response_text("alpha")),
        latency_ms=10,
        cost_usd=0.005,
        cost_known=True,
    )
    paths.response_text("alpha").write_text("Some response.")
    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=[entry],
        cost_usd=0.005,
        cost_known=True,
        wall_ms=10,
    )
    artifacts.write_manifest(paths, handle.model_dump())

    # Stub the LiteLLM call so the synth path returns deterministic text + cost.
    class FakeMsg:
        content = "synth body"

    class FakeChoice:
        message = FakeMsg()
        finish_reason = "stop"

    class FakeResp:
        choices = [FakeChoice()]

    async def fake_acompletion(**kwargs):
        return FakeResp()

    monkeypatch.setattr(synth_mod.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        synth_mod.litellm,
        "completion_cost",
        lambda **kw: 0.05,
    )

    result = await synth_mod.synthesise(paths.run_id)
    assert result.status is synth_mod.SynthStatus.OK
    assert abs(result.cost_usd - 0.05) < 1e-9

    # Manifest on disk now reflects the synth spend + synthesiser badge.
    on_disk = json.loads(paths.manifest_json.read_text())
    assert abs(on_disk["cost_usd"] - 0.05) < 1e-9
    assert "synthesiser" in on_disk
