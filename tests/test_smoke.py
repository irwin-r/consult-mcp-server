"""Smoke tests. The unit slice runs offline (no API keys); the live slice
hits real providers only when relevant API keys are present.

Run: `pytest -v`
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from consult import artifacts, refine as refine_mod, registry
from consult.runner import _build_per_slug_prompt, _make_slug, estimate_cost
from consult.status import classify
from consult.types import ArbiterVerdict, Capsule, ManifestEntry, ModelSpec, RunHandle, Status


# ---- Pure-Python tests (no network) ----------------------------------------


def test_registry_loads_defaults():
    cfg = registry.models_config()
    assert "models" in cfg
    assert "claude-haiku" in cfg["models"]
    assert "tiers" in cfg
    assert "quick" in cfg["tiers"]


def test_registry_resolve_alias_and_raw_id():
    entry = registry.resolve_model("claude-haiku")
    assert entry["litellm_id"].startswith("anthropic/")
    raw = registry.resolve_model("openrouter/some/model")
    assert raw["litellm_id"] == "openrouter/some/model"


def test_stance_lookup_and_passthrough():
    assert "security" in registry.resolve_stance("security")
    assert registry.resolve_stance("You are a freeform stance.") == "You are a freeform stance."
    assert registry.resolve_stance(None) == ""


def test_slug_and_prompt_assembly():
    spec = ModelSpec(model="claude-haiku", stance="security")
    assert _make_slug(spec, 0, blinded=False).startswith("claude-haiku")
    assert _make_slug(spec, 2, blinded=True) == "panelist-gamma"


def test_footer_injected_on_every_prompt():
    """Capsule confidence extraction depends on the footer being present."""
    with_stance = _build_per_slug_prompt("How to ship X?", "You are an SRE.")
    assert with_stance.startswith("You are an SRE.")
    assert "How to ship X?" in with_stance
    assert "CONFIDENCE:" in with_stance
    assert "KEY_REASON:" in with_stance

    no_stance = _build_per_slug_prompt("How to ship X?", "")
    assert no_stance.startswith("How to ship X?")
    assert "CONFIDENCE:" in no_stance
    assert "KEY_REASON:" in no_stance


def test_status_classifier_handles_exceptions():
    s, _, _ = classify(None, exception=RuntimeError("rate limit hit"))
    assert s == Status.RATE_LIMITED
    s, _, _ = classify(None, exception=TimeoutError("timed out"))
    assert s == Status.TIMEOUT


def test_run_handle_usable_parametric():
    entries = [
        ManifestEntry(
            slug=f"m{i}", model_id=f"openrouter/x{i}/y", status=Status.OK,
            resource_uri="consult://x", body_path="/tmp/x",
        )
        for i in range(4)
    ]
    entries[3].model_id = "openai/gpt-5"
    handle = RunHandle(
        run_id="t", artifacts_dir="/tmp/t", manifest=entries, cost_usd=0, wall_ms=0
    )
    assert handle.usable() is True
    # Set all-OR — only one provider
    for e in entries:
        e.model_id = "openrouter/foo/bar"
    assert handle.usable(min_providers=2) is False


def test_artifacts_create_and_uri():
    paths = artifacts.create_run()
    assert paths.root.exists()
    assert paths.responses.exists()
    uri = paths.resource_uri("alpha")
    rid, slug = artifacts.parse_resource_uri(uri)
    assert rid == paths.run_id
    assert slug == "alpha"
    # cleanup
    import shutil
    shutil.rmtree(paths.root)


# ---- Refine offline tests --------------------------------------------------


def test_refine_suffix_specs_round_indexes_slugs():
    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="grok", stance="contrarian")]
    r2 = refine_mod._suffix_specs(specs, 2)
    assert r2[0].slug.endswith(".r2")
    assert r2[1].slug.endswith(".r2")
    # Same model index should produce stable base slug
    r2b = refine_mod._suffix_specs(specs, 2)
    assert [s.slug for s in r2] == [s.slug for s in r2b]


def test_refinement_prompt_includes_gaps_and_focus():
    manifest = [
        ManifestEntry(
            slug="m-1.r1",
            model_id="anthropic/x",
            status=Status.OK,
            resource_uri="consult://x",
            body_path="/tmp/x",
            capsule=Capsule(position="A says X", recommendation="do X"),
        )
    ]
    verdict = ArbiterVerdict(
        round=1, score=0.4, gaps=["cost not discussed"], next_round_focus="address cost"
    )
    out = refine_mod._build_refinement_prompt("Should we ship X?", 2, manifest, verdict)
    assert "Should we ship X?" in out
    assert "cost not discussed" in out
    assert "address cost" in out
    assert "m-1.r1" in out


def test_refine_validates_max_rounds():
    with pytest.raises(ValueError, match="max_rounds"):
        asyncio.get_event_loop().run_until_complete(
            refine_mod.refine("q", [ModelSpec(model="claude-haiku")], max_rounds=5)
        )


def test_arbiter_json_extractor_tolerates_fences():
    fenced = '```json\n{"score": 0.7, "gaps": ["x"], "next_round_focus": "", "reasoning": ""}\n```'
    data = refine_mod._extract_json(fenced)
    assert data["score"] == 0.7
    assert data["gaps"] == ["x"]


# ---- Live tests (gated on API keys) ----------------------------------------


HAVE_KEYS = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
)


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
def test_estimate_cost_smoke():
    specs = [ModelSpec(model="claude-haiku")]
    est = estimate_cost(specs, "say hello in five words")
    # Cost should be > 0 if LiteLLM knows the price; allow 0 since prices change
    assert est >= 0


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
@pytest.mark.asyncio
async def test_tiny_panel_dry_run():
    from consult.runner import fanout

    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="openrouter/x-ai/grok-4.3")]
    handle = await fanout("ping", specs, dry_run=True)
    assert handle.partial
    assert "dry_run" in (handle.partial_reason or "")
