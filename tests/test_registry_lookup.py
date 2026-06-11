"""Model/tier/stance resolution.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

from consult import registry
from consult.types import ModelSpec


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


def test_expand_specs_multi_instance_and_passthrough():
    """`model:N` sugar expands to N specs; bare model strings pass through.
    Stance and custom slug are preserved on every expanded instance.
    """
    from consult.runner import expand_specs

    raw = [
        ModelSpec(model="claude-haiku:3", stance="skeptic"),
        ModelSpec(model="gpt-pro"),
        ModelSpec(model="openrouter/foo/bar"),  # no colon — passthrough
    ]
    expanded = expand_specs(raw)
    assert len(expanded) == 5  # 3 + 1 + 1
    assert [s.model for s in expanded] == [
        "claude-haiku",
        "claude-haiku",
        "claude-haiku",
        "gpt-pro",
        "openrouter/foo/bar",
    ]
    # Stance survives expansion
    assert all(s.stance == "skeptic" for s in expanded[:3])

    # Idempotent: re-expanding already-expanded specs is a no-op
    assert expand_specs(expanded) == expanded


def test_registry_resolve_rubric_loads_named_packaged_rubrics():
    """The four packaged rubrics resolve by name."""
    from consult import registry as reg

    for name in ("consensus", "code_review", "research_brief", "critique"):
        text = reg.resolve_rubric(name)
        assert "{n}" in text, f"rubric {name} should have an n-placeholder"
        # Should look like a real rubric, not a one-line stub
        assert len(text) > 200, f"rubric {name} suspiciously short: {len(text)}"


def test_registry_resolve_rubric_passes_literal_through():
    """Unknown names → returned as-is (literal rubric)."""
    from consult import registry as reg

    literal = "You have {n} responses. Write a haiku."
    assert reg.resolve_rubric(literal) == literal


def test_registry_list_rubrics_includes_packaged():
    from consult import registry as reg

    rubrics = reg.list_rubrics()
    for expected in ("consensus", "code_review", "research_brief", "critique"):
        assert expected in rubrics, (expected, rubrics)


def test_max_input_tokens_registry_override():
    """Explicit max_input_tokens in the registry entry wins over LiteLLM."""
    from consult.runner import _max_input_tokens

    entry = {"max_input_tokens": 50_000}
    assert _max_input_tokens("openrouter/some/model", entry) == 50_000
