"""Registry loader tests — user-override merge semantics."""

from __future__ import annotations

import json

import pytest

from consult import registry


@pytest.fixture()
def user_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_USER_CONFIG", tmp_path)
    return tmp_path


def test_no_user_file_returns_packaged_config(user_config_dir):
    cfg = registry._load_json("models.json")
    assert "claude-haiku" in cfg["models"]
    assert "quick" in cfg["tiers"]


def test_user_file_merges_over_packaged(user_config_dir):
    """A user file with one extra model keeps every packaged model, tier,
    and default — the old replace-wholesale behaviour dropped them all.
    """
    (user_config_dir / "models.json").write_text(
        json.dumps(
            {
                "models": {
                    "my-local": {
                        "litellm_id": "ollama/llama3",
                        "provider": "ollama",
                    }
                }
            }
        )
    )
    cfg = registry._load_json("models.json")
    assert "my-local" in cfg["models"]
    assert "claude-haiku" in cfg["models"]  # packaged survives
    assert "quick" in cfg["tiers"]  # untouched sections survive
    assert cfg["defaults"]["synthesiser"]  # defaults survive


def test_user_override_of_single_field_keeps_other_fields(user_config_dir):
    (user_config_dir / "models.json").write_text(
        json.dumps({"models": {"claude-haiku": {"default_timeout_s": 999}}})
    )
    cfg = registry._load_json("models.json")
    entry = cfg["models"]["claude-haiku"]
    assert entry["default_timeout_s"] == 999
    assert entry["litellm_id"].startswith("anthropic/")  # not clobbered


def test_null_deletes_packaged_key(user_config_dir):
    """JSON null is the explicit remove-this-entry escape hatch — e.g. an
    operator stripping aggregator models for privacy.
    """
    (user_config_dir / "models.json").write_text(json.dumps({"models": {"deepseek": None}}))
    cfg = registry._load_json("models.json")
    assert "deepseek" not in cfg["models"]
    assert "claude-haiku" in cfg["models"]


def test_tier_list_replaces_wholesale(user_config_dir):
    (user_config_dir / "models.json").write_text(json.dumps({"tiers": {"quick": ["claude-haiku"]}}))
    cfg = registry._load_json("models.json")
    assert cfg["tiers"]["quick"] == ["claude-haiku"]
    assert len(cfg["tiers"]["standard"]) > 1  # other tiers untouched


def test_non_object_user_file_raises(user_config_dir):
    (user_config_dir / "models.json").write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError, match="JSON object"):
        registry._load_json("models.json")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}, "keep": True}
    overlay = {"a": {"b": 9, "c": None}}
    merged = registry._deep_merge(base, overlay)
    assert merged == {"a": {"b": 9}, "keep": True}
    assert base == {"a": {"b": 1, "c": 2}, "keep": True}
    assert overlay == {"a": {"b": 9, "c": None}}


def test_stances_merge_too(user_config_dir):
    (user_config_dir / "stances.json").write_text(json.dumps({"pirate": "Argue like a pirate."}))
    cfg = registry._load_json("stances.json")
    assert cfg["pirate"] == "Argue like a pirate."
    assert "security" in cfg  # packaged stances survive
