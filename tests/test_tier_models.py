"""Tier panels must use models the engine can actually call.

A `mode=responses` model (e.g. openai/gpt-5.5-pro, openai/gpt-5.3-codex) 404s on
chat completions, so the engine routes it through the Responses adapter — but
only when its registry entry is marked `"mode": "responses"`. This test ensures
every tiered model is either chat-completable or explicitly marked for Responses
routing, and that none uses the legacy `text-completion-openai/` endpoint.
"""

from __future__ import annotations

import litellm

from consult import registry


def test_tier_responses_models_are_marked_for_routing():
    cfg = registry.models_config()
    models = cfg["models"]
    tiered = {alias for members in cfg["tiers"].values() for alias in members}
    assert tiered, "expected at least one tier with members"

    offenders: list[str] = []
    for alias in sorted(tiered):
        entry = models.get(alias)
        assert entry is not None, f"tier references unknown model alias {alias!r}"
        lid = entry["litellm_id"]
        if lid.startswith("text-completion-openai/"):
            offenders.append(f"{alias} ({lid}): legacy text-completion endpoint")
            continue
        try:
            mode = litellm.get_model_info(lid).get("mode")
        except Exception:
            mode = None  # openrouter / unknown ids — no metadata, assumed chat
        if mode == "responses" and entry.get("mode") != "responses":
            offenders.append(
                f"{alias} ({lid}): mode=responses but the registry entry is not marked "
                '"mode": "responses", so the engine would call it as chat and 404'
            )

    assert not offenders, "tier models the engine can't call:\n  " + "\n  ".join(offenders)


def test_gpt_pro_excluded_from_standard_tier():
    """gpt-pro (Responses route, medium reasoning) burns its whole completion
    budget on reasoning and truncates with an empty capsule on standard panels,
    where it was the most expensive panellist by ~8x for zero usable value
    (issue #67). It stays in deep/review, which opt into heavier, costlier runs.
    """
    tiers = registry.models_config()["tiers"]
    assert "gpt-pro" not in tiers["standard"]
    assert "gpt-pro" in tiers["deep"]
    assert "gpt-pro" in tiers["review"]
