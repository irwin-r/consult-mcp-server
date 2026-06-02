"""Tier panels must use chat-completable models.

A model whose litellm id is `mode=responses` (e.g. openai/gpt-5.5-pro) or uses
the legacy `text-completion-openai/` endpoint (e.g. gpt-5.3-codex) 404s on the
chat-completions endpoint the engine drives via `litellm.acompletion`. Those
must not appear in any default tier. This guards against re-adding that class
of model to a tier — the gpt-pro / gpt-codex 404 a live panel hit.
"""

from __future__ import annotations

import litellm

from consult import registry


def test_no_tier_model_uses_responses_or_text_completion_endpoint():
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
        if mode == "responses":
            offenders.append(f"{alias} ({lid}): mode=responses, needs the Responses API")

    assert not offenders, "tier models not chat-completable:\n  " + "\n  ".join(offenders)
