"""Model + stance registry loader. Sources from config/*.json next to the
package, with optional overrides at ~/.consult/models.json and
~/.consult/stances.json.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_PKG_CONFIG = Path(__file__).parent.parent / "config"
_USER_CONFIG = Path(os.path.expanduser("~/.consult"))


def _load_json(name: str) -> dict[str, Any]:
    user = _USER_CONFIG / name
    if user.exists():
        return json.loads(user.read_text())
    pkg = _PKG_CONFIG / name
    return json.loads(pkg.read_text())


@lru_cache(maxsize=1)
def models_config() -> dict[str, Any]:
    return _load_json("models.json")


@lru_cache(maxsize=1)
def stances_config() -> dict[str, str]:
    return _load_json("stances.json")


def resolve_model(alias_or_id: str) -> dict[str, Any]:
    """Look up by registry alias first, then accept a raw LiteLLM ID.

    Returns a dict with at least {alias, litellm_id, default_budget_tokens,
    default_timeout_s, provider}. Raises KeyError if neither matches and the
    string doesn't look like a LiteLLM ID.
    """
    cfg = models_config()
    models = cfg["models"]
    if alias_or_id in models:
        entry = dict(models[alias_or_id])
        entry["alias"] = alias_or_id
        return entry
    # Allow raw LiteLLM IDs like "openrouter/x-ai/grok-4.3" — synthesise a row
    if "/" in alias_or_id or alias_or_id.startswith(("gpt-", "claude-", "gemini-")):
        return {
            "alias": alias_or_id,
            "litellm_id": alias_or_id,
            "default_budget_tokens": 8000,
            "default_timeout_s": 180,
            "provider": alias_or_id.split("/")[0] if "/" in alias_or_id else "openai",
        }
    raise KeyError(f"Unknown model: {alias_or_id}")


def resolve_tier(tier: str) -> list[str]:
    cfg = models_config()
    tiers = cfg.get("tiers", {})
    if tier not in tiers:
        raise KeyError(f"Unknown tier: {tier}. Available: {list(tiers)}")
    return list(tiers[tier])


def resolve_stance(key_or_prompt: str | None) -> str:
    """Return the stance prompt text. If `key_or_prompt` matches a registered
    key, return that. Otherwise treat it as a literal prompt (custom stance).
    """
    if not key_or_prompt:
        return ""
    stances = stances_config()
    if key_or_prompt in stances:
        return stances[key_or_prompt]
    return key_or_prompt


def default_synthesiser() -> str:
    return models_config().get("defaults", {}).get("synthesiser", "gemini-pro")


def default_capsule_extractor() -> str:
    return models_config().get("defaults", {}).get("capsule_extractor", "claude-haiku")


def default_max_run_usd() -> float:
    env = os.environ.get("CONSULT_MAX_RUN_USD")
    if env:
        return float(env)
    return float(models_config().get("defaults", {}).get("max_run_usd", 5.0))
