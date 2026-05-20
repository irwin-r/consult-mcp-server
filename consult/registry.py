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

_PKG_CONFIG = Path(__file__).parent / "config"
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
    # Allow raw LiteLLM IDs like "openrouter/x-ai/grok-4.3" — synthesise a row.
    if "/" in alias_or_id or alias_or_id.startswith(("gpt-", "claude-", "gemini-")):
        return {
            "alias": alias_or_id,
            "litellm_id": alias_or_id,
            "default_budget_tokens": 8000,
            "default_timeout_s": 180,
            "provider": _infer_provider(alias_or_id),
        }
    raise KeyError(f"Unknown model: {alias_or_id}")


def _infer_provider(litellm_id: str) -> str:
    """Best-effort provider inference for raw IDs without an explicit prefix.

    `litellm_id` with a `/` uses the leading segment as the provider —
    matches LiteLLM's own routing. For bare names like
    `claude-3-5-sonnet-latest` (allowed because LiteLLM accepts them as
    Anthropic shortcuts), defaulting to "openai" wrongly puts the call
    into the OpenAI rate-limit bucket and disables Anthropic cache_control
    handling. Map a few known prefixes; fall back to openai only when
    nothing matches (preserves the prior behaviour for the OpenAI shortcuts).
    """
    if "/" in litellm_id:
        return litellm_id.split("/")[0]
    if litellm_id.startswith("claude-"):
        return "anthropic"
    if litellm_id.startswith("gemini-"):
        return "google"
    if litellm_id.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    return "openai"


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


def list_rubrics() -> list[str]:
    """List available rubric names (user overrides + package defaults)."""
    seen: set[str] = set()
    for d in (_USER_CONFIG / "rubrics", _PKG_CONFIG / "rubrics"):
        if d.exists():
            for p in d.glob("*.md"):
                seen.add(p.stem)
    return sorted(seen)


def resolve_rubric(name_or_text: str) -> str:
    """Resolve a rubric reference.

    Lookup order (user overrides first, then package defaults):
    1. `~/.consult/rubrics/<name>.md`
    2. `consult/config/rubrics/<name>.md`
    3. Otherwise return `name_or_text` as a literal rubric string — preserves
       backwards compat with `synthesise(rubric=<literal multi-line string>)`.

    Callers passing an unknown short name get the literal back; that will
    show up clearly in `synth_input.txt`. Use `list_rubrics()` to discover
    the available named rubrics.
    """
    user_path = _USER_CONFIG / "rubrics" / f"{name_or_text}.md"
    if user_path.exists():
        return user_path.read_text()
    pkg_path = _PKG_CONFIG / "rubrics" / f"{name_or_text}.md"
    if pkg_path.exists():
        return pkg_path.read_text()
    return name_or_text


def provider_concurrency() -> dict[str, int]:
    """Returns provider → max concurrent in-flight LiteLLM calls.

    Sourced from `models.json:defaults.concurrency`, then overlaid with the
    `CONSULT_PROVIDER_CONCURRENCY` env var (comma-separated `provider:N` pairs).
    The key `"default"` applies to any provider not explicitly listed — useful
    for raw LiteLLM IDs that resolve to providers absent from the registry.

    `openai` defaults to 2 because the FRICTION log records OpenAI rate-limits
    on every panel run from a shared key; the other providers haven't shown
    the same pattern and default to 5.

    Any non-positive value (env typo of `openai:0`, negative number) is
    floored to 1: a Semaphore(0) blocks the first acquire forever, which
    would silently hang every panellist for that provider past the per-call
    timeout. Floor=1 surfaces as "everything serialised through that
    provider" — slow, but not a deadlock.
    """
    cfg_caps = models_config().get("defaults", {}).get("concurrency", {})
    out: dict[str, int] = {"default": 5}
    for k, v in cfg_caps.items():
        out[k] = max(1, int(v))
    env = os.environ.get("CONSULT_PROVIDER_CONCURRENCY")
    if env:
        for pair in env.split(","):
            if ":" not in pair:
                continue
            provider, limit = pair.split(":", 1)
            try:
                out[provider.strip()] = max(1, int(limit))
            except ValueError:
                continue
    return out
