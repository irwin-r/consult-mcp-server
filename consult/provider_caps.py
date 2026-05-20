"""Per-provider capability flags.

Some providers reject parameters that the rest of the panel accepts. The
canonical case is `temperature`: Gemini-3 warns against values < 1.0 and
the newest Claude Opus rejects the parameter outright. A substring match
on the LiteLLM ID is brittle (caught the first case, missed the second),
so the panel uses this table — patternable via models.json or an env
override — to decide which kwargs to send.

Patterns are case-insensitive substring matches against the LiteLLM ID
(provider/model). The defaults below capture today's known
non-temperature-accepting models; operators extend the list via
~/.consult/models.json:defaults.capabilities.no_temperature without
patching code.
"""

from __future__ import annotations

import os
from functools import lru_cache

from . import registry

# Built-in deny list. Substrings, case-insensitive, matched against the
# resolved LiteLLM ID. `gemini` covers `gemini/gemini-...` and
# `openrouter/google/gemini-...`. `claude-opus-4-7` covers the specific
# Anthropic model that started rejecting temperature; `claude-opus-5`
# is included pre-emptively for the next bump.
_DEFAULT_NO_TEMPERATURE = (
    "gemini",
    "claude-opus-4-7",
    "claude-opus-5",
    # GPT-5 reasoning models reject temperature when reasoning_effort is set
    "gpt-5",
)


@lru_cache(maxsize=1)
def _no_temperature_patterns() -> tuple[str, ...]:
    """Combined deny list: built-in + models.json overlay + env override.

    Env `CONSULT_NO_TEMPERATURE` is a comma-separated list of substrings;
    `CONSULT_NO_TEMPERATURE_RESET=1` discards built-ins and uses only the
    env value (escape hatch when a built-in pattern is wrong).
    """
    patterns: list[str] = []
    if os.environ.get("CONSULT_NO_TEMPERATURE_RESET", "0") != "1":
        patterns.extend(_DEFAULT_NO_TEMPERATURE)
    cfg_caps = registry.models_config().get("defaults", {}).get("capabilities", {})
    patterns.extend(cfg_caps.get("no_temperature", []) or [])
    env = os.environ.get("CONSULT_NO_TEMPERATURE")
    if env:
        patterns.extend(p.strip() for p in env.split(",") if p.strip())
    return tuple(p.lower() for p in patterns)


def supports_temperature(litellm_id: str) -> bool:
    """Return True if the provider/model accepts a `temperature` kwarg.

    Conservative on unknowns: returns True by default (matches LiteLLM's
    behaviour) so a previously-working unknown model keeps working. Override
    via `CONSULT_NO_TEMPERATURE` or models.json:defaults.capabilities.no_temperature.
    """
    if not litellm_id:
        return True
    lower = litellm_id.lower()
    return all(pat not in lower for pat in _no_temperature_patterns())


def apply_temperature(kwargs: dict, litellm_id: str, value: float) -> dict:
    """Mutate `kwargs` in place to add temperature when the provider supports it.

    Returns `kwargs` for chaining. Centralised so capsule/synth/refine never
    re-implement the substring check. The mutation is local — callers that
    care about purity should pass a copy.
    """
    if supports_temperature(litellm_id):
        kwargs["temperature"] = value
    return kwargs


def reset_cache() -> None:
    """Drop the cached pattern list. For tests that mutate the env mid-run."""
    _no_temperature_patterns.cache_clear()
