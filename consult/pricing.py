"""Gap-fill LiteLLM's price tables from models.json `pricing` blocks.

LiteLLM ships a static price/context table that lags provider launches by
days to weeks. A registry model missing from that table degrades two
guards at once: `estimate_cost` returns `all_known=False` so the
`max_run_usd` gate can't validate the panel, and post-call accounting
(`litellm.completion_cost` in fanout/synth/refine/peer_rank) logs a
warning and under-reports actual spend. The 2026-07 registry refresh made
this the common case — 13 of 24 packaged models were unknown to the
shipped table.

The fix: models.json entries may carry a `pricing` block
(`{"input_usd_per_mtok": X, "output_usd_per_mtok": Y}`) plus the existing
top-level `max_input_tokens` override. `ensure_registered()` feeds those
into `litellm.register_model` once per process.

Gap-fill only: an ID LiteLLM already prices keeps the shipped numbers.
The library's table updates with upgrades; the registry block is the
fallback for models newer than the installed tables, not an override.
Best-effort by design — a malformed block logs and skips, and estimation
degrades to `all_known=False` exactly as before.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from . import registry
from .redact import redact_exc

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def ensure_registered() -> int:
    """Register `pricing` blocks with LiteLLM; returns how many were added.

    `lru_cache` makes repeat calls free, so hot paths (cost estimation,
    fan-out, synthesis) call this unconditionally at entry. Tests that
    swap the registry config should `ensure_registered.cache_clear()`
    alongside the registry cache clears.
    """
    import litellm  # deferred: keep `consult.registry` importable without litellm

    registered = 0
    for _alias, entry in registry.models_config().get("models", {}).items():
        pricing = entry.get("pricing")
        litellm_id = entry.get("litellm_id")
        if not (isinstance(pricing, dict) and litellm_id):
            continue
        try:
            litellm.cost_per_token(model=litellm_id, prompt_tokens=1, completion_tokens=1)
            continue  # already priced — shipped tables win
        except Exception:  # noqa: BLE001 — unknown model, fall through and register
            pass
        try:
            row: dict[str, object] = {
                "input_cost_per_token": float(pricing["input_usd_per_mtok"]) / 1e6,
                "output_cost_per_token": float(pricing["output_usd_per_mtok"]) / 1e6,
                "litellm_provider": str(litellm_id).split("/", 1)[0],
                "mode": "responses" if entry.get("mode") == "responses" else "chat",
            }
            if isinstance(entry.get("max_input_tokens"), int):
                row["max_input_tokens"] = entry["max_input_tokens"]
            litellm.register_model({str(litellm_id): row})
            registered += 1
        except Exception as e:  # noqa: BLE001 — never block the run on a bad block
            logger.warning("pricing: could not register %s (%s)", litellm_id, redact_exc(e))
    if registered:
        logger.info("pricing: registered %d model(s) missing from LiteLLM tables", registered)
    return registered
