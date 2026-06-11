"""Panel cost estimation against LiteLLM's price tables.

`aestimate_cost` resolves `estimate_cost` through the package facade at
call time, so tests that monkeypatch `consult.runner.estimate_cost` with
a plain lambda keep affecting the async path — the documented contract
from the single-module days.
"""

from __future__ import annotations

import asyncio
import logging

import litellm

from consult import runner as _facade

from .. import registry
from ..capsule import MAX_TOKENS_BY_KIND
from ..redact import redact_exc
from ..types import ModelSpec

logger = logging.getLogger(__name__)


async def aestimate_cost(
    specs: list[ModelSpec],
    prompt: str,
    *,
    capsule_kind: str = "decision",
) -> tuple[float, bool]:
    """Async wrapper around `estimate_cost`.

    `litellm.token_counter` is blocking and on a cache miss takes 50-200ms
    while it loads the tokenizer; large panels with new aliases can stall
    heartbeat ticks for noticeable real-time. Offloading to a thread keeps
    the event loop responsive. Test monkeypatches still bind to the sync
    `estimate_cost` symbol — this wrapper picks up whatever's currently
    bound there, so test setup is unchanged.
    """
    return await asyncio.to_thread(_facade.estimate_cost, specs, prompt, capsule_kind=capsule_kind)


def estimate_cost(
    specs: list[ModelSpec],
    prompt: str,
    *,
    capsule_kind: str = "decision",
) -> tuple[float, bool]:
    """Returns (total_estimate, all_known).

    Uses LiteLLM's per-token price tables via `cost_per_token()`. Models that
    don't have pricing data set `all_known=False`; caller must treat unknown
    costs conservatively (a panel with even one unknown-cost spec cannot be
    validated against `max_run_usd`).

    Unknown-alias specs are treated as cost-unknown (rather than raising) so
    a single typo can't abort the panel here; `_call_one` surfaces the alias
    as a per-spec Status.ERROR.

    Sync by design so tests can monkeypatch it with a plain lambda; the
    async fan-out paths call `aestimate_cost()` to keep the event loop
    free during the blocking `token_counter` lookup.
    """
    total = 0.0
    all_known = True
    for spec in specs:
        try:
            entry = registry.resolve_model(spec.model)
        except KeyError:
            all_known = False
            continue
        # CLI panellists have no per-call dollar cost: the user's CLI
        # auth covers usage. Estimating them as $0 is honest (not
        # cost-unknown — that would inappropriately make the cap-check
        # conservative for what is genuinely free at this layer).
        if entry.get("provider") == "cli":
            continue
        litellm_id = entry.get("litellm_id")
        if not litellm_id:
            all_known = False
            continue
        try:
            tin = litellm.token_counter(model=litellm_id, text=prompt)
            # Match _call_one: estimate output by capsule_kind, not per-model
            # default. Keeps the cap-check honest after the dimension flip.
            tout = MAX_TOKENS_BY_KIND.get(capsule_kind, MAX_TOKENS_BY_KIND["decision"])
            # cost_per_token returns the TOTAL prompt/completion cost for
            # the given token counts, not per-token rates.
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=litellm_id, prompt_tokens=tin, completion_tokens=tout
            )
            if prompt_cost is None or completion_cost is None:
                all_known = False
                continue
            total += prompt_cost + completion_cost
        except Exception as e:  # noqa: BLE001
            logger.warning("estimate_cost: no price for %s (%s)", litellm_id, redact_exc(e))
            all_known = False
            continue
    return total, all_known
