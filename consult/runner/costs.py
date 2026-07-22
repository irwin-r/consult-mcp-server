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

from .. import pricing, registry
from ..redact import redact_exc
from ..types import ModelSpec
from .specs import output_budget

logger = logging.getLogger(__name__)


async def aestimate_cost(
    specs: list[ModelSpec],
    prompt: str,
    *,
    capsule_kind: str = "decision",
    max_output_tokens: int | None = None,
) -> tuple[float, bool]:
    """Async wrapper around `estimate_cost`.

    `litellm.token_counter` is blocking and on a cache miss takes 50-200ms
    while it loads the tokenizer; large panels with new aliases can stall
    heartbeat ticks for noticeable real-time. Offloading to a thread keeps
    the event loop responsive. Test monkeypatches still bind to the sync
    `estimate_cost` symbol — this wrapper picks up whatever's currently
    bound there, so test setup is unchanged. `max_output_tokens` is
    forwarded only when set, so patched lambdas without the kwarg survive.
    """
    kwargs: dict[str, object] = {"capsule_kind": capsule_kind}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return await asyncio.to_thread(lambda: _facade.estimate_cost(specs, prompt, **kwargs))


def estimate_cost(
    specs: list[ModelSpec],
    prompt: str,
    *,
    capsule_kind: str = "decision",
    max_output_tokens: int | None = None,
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
    pricing.ensure_registered()
    total = 0.0
    all_known = True
    for spec in specs:
        try:
            entry = registry.resolve_model(spec.model)
        except KeyError:
            all_known = False
            continue
        litellm_id = entry.get("litellm_id")
        if not litellm_id:
            all_known = False
            continue
        try:
            tin = litellm.token_counter(model=litellm_id, text=prompt)
            # Match _call_one: the estimate prices the same `output_budget`
            # ceiling the call will actually grant. Conservative by design —
            # the gate must hold even if the model fills its whole budget.
            tout = output_budget(entry, capsule_kind, max_output_tokens)
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


def estimate_drivers(
    specs: list[ModelSpec],
    prompt: str,
    *,
    capsule_kind: str = "decision",
    max_output_tokens: int | None = None,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Per-spec cost estimates, highest first, for the over-cap message.

    With per-model output budgets, one expensive flagship can dominate the
    panel estimate; a bare "estimated cost exceeds cap" rejection gives the
    caller nothing to act on. Naming the drivers turns the block into a
    choice: raise the cap, or drop/replace the named models.

    Only known-priced specs appear (an unpriced spec can't drive the
    estimate the gate sees). Best-effort: any per-spec failure just omits
    that spec.
    """
    pricing.ensure_registered()
    per_spec: list[tuple[str, float]] = []
    for spec in specs:
        try:
            entry = registry.resolve_model(spec.model)
            litellm_id = entry.get("litellm_id")
            if not litellm_id:
                continue
            tin = litellm.token_counter(model=litellm_id, text=prompt)
            tout = output_budget(entry, capsule_kind, max_output_tokens)
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=litellm_id, prompt_tokens=tin, completion_tokens=tout
            )
            if prompt_cost is None or completion_cost is None:
                continue
            per_spec.append((spec.model, prompt_cost + completion_cost))
        except Exception:  # noqa: BLE001 — message enrichment only
            continue
    per_spec.sort(key=lambda kv: kv[1], reverse=True)
    return per_spec[:top_n]
