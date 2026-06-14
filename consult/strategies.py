"""Pluggable refine strategies.

The default refine flow runs the same panel each round until the arbiter
converges (or the round cap is hit). Strategies are a per-round hook that
lets callers customise *what happens between rounds* — which panellists
carry over, what weight each gets — without forking refine.refine().

Inspired by llm-consortium's strategy plugin system. consult ships only
the default for now:

- "default" — every panellist runs every round (current behaviour).

The "elimination" strategy (drop the most-divergent panellist each round)
was retired as unused complexity (issue #59); the hook stays so a future
strategy is a matter of subclassing `Strategy` and registering it in
`_STRATEGIES`. Refine.refine() consults the strategy via `before_round(...)`
to decide what panel to run this round.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import ManifestEntry, ModelSpec


class Strategy(ABC):
    """Per-round customisation hook for `refine.refine()`.

    Strategies are stateful across rounds — they receive the prior round's
    manifest (with capsules) and decide what the next round's panel should
    look like.
    """

    @abstractmethod
    def before_round(
        self,
        *,
        round_num: int,
        base_specs: list[ModelSpec],
        prior_manifest: list[ManifestEntry] | None,
    ) -> list[ModelSpec]:
        """Return the spec list for round `round_num`.

        Round 1 always sees the full `base_specs` — strategies typically
        only filter / reweight from round 2 onwards (where there's a
        prior_manifest to reason about).
        """


class DefaultStrategy(Strategy):
    """No-op strategy — every panellist runs every round. Default."""

    def before_round(
        self,
        *,
        round_num: int,
        base_specs: list[ModelSpec],
        prior_manifest: list[ManifestEntry] | None,
    ) -> list[ModelSpec]:
        return list(base_specs)


_STRATEGIES: dict[str, type[Strategy]] = {
    "default": DefaultStrategy,
}


def strategy_for(name: str) -> Strategy:
    """Resolve a strategy name to an instance. Raises ValueError on
    unknown names so the caller fails fast at the refine boundary."""
    cls = _STRATEGIES.get(name)
    if cls is None:
        available = ", ".join(sorted(_STRATEGIES.keys()))
        raise ValueError(f"Unknown refine strategy {name!r}. Available: {available}")
    return cls()


def list_strategies() -> list[str]:
    return sorted(_STRATEGIES.keys())
