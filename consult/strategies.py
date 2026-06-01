"""Pluggable refine strategies.

The default refine flow runs the same panel each round until the arbiter
converges (or the round cap is hit). Strategies let callers customise
*what happens between rounds* — which panellists carry over, what
weight each gets, whether the synth picks from the final round or
averages — without forking refine.refine().

Inspired by llm-consortium's strategy plugin system (default, voting,
elimination, role, semantic). consult ships two for now:

- "default" — every panellist runs every round (current behaviour).
- "elimination" — after each round, the panellist whose capsule
  is *most distant* from the panel medoid is dropped from the next
  round. Idea: outlier panellists contribute noise; eliminating them
  tightens the consensus signal each subsequent arbiter evaluates.

Adding a new strategy is a matter of subclassing `Strategy` and
registering it in `_STRATEGIES`. Refine.refine() consults the strategy
via `before_round(...)` to decide what panel to run this round.

This module's behaviours are NOT wired into the default flow unless the
caller explicitly passes `strategy="elimination"`. Default semantics
are unchanged.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from difflib import SequenceMatcher

from .types import (
    Capsule,
    ManifestEntry,
    ModelSpec,
    ResearchCapsule,
    ReviewCapsule,
    Status,
)
from .voting import _feature_string

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Per-round customisation hook for `refine.refine()`.

    Strategies are stateful across rounds — they receive prior-round
    `handle` (with capsules) and prior `verdict`, and decide what the
    next round's panel should look like.
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


class EliminationStrategy(Strategy):
    """Drop the most-divergent panellist from the next round.

    Each round 2+, computes each panellist's mean similarity to the rest
    of the panel (via `voting._feature_string` + difflib). The panellist
    with the LOWEST mean similarity — i.e. the largest distance from the
    panel consensus — is removed from the next round's spec list.

    Round 1 runs the full panel. Each subsequent round drops one
    panellist; with `max_rounds=3` and a 5-panellist start, round 2 has
    4 panellists and round 3 has 3.

    Edge cases:
    - 2 or fewer usable panellists: elimination is a no-op (we'd
      otherwise drop down to one panellist, which defeats the consensus
      purpose entirely).
    - First-round eliminees never come back — the strategy is
      monotone. This is intentional; a panellist that disagreed once
      probably disagrees structurally, not noisily.
    - When the prior round has all-failed capsules, the strategy can't
      compute distances and falls back to the default (no elimination).
    """

    def __init__(self) -> None:
        # Tracks panellists eliminated in prior rounds so we don't
        # re-introduce them. Keyed by spec.model + spec.stance + spec.slug
        # for stability.
        self._eliminated: set[tuple[str, str | None, str | None]] = set()

    def before_round(
        self,
        *,
        round_num: int,
        base_specs: list[ModelSpec],
        prior_manifest: list[ManifestEntry] | None,
    ) -> list[ModelSpec]:
        if round_num == 1:
            return list(base_specs)
        if prior_manifest is None:
            return list(base_specs)

        worst_slug = self._compute_worst(prior_manifest)
        if worst_slug is None:
            logger.info(
                "EliminationStrategy: no clear outlier in round %d; keeping full panel",
                round_num - 1,
            )
            return [s for s in base_specs if (s.model, s.stance, s.slug) not in self._eliminated]

        # `worst_slug` carries the `.r<round_num-1>` suffix from
        # _suffix_specs. Map back to the base ModelSpec by stripping
        # the `-<i>.r<n>` and matching on model+stance order.
        # Simpler: iterate base_specs and pick the one whose generated
        # round-N slug would equal worst_slug. But we don't have the
        # slug-derivation here. Use a positional match instead — the
        # i-th base spec corresponds to round-N slug `<base>-<i>.r<N>`.
        worst_base_idx = self._find_worst_base_index(worst_slug, base_specs)
        if worst_base_idx is None:
            return list(base_specs)
        target = base_specs[worst_base_idx]
        key = (target.model, target.stance, target.slug)
        self._eliminated.add(key)
        logger.info(
            "EliminationStrategy: eliminating %s (slug=%s) for round %d",
            target.model,
            worst_slug,
            round_num,
        )

        usable_specs = [s for s in base_specs if (s.model, s.stance, s.slug) not in self._eliminated]
        # Floor at 2 panellists — eliminating below that loses the
        # consensus signal. Stop further eliminations rather than
        # walking the panel to zero.
        if len(usable_specs) < 2:
            return list(base_specs)
        return usable_specs

    def _compute_worst(
        self,
        prior_manifest: list[ManifestEntry],
    ) -> str | None:
        """Return the slug of the panellist with the lowest mean
        similarity to the rest of the panel. None when there's no clear
        outlier (e.g. all-failed manifest, or insufficient usable
        capsules)."""
        usable = [
            m
            for m in prior_manifest
            if m.status in (Status.OK, Status.TRUNCATED) and m.capsule is not None and _feature_string(m)
        ]
        if len(usable) < 3:
            # 2 or fewer: dropping one leaves at most one — pointless
            return None
        features = [_feature_string(m) for m in usable]
        worst_slug: str | None = None
        worst_score = float("inf")
        for i, entry in enumerate(usable):
            others = features[:i] + features[i + 1 :]
            mean_sim = sum(SequenceMatcher(None, features[i], o).ratio() for o in others) / len(others)
            if mean_sim < worst_score:
                worst_score = mean_sim
                worst_slug = entry.slug
        return worst_slug

    def _find_worst_base_index(
        self,
        worst_slug: str,
        base_specs: list[ModelSpec],
    ) -> int | None:
        """Map a round-suffixed slug back to its base-spec index.

        Refine's `_suffix_specs` builds slugs as
        `<base>-<i>.r<round_num>`. The base part may or may not match
        the spec's `model` field directly. The most robust mapping is
        via the index `i`. We extract it from the slug by stripping
        the `.r<N>` and the trailing `-<i>` (since refine always adds
        one) — but this requires knowing the actual derivation rules.

        Approach: scan base_specs and find the one whose
        `_make_slug_for_round(i, round_num)` would produce
        `worst_slug`. Since we don't import the runner private here,
        we approximate: strip the `.r<n>` suffix and check if it
        starts with a known model alias.
        """
        # Strip the `.r<n>` suffix
        base_part = worst_slug
        rfind = base_part.rfind(".r")
        if rfind != -1:
            base_part = base_part[:rfind]
        # Strip the trailing `-<i>` (the panel-index suffix from
        # _suffix_specs)
        rdash = base_part.rfind("-")
        if rdash != -1:
            tail = base_part[rdash + 1 :]
            if tail.isdigit():
                idx = int(tail)
                if 0 <= idx < len(base_specs):
                    return idx
        return None


_STRATEGIES: dict[str, type[Strategy]] = {
    "default": DefaultStrategy,
    "elimination": EliminationStrategy,
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


# Reference these to silence linters that flag unused imports in
# module-level type hints.
_ = (Capsule, ReviewCapsule, ResearchCapsule)
