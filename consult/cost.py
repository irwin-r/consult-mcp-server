"""Cost accumulation with the None-means-unknown convention in one place.

Every multi-stage tool (consult, refine, sequence) sums spend across
fanout, capsule extraction, arbiter calls, and synthesis, and must flip a
"total is only a lower bound" flag the moment any stage's price is
unknown. Hand-threading a `(cumulative_cost, cost_all_known)` pair through
each stage produced a recurring bug class — dropped extractor cost,
dropped synth cost, an unknown-priced arbiter leaving the flag True (each
shipped as its own fix). The meter owns the two-field invariant so a new
stage can't get it half-right.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostMeter:
    """Running spend across the stages of one run.

    `total` only ever includes amounts we actually know; `known=False`
    marks it as a lower bound (some stage billed an unknown amount).
    Mirrors the `ManifestEntry` convention: `cost_usd=None` means
    unknown, never free.
    """

    total: float = 0.0
    known: bool = True

    def add(self, cost_usd: float | None, cost_known: bool = True) -> None:
        """Record one stage's spend.

        `cost_usd=None` means the stage billed an unknown amount: nothing
        is added and the total degrades to a lower bound. A real zero
        (`0.0, known=True`) adds nothing and keeps the total exact.
        """
        if cost_usd is not None:
            self.total += cost_usd
        if cost_usd is None or not cost_known:
            self.known = False

    def mark_unknown(self) -> None:
        """Degrade the total to a lower bound without adding spend.

        For signals that aren't a bill — e.g. a pre-flight estimate that
        couldn't price every panellist, which sequence surfaces as
        `cost_known=False` on the final result.
        """
        self.known = False
