"""Build the per-run calibration block (issue #52).

The engine already blinds and shuffles the synth's input, scores disagreement,
and records each panellist's family / privacy tier / stance. None of it showed
up in the result, so a reviewing agent once misread `synthesise(anonymised=true)`
as a broken blinding feature. `build()` collects those signals into one
`Calibration` so the disclosure travels with the answer.

Family and privacy tier are recovered from the registry by the manifest's
`model_id` (which stays real even on blinded runs), so this works without
threading the original specs through.
"""

from __future__ import annotations

from collections import Counter

from . import registry
from .types import Calibration, ManifestEntry, Status

_USABLE = (Status.OK, Status.TRUNCATED)


def _litellm_index() -> dict[str, dict]:
    """Map litellm_id -> registry entry, for family / privacy-tier recovery."""
    models = registry.models_config().get("models", {})
    return {e["litellm_id"]: e for e in models.values() if e.get("litellm_id")}


def build(
    manifest: list[ManifestEntry],
    *,
    blinded: bool,
    disagreement: float | None,
) -> Calibration:
    """Assemble the calibration block from a panel manifest.

    `disagreement` is passed in (computed post-capsule by the caller); it is
    `None` for panels with fewer than two usable capsules to compare.
    """
    index = _litellm_index()
    status_counts: Counter[str] = Counter()
    # spend per status, with an unknown flag so a single unpriced entry marks
    # the whole status's spend as unknown rather than silently understating it.
    spend: dict[str, float] = {}
    spend_unknown: set[str] = set()
    families: Counter[str] = Counter()
    privacy: Counter[str] = Counter()
    stances: set[str] = set()

    for m in manifest:
        status = m.status.value
        status_counts[status] += 1
        if m.cost_usd is None or not m.cost_known:
            spend_unknown.add(status)
        else:
            spend[status] = spend.get(status, 0.0) + m.cost_usd
        if m.status in _USABLE:
            entry = index.get(m.model_id or "")
            if entry:
                if entry.get("family"):
                    families[entry["family"]] += 1
                if entry.get("privacy_tier"):
                    privacy[entry["privacy_tier"]] += 1
            stances.add(m.persona or "neutral")

    spend_by_status: dict[str, float | None] = {
        s: (None if s in spend_unknown else round(spend.get(s, 0.0), 6)) for s in status_counts
    }
    usable = sum(status_counts.get(s.value, 0) for s in _USABLE)

    return Calibration(
        blinded=blinded,
        disagreement=disagreement,
        panellists=len(manifest),
        usable=usable,
        status_counts=dict(status_counts),
        spend_by_status=spend_by_status,
        family_diversity=len(families),
        families=dict(families),
        privacy_tiers=dict(privacy),
        stance_coverage=sorted(stances),
    )
