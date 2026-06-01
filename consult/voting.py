"""Medoid voting over `model:N` stochastic-averaging panels.

When the caller uses the `model:N` syntax (`claude-haiku:3` → 3 instances
of the same model), the resulting manifest carries N entries that
differ only in seed-induced variance. The "More Agents Is All You Need"
paper (arxiv 2402.05120) shows that picking the *medoid* — the response
whose cumulative similarity to the group is maximal — out-performs both
random selection and naive ensembling for repeat-sample voting.

This module is a side-car helper, NOT wired into the default flow:

- `orchestrate.consult()`, `runner.fanout()`, etc. still return the full
  N-entry manifest. Callers who want the medoid call
  `manifest_after_medoid(handle.manifest)` explicitly.

- The similarity function is stdlib `difflib.SequenceMatcher.ratio()`
  over a per-capsule feature string. No embedding model dependency, no
  network call, fast. The signal is rough but the consensus among same-
  model-same-prompt instances is also usually high, so the medoid is
  easy to find with a coarse metric.

- Empty / failed capsules score 0 cumulative similarity and naturally
  lose the vote.

- Grouping is by `(round, model_id)`: refine's `.r<n>` round-suffixed
  slugs are kept separate so we don't fold round-1 and round-2 of the
  same model into one bucket.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from .types import (
    Capsule,
    ManifestEntry,
    ResearchCapsule,
    ReviewCapsule,
    Status,
)

logger = logging.getLogger(__name__)


_ROUND_SUFFIX_RE = re.compile(r"\.r(\d+)$")


def _round_from_slug(slug: str) -> int:
    """Parse the `.r<n>` round number from a refine slug, or 0 if absent."""
    m = _ROUND_SUFFIX_RE.search(slug)
    return int(m.group(1)) if m else 0


def _feature_string(entry: ManifestEntry) -> str:
    """Reduce a manifest entry's capsule to a similarity-feature string.

    The string is what `SequenceMatcher.ratio()` compares. We use the
    structured semantic-load-bearing fields rather than the raw body so
    differences in phrasing don't dominate the similarity score.

    Returns "" for entries with no usable capsule (unusable status or
    missing capsule) — those entries score 0 against everything and lose
    the vote.
    """
    if entry.status not in (Status.OK, Status.TRUNCATED) or entry.capsule is None:
        return ""
    cap = entry.capsule
    if isinstance(cap, Capsule):
        parts = [cap.position, cap.recommendation, *cap.key_points]
    elif isinstance(cap, ReviewCapsule):
        # Findings dominate review capsules; flatten them as
        # `severity|category|summary` so two reviews flagging the same
        # issue score high regardless of suggestion phrasing.
        parts = [cap.overall_verdict]
        parts.extend(f"{f.severity}|{f.category}|{f.summary}" for f in cap.findings)
    elif isinstance(cap, ResearchCapsule):
        parts = [*cap.claims, *cap.evidence, *cap.uncertainties]
    else:  # pragma: no cover — discriminated union exhausted above
        parts = []
    return "\n".join(p for p in parts if p)


def _cumulative_similarity(target: str, others: list[str]) -> float:
    """Σ ratio(target, o) for o in others. Empty target ⇒ 0 (always loses)."""
    if not target:
        return 0.0
    return sum(SequenceMatcher(None, target, o).ratio() for o in others if o)


def medoid_slugs(manifest: list[ManifestEntry]) -> dict[tuple[int, str], str]:
    """Pick the medoid slug per `(round, model_id)` group.

    Returns `{(round, model_id): medoid_slug}` covering every group with
    at least one usable entry. Singleton groups map to the single entry's
    slug trivially.

    Implementation note: the score is summed similarity to every OTHER
    entry in the group (the paper's formula explicitly excludes self-
    similarity, which is always 1.0 and would shift the argmax for
    nothing). Ties broken by lexicographic slug order — stable across
    runs.
    """
    # Bucket entries by (round, model_id). Entries with no model_id (e.g.
    # ERROR-status panellists where alias resolution failed) get a
    # synthetic key per-slug so they never group with each other.
    groups: dict[tuple[int, str], list[ManifestEntry]] = {}
    for entry in manifest:
        if entry.model_id is None:
            # Use slug as the key so each unknown-alias entry is its own
            # singleton group. Prevents accidentally medoid-collapsing
            # several "ERROR: unknown alias" rows.
            key = (_round_from_slug(entry.slug), f"__nomid__:{entry.slug}")
        else:
            key = (_round_from_slug(entry.slug), entry.model_id)
        groups.setdefault(key, []).append(entry)

    out: dict[tuple[int, str], str] = {}
    for key, group in groups.items():
        usable = [e for e in group if _feature_string(e)]
        if not usable:
            # Whole group failed; skip — caller's responsibility to
            # decide what to do with a wholly-unusable bucket.
            continue
        if len(usable) == 1:
            out[key] = usable[0].slug
            continue
        features = [_feature_string(e) for e in usable]
        # Score each candidate by Σ similarity to the others.
        best_slug = ""
        best_score = -1.0
        for i, entry in enumerate(usable):
            others = features[:i] + features[i + 1 :]
            score = _cumulative_similarity(features[i], others)
            # Lex-sort by slug as tiebreaker for stability across runs.
            if score > best_score or (score == best_score and entry.slug < best_slug):
                best_score = score
                best_slug = entry.slug
        out[key] = best_slug
        logger.debug(
            "medoid for %s (%d candidates): %s (score=%.3f)",
            key,
            len(usable),
            best_slug,
            best_score,
        )
    return out


def panel_disagreement(manifest: list[ManifestEntry]) -> float | None:
    """Compute a panel-wide disagreement score in [0, 1].

    0.0 = perfect agreement (all usable capsules say the same thing).
    1.0 = no two panellists agree on anything.

    Formula: `1 - mean(pairwise similarity)` across every usable
    (Status.OK / TRUNCATED) capsule pair. Same feature-extraction +
    similarity logic as `medoid_slugs` — `difflib.SequenceMatcher.ratio()`
    over the capsule's structured fields.

    Returns `None` (not 0.0 or 1.0) when the panel has fewer than two
    usable capsules — there is no pair to compare. Callers must handle
    `None` rather than treating it as full agreement.

    Used by `orchestrate.consult(gate_synth_at_agreement=...)` to decide
    whether the flagship synth is worth the spend, per MAgICoRe / FrugalGPT:
    high agreement ⇒ aggregation is cheap and lossless; low agreement
    ⇒ flagship synth earns its cost. The metric is also exposed on
    `RunResult.disagreement` so callers can route on it themselves.
    """
    features = [_feature_string(e) for e in manifest if _feature_string(e)]
    if len(features) < 2:
        return None
    sims: list[float] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sims.append(SequenceMatcher(None, features[i], features[j]).ratio())
    mean_sim = sum(sims) / len(sims)
    # Clamp into [0, 1] to absorb floating-point drift on near-identical
    # capsules where mean_sim could end up at 1.0000000002 or similar.
    return max(0.0, min(1.0, 1.0 - mean_sim))


def manifest_after_medoid(
    manifest: list[ManifestEntry],
) -> list[ManifestEntry]:
    """Filter `manifest` to one entry per `(round, model_id)` group.

    For each group, the medoid (per `medoid_slugs`) is kept; the rest are
    dropped. Wholly-unusable groups (no capsules or all failed) pass
    through unchanged so the caller can still see the failure rows.

    Original manifest order is preserved among the kept entries — the
    medoid takes the position of whichever entry in the group it was.
    """
    medoid_by_key = medoid_slugs(manifest)
    if not medoid_by_key:
        return list(manifest)
    medoid_slugs_set = set(medoid_by_key.values())

    # Collect the keys we picked medoids for; entries in *other* groups
    # (failed groups) keep all their entries.
    chosen_keys = set(medoid_by_key.keys())

    out: list[ManifestEntry] = []
    for entry in manifest:
        if entry.model_id is None:
            key = (_round_from_slug(entry.slug), f"__nomid__:{entry.slug}")
        else:
            key = (_round_from_slug(entry.slug), entry.model_id)
        if key not in chosen_keys:
            # Wholly-failed group — pass every entry through, no folding.
            out.append(entry)
        elif entry.slug in medoid_slugs_set:
            out.append(entry)
        # Otherwise drop: this entry is a non-medoid in a usable group.
    return out
