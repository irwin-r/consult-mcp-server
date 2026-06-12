"""Peer-rank pass — each panellist ranks the (anonymised) responses of
their peers, then we aggregate via Borda count.

This is the llm-council pattern (https://virtuslab.com/blog/ai/llm-council):
after the initial fanout, ask each panellist to rank the rest. Because
ranks dilute the influence of any single rogue judge, the aggregate is
more robust than a single judge model's verdict — particularly when the
judge is also one of the panel (which avoids the "judge model fixates on
its own house style" failure mode).

Implementation notes:

- Each ranker sees the manifest with its OWN response excluded, and
  every other panellist relabelled to Alpha/Beta/... — the per-ranker
  view is independently blinded so a ranker can't recognise its own
  family from style.
- The rankings come back as a list of blind labels; we map back to
  slugs via the per-ranker label dict.
- Aggregation is Borda count: position-1 → (N-1) points, position-N → 0
  points, summed across rankers. Higher count = preferred. Ties broken
  by slug lex order for stability.
- Rankers that emit malformed JSON, or rank a label that isn't on their
  blind list, or that drop labels, are skipped entirely for that
  round's contribution — partial rankings are not Borda-friendly.
- This is intentionally NOT wired into `orchestrate.consult` by default
  — N extra calls per panel is a real cost and most callers don't need
  it. Opt in by calling `peer_rank.peer_rank_run(handle)` after fanout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, cast

import litellm

from . import registry
from .jsonparse import extract_json
from .synth import _blind_label
from .types import ManifestEntry, Status

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankerOutcome:
    """One ranker's contribution to a peer-rank pass.

    `pairs` is `[(rank_position, ranked_slug), ...]`, empty when the
    ranker produced no usable ranking. `reason` is None on success and a
    short explanation otherwise (call failure, parse error, label
    misuse), so a dropped ranker is attributable from the serialised
    result. `cost_usd`/`cost_known` follow the ManifestEntry convention;
    a dropped ranker's spend still counts toward the pass total.
    """

    slug: str
    pairs: list[tuple[int, str]]
    reason: str | None = None
    cost_usd: float = 0.0
    cost_known: bool = True

    @property
    def failed(self) -> bool:
        return self.reason is not None


@dataclass(frozen=True)
class PeerRanking:
    """Aggregate ranking produced by a peer-rank pass.

    `ranks` is the canonical output: a list of (slug, borda_count) in
    descending order (most-preferred first). `per_ranker` is the raw
    per-ranker forensics, one `RankerOutcome` per ranker.

    `cost_usd` and `cost_known` follow the same convention as
    ManifestEntry. `cost_known` is conservatively marked False when a
    ranker raised outright, since that call's spend may be unknowable.
    """

    ranks: list[tuple[str, int]]
    per_ranker: list[RankerOutcome]
    cost_usd: float = 0.0
    cost_known: bool = True


_PEER_RANK_PROMPT = """\
Below are answers from {n} OTHER models to the same question. Models are \
anonymised as Alpha, Beta, etc. You did NOT write any of these (your own \
response was excluded from this list).

Question:
{question}

Responses to rank:
{blocks}

Rank the responses from best to worst, considering:
- Correctness: is the analysis sound?
- Completeness: does it address the full question?
- Calibration: does the stated confidence match the strength of evidence?

Return EXACTLY this JSON object (no commentary, no markdown fences):

{{"ranking": {example}}}

— the first label is best, the last is worst. Use every label exactly \
once. Do NOT invent labels.
"""


def _blocks_for_ranker(
    others: list[ManifestEntry],
    bodies: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Render the body blocks for a single ranker's view.

    Returns (joined_blocks_text, label_to_slug). Order is shuffled
    independently per ranker (different rankers see different orders) —
    position-bias mitigation just like the synth's input.
    """
    shuffled = list(others)
    random.shuffle(shuffled)
    label_to_slug: dict[str, str] = {}
    blocks: list[str] = []
    for i, entry in enumerate(shuffled):
        label = _blind_label(i)
        label_to_slug[label] = entry.slug
        body = bodies.get(entry.slug, "")
        blocks.append(f"[{label}]\n{body.strip()}")
    return "\n\n".join(blocks), label_to_slug


async def _ask_one_ranker(
    *,
    ranker: ManifestEntry,
    others: list[ManifestEntry],
    bodies: dict[str, str],
    question: str,
) -> RankerOutcome:
    """Run the peer-rank prompt against a single ranker.

    Returns a `RankerOutcome`. `pairs` is empty and `reason` set when
    the ranker errored, returned malformed JSON, or misused labels.
    Cost is captured even on parse failure.

    The ranker is the panellist's model (we use `ranker.model_id`); if
    the model_id was lost (e.g. blinded run) we use the ranker's slug
    as a registry alias (best-effort).
    """
    if not others:
        # Nothing to rank — not a failure, just no work to do.
        return RankerOutcome(slug=ranker.slug, pairs=[])

    blocks, label_to_slug = _blocks_for_ranker(others, bodies)
    # The example must use this ranker's real labels. A hardcoded
    # three-label example taught models to copy it verbatim: on a
    # two-peer ranking, 7 of 30 nano-tier rankers invented "Gamma" and
    # were dropped. label_to_slug insertion order matches the rendered
    # block order, so the example lists labels as the ranker sees them.
    example = json.dumps(list(label_to_slug))
    prompt = _PEER_RANK_PROMPT.format(
        n=len(others),
        question=question,
        blocks=blocks,
        example=example,
    )

    # Resolve the ranker's model. Prefer model_id (litellm-resolvable)
    # since slugs may carry round suffixes. Fall back to slug as alias.
    litellm_id: str = ranker.model_id or ""
    if not litellm_id:
        try:
            litellm_id = registry.resolve_model(ranker.slug).get("litellm_id") or ""
        except KeyError:
            logger.warning(
                "peer_rank: ranker %s has no model_id and slug isn't an alias — skipping",
                ranker.slug,
            )
            # No call was made, so the zero spend is known-true.
            return RankerOutcome(
                slug=ranker.slug,
                pairs=[],
                reason="no model_id and slug is not a registry alias",
            )

    try:
        resp = cast(
            Any,
            await asyncio.wait_for(
                litellm.acompletion(
                    model=litellm_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=512,
                ),
                timeout=120,
            ),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "peer_rank: %s rank call failed (%s)",
            ranker.slug,
            e,
        )
        return RankerOutcome(
            slug=ranker.slug,
            pairs=[],
            reason=f"ranker raised {type(e).__name__}: {e}"[:200],
            cost_known=False,
        )

    try:
        cost = litellm.completion_cost(completion_response=resp)
        cost_value = float(cost) if cost is not None else 0.0
        cost_known = cost is not None
    except Exception:  # noqa: BLE001
        cost_value = 0.0
        cost_known = False

    def _dropped(reason: str) -> RankerOutcome:
        return RankerOutcome(
            slug=ranker.slug,
            pairs=[],
            reason=reason[:200],
            cost_usd=cost_value,
            cost_known=cost_known,
        )

    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError, TypeError):
        return _dropped("response carried no content")

    data = extract_json(content)
    if not isinstance(data, dict):
        logger.warning(
            "peer_rank: %s returned non-JSON ranking; sample=%r",
            ranker.slug,
            content[:120].replace("\n", " "),
        )
        return _dropped("non-JSON ranking")

    raw_ranking = data.get("ranking")
    if not isinstance(raw_ranking, list):
        return _dropped("ranking field missing or not a list")

    # Validate: every entry must be a known label, and every label must
    # appear exactly once. Partial / wrong rankings produce no Borda
    # contribution — falling back to a default ranking would silently
    # bias the aggregate.
    seen: set[str] = set()
    pairs: list[tuple[int, str]] = []
    for position, label in enumerate(raw_ranking, start=1):
        if not isinstance(label, str) or label not in label_to_slug:
            logger.warning(
                "peer_rank: %s emitted unknown/non-string label %r",
                ranker.slug,
                label,
            )
            return _dropped(f"unknown or non-string label {label!r}")
        if label in seen:
            logger.warning(
                "peer_rank: %s emitted duplicate label %s",
                ranker.slug,
                label,
            )
            return _dropped(f"duplicate label {label}")
        seen.add(label)
        pairs.append((position, label_to_slug[label]))
    if seen != set(label_to_slug.keys()):
        # Dropped labels — incomplete ranking
        logger.warning(
            "peer_rank: %s ranked only %d/%d labels",
            ranker.slug,
            len(seen),
            len(label_to_slug),
        )
        return _dropped(f"ranked only {len(seen)}/{len(label_to_slug)} labels")

    return RankerOutcome(
        slug=ranker.slug,
        pairs=pairs,
        cost_usd=cost_value,
        cost_known=cost_known,
    )


async def peer_rank_run(
    manifest: list[ManifestEntry],
    bodies: dict[str, str],
    *,
    question: str,
) -> PeerRanking:
    """Run a peer-rank pass over a finished panel.

    `manifest` is the post-capsule manifest from `runner.fanout` (and,
    optionally, `capsule.annotate`). `bodies` maps slug to the
    panellist's full body text (read from `paths.response_text(slug)`).
    `question` is the original prompt (we re-include it in the rank
    prompt for context — the ranker shouldn't be reasoning blind).

    Returns a `PeerRanking` with aggregate Borda counts. Failing
    rankers (timeout, parse error, malformed ranking) contribute zero to
    the aggregate — their score on the rank scale is uncountable, so
    surrendering one ranker's input is safer than imputing. Each such
    ranker's `RankerOutcome` carries the drop reason and its spend.

    Cost is the sum of all ranker calls; `cost_known=False` if any
    ranker call's price was unknown.
    """
    usable = [m for m in manifest if m.status in (Status.OK, Status.TRUNCATED)]
    if len(usable) < 2:
        return PeerRanking(ranks=[], per_ranker=[])

    # Run all rankers concurrently — each takes ~5-15s on a flagship,
    # so serial would dominate wall-time on wide panels.
    async def _one(ranker: ManifestEntry):
        others = [m for m in usable if m.slug != ranker.slug]
        return await _ask_one_ranker(
            ranker=ranker,
            others=others,
            bodies=bodies,
            question=question,
        )

    results = await asyncio.gather(*(_one(r) for r in usable), return_exceptions=True)

    # Borda count: rank-1 ⇒ (N-1) pts, rank-(N-1) ⇒ 0 pts.
    n = len(usable)
    points: dict[str, int] = {m.slug: 0 for m in usable}
    per_ranker: list[RankerOutcome] = []
    total_cost = 0.0
    all_known = True
    for ranker, result in zip(usable, results, strict=True):
        if isinstance(result, BaseException):
            # Defensive: _ask_one_ranker catches its own failures, so
            # only a bug reaches here. Any spend from a billed-then-
            # crashed call is lost, hence cost 0 with cost_known False.
            logger.warning("peer_rank: %s ranker raised %r", ranker.slug, result)
            per_ranker.append(
                RankerOutcome(
                    slug=ranker.slug,
                    pairs=[],
                    reason=f"ranker raised {type(result).__name__}: {result}"[:200],
                    cost_known=False,
                )
            )
            all_known = False
            continue
        per_ranker.append(result)
        total_cost += result.cost_usd
        if not result.cost_known:
            all_known = False
        for position, slug in result.pairs:
            # rank-1 worth most; rank-(N-1) worth 0 (N-1 others ranked, so
            # max points = N-2 for the best, min = 0 for the worst)
            points[slug] += n - 1 - position

    # Sort: highest points first, then lex by slug for stability
    ranked = sorted(points.items(), key=lambda kv: (-kv[1], kv[0]))
    return PeerRanking(
        ranks=ranked,
        per_ranker=per_ranker,
        cost_usd=total_cost,
        cost_known=all_known,
    )


__all__ = [
    "PeerRanking",
    "RankerOutcome",
    "peer_rank_run",
]
