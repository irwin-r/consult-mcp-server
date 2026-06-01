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
import logging
import random
from dataclasses import dataclass

import litellm

from . import registry
from .jsonparse import extract_json
from .synth import _blind_label
from .types import ManifestEntry, Status

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerRanking:
    """Aggregate ranking produced by a peer-rank pass.

    `ranks` is the canonical output: a list of (slug, borda_count) in
    descending order (most-preferred first). `per_ranker` is the raw
    per-ranker contributions for forensics — each entry is `(ranker_slug,
    [(rank_position, ranked_slug), ...])`. Rankers that produced no
    usable ranking appear with an empty inner list.

    `cost_usd` and `cost_known` follow the same convention as ManifestEntry.
    """

    ranks: list[tuple[str, int]]
    per_ranker: list[tuple[str, list[tuple[int, str]]]]
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

{{"ranking": ["Alpha", "Beta", "Gamma"]}}

— the first label is best, the last is worst. Use every label exactly \
once. Do NOT invent labels.
"""


def _blocks_for_ranker(
    others: list[ManifestEntry], bodies: dict[str, str],
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
) -> tuple[list[tuple[int, str]], float, bool]:
    """Run the peer-rank prompt against a single ranker.

    Returns `(ranked_pairs, cost_usd, cost_known)` where `ranked_pairs`
    is a list of (rank_position, ranker_slug_seen_as_real_slug) — empty
    when the ranker errored or returned malformed JSON or used unknown
    labels. Cost is captured even on parse failure.

    The ranker is the panellist's model (we use `ranker.model_id`); if
    the model_id was lost (e.g. blinded run) we use the ranker's slug
    as a registry alias (best-effort).
    """
    if not others:
        # Nothing to rank — return immediately with no cost.
        return [], 0.0, True

    blocks, label_to_slug = _blocks_for_ranker(others, bodies)
    prompt = _PEER_RANK_PROMPT.format(
        n=len(others), question=question, blocks=blocks,
    )

    # Resolve the ranker's model. Prefer model_id (litellm-resolvable)
    # since slugs may carry round suffixes. Fall back to slug as alias.
    litellm_id: str = ranker.model_id or ""
    if not litellm_id:
        try:
            litellm_id = registry.resolve_model(ranker.slug)["litellm_id"]
        except KeyError:
            logger.warning(
                "peer_rank: ranker %s has no model_id and slug isn't an "
                "alias — skipping",
                ranker.slug,
            )
            return [], 0.0, True

    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=litellm_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=512,
            ),
            timeout=120,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "peer_rank: %s rank call failed (%s)", ranker.slug, e,
        )
        return [], 0.0, False

    try:
        cost = litellm.completion_cost(completion_response=resp)
        cost_value = float(cost) if cost is not None else 0.0
        cost_known = cost is not None
    except Exception:  # noqa: BLE001
        cost_value = 0.0
        cost_known = False

    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError, TypeError):
        return [], cost_value, cost_known

    data = extract_json(content)
    if not isinstance(data, dict):
        logger.warning(
            "peer_rank: %s returned non-JSON ranking; sample=%r",
            ranker.slug, content[:120].replace("\n", " "),
        )
        return [], cost_value, cost_known

    raw_ranking = data.get("ranking")
    if not isinstance(raw_ranking, list):
        return [], cost_value, cost_known

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
                ranker.slug, label,
            )
            return [], cost_value, cost_known
        if label in seen:
            logger.warning(
                "peer_rank: %s emitted duplicate label %s", ranker.slug, label,
            )
            return [], cost_value, cost_known
        seen.add(label)
        pairs.append((position, label_to_slug[label]))
    if seen != set(label_to_slug.keys()):
        # Dropped labels — incomplete ranking
        logger.warning(
            "peer_rank: %s ranked only %d/%d labels",
            ranker.slug, len(seen), len(label_to_slug),
        )
        return [], cost_value, cost_known

    return pairs, cost_value, cost_known


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
    surrendering one ranker's input is safer than imputing.

    Cost is the sum of all ranker calls; `cost_known=False` if any
    ranker call's price was unknown.
    """
    usable = [
        m for m in manifest if m.status in (Status.OK, Status.TRUNCATED)
    ]
    if len(usable) < 2:
        return PeerRanking(ranks=[], per_ranker=[])

    # Run all rankers concurrently — each takes ~5-15s on a flagship,
    # so serial would dominate wall-time on wide panels.
    async def _one(ranker: ManifestEntry):
        others = [m for m in usable if m.slug != ranker.slug]
        return await _ask_one_ranker(
            ranker=ranker, others=others, bodies=bodies, question=question,
        )

    results = await asyncio.gather(
        *(_one(r) for r in usable), return_exceptions=True
    )

    # Borda count: rank-1 ⇒ (N-1) pts, rank-(N-1) ⇒ 0 pts.
    n = len(usable)
    points: dict[str, int] = {m.slug: 0 for m in usable}
    per_ranker: list[tuple[str, list[tuple[int, str]]]] = []
    total_cost = 0.0
    all_known = True
    for ranker, result in zip(usable, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("peer_rank: %s ranker raised %r", ranker.slug, result)
            per_ranker.append((ranker.slug, []))
            all_known = False
            continue
        pairs, cost, cost_known = result
        per_ranker.append((ranker.slug, pairs))
        total_cost += cost
        if not cost_known:
            all_known = False
        for position, slug in pairs:
            # rank-1 worth most; rank-(N-1) worth 0 (N-1 others ranked, so
            # max points = N-2 for the best, min = 0 for the worst)
            points[slug] += (n - 1 - position)

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
    "peer_rank_run",
]
