"""Tests for `consult.peer_rank` — Borda-aggregated peer ranking pass."""

from __future__ import annotations

import json

import pytest

from consult import peer_rank as peer_rank_mod
from consult.peer_rank import PeerRanking, peer_rank_run
from consult.types import Capsule, ManifestEntry, Status


def _entry(slug: str, model_id: str) -> ManifestEntry:
    return ManifestEntry(
        slug=slug,
        model_id=model_id,
        status=Status.OK,
        resource_uri=f"consult://x/{slug}",
        body_path=f"/x/{slug}",
        cost_usd=0.01,
        cost_known=True,
        capsule=Capsule(position=f"{slug} position", recommendation="r"),
    )


def _fake_response(payload: dict):
    """Synthesise a litellm completion-like object."""

    class _Choice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _Resp:
        choices = [_Choice(json.dumps(payload))]

    return _Resp()


@pytest.mark.asyncio
async def test_peer_rank_borda_aggregates_per_ranker_contributions(monkeypatch):
    """3 panellists rank each other. Each ranker sees 2 others, so the
    Borda points per ranker run from N-2=1 (best) to 0 (worst).

    Rigged so:
    - a, b, c are the panellists
    - a ranks: [b, c]   → b gets 1, c gets 0
    - b ranks: [a, c]   → a gets 1, c gets 0
    - c ranks: [a, b]   → a gets 1, b gets 0
    Aggregate: a=2, b=1, c=0.
    """
    manifest = [
        _entry("a", "x/a"),
        _entry("b", "x/b"),
        _entry("c", "x/c"),
    ]
    bodies = {"a": "A body", "b": "B body", "c": "C body"}

    # The rankings the rankers WOULD return. We can't predict which Greek
    # label they'd see (per-ranker shuffle), so we capture the label_to_slug
    # mapping by mocking `_blocks_for_ranker` to a deterministic order.
    monkeypatch.setattr(
        peer_rank_mod,
        "_blocks_for_ranker",
        lambda others, bodies: (
            "\n\n".join(f"[{['Alpha', 'Beta', 'Gamma'][i]}]\n{bodies[e.slug]}" for i, e in enumerate(others)),
            {label: entry.slug for label, entry in zip(["Alpha", "Beta", "Gamma"], others, strict=False)},
        ),
    )

    # Each ranker is called with their own `others` (excluding self).
    # The ranker for "a" sees [b, c] → Alpha=b, Beta=c. Returns ["Alpha", "Beta"].
    # For "b": [a, c] → Alpha=a, Beta=c. Returns ["Alpha", "Beta"].
    # For "c": [a, b] → Alpha=a, Beta=b. Returns ["Alpha", "Beta"].
    # Each ranker ranks Alpha first, so we hardcode: ["Alpha", "Beta"].
    async def fake_acompletion(**kw):
        return _fake_response({"ranking": ["Alpha", "Beta"]})

    monkeypatch.setattr(
        "consult.peer_rank.litellm.acompletion",
        fake_acompletion,
    )
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )

    result = await peer_rank_run(manifest, bodies, question="Q?")

    # Each ranker contributes (N-1) - position points to the slug it
    # ranked at that position; here N=2 others per ranker, so position-1
    # is worth (2-1) = 1 point, position-2 is worth 0.
    points = dict(result.ranks)
    # a ranks b first; b ranks a first; c ranks a first → a=2, b=1, c=0
    assert points["a"] == 2
    assert points["b"] == 1
    assert points["c"] == 0
    # Order is descending by points
    assert result.ranks == [("a", 2), ("b", 1), ("c", 0)]
    # Cost rolled up across 3 ranker calls
    assert result.cost_usd == pytest.approx(0.003, abs=1e-9)


@pytest.mark.asyncio
async def test_peer_rank_skips_failed_rankers(monkeypatch):
    """A ranker that times out / raises must not pollute the aggregate;
    other rankers' contributions are kept."""
    manifest = [_entry("a", "x/a"), _entry("b", "x/b"), _entry("c", "x/c")]
    bodies = {"a": "A", "b": "B", "c": "C"}

    monkeypatch.setattr(
        peer_rank_mod,
        "_blocks_for_ranker",
        lambda others, bodies: (
            "\n\n".join(f"[{['Alpha', 'Beta', 'Gamma'][i]}]\n{bodies[e.slug]}" for i, e in enumerate(others)),
            {label: entry.slug for label, entry in zip(["Alpha", "Beta", "Gamma"], others, strict=False)},
        ),
    )

    call_n = {"n": 0}

    async def maybe_fail(**kw):
        call_n["n"] += 1
        if call_n["n"] == 1:
            raise TimeoutError("first ranker timed out")
        return _fake_response({"ranking": ["Alpha", "Beta"]})

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", maybe_fail)
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )

    result = await peer_rank_run(manifest, bodies, question="Q?")
    # Aggregate completed (didn't propagate the exception)
    assert isinstance(result, PeerRanking)
    # Cost reflects only the successful calls
    assert result.cost_usd == pytest.approx(0.002, abs=1e-9)
    # All 3 rankers appear in per_ranker, but the failed one has empty pairs
    failed = [r for r in result.per_ranker if not r[1]]
    assert len(failed) == 1


@pytest.mark.asyncio
async def test_peer_rank_rejects_incomplete_ranking(monkeypatch):
    """If a ranker omits a label or repeats one, that ranker's contribution
    is dropped entirely (don't impute / partial Borda would silently bias)."""
    manifest = [_entry("a", "x/a"), _entry("b", "x/b"), _entry("c", "x/c")]
    bodies = {"a": "A", "b": "B", "c": "C"}

    monkeypatch.setattr(
        peer_rank_mod,
        "_blocks_for_ranker",
        lambda others, bodies: (
            "\n\n".join(f"[{['Alpha', 'Beta', 'Gamma'][i]}]\n{bodies[e.slug]}" for i, e in enumerate(others)),
            {label: entry.slug for label, entry in zip(["Alpha", "Beta", "Gamma"], others, strict=False)},
        ),
    )

    # One ranker drops a label. The peer_rank aggregator must skip its
    # contribution rather than counting only the rank-1 slug.
    async def bad_ranker(**kw):
        return _fake_response({"ranking": ["Alpha"]})  # missing Beta

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", bad_ranker)
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )

    result = await peer_rank_run(manifest, bodies, question="Q?")
    # No usable rankings ⇒ all slugs at 0 points
    for _slug, pts in result.ranks:
        assert pts == 0


@pytest.mark.asyncio
async def test_peer_rank_too_few_panellists_returns_empty():
    """Need at least 2 panellists to rank — fewer is degenerate."""
    one = [_entry("a", "x/a")]
    result = await peer_rank_run(one, {"a": "body"}, question="Q?")
    assert result.ranks == []
    assert result.cost_usd == 0.0


@pytest.mark.asyncio
async def test_peer_rank_filters_failed_panellists(monkeypatch):
    """Status.ERROR / TIMEOUT / EMPTY entries are not included in the
    pool of rankers OR the pool of rankable responses."""
    manifest = [
        _entry("a", "x/a"),
        _entry("b", "x/b"),
        ManifestEntry(
            slug="c",
            model_id="x/c",
            status=Status.ERROR,
            resource_uri="x",
            body_path="/x/c",
            error="boom",
        ),
    ]
    bodies = {"a": "A", "b": "B"}

    monkeypatch.setattr(
        peer_rank_mod,
        "_blocks_for_ranker",
        lambda others, bodies: (
            "\n\n".join(f"[{['Alpha', 'Beta'][i]}]\n{bodies[e.slug]}" for i, e in enumerate(others)),
            {label: entry.slug for label, entry in zip(["Alpha", "Beta"], others, strict=False)},
        ),
    )

    async def fake_acompletion(**kw):
        # 2 usable panellists ⇒ 1 other per ranker ⇒ ranking of length 1
        return _fake_response({"ranking": ["Alpha"]})

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )

    result = await peer_rank_run(manifest, bodies, question="Q?")
    # Only a and b appear (c was filtered)
    slugs_in_result = {s for s, _ in result.ranks}
    assert slugs_in_result == {"a", "b"}


@pytest.mark.asyncio
async def test_peer_rank_all_rankers_fail_returns_zeroed_ranking(monkeypatch):
    """(issue #44) Every ranker returning garbage must yield a zero-point
    Borda board with empty per-ranker contributions, not an exception."""
    from types import SimpleNamespace

    from consult import peer_rank
    from consult.types import Capsule, ManifestEntry, Status

    async def garbage(*a, **kw):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="definitely not json"),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setattr(peer_rank.litellm, "acompletion", garbage)
    monkeypatch.setattr(peer_rank.litellm, "completion_cost", lambda **kw: 0.001)

    def entry(slug):
        return ManifestEntry(
            slug=slug,
            model_id=f"x/{slug}",
            status=Status.OK,
            resource_uri=f"consult://x/{slug}",
            body_path=f"/x/{slug}",
            capsule=Capsule(position=f"{slug} position"),
        )

    ranking = await peer_rank.peer_rank_run(
        [entry("a-0"), entry("b-1")],
        {"a-0": "body a", "b-1": "body b"},
        question="q",
    )
    assert ranking.ranks == [("a-0", 0), ("b-1", 0)]
    assert all(pairs == [] for _ranker, pairs in ranking.per_ranker)
    assert ranking.cost_usd == pytest.approx(0.002)
    assert ranking.cost_known is True
