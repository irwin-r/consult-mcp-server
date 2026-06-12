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
    # Usable rankings carry no drop reason
    assert all(o.reason is None and not o.failed for o in result.per_ranker)


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
    # All 3 rankers appear in per_ranker; the failed one carries the
    # drop reason and an unknown (zero) spend.
    failed = [o for o in result.per_ranker if o.failed]
    assert len(failed) == 1
    assert failed[0].pairs == []
    assert failed[0].reason == "ranker raised TimeoutError: first ranker timed out"
    assert failed[0].cost_usd == 0.0
    assert failed[0].cost_known is False
    # An unknowable ranker spend makes the pass total unknowable too
    assert result.cost_known is False


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
    assert all(o.pairs == [] for o in ranking.per_ranker)
    assert all(o.reason == "non-JSON ranking" for o in ranking.per_ranker)
    # Parse failures still billed a real call — that spend is preserved
    # per outcome and lands in the pass total.
    assert all(o.cost_usd == pytest.approx(0.001) for o in ranking.per_ranker)
    assert ranking.cost_usd == pytest.approx(0.002)
    assert ranking.cost_known is True


# --- RankerOutcome forensics (issue #58) -----------------------------------
#
# A dropped ranker must say WHY it was dropped and what it cost. These
# exercise `_ask_one_ranker` directly, one test per failure exit.


def _ranker_view(monkeypatch):
    """Pin the per-ranker shuffle so labels are deterministic."""
    monkeypatch.setattr(
        peer_rank_mod,
        "_blocks_for_ranker",
        lambda others, bodies: (
            "\n\n".join(f"[{['Alpha', 'Beta', 'Gamma'][i]}]\n{bodies[e.slug]}" for i, e in enumerate(others)),
            {label: entry.slug for label, entry in zip(["Alpha", "Beta", "Gamma"], others, strict=False)},
        ),
    )


async def _run_one_ranker(monkeypatch, payload):
    _ranker_view(monkeypatch)

    async def fake_acompletion(**kw):
        return _fake_response(payload)

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )
    return await peer_rank_mod._ask_one_ranker(
        ranker=_entry("r", "x/r"),
        others=[_entry("a", "x/a"), _entry("b", "x/b")],
        bodies={"a": "A", "b": "B"},
        question="Q?",
    )


@pytest.mark.asyncio
async def test_ask_one_ranker_success_has_no_reason(monkeypatch):
    outcome = await _run_one_ranker(monkeypatch, {"ranking": ["Alpha", "Beta"]})
    assert outcome.slug == "r"
    assert outcome.pairs == [(1, "a"), (2, "b")]
    assert outcome.reason is None
    assert not outcome.failed
    assert outcome.cost_usd == pytest.approx(0.001)
    assert outcome.cost_known is True


@pytest.mark.asyncio
async def test_ask_one_ranker_prompt_example_uses_actual_labels(monkeypatch):
    """The JSON example in the rank prompt is built from the ranker's
    real labels. The old hardcoded three-label example taught models to
    copy it verbatim and invent labels on two-peer rankings."""
    _ranker_view(monkeypatch)
    captured: dict = {}

    async def capture(**kw):
        captured["prompt"] = kw["messages"][0]["content"]
        return _fake_response({"ranking": ["Alpha", "Beta"]})

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", capture)
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )
    outcome = await peer_rank_mod._ask_one_ranker(
        ranker=_entry("r", "x/r"),
        others=[_entry("a", "x/a"), _entry("b", "x/b")],
        bodies={"a": "A", "b": "B"},
        question="Q?",
    )
    assert not outcome.failed
    assert '{"ranking": ["Alpha", "Beta"]}' in captured["prompt"]
    assert "Gamma" not in captured["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        ({"ranking": ["Zeta", "Beta"]}, "unknown or non-string label 'Zeta'"),
        ({"ranking": [42, "Beta"]}, "unknown or non-string label 42"),
        ({"ranking": ["Alpha", "Alpha"]}, "duplicate label Alpha"),
        ({"ranking": ["Alpha"]}, "ranked only 1/2 labels"),
        ({"ranking": "Alpha"}, "ranking field missing or not a list"),
        ({"verdict": "fine"}, "ranking field missing or not a list"),
    ],
)
async def test_ask_one_ranker_reason_per_failure_path(monkeypatch, payload, expected_reason):
    outcome = await _run_one_ranker(monkeypatch, payload)
    assert outcome.failed
    assert outcome.reason == expected_reason
    assert outcome.pairs == []
    # The call was billed before the parse failed — spend is preserved
    assert outcome.cost_usd == pytest.approx(0.001)
    assert outcome.cost_known is True


@pytest.mark.asyncio
async def test_ask_one_ranker_non_json_reason_preserves_cost(monkeypatch):
    from types import SimpleNamespace

    _ranker_view(monkeypatch)

    async def garbage(**kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="definitely not json"))]
        )

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", garbage)
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )
    outcome = await peer_rank_mod._ask_one_ranker(
        ranker=_entry("r", "x/r"),
        others=[_entry("a", "x/a"), _entry("b", "x/b")],
        bodies={"a": "A", "b": "B"},
        question="Q?",
    )
    assert outcome.reason == "non-JSON ranking"
    assert outcome.cost_usd == pytest.approx(0.001)
    assert outcome.cost_known is True


@pytest.mark.asyncio
async def test_ask_one_ranker_no_content_reason(monkeypatch):
    from types import SimpleNamespace

    _ranker_view(monkeypatch)

    async def empty(**kw):
        return SimpleNamespace(choices=[])

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", empty)
    monkeypatch.setattr(
        "consult.peer_rank.litellm.completion_cost",
        lambda completion_response=None: 0.001,
    )
    outcome = await peer_rank_mod._ask_one_ranker(
        ranker=_entry("r", "x/r"),
        others=[_entry("a", "x/a")],
        bodies={"a": "A"},
        question="Q?",
    )
    assert outcome.reason == "response carried no content"
    assert outcome.cost_usd == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_ask_one_ranker_unresolvable_model_is_flagged(monkeypatch):
    """No model_id and a slug that isn't a registry alias: flagged as a
    failure, but with a known-zero spend (no call was made)."""
    _ranker_view(monkeypatch)

    def no_such_alias(slug):
        raise KeyError(slug)

    monkeypatch.setattr(peer_rank_mod.registry, "resolve_model", no_such_alias)
    ranker = ManifestEntry(
        slug="not-an-alias",
        model_id=None,
        status=Status.OK,
        resource_uri="consult://x/not-an-alias",
        body_path="/x/not-an-alias",
    )
    outcome = await peer_rank_mod._ask_one_ranker(
        ranker=ranker,
        others=[_entry("a", "x/a")],
        bodies={"a": "A"},
        question="Q?",
    )
    assert outcome.failed
    assert outcome.reason == "no model_id and slug is not a registry alias"
    assert outcome.cost_usd == 0.0
    assert outcome.cost_known is True


@pytest.mark.asyncio
async def test_peer_rank_gather_exception_becomes_outcome(monkeypatch):
    """An exception that escapes `_ask_one_ranker` entirely (a bug, not a
    provider failure) still lands in per_ranker with a reason, and the
    pass total is marked unknowable."""
    manifest = [_entry("a", "x/a"), _entry("b", "x/b")]

    async def boom(**kw):
        raise RuntimeError("loop bug")

    monkeypatch.setattr(peer_rank_mod, "_ask_one_ranker", boom)

    result = await peer_rank_run(manifest, {"a": "A", "b": "B"}, question="Q?")
    assert all(o.failed for o in result.per_ranker)
    assert all(o.reason == "ranker raised RuntimeError: loop bug" for o in result.per_ranker)
    assert all(o.cost_usd == 0.0 and o.cost_known is False for o in result.per_ranker)
    assert result.cost_usd == 0.0
    assert result.cost_known is False


@pytest.mark.asyncio
async def test_ask_one_ranker_truncates_long_reasons(monkeypatch):
    _ranker_view(monkeypatch)

    async def fail_loudly(**kw):
        raise RuntimeError("x" * 500)

    monkeypatch.setattr("consult.peer_rank.litellm.acompletion", fail_loudly)
    outcome = await peer_rank_mod._ask_one_ranker(
        ranker=_entry("r", "x/r"),
        others=[_entry("a", "x/a")],
        bodies={"a": "A"},
        question="Q?",
    )
    assert outcome.failed
    assert len(outcome.reason) == 200
    assert outcome.reason.startswith("ranker raised RuntimeError: xxx")
