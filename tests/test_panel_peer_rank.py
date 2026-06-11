"""The opt-in peer_rank flag on the panel tool.

peer_rank.py shipped as a library-only side-car with no way to reach it
through MCP; the panel tool now exposes it behind a default-off flag.
"""

from __future__ import annotations

import pytest

from consult import artifacts, peer_rank, task_store  # noqa: F401  (task_store: isolation hygiene)
from consult.mcp import handlers
from consult.types import Capsule, ManifestEntry, RunHandle, Status


@pytest.fixture()
def _fake_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setenv("CONSULT_HEARTBEAT_INTERVAL_S", "0")

    async def fake_fanout(prompt, specs, **kwargs):
        paths = artifacts.create_run()
        manifest = []
        for i, spec in enumerate(specs):
            slug = f"{spec.model}-{i}"
            body = paths.response_text(slug)
            body.parent.mkdir(parents=True, exist_ok=True)
            body.write_text(f"{slug} body")
            manifest.append(
                ManifestEntry(
                    slug=slug,
                    model_id=f"x/{spec.model}",
                    status=Status.OK,
                    resource_uri=f"consult://x/{slug}",
                    body_path=str(body),
                    latency_ms=10,
                    cost_usd=0.01,
                    cost_known=True,
                    capsule=Capsule(position=f"{slug} position"),
                )
            )
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=manifest,
            cost_usd=0.01 * len(specs),
            cost_known=True,
            wall_ms=10,
        )

    monkeypatch.setattr("consult.mcp.handlers.runner.fanout", fake_fanout)

    async def fake_render(result):
        return result

    monkeypatch.setattr("consult.mcp.handlers._augment_result", fake_render)


async def test_panel_peer_rank_attaches_ranking_and_cost(_fake_panel, monkeypatch):
    async def fake_rank(manifest, bodies, *, question):
        assert question == "rank me"
        assert set(bodies) == {"claude-haiku-0", "gpt-mini-1"}
        return peer_rank.PeerRanking(
            ranks=[("claude-haiku-0", 1), ("gpt-mini-1", 0)],
            per_ranker=[("claude-haiku-0", [(1, "gpt-mini-1")])],
            cost_usd=0.02,
            cost_known=True,
        )

    monkeypatch.setattr("consult.peer_rank.peer_rank_run", fake_rank)

    result = await handlers.panel(
        {
            "prompt": "rank me",
            "models": [{"model": "claude-haiku"}, {"model": "gpt-mini"}],
            "extract_capsules": False,
            "peer_rank": True,
        }
    )
    assert result["peer_ranking"]["ranks"] == [["claude-haiku-0", 1], ["gpt-mini-1", 0]]
    assert result["cost_usd"] == pytest.approx(0.02 + 0.02)  # panel + ranking


async def test_panel_peer_rank_off_by_default(_fake_panel, monkeypatch):
    called = {"n": 0}

    async def fake_rank(*a, **kw):
        called["n"] += 1
        return peer_rank.PeerRanking(ranks=[], per_ranker=[])

    monkeypatch.setattr("consult.peer_rank.peer_rank_run", fake_rank)

    result = await handlers.panel(
        {
            "prompt": "no ranking",
            "models": [{"model": "claude-haiku"}],
            "extract_capsules": False,
        }
    )
    assert "peer_ranking" not in result
    assert called["n"] == 0


async def test_panel_peer_rank_failure_keeps_panel_result(_fake_panel, monkeypatch):
    async def explode(*a, **kw):
        raise RuntimeError("ranker meltdown")

    monkeypatch.setattr("consult.peer_rank.peer_rank_run", explode)

    result = await handlers.panel(
        {
            "prompt": "q",
            "models": [{"model": "claude-haiku"}],
            "extract_capsules": False,
            "peer_rank": True,
        }
    )
    assert "peer_ranking" not in result
    assert result["manifest"]  # the panel result survived the side-car failure
