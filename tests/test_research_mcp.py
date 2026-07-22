"""MCP surface for the research tool (issue #92 PR 4).

Covers the env-gated listing, the handler's dossier-as-resource contract,
the presence-sensitive max_run_usd mapping (absent = default cap, explicit
null = uncapped), and the dossier resource kind end to end.
"""

from __future__ import annotations

import pytest

from consult import artifacts
from consult import research as research_mod
from consult.mcp import handlers
from consult.types import Brief, ResearchGap, ResearchResult, ResearchVerdict


def _fake_result(run_id: str = "20260722-000000-11111") -> ResearchResult:
    brief = Brief.model_validate(
        {
            "assumptions": ["AU market"],
            "sections": [
                {"id": "niche", "title": "Niche", "goal": "g", "acceptance": "x"},
                {"id": "brand", "title": "Brand", "goal": "g", "acceptance": "x"},
            ],
        }
    )
    verdict = ResearchVerdict(
        round=1,
        section_status={"niche": "accepted", "brand": "draft"},
        blocking_gaps=[ResearchGap(id="g1-1", text="brand voice missing", section_id="brand")],
    )
    return ResearchResult(
        run_id=run_id,
        brief=brief,
        rounds_completed=1,
        verdicts=[verdict],
        dossier="## Niche\n\nbody\n",
        converged=False,
        stop_reason="max_rounds",
        open_gaps=list(verdict.blocking_gaps),
        cost_usd=1.23,
        cost_known=True,
        wall_ms=1000,
    )


@pytest.mark.asyncio
async def test_research_hidden_from_default_surface(monkeypatch):
    from consult.mcp import server as server_mod

    monkeypatch.delenv("CONSULT_ENABLE_RESEARCH", raising=False)
    tools = await server_mod.handle_list_tools()
    assert "research" not in {t.name for t in tools}


@pytest.mark.asyncio
async def test_research_advertised_when_enabled(monkeypatch):
    from consult.mcp import server as server_mod

    monkeypatch.setenv("CONSULT_ENABLE_RESEARCH", "1")
    tools = await server_mod.handle_list_tools()
    by_name = {t.name: t for t in tools}
    assert "research" in by_name
    schema = by_name["research"].inputSchema
    assert schema["required"] == ["prompt"]
    # Nullable cap is the uncapped opt-in; the schema must allow null.
    assert schema["properties"]["max_run_usd"]["type"] == ["number", "null"]
    # Dispatch entry exists so task mode works without extra plumbing.
    assert server_mod._HANDLERS["research"] is handlers.research


@pytest.mark.asyncio
async def test_handler_ships_dossier_as_resource_not_inline(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    captured: dict = {}

    async def fake_research(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        result = _fake_result()
        run_dir = tmp_path / result.run_id
        run_dir.mkdir()
        (run_dir / "journal.jsonl").write_text('{"phase": "brief"}\n')
        return result

    monkeypatch.setattr(research_mod, "research", fake_research)

    payload = await handlers.research({"prompt": "goal"})

    assert "dossier" not in payload
    assert payload["dossier_uri"] == "consult://runs/20260722-000000-11111/dossier/dossier.md"
    assert payload["dossier_chars"] == len("## Niche\n\nbody\n")
    assert payload["journal_path"].endswith("journal.jsonl")
    # Deterministic summary carries statuses and gaps without a model call.
    assert "Niche [niche]: accepted" in payload["summary"]
    assert "Brand [brand]: draft" in payload["summary"]
    assert "[g1-1] brand voice missing" in payload["summary"]
    # Absent cap must NOT be forwarded (engine default applies).
    assert "max_run_usd" not in captured


@pytest.mark.asyncio
async def test_handler_forwards_explicit_null_cap_as_uncapped(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    captured: dict = {}

    async def fake_research(prompt, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(research_mod, "research", fake_research)

    await handlers.research({"prompt": "goal", "max_run_usd": None})

    assert "max_run_usd" in captured and captured["max_run_usd"] is None


def test_parse_resource_uri_accepts_dossier_kind():
    run_id, kind, name = artifacts.parse_resource_uri(
        "consult://runs/20260722-000000-11111/dossier/dossier.md"
    )
    assert (run_id, kind, name) == ("20260722-000000-11111", "dossier", "dossier.md")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/x/notakind/y")


@pytest.mark.asyncio
async def test_read_resource_serves_the_dossier(monkeypatch, tmp_path):
    from pydantic import AnyUrl

    from consult.mcp import server as server_mod

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    run_dir = tmp_path / "20260722-000000-22222"
    run_dir.mkdir()
    (run_dir / "dossier.md").write_text("## Niche\n\nthe goods\n")

    text = await server_mod.handle_read_resource(
        AnyUrl("consult://runs/20260722-000000-22222/dossier/dossier.md")
    )
    assert "the goods" in text

    with pytest.raises(FileNotFoundError):
        await server_mod.handle_read_resource(AnyUrl("consult://runs/20260722-000000-22222/dossier/other.md"))
