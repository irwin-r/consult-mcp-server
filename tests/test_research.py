"""Research engine loop tests — scripted director, fake sub-runs.

Every test drives `research()` offline: `_director_json` is replaced with a
scripted queue per role label, and `orchestrate.consult` with a stub that
returns a duck-typed RunResult. The suite covers the panel-mandated rails
from issue #92: fail-closed plans, the per-round budget gate, stall
detection, the gap reopen guard, item-failure trapping, and the journal.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from consult import artifacts, orchestrate, research, runner
from consult.types import Brief, ResearchGap

BRIEF_DATA = {
    "assumptions": ["AU market", "DTC physical product"],
    "sections": [
        {
            "id": "niche",
            "title": "Niche",
            "goal": "pick a niche",
            "acceptance": "names 3 niches with evidence",
        },
        {"id": "brand", "title": "Brand", "goal": "brand identity", "acceptance": "name plus voice"},
    ],
}

PLAN_R1 = {
    "items": [
        {"id": "r1-1", "kind": "consult", "question": "Pick the niche.", "section_ids": ["niche"]},
        {"id": "r1-2", "kind": "panel", "question": "Design the brand.", "section_ids": ["brand"]},
    ]
}

JUDGE_ACCEPT = {
    "section_status": {"niche": "accepted", "brand": "accepted"},
    "blocking_gaps": [],
    "next_focus": "",
    "reasoning": "both sections meet their bars",
}


class ScriptedDirector:
    """Queue-per-label fake for `_director_json`. A queue entry of None
    simulates a call that failed across every candidate."""

    def __init__(self, brief=None, plans=(), judges=()):
        self.queues = {"brief": [brief], "plan": list(plans), "judge": list(judges)}
        self.calls: list[str] = []

    async def __call__(self, prompt, alias, *, label, **_kw):
        self.calls.append(label)
        queue = self.queues[label]
        data = queue.pop(0) if queue else None
        if data is None:
            return None, 0.01, True, "json_parse_failed"
        return data, 0.01, True, None


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """Route runs into tmp, stub the estimator, record sub-run calls."""
    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "aestimate_cost", _fake_estimate)
    calls: list[dict] = []

    async def fake_consult(
        question, *, tier="standard", capsule_kind="decision", max_run_usd=None, max_output_tokens=None, **kw
    ):
        calls.append({"question": question, "tier": tier, "max_run_usd": max_run_usd, **kw})
        return SimpleNamespace(
            run_id=f"sub-{len(calls)}",
            synthesis=f"ANSWER to: {question}",
            cost_usd=0.1,
            cost_known=True,
            partial=False,
            partial_reason=None,
        )

    monkeypatch.setattr(orchestrate, "consult", fake_consult)
    return SimpleNamespace(tmp=tmp_path, consult_calls=calls, monkeypatch=monkeypatch)


async def _fake_estimate(specs, prompt, **kw):
    return 0.05, True


def _journal_phases(tmp_path, run_id):
    lines = (tmp_path / run_id / "journal.jsonl").read_text().splitlines()
    return [json.loads(line)["phase"] for line in lines]


@pytest.mark.asyncio
async def test_converges_when_judge_accepts(rig):
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1], judges=[JUDGE_ACCEPT])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("build an e-commerce brand", max_rounds=6)

    assert result.converged is True
    assert result.stop_reason == "accepted"
    assert result.rounds_completed == 1
    assert result.partial is False
    assert "ANSWER to: Pick the niche." in result.dossier
    assert "ANSWER to: Design the brand." in result.dossier
    assert result.cost_usd == pytest.approx(0.01 * 3 + 0.1 * 2)
    assert len(rig.consult_calls) == 2

    phases = _journal_phases(rig.tmp, result.run_id)
    assert phases == ["brief", "plan", "item_result", "item_result", "assembled", "verdict"]
    run_dir = rig.tmp / result.run_id
    for name in ("brief.json", "dossier.md", "verdicts.json", "manifest.json"):
        assert (run_dir / name).exists()
    # The judge's dossier view is provenance-free; the disk copy is not.
    assert "source: run sub-" in (run_dir / "dossier.md").read_text()

    # Ledger contract: the parent manifest carries DIRECTOR-ONLY spend
    # (sub-runs self-report), with the full total alongside for humans.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["cost_usd"] == pytest.approx(0.01 * 3)
    assert manifest["total_cost_usd"] == pytest.approx(result.cost_usd)

    # Patient sub-runs: dropout off, timeouts floored at the default.
    assert all(c["tail_dropout_s"] == 0.0 for c in rig.consult_calls)
    assert all(c["timeout_floor_s"] == research.DEFAULT_MODEL_TIMEOUT_FLOOR_S for c in rig.consult_calls)


@pytest.mark.asyncio
async def test_plan_targeting_unknown_section_is_fail_closed(rig):
    bad_plan = {"items": [{"id": "r1-1", "kind": "consult", "question": "q", "section_ids": ["nope"]}]}
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[bad_plan])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal")

    assert result.stop_reason == "director_error"
    assert result.partial is True
    assert result.rounds_completed == 0
    assert rig.consult_calls == []  # nothing executed from a rejected plan


@pytest.mark.asyncio
async def test_plan_with_hallucinated_tier_is_fail_closed(rig):
    """An unknown tier used to crash the budget gate with an
    UnknownModelError instead of rejecting the plan (review finding)."""
    bad_plan = {
        "items": [
            {
                "id": "r1-1",
                "kind": "consult",
                "question": "q",
                "section_ids": ["niche"],
                "tier": "no-such-tier",
            }
        ]
    }
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[bad_plan])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_run_usd=25.0)

    assert result.stop_reason == "director_error"
    assert result.partial is True
    assert rig.consult_calls == []


@pytest.mark.asyncio
async def test_plan_with_overlapping_sections_is_fail_closed(rig):
    overlapping = {
        "items": [
            {"id": "r1-1", "kind": "consult", "question": "a", "section_ids": ["niche"]},
            {"id": "r1-2", "kind": "panel", "question": "b", "section_ids": ["niche"]},
        ]
    }
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[overlapping])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal")

    assert result.stop_reason == "director_error"
    assert rig.consult_calls == []


@pytest.mark.asyncio
async def test_strict_cap_refuses_after_accrued_cost_goes_unknown(rig):
    """Once an item's spend is unknown, gating a known cap against a
    known-understated meter is spending blind (review finding)."""

    async def unknown_cost_consult(question, **kw):
        rig.consult_calls.append({"question": question})
        return SimpleNamespace(
            run_id="sub-u",
            synthesis="ANSWER",
            cost_usd=None,
            cost_known=False,
            partial=False,
            partial_reason=None,
        )

    rig.monkeypatch.setattr(orchestrate, "consult", unknown_cost_consult)
    judge_r1 = {
        "section_status": {"niche": "accepted", "brand": "draft"},
        "blocking_gaps": [{"id": "g1", "text": "brand thin", "section_id": "brand"}],
    }
    replan = {"items": [{"id": "r2-1", "kind": "consult", "question": "more", "section_ids": ["brand"]}]}
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1, replan], judges=[judge_r1])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_run_usd=25.0)

    assert result.stop_reason == "budget"
    assert "accrued spend includes unknown pricing" in (result.partial_reason or "")
    assert result.rounds_completed == 1  # round 2 never executed


@pytest.mark.asyncio
async def test_crash_mid_round_still_persists_manifest(rig):
    class ExplodingDirector(ScriptedDirector):
        async def __call__(self, prompt, alias, *, label, **kw):
            if label == "judge":
                raise RuntimeError("director exploded")
            return await super().__call__(prompt, alias, label=label, **kw)

    director = ExplodingDirector(brief=BRIEF_DATA, plans=[PLAN_R1])
    rig.monkeypatch.setattr(research, "_director_json", director)

    with pytest.raises(RuntimeError, match="director exploded"):
        await research.research("goal")

    run_dir = next(p for p in rig.tmp.iterdir() if (p / "journal.jsonl").exists())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["partial"] is True
    assert "aborted before finalise" in manifest["partial_reason"]
    assert (run_dir / "dossier.md").exists()


@pytest.mark.asyncio
async def test_plan_json_failure_is_fail_closed(rig):
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[None])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal")

    assert result.stop_reason == "director_error"
    assert result.partial is True
    assert rig.consult_calls == []


@pytest.mark.asyncio
async def test_budget_gate_refuses_round_before_execution(rig):
    async def pricey_estimate(specs, prompt, **kw):
        return 50.0, True

    rig.monkeypatch.setattr(runner, "aestimate_cost", pricey_estimate)
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_run_usd=5.0)

    assert result.stop_reason == "budget"
    assert result.partial is True
    assert "exceed cap" in (result.partial_reason or "")
    assert rig.consult_calls == []


@pytest.mark.asyncio
async def test_strict_cap_with_unknown_pricing_refuses(rig):
    async def unknown_estimate(specs, prompt, **kw):
        return 0.0, False

    rig.monkeypatch.setattr(runner, "aestimate_cost", unknown_estimate)
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_run_usd=5.0)

    assert result.stop_reason == "budget"
    assert "unknown pricing" in (result.partial_reason or "")
    assert rig.consult_calls == []


@pytest.mark.asyncio
async def test_uncapped_skips_the_gate_entirely(rig):
    async def exploding_estimate(specs, prompt, **kw):
        raise AssertionError("uncapped runs must not price rounds")

    rig.monkeypatch.setattr(runner, "aestimate_cost", exploding_estimate)
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1], judges=[JUDGE_ACCEPT])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_run_usd=None)

    assert result.converged is True
    assert len(rig.consult_calls) == 2
    # Uncapped parent passes no child cap slice either.
    assert all(c["max_run_usd"] is None for c in rig.consult_calls)


@pytest.mark.asyncio
async def test_stall_stops_after_two_flat_rounds(rig):
    flat_judge = {
        "section_status": {"niche": "accepted", "brand": "draft"},
        "blocking_gaps": [{"id": "g1-1", "text": "brand voice missing", "section_id": "brand"}],
        "next_focus": "fix brand",
        "reasoning": "",
    }
    replan = {"items": [{"id": "rN-1", "kind": "consult", "question": "fix brand", "section_ids": ["brand"]}]}
    director = ScriptedDirector(
        brief=BRIEF_DATA,
        plans=[PLAN_R1, replan, replan, replan, replan],
        judges=[flat_judge, flat_judge, flat_judge, flat_judge],
    )
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_rounds=6)

    # Round 1 sets the baseline, rounds 2 and 3 show no progress → stalled.
    assert result.stop_reason == "stalled"
    assert result.converged is False
    assert result.partial is False
    assert result.rounds_completed == 3
    assert [g.id for g in result.open_gaps] == ["g1-1"]


@pytest.mark.asyncio
async def test_reopen_guard_drops_resolved_gap(rig):
    judge_r1 = {
        "section_status": {"niche": "accepted", "brand": "draft"},
        "blocking_gaps": [{"id": "g1-1", "text": "brand voice missing", "section_id": "brand"}],
    }
    # Round 2: g1-1 not reported → resolved. Judge re-raises it in round 3;
    # the guard drops it, so acceptance stands.
    judge_r2 = {
        "section_status": {"niche": "accepted", "brand": "draft"},
        "blocking_gaps": [{"id": "g2-1", "text": "brand name unchecked", "section_id": "brand"}],
    }
    judge_r3 = {
        "section_status": {"niche": "accepted", "brand": "accepted"},
        "blocking_gaps": [{"id": "G1-1", "text": "brand voice missing", "section_id": "brand"}],
    }
    replan = {"items": [{"id": "rN-1", "kind": "consult", "question": "fix brand", "section_ids": ["brand"]}]}
    director = ScriptedDirector(
        brief=BRIEF_DATA,
        plans=[PLAN_R1, replan, replan],
        judges=[judge_r1, judge_r2, judge_r3],
    )
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_rounds=6)

    assert result.stop_reason == "accepted"
    assert result.converged is True
    assert result.verdicts[2].blocking_gaps == []  # g1-1 filtered by the guard


@pytest.mark.asyncio
async def test_item_failure_is_trapped_and_recorded(rig):
    async def flaky_consult(question, **kw):
        if "Design" in question:
            raise RuntimeError("provider exploded")
        rig.consult_calls.append({"question": question})
        return SimpleNamespace(
            run_id="sub-ok",
            synthesis="ANSWER",
            cost_usd=0.1,
            cost_known=True,
            partial=False,
            partial_reason=None,
        )

    rig.monkeypatch.setattr(orchestrate, "consult", flaky_consult)
    accept_niche_only = {
        "section_status": {"niche": "accepted", "brand": "accepted"},
        "blocking_gaps": [],
    }
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1], judges=[accept_niche_only])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal")

    statuses = {r.item_id: r.status for r in result.rounds[0].results}
    assert statuses == {"r1-1": "ok", "r1-2": "error"}
    assert result.cost_known is False  # the failed item's spend is unknown
    assert result.stop_reason == "accepted"


@pytest.mark.asyncio
async def test_max_rounds_exit_is_not_partial(rig):
    improving = [
        {
            "section_status": {"niche": "accepted", "brand": "draft"},
            "blocking_gaps": [{"id": "g1", "text": "brand thin", "section_id": "brand"}],
        },
        {
            "section_status": {"niche": "accepted", "brand": "draft"},
            "blocking_gaps": [],
        },
    ]
    replan = {
        "items": [{"id": "rN-1", "kind": "consult", "question": "more brand", "section_ids": ["brand"]}]
    }
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1, replan], judges=improving)
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_rounds=2)

    assert result.stop_reason == "max_rounds"
    assert result.converged is False
    assert result.partial is False
    assert result.rounds_completed == 2


@pytest.mark.asyncio
async def test_judge_failing_twice_aborts(rig):
    replan = {"items": [{"id": "rN-1", "kind": "consult", "question": "again", "section_ids": ["brand"]}]}
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[PLAN_R1, replan], judges=[None, None])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal")

    assert result.stop_reason == "director_error"
    assert result.partial is True
    assert result.rounds_completed == 2
    assert all(not v.parsed_ok for v in result.verdicts)


@pytest.mark.asyncio
async def test_brief_failure_returns_partial(rig):
    director = ScriptedDirector(brief=None)
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal")

    assert result.stop_reason == "director_error"
    assert result.partial is True
    assert result.brief is None
    assert result.rounds_completed == 0
    assert rig.consult_calls == []


# ---- Parser units ------------------------------------------------------------


def test_norm_id_normalises_llm_variants():
    assert research._norm_id("Market_Research") == "market-research"
    assert research._norm_id("niche selection!") == "niche-selection"
    assert research._norm_id("  Brand  ") == "brand"


def test_parse_plan_truncates_to_four_items():
    brief = Brief.model_validate(
        {"assumptions": [], "sections": [{"id": "a", "title": "A", "goal": "g", "acceptance": "x"}]}
    )
    data = {
        "items": [
            {"id": f"i{n}", "kind": "evidence", "question": "q", "section_ids": ["a"]} for n in range(6)
        ]
    }
    items = research._parse_plan(data, brief, 1)
    assert items is not None and len(items) == 4


def test_parse_verdict_defaults_unscored_sections_to_draft():
    brief = Brief.model_validate(
        {
            "assumptions": [],
            "sections": [
                {"id": "a", "title": "A", "goal": "g", "acceptance": "x"},
                {"id": "b", "title": "B", "goal": "g", "acceptance": "x"},
            ],
        }
    )
    verdict = research._parse_verdict(
        {"section_status": {"a": "accepted"}},
        brief,
        1,
        set(),
        cost_usd=0.0,
        cost_known=True,
        sections_with_content={"a", "b"},
    )
    assert verdict.section_status == {"a": "accepted", "b": "draft"}
    bare = research._parse_verdict(
        {"section_status": {"a": "accepted"}}, brief, 1, set(), cost_usd=0.0, cost_known=True
    )
    assert bare.section_status == {"a": "accepted", "b": "missing"}
    assert verdict.accepted(brief) is False


def test_verdict_accepted_requires_no_gaps():
    brief = Brief.model_validate(
        {"assumptions": [], "sections": [{"id": "a", "title": "A", "goal": "g", "acceptance": "x"}]}
    )
    verdict = research._parse_verdict(
        {
            "section_status": {"a": "accepted"},
            "blocking_gaps": [{"id": "g1", "text": "still wrong", "section_id": "a"}],
        },
        brief,
        1,
        set(),
        cost_usd=0.0,
        cost_known=True,
    )
    assert verdict.accepted(brief) is False
    verdict_clean = research._parse_verdict(
        {"section_status": {"a": "accepted"}}, brief, 1, set(), cost_usd=0.0, cost_known=True
    )
    assert verdict_clean.accepted(brief) is True


def test_reopen_guard_unit():
    brief = Brief.model_validate(
        {"assumptions": [], "sections": [{"id": "a", "title": "A", "goal": "g", "acceptance": "x"}]}
    )
    verdict = research._parse_verdict(
        {
            "section_status": {"a": "draft"},
            "blocking_gaps": [
                {"id": "resolved-1", "text": "old", "section_id": "a"},
                {"id": "fresh-1", "text": "new", "section_id": "a"},
            ],
        },
        brief,
        3,
        {"resolved-1"},
        cost_usd=0.0,
        cost_known=True,
    )
    assert [g.id for g in verdict.blocking_gaps] == ["fresh-1"]
    assert isinstance(verdict.blocking_gaps[0], ResearchGap)


@pytest.mark.asyncio
async def test_evidence_item_feeds_later_rounds_not_sections(rig):
    from consult import evidence as evidence_mod

    pack = evidence_mod.EvidencePack(
        run_id="ev-run-1",
        records=[
            evidence_mod.EvidenceRecord(
                url="https://ex.example/prices",
                title="Price survey",
                claims=["rivals charge nine dollars flat"],
                model_id="fake/sonar-pro",
                slug="sonar-pro-1",
            )
        ],
        cost_usd=0.05,
    )

    async def fake_gather(question, **kw):
        return pack

    rig.monkeypatch.setattr(evidence_mod, "gather_evidence", fake_gather)
    plan_r1 = {
        "items": [{"id": "r1-ev", "kind": "evidence", "question": "find prices", "section_ids": ["niche"]}]
    }
    judge_r1 = {
        "section_status": {"niche": "draft", "brand": "draft"},
        "blocking_gaps": [{"id": "g1", "text": "sections need content", "section_id": "niche"}],
    }
    plan_r2 = {
        "items": [
            {
                "id": "r2-1",
                "kind": "consult",
                "question": "write both sections",
                "section_ids": ["niche", "brand"],
            }
        ]
    }
    director = ScriptedDirector(brief=BRIEF_DATA, plans=[plan_r1, plan_r2], judges=[judge_r1, JUDGE_ACCEPT])
    rig.monkeypatch.setattr(research, "_director_json", director)

    result = await research.research("goal", max_rounds=4)

    assert result.converged is True
    # The evidence item recorded its pass run but wrote no section body.
    assert result.rounds[0].results[0].run_id == "ev-run-1"
    # Round 2's consult question carries the framed evidence pack.
    assert len(rig.consult_calls) == 1
    question = rig.consult_calls[0]["question"]
    assert question.startswith("write both sections")
    assert "BEGIN UNTRUSTED WEB EVIDENCE" in question
    assert "https://ex.example/prices" in question
    # The dossier body is the consult answer, not an empty evidence body.
    assert "ANSWER to: write both sections" in result.dossier
