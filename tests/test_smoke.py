"""Smoke tests. The unit slice runs offline (no API keys); the live slice
hits real providers only when relevant API keys are present.

Run: `pytest -v`
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from consult import artifacts, refine as refine_mod, registry
from consult.runner import _build_per_slug_prompt, _make_slug, estimate_cost
from consult.status import classify
from consult.types import ArbiterVerdict, Capsule, ManifestEntry, ModelSpec, RunHandle, Status


# ---- Pure-Python tests (no network) ----------------------------------------


def test_registry_loads_defaults():
    cfg = registry.models_config()
    assert "models" in cfg
    assert "claude-haiku" in cfg["models"]
    assert "tiers" in cfg
    assert "quick" in cfg["tiers"]


def test_registry_resolve_alias_and_raw_id():
    entry = registry.resolve_model("claude-haiku")
    assert entry["litellm_id"].startswith("anthropic/")
    raw = registry.resolve_model("openrouter/some/model")
    assert raw["litellm_id"] == "openrouter/some/model"


def test_stance_lookup_and_passthrough():
    assert "security" in registry.resolve_stance("security")
    assert registry.resolve_stance("You are a freeform stance.") == "You are a freeform stance."
    assert registry.resolve_stance(None) == ""


def test_expand_specs_multi_instance_and_passthrough():
    """`model:N` sugar expands to N specs; bare model strings pass through.
    Stance and custom slug are preserved on every expanded instance.
    """
    from consult.runner import expand_specs

    raw = [
        ModelSpec(model="claude-haiku:3", stance="skeptic"),
        ModelSpec(model="gpt-pro"),
        ModelSpec(model="openrouter/foo/bar"),  # no colon — passthrough
    ]
    expanded = expand_specs(raw)
    assert len(expanded) == 5  # 3 + 1 + 1
    assert [s.model for s in expanded] == [
        "claude-haiku", "claude-haiku", "claude-haiku",
        "gpt-pro",
        "openrouter/foo/bar",
    ]
    # Stance survives expansion
    assert all(s.stance == "skeptic" for s in expanded[:3])

    # Idempotent: re-expanding already-expanded specs is a no-op
    assert expand_specs(expanded) == expanded


def test_expand_specs_rejects_zero_count():
    """`model:0` is almost certainly a typo and must fail loudly rather
    than silently dropping the spec from the panel.
    """
    from consult.runner import expand_specs

    with pytest.raises(ValueError, match="must be ≥1"):
        expand_specs([ModelSpec(model="claude-haiku:0")])


def test_slug_and_prompt_assembly():
    spec = ModelSpec(model="claude-haiku", stance="security")
    assert _make_slug(spec, 0, blinded=False).startswith("claude-haiku")
    assert _make_slug(spec, 2, blinded=True) == "panelist-gamma"


def test_footer_injected_on_every_prompt():
    """Capsule confidence extraction depends on the footer being present."""
    with_stance = _build_per_slug_prompt("How to ship X?", "You are an SRE.")
    assert with_stance.startswith("You are an SRE.")
    assert "How to ship X?" in with_stance
    assert "CONFIDENCE:" in with_stance
    assert "KEY_REASON:" in with_stance

    no_stance = _build_per_slug_prompt("How to ship X?", "")
    assert no_stance.startswith("How to ship X?")
    assert "CONFIDENCE:" in no_stance
    assert "KEY_REASON:" in no_stance


def test_status_classifier_handles_exceptions():
    s, _, _ = classify(None, exception=RuntimeError("rate limit hit"))
    assert s == Status.RATE_LIMITED
    s, _, _ = classify(None, exception=TimeoutError("timed out"))
    assert s == Status.TIMEOUT


def test_run_handle_usable_parametric():
    entries = [
        ManifestEntry(
            slug=f"m{i}", model_id=f"openrouter/x{i}/y", status=Status.OK,
            resource_uri="consult://x", body_path="/tmp/x",
        )
        for i in range(4)
    ]
    entries[3].model_id = "openai/gpt-5"
    handle = RunHandle(
        run_id="t", artifacts_dir="/tmp/t", manifest=entries, cost_usd=0, wall_ms=0
    )
    assert handle.usable() is True
    # Set all-OR — only one provider
    for e in entries:
        e.model_id = "openrouter/foo/bar"
    assert handle.usable(min_providers=2) is False


def test_artifacts_create_and_uri():
    paths = artifacts.create_run()
    assert paths.root.exists()
    assert paths.responses.exists()
    uri = paths.resource_uri("alpha")
    rid, slug = artifacts.parse_resource_uri(uri)
    assert rid == paths.run_id
    assert slug == "alpha"
    # cleanup
    import shutil
    shutil.rmtree(paths.root)


# ---- Refine offline tests --------------------------------------------------


def test_refine_suffix_specs_round_indexes_slugs():
    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="grok", stance="contrarian")]
    r2 = refine_mod._suffix_specs(specs, 2)
    assert r2[0].slug.endswith(".r2")
    assert r2[1].slug.endswith(".r2")
    # Same model index should produce stable base slug
    r2b = refine_mod._suffix_specs(specs, 2)
    assert [s.slug for s in r2] == [s.slug for s in r2b]


def test_refinement_prompt_includes_gaps_and_focus():
    manifest = [
        ManifestEntry(
            slug="m-1.r1",
            model_id="anthropic/x",
            status=Status.OK,
            resource_uri="consult://x",
            body_path="/tmp/x",
            capsule=Capsule(position="A says X", recommendation="do X"),
        )
    ]
    verdict = ArbiterVerdict(
        round=1, score=0.4, gaps=["cost not discussed"], next_round_focus="address cost"
    )
    out = refine_mod._build_refinement_prompt("Should we ship X?", 2, manifest, verdict)
    assert "Should we ship X?" in out
    assert "cost not discussed" in out
    assert "address cost" in out
    assert "m-1.r1" in out


@pytest.mark.asyncio
async def test_refine_continuation_prepends_prior_synthesis(tmp_path, monkeypatch):
    """A valid continuation_id loads the prior run's synthesis.md and
    prepends it as 'Prior consultation summary' before the follow-up.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()
    (prior.root / "synthesis.md").write_text("ANSWER: pick DuckDB.")

    result = _apply_continuation("Now what about Polars for ETL?", prior.run_id)
    assert "Prior consultation summary" in result
    assert "ANSWER: pick DuckDB." in result
    assert "Follow-up question" in result
    assert "Now what about Polars for ETL?" in result


def test_refine_continuation_none_or_empty_is_passthrough(tmp_path, monkeypatch):
    """No continuation_id (or empty string) leaves the prompt untouched."""
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    assert _apply_continuation("hello", None) == "hello"
    assert _apply_continuation("hello", "") == "hello"


def test_refine_continuation_unknown_id_raises(tmp_path, monkeypatch):
    """An unknown continuation_id must raise — silently dropping the prior
    context would leave the caller thinking the new round had it.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="continuation_id not found"):
        _apply_continuation("hello", "20990101-000000-99999")


def test_refine_continuation_missing_synthesis_raises(tmp_path, monkeypatch):
    """A run that exists but has no synthesis.md (e.g. dry-run, cap-aborted)
    can't be a continuation source — fail clearly rather than prepend empty.
    """
    from consult.refine import _apply_continuation

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    prior = artifacts.create_run()  # no synthesis.md written
    with pytest.raises(ValueError, match="no synthesis.md"):
        _apply_continuation("hello", prior.run_id)


@pytest.mark.asyncio
async def test_refine_validates_max_rounds():
    with pytest.raises(ValueError, match="max_rounds"):
        await refine_mod.refine("q", [ModelSpec(model="claude-haiku")], max_rounds=5)


def test_arbiter_json_extractor_tolerates_fences():
    fenced = '```json\n{"score": 0.7, "gaps": ["x"], "next_round_focus": "", "reasoning": ""}\n```'
    data = refine_mod._extract_json(fenced)
    assert data["score"] == 0.7
    assert data["gaps"] == ["x"]


# ---- New (post-review) offline tests --------------------------------------


@pytest.mark.asyncio
async def test_fanout_dry_run_returns_partial():
    """Dry run must never make a billable call and must explain itself."""
    from consult.runner import fanout

    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="claude-sonnet")]
    handle = await fanout("any prompt", specs, dry_run=True)
    assert handle.partial is True
    assert handle.partial_reason and "dry_run" in handle.partial_reason
    assert handle.manifest == []
    assert handle.cost_usd == 0.0


@pytest.mark.asyncio
async def test_call_one_unknown_alias_returns_error_entry(tmp_path, monkeypatch):
    """An unknown alias must surface as a per-spec Status.ERROR rather than
    crashing the panel. Regression guard: KeyError out of `resolve_model`
    previously propagated through `asyncio.gather` and aborted every sibling.
    """
    from consult.runner import _call_one

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    spec = ModelSpec(model="definitely-not-a-real-alias")
    entry = await _call_one(spec, "bogus-0", "prompt", paths)
    assert entry.status is Status.ERROR
    assert entry.error and "definitely-not-a-real-alias" in entry.error
    assert entry.model_id is None
    assert entry.cost_known is True  # no call was billable


def test_estimate_cost_skips_unknown_alias_without_raising():
    """Unknown aliases mark cost_known=False but must not raise — `fanout`
    relies on this so a typo doesn't abort the run before any panel work.
    """
    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="bogus-xyz")]
    total, all_known = estimate_cost(specs, "hello")
    assert total >= 0.0
    assert all_known is False


@pytest.mark.asyncio
async def test_fanout_cost_cap_returns_partial(monkeypatch):
    """Setting max_run_usd to 0 must abort before any model call."""
    from consult import runner
    from consult.runner import fanout

    # Force a non-zero estimate so the cap path is exercised
    monkeypatch.setattr(runner, "estimate_cost", lambda specs, prompt: (0.99, True))
    specs = [ModelSpec(model="claude-haiku")]
    handle = await fanout("p", specs, max_run_usd=0.01)
    assert handle.partial is True
    assert handle.partial_reason and "exceeds cap" in handle.partial_reason
    assert "known-priced" not in handle.partial_reason  # all_known=True path
    assert handle.manifest == []


@pytest.mark.asyncio
async def test_fanout_cost_cap_message_discloses_partial_pricing(monkeypatch):
    """When estimate_cost returns all_known=False, the cap message must say
    so — otherwise the displayed estimate (only the known-priced portion)
    looks misleadingly low. Mirrors the dry_run branch.
    """
    from consult import runner
    from consult.runner import fanout

    monkeypatch.setattr(runner, "estimate_cost", lambda specs, prompt: (0.50, False))
    specs = [ModelSpec(model="claude-haiku")]
    handle = await fanout("p", specs, max_run_usd=0.01)
    assert handle.partial is True
    assert handle.partial_reason
    assert "known-priced portion only" in handle.partial_reason
    assert "exceeds cap" in handle.partial_reason
    assert handle.cost_known is False


def test_capsule_extract_json_recovers_prose_and_fences():
    """The capsule contract depends on this — one regex change breaks all callers."""
    from consult.capsule import _extract_json

    assert _extract_json('{"score": 0.5}') == {"score": 0.5}
    assert _extract_json('```json\n{"k": "v"}\n```') == {"k": "v"}
    assert _extract_json('prefix\n{"k": 1}\nsuffix') == {"k": 1}
    assert _extract_json("definitely not json") is None
    assert _extract_json("") is None


def test_parse_resource_uri_rejects_malformed():
    """Permissive parsing would be a path-traversal hazard."""
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("http://example.com/runs/abc/responses/x")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc/responses/x/extra")
    with pytest.raises(ValueError):
        artifacts.parse_resource_uri("consult://runs/abc/capsules/x")
    # Happy path still works
    rid, slug = artifacts.parse_resource_uri("consult://runs/r1/responses/alpha.r2")
    assert rid == "r1"
    assert slug == "alpha.r2"


def test_status_classifier_normal_responses():
    """Cover the OK/TRUNCATED/EMPTY/CONTENT_FILTERED/MALFORMED response paths.

    Today only the exception branch is tested — the body classification logic
    could return wrong statuses silently if not exercised.
    """
    from types import SimpleNamespace

    from consult.status import classify

    def make_resp(content, finish):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason=finish
                )
            ]
        )

    s, _, body = classify(make_resp("real content", "stop"))
    assert s == Status.OK
    assert body == "real content"

    s, _, body = classify(make_resp("partial", "length"))
    assert s == Status.TRUNCATED
    assert body == "partial"

    s, _, _ = classify(make_resp("", "length"))
    assert s == Status.TRUNCATED  # empty body + length still TRUNCATED

    s, _, _ = classify(make_resp("    \n  \n", "stop"))
    assert s == Status.EMPTY  # whitespace-only body (OR thinking-model fail mode)

    s, _, _ = classify(make_resp("blocked", "content_filter"))
    assert s == Status.CONTENT_FILTERED

    s, _, _ = classify(SimpleNamespace(choices=[]))
    assert s == Status.MALFORMED

    s, _, _ = classify(None)
    assert s == Status.EMPTY


def test_synth_build_input_anonymised_and_filters_failures():
    """Privacy-relevant for blinded mode + correctness for the synthesiser input.

    ERROR/EMPTY entries must be excluded from the synthesiser input, and
    anonymised mode must not leak model IDs.
    """
    from consult.synth import _build_input

    manifest = [
        {
            "slug": "panelist-alpha",
            "model_id": "anthropic/claude-opus-4-7",
            "persona": "contrarian",
            "confidence": 0.8,
            "status": "OK",
        },
        {
            "slug": "panelist-beta",
            "model_id": "openai/gpt-5.5",
            "persona": None,
            "confidence": None,
            "status": "EMPTY",  # must be filtered out
        },
        {
            "slug": "panelist-gamma",
            "model_id": "gemini/gemini-3.1-pro-preview",
            "persona": None,
            "confidence": 0.6,
            "status": "TRUNCATED",  # truncated-with-body stays in
        },
    ]
    bodies = {
        "panelist-alpha": "alpha body",
        "panelist-beta": "",
        "panelist-gamma": "gamma body",
    }
    rubric = "rubric {n}"

    blinded = _build_input(manifest, bodies, rubric=rubric, anonymised=True)
    assert "anthropic/claude-opus-4-7" not in blinded
    assert "openai/gpt-5.5" not in blinded
    assert "panelist-alpha" in blinded
    assert "panelist-gamma" in blinded
    assert "panelist-beta" not in blinded  # filtered
    assert "alpha body" in blinded
    assert "rubric 2" in blinded  # only OK + TRUNCATED counted

    unblinded = _build_input(manifest, bodies, rubric=rubric, anonymised=False)
    assert "anthropic/claude-opus-4-7" in unblinded
    assert "gemini/gemini-3.1-pro-preview" in unblinded


def test_manifest_entry_validates_error_requirement():
    """Constructing an ERROR/TIMEOUT entry without an error string must fail."""
    import pydantic

    base = dict(
        slug="x", status=Status.ERROR, resource_uri="consult://x", body_path="/tmp/x"
    )
    with pytest.raises(pydantic.ValidationError):
        ManifestEntry(**base)
    # With error, it succeeds
    ManifestEntry(**base, error="auth failed")


def test_run_handle_validates_partial_coupling():
    """partial=True ⇔ partial_reason set."""
    import pydantic

    base = dict(run_id="r", artifacts_dir="/tmp/r", manifest=[], cost_usd=0.0, wall_ms=0)
    with pytest.raises(pydantic.ValidationError):
        RunHandle(**base, partial=True)  # no reason
    with pytest.raises(pydantic.ValidationError):
        RunHandle(**base, partial=False, partial_reason="oops")  # reason without partial


def test_daily_ledger_aggregates_costs_status_and_panel_size(tmp_path, monkeypatch):
    """Daily ledger reads every run dir whose ID starts with YYYYMMDD,
    aggregates cost + cost_known, and records per-status counts. Malformed
    or missing manifests are skipped without aborting the scan.
    """
    from datetime import date as date_cls

    from consult import ledger

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)

    # Two real runs on the target date, one on a different date, plus a
    # malformed-manifest run that should be skipped without crashing.
    def make_run(rid: str, cost: float, cost_known: bool, statuses: list[str]):
        d = tmp_path / rid
        d.mkdir()
        manifest = {
            "run_id": rid,
            "cost_usd": cost,
            "cost_known": cost_known,
            "manifest": [{"status": s} for s in statuses],
        }
        (d / "manifest.json").write_text(json.dumps(manifest))

    make_run("20260101-100000-1", 0.40, True, ["OK", "OK", "RATE_LIMITED"])
    make_run("20260101-110000-2", 0.15, False, ["OK"])
    make_run("20260102-100000-3", 99.0, True, ["OK"])  # different day, ignored

    # Malformed manifest — bytes that aren't JSON
    bad = tmp_path / "20260101-120000-9"
    bad.mkdir()
    (bad / "manifest.json").write_text("{this is not json")

    # Bare directory with no manifest at all — also skipped
    (tmp_path / "20260101-130000-9").mkdir()

    led = ledger.daily_ledger(date_cls(2026, 1, 1))
    assert led.date == date_cls(2026, 1, 1)
    assert len(led.runs) == 2  # malformed + manifest-less skipped, other day excluded
    assert led.total_usd == pytest.approx(0.55)
    assert led.total_known is False  # one of the two had cost_known=False

    by_id = {r.run_id: r for r in led.runs}
    assert by_id["20260101-100000-1"].status_counts == {"OK": 2, "RATE_LIMITED": 1}
    assert by_id["20260101-100000-1"].panel_size == 3
    assert by_id["20260101-110000-2"].panel_size == 1


def test_daily_ledger_empty_day_returns_zero(tmp_path, monkeypatch):
    """A day with no runs must return a well-formed empty ledger (not raise),
    and total_known=True since there's nothing unknown about $0."""
    from datetime import date as date_cls

    from consult import ledger

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    led = ledger.daily_ledger(date_cls(2030, 6, 15))
    assert led.runs == []
    assert led.total_usd == 0.0
    assert led.total_known is True


def test_refine_result_validates_partial_coupling_and_surfaces_reason():
    """RefineResult must (a) enforce partial⇔partial_reason coupling and
    (b) actually accept partial_reason at all — previously the field didn't
    exist on the model, so refine() silently dropped it via Pydantic's
    "ignore extras" default and callers had no way to learn why a run
    stopped early (e.g. cost-cap or unknown-pricing refusal).
    """
    import pydantic

    from consult.types import RefineResult

    base = dict(
        run_id="r",
        rounds_completed=1,
        final_manifest=[],
        verdicts=[],
        synthesis="x",
        converged=False,
        threshold=0.85,
        cost_usd=0.0,
        wall_ms=0,
    )
    # Happy path: partial_reason actually round-trips through the model
    rr = RefineResult(**base, partial=True, partial_reason="cost cap exceeded")
    assert rr.partial_reason == "cost cap exceeded"

    # Validator catches the inconsistent states
    with pytest.raises(pydantic.ValidationError):
        RefineResult(**base, partial=True)  # no reason
    with pytest.raises(pydantic.ValidationError):
        RefineResult(**base, partial=False, partial_reason="oops")


# ---- Live tests (gated on API keys) ----------------------------------------


HAVE_KEYS = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
)


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
def test_estimate_cost_smoke():
    specs = [ModelSpec(model="claude-haiku")]
    est, _ = estimate_cost(specs, "say hello in five words")
    assert est >= 0


@pytest.mark.skipif(not HAVE_KEYS, reason="no API keys present")
@pytest.mark.asyncio
async def test_tiny_panel_dry_run():
    from consult.runner import fanout

    specs = [ModelSpec(model="claude-haiku"), ModelSpec(model="openrouter/x-ai/grok-4.3")]
    handle = await fanout("ping", specs, dry_run=True)
    assert handle.partial
    assert "dry_run" in (handle.partial_reason or "")
