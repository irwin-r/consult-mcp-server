"""LiteLLM response classification.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import json

import pytest

from consult import artifacts
from consult.status import classify
from consult.types import Status


def test_status_classifier_handles_exceptions():
    s, _, _ = classify(None, exception=RuntimeError("rate limit hit"))
    assert s == Status.RATE_LIMITED
    s, _, _ = classify(None, exception=TimeoutError("timed out"))
    assert s == Status.TIMEOUT


def test_status_classifier_normal_responses():
    """Cover the OK/TRUNCATED/EMPTY/CONTENT_FILTERED/MALFORMED response paths.

    Today only the exception branch is tested — the body classification logic
    could return wrong statuses silently if not exercised.
    """
    from types import SimpleNamespace

    from consult.status import classify

    def make_resp(content, finish):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)]
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


def test_status_classifier_salvages_reasoning_content():
    """OR thinking models (kimi k2.6, glm-5.1) can return an empty `content`
    with the actual text in `reasoning_content`. classify() must salvage it
    rather than discard a completed (stop) or partially-useful (length)
    response — run 20260612-005818 binned a finished kimi answer (52k chars
    of reasoning_content, zero content) as EMPTY.
    """
    from types import SimpleNamespace

    def make_resp(content, finish, reasoning=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, reasoning_content=reasoning),
                    finish_reason=finish,
                )
            ]
        )

    # Completed answer that landed in the reasoning channel → OK, salvaged.
    s, _, body = classify(make_resp("", "stop", reasoning="the actual findings"))
    assert s == Status.OK
    assert body == "the actual findings"

    # Reasoning truncated before any content → TRUNCATED with the partial body.
    s, _, body = classify(make_resp("", "length", reasoning="partial reasoning"))
    assert s == Status.TRUNCATED
    assert body == "partial reasoning"

    # Whitespace-only reasoning doesn't rescue anything.
    s, _, _ = classify(make_resp("", "stop", reasoning="   \n"))
    assert s == Status.EMPTY

    # `content` wins when present; reasoning stays untouched.
    s, _, body = classify(make_resp("real", "stop", reasoning="ignored"))
    assert s == Status.OK
    assert body == "real"


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
