"""Daily cost ledger.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

from consult import artifacts


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
