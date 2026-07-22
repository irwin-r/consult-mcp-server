"""Pricing gap-fill tests — models.json `pricing` blocks → litellm.register_model."""

from __future__ import annotations

import json

import litellm
import pytest

from consult import pricing, registry


@pytest.fixture()
def fresh_registry(tmp_path, monkeypatch):
    """Point the user overlay at tmp and clear every cache the test touches.

    `models_config()` and `ensure_registered()` are both process-cached;
    without the clears a test would register against the previous test's
    config (or leak its tmp overlay into later tests).
    """
    monkeypatch.setattr(registry, "_USER_CONFIG", tmp_path)
    registry.models_config.cache_clear()
    pricing.ensure_registered.cache_clear()
    yield tmp_path
    registry.models_config.cache_clear()
    pricing.ensure_registered.cache_clear()


def _write_overlay(tmp_path, models: dict) -> None:
    (tmp_path / "models.json").write_text(json.dumps({"models": models}))


def test_registers_unpriced_model_with_pricing_block(fresh_registry, monkeypatch):
    _write_overlay(
        fresh_registry,
        {
            "test-model": {
                "litellm_id": "openrouter/acme/test-1",
                "provider": "openrouter",
                "max_input_tokens": 42000,
                "pricing": {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 8.0},
            }
        },
    )
    monkeypatch.setattr(litellm, "cost_per_token", lambda **kw: (_ for _ in ()).throw(Exception("unknown")))
    captured: list[dict] = []
    monkeypatch.setattr(litellm, "register_model", lambda mc: captured.append(mc))

    count = pricing.ensure_registered()

    rows = {k: v for mc in captured for k, v in mc.items()}
    assert "openrouter/acme/test-1" in rows
    row = rows["openrouter/acme/test-1"]
    assert row["input_cost_per_token"] == pytest.approx(2.0 / 1e6)
    assert row["output_cost_per_token"] == pytest.approx(8.0 / 1e6)
    assert row["litellm_provider"] == "openrouter"
    assert row["mode"] == "chat"
    assert row["max_input_tokens"] == 42000
    assert count == len(rows)


def test_already_priced_models_left_alone(fresh_registry, monkeypatch):
    """Shipped tables win: a model litellm can already price is never
    re-registered, so library upgrades keep supplying fresher numbers.
    """
    monkeypatch.setattr(litellm, "cost_per_token", lambda **kw: (0.001, 0.002))
    captured: list[dict] = []
    monkeypatch.setattr(litellm, "register_model", lambda mc: captured.append(mc))

    assert pricing.ensure_registered() == 0
    assert captured == []


def test_malformed_pricing_block_skips_without_raising(fresh_registry, monkeypatch, caplog):
    _write_overlay(
        fresh_registry,
        {
            "bad-model": {
                "litellm_id": "openrouter/acme/bad-1",
                "provider": "openrouter",
                "pricing": {"input_usd_per_mtok": "not-a-number"},
            },
            # Packaged entries without a pricing block must be ignored, and a
            # None pricing (JSON null override) must not trip isinstance.
            "null-model": {"litellm_id": "openrouter/acme/null-1", "pricing": None},
        },
    )
    monkeypatch.setattr(litellm, "cost_per_token", lambda **kw: (_ for _ in ()).throw(Exception("unknown")))
    captured: list[dict] = []
    monkeypatch.setattr(litellm, "register_model", lambda mc: captured.append(mc))

    pricing.ensure_registered()  # must not raise

    rows = {k for mc in captured for k in mc}
    assert "openrouter/acme/bad-1" not in rows
    assert "openrouter/acme/null-1" not in rows


def test_responses_mode_entry_registers_as_responses(fresh_registry, monkeypatch):
    _write_overlay(
        fresh_registry,
        {
            "resp-model": {
                "litellm_id": "openai/test-pro",
                "provider": "openai",
                "mode": "responses",
                "pricing": {"input_usd_per_mtok": 5.0, "output_usd_per_mtok": 30.0},
            }
        },
    )
    monkeypatch.setattr(litellm, "cost_per_token", lambda **kw: (_ for _ in ()).throw(Exception("unknown")))
    captured: list[dict] = []
    monkeypatch.setattr(litellm, "register_model", lambda mc: captured.append(mc))

    pricing.ensure_registered()

    rows = {k: v for mc in captured for k, v in mc.items()}
    assert rows["openai/test-pro"]["mode"] == "responses"


def test_estimate_cost_triggers_registration(monkeypatch):
    """The estimate path is the first litellm price lookup on a dry run;
    it must gap-fill before pricing the panel.
    """
    from consult.runner import costs

    calls: list[int] = []
    monkeypatch.setattr(pricing, "ensure_registered", lambda: calls.append(1))
    costs.estimate_cost([], "prompt")
    assert calls


def test_packaged_registry_prices_every_tiered_model():
    """End-to-end guard for the packaged config: after gap-fill, every model
    reachable through a tier must be priceable, or the `max_run_usd` gate
    silently degrades to warn-don't-block (the bug this module fixes).
    """
    registry.models_config.cache_clear()
    pricing.ensure_registered.cache_clear()
    try:
        pricing.ensure_registered()
        cfg = registry.models_config()
        tiered = {alias for members in cfg["tiers"].values() for alias in members}
        unpriced = []
        for alias in sorted(tiered):
            lid = cfg["models"][alias]["litellm_id"]
            try:
                pc, cc = litellm.cost_per_token(model=lid, prompt_tokens=100, completion_tokens=100)
            except Exception:
                unpriced.append(lid)
                continue
            if pc is None or cc is None:
                unpriced.append(lid)
        assert not unpriced, "tiered models litellm cannot price after gap-fill:\n  " + "\n  ".join(unpriced)
    finally:
        registry.models_config.cache_clear()
        pricing.ensure_registered.cache_clear()
