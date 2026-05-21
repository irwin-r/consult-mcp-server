"""Tests for the `consult.telemetry` no-op-when-off helpers.

Real OTel emission can't be unit-tested without spinning up an exporter;
that's covered by integration-time inspection. These tests pin the
contract that matters for callers: telemetry is a zero-overhead no-op
when not configured, and never raises.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_telemetry_module():
    """The telemetry module caches its tracer resolution. Reload it
    before each test so monkeypatching the env var works."""
    import consult.telemetry as telemetry
    importlib.reload(telemetry)
    yield
    importlib.reload(telemetry)


def test_span_yields_none_when_otel_endpoint_unset(monkeypatch):
    """Without `OTEL_EXPORTER_OTLP_ENDPOINT` the context manager yields
    `None` — call sites use this as the "telemetry off" signal."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    import consult.telemetry as telemetry

    with telemetry.span("test.op", attributes={"k": "v"}) as sp:
        assert sp is None


def test_set_attribute_is_noop_on_none_span(monkeypatch):
    """`set_attribute(None, ...)` must not raise — call sites unconditionally
    invoke it after `with` yields."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    import consult.telemetry as telemetry
    # Multiple set_attribute calls on None — should be silent no-ops
    telemetry.set_attribute(None, "anything", 42)
    telemetry.set_attribute(None, "complex", {"unserialisable": object()})
    telemetry.set_attribute(None, "list", ["a", "b"])


def test_record_exception_is_noop_on_none_span(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    import consult.telemetry as telemetry
    telemetry.record_exception(None, RuntimeError("boom"))


def test_set_attribute_falls_back_to_str_on_unserialisable_value(monkeypatch):
    """When OTel IS enabled but the value can't be encoded, the helper
    must fall back to `str(value)` rather than raising — observability
    must not abort the call."""

    class _FakeSpan:
        def __init__(self):
            self.values = []

        def set_attribute(self, key, value):
            self.values.append((key, value))
            if isinstance(value, dict):
                # Mimic SDK rejecting nested dicts
                raise TypeError("dict not supported")

    import consult.telemetry as telemetry
    span = _FakeSpan()
    telemetry.set_attribute(span, "k", {"nested": "dict"})
    # First attempt with original value, second with str()
    assert len(span.values) == 2
    assert isinstance(span.values[1][1], str)


def test_span_no_error_when_opentelemetry_not_installed(monkeypatch):
    """Even with the env var set, if `opentelemetry` can't be imported
    the helper must still yield `None` cleanly."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # Simulate the import failing
    import sys
    real = sys.modules.pop("opentelemetry", None)
    real_trace = sys.modules.pop("opentelemetry.trace", None)
    sys.modules["opentelemetry"] = None  # forces ImportError on `from opentelemetry import trace`
    try:
        import consult.telemetry as telemetry
        importlib.reload(telemetry)
        with telemetry.span("test.op") as sp:
            assert sp is None
    finally:
        sys.modules.pop("opentelemetry", None)
        if real is not None:
            sys.modules["opentelemetry"] = real
        if real_trace is not None:
            sys.modules["opentelemetry.trace"] = real_trace
