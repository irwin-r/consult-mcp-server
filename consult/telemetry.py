"""OpenTelemetry instrumentation — optional, no-op when not installed.

Emits spans following the OpenTelemetry GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/) so consult slots
into any AI-observability stack (Langfuse, Phoenix, Helicone, Honeycomb,
Datadog) without consult-specific shims.

Spans emitted:

- `consult.run` — the outer orchestrate.consult / refine.refine call
  with `app.consult.run_id`, `app.consult.tier`, `app.consult.tool_name`
  attributes.

- `gen_ai.chat <model>` — one per panellist with `gen_ai.system`,
  `gen_ai.request.model`, `gen_ai.response.model`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.response.finish_reasons`, plus consult's `app.consult.slug`
  and `app.consult.cost_usd`.

Activation: only when the `opentelemetry-api` package is importable AND
the `OTEL_EXPORTER_OTLP_ENDPOINT` env var is set. Otherwise the
context-manager helpers are no-ops — no overhead and no need to
conditionalise call sites. The opt-in via env var matches the OTel
ecosystem convention.

Install with `pip install consult-mcp-server[otel]` to pull the runtime
dependency. The exporter (OTLP, console, etc.) is the caller's
responsibility — consult only emits spans into whatever tracer provider
the host process configured.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

logger = logging.getLogger(__name__)


# Resolved on first use, cached. `None` means OTel isn't installed; a
# `Tracer` instance means it is and we should emit. Lazily evaluated so
# import-time cost is zero when OTel isn't around.
_tracer_singleton: Any = None
_resolution_attempted: bool = False


def _get_tracer() -> Any:
    """Resolve (and cache) the OpenTelemetry tracer for `consult`.

    Returns `None` when:
    - `opentelemetry.trace` cannot be imported (extras not installed)
    - the user hasn't configured OTel (no `OTEL_EXPORTER_OTLP_ENDPOINT`
      env var). The presence-of-endpoint check is the lightest gate —
      with an endpoint, the host process is signalling "I want
      telemetry"; without, we stay silent.
    """
    global _tracer_singleton, _resolution_attempted
    if _resolution_attempted:
        return _tracer_singleton
    _resolution_attempted = True

    # Cheap gate: skip the import if the user hasn't opted in via env.
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None

    try:
        from opentelemetry import trace  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.debug(
            "OTEL_EXPORTER_OTLP_ENDPOINT set but `opentelemetry` not "
            "installed; install with `pip install consult-mcp-server[otel]`"
        )
        return None

    _tracer_singleton = trace.get_tracer("consult")
    return _tracer_singleton


@contextmanager
def span(name: str, *, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Context manager that yields a span (or `None` when OTel is off).

    Call sites should check whether the yielded value is truthy before
    setting additional attributes:

        with telemetry.span("gen_ai.chat", attributes={...}) as sp:
            ...do work...
            if sp is not None:
                sp.set_attribute("gen_ai.usage.output_tokens", n)

    A wholly-OTel-off code path yields `None`, never raises, and never
    imports `opentelemetry`. Adding instrumentation to a new call site
    is a one-block addition with zero cost in the default config.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    # Attributes pass-through to OTel — values are converted by the SDK.
    # We intentionally don't pre-validate; an upstream change to the
    # semantic conventions shouldn't require a consult release.
    with tracer.start_as_current_span(name, attributes=attributes or {}) as sp:
        yield sp


def set_attribute(span_or_none: Any, key: str, value: Any) -> None:
    """Safely set an attribute on a span returned by `span()`.

    A `None` span (OTel off) is a no-op. Values that the OTel SDK can't
    encode are coerced to `str()` so a call-site change doesn't crash
    the run on an unexpected type.
    """
    if span_or_none is None:
        return
    try:
        span_or_none.set_attribute(key, value)
    except Exception:  # noqa: BLE001 — observability must not abort work
        try:
            span_or_none.set_attribute(key, str(value))
        except Exception:  # noqa: BLE001
            logger.debug("telemetry: failed to set attribute %s", key)


def record_exception(span_or_none: Any, exc: BaseException) -> None:
    """Record an exception on the span. No-op when OTel is off."""
    if span_or_none is None:
        return
    with suppress(Exception):
        span_or_none.record_exception(exc)
