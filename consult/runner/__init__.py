"""Async fan-out runner. Calls LiteLLM in parallel, classifies responses,
writes artifacts, and assembles a RunHandle.

This package replaced the single 1,900-line runner.py module; the public
surface is unchanged. Everything importable from `consult.runner` before
the split — including the underscore helpers tests exercise directly —
is re-exported here, and the internal call paths resolve `_call_one`,
`estimate_cost` / `aestimate_cost`, and the retry-class helpers through
this module at call time, so `monkeypatch.setattr(consult.runner, ...)`
behaves exactly as it did on the flat module.

Layout:
- transport.py — LiteLLM retry/streaming/Responses adapter, provider
  semaphores, message construction
- fit.py — context-budget fitting and attachment-aware trimming
- specs.py — `model:N` expansion, slug derivation, per-slug prompts
- costs.py — panel cost estimation
- fanout.py — `_call_one`, slow-tail dropout, `fanout()`
"""

from __future__ import annotations

import litellm  # noqa: F401 — tests patch `consult.runner.litellm.<fn>` directly

from ..progress import ProgressCallback  # noqa: F401
from ..status import classify  # noqa: F401
from .costs import aestimate_cost, estimate_cost, estimate_drivers  # noqa: F401
from .fanout import (  # noqa: F401
    _call_one,
    _dropout_entry,
    _gather_with_tail_dropout,
    _PanelProgress,
    _write_text_async,
    fanout,
)
from .fit import (  # noqa: F401
    _fit_prompt_to_context,
    _max_input_tokens,
    _replace_attachments_with_stubs,
    concat_turn_text,
)
from .specs import (  # noqa: F401
    _DEFAULT_MAX_PANEL_SIZE,
    _build_per_slug_prompt,
    _make_slug,
    _make_slugs,
    _max_panel_size,
    expand_specs,
    sanitise_derived_slug,
)
from .transport import (  # noqa: F401
    _acompletion_with_retry,
    _aresponses_as_completion,
    _bare_api_error_class,
    _format_error_message,
    _get_provider_sems,
    _rate_limit_class,
    _stream_acompletion,
    _stream_partial_interval_s,
    _transient_error_classes,
    apply_web_search,
    build_messages,
    configure_litellm,
)
