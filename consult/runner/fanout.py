"""The fan-out orchestrator: per-panellist calls, slow-tail dropout,
progress plumbing, and `fanout()` itself.

`_call_one` and `aestimate_cost` are resolved through the package facade
at call time so `monkeypatch.setattr(runner, "_call_one", ...)` and the
sync `estimate_cost` patch idiom keep working as they did when runner
was a single module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm

from consult import runner as _facade

from .. import artifacts, citations, context, pricing, registry, telemetry
from .. import attachments as attachments_mod
from ..envutil import env_float
from ..progress import (
    Heartbeat,
    PanellistCompleted,
    PanellistPartial,
    PanellistStarted,
    PhaseStarted,
    ProgressCallback,
    ProgressEvent,
    append_progress_log,
)
from ..redact import redact_exc, redact_traceback, scrub_exception_attrs
from ..status import classify
from ..types import ManifestEntry, ModelSpec, RunHandle, Status
from .fit import _fit_prompt_to_context, concat_turn_text
from .specs import _build_per_slug_prompt, _make_slugs, expand_specs, output_budget
from .transport import (
    _acompletion_with_retry,
    _aresponses_as_completion,
    _format_error_message,
    _get_provider_sems,
    _stream_acompletion,
    apply_web_search,
    build_messages,
    configure_litellm,
)

logger = logging.getLogger(__name__)


# Sent as the follow-up user turn when a panellist's response was cut by the
# output cap. The model sees its own partial answer as the preceding
# assistant turn, so "continue from where it stopped" is well-defined.
_CONTINUATION_NUDGE = (
    "Your previous message was cut off by an output-length limit. Continue "
    "it from EXACTLY where it stopped — do not repeat anything you already "
    "wrote, do not add a preamble or an apology, resume mid-sentence if "
    "necessary. If the instructions asked for a CONFIDENCE/KEY_REASON "
    "footer, make sure your continuation ends with it."
)


def _join_notes(*notes: str | None) -> str | None:
    """Merge optional manifest-note fragments, dropping Nones."""
    joined = "; ".join(n for n in notes if n)
    return joined or None


async def _write_text_async(path: Path, content: str) -> None:
    """Off-loop write helper. Wraps `Path.write_text` in `asyncio.to_thread`
    so per-panellist artifact writes inside `_call_one` don't block the
    event loop while N parallel panellists finish around the same time.
    Sync-in-async writes on a 10-spec panel previously stacked ~30 blocking
    syscalls that could push past per-call timeouts.
    """
    await asyncio.to_thread(path.write_text, content)


async def _call_one(
    spec: ModelSpec,
    slug: str,
    per_slug_prompt: str,
    paths: artifacts.RunPaths,
    provider_sems: dict[str, asyncio.Semaphore] | None = None,
    *,
    stream: bool = False,
    on_partial: Callable[[int, int], Awaitable[None]] | None = None,
    on_activity: Callable[[], None] | None = None,
    prior_turns: list[dict[str, Any]] | None = None,
    capsule_kind: str = "decision",
    max_output_tokens: int | None = None,
    web_search: bool = False,
    timeout_floor_s: float | None = None,
) -> ManifestEntry:
    # An unknown alias must fail this single panellist, not the whole panel.
    # `asyncio.gather` without return_exceptions=True would otherwise cancel
    # every sibling call when the KeyError propagates out.
    try:
        entry = registry.resolve_model(spec.model)
    except KeyError as e:
        await _write_text_async(paths.response_text(slug), "")
        # Unknown alias — no provider call was made, so the cost is genuinely
        # zero (NOT unknown). cost_usd=0.0 + cost_known=True is the correct
        # encoding; using cost_usd=None would now fail the cost-invariant
        # validator on ManifestEntry.
        return ManifestEntry(
            slug=slug,
            model_id=None,
            persona=spec.stance if spec.stance else None,
            status=Status.ERROR,
            finish_reason=None,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=0,
            tokens_in=None,
            tokens_out=None,
            cost_usd=0.0,
            cost_known=True,
            error=str(e),
            confidence=None,
            capsule=None,
        )
    # CLI panellists don't have a LiteLLM ID — they invoke a subprocess.
    # Use `litellm_id` when present, otherwise fall back to the spec's
    # alias for logging/diagnostics so error messages stay attributable.
    litellm_id = entry.get("litellm_id") or spec.model
    # Output budget: kind cap vs model default vs caller override — see
    # `output_budget` for the reasoning behind each dimension. Issue #55:
    # the kind cap alone bound below reasoning burn, so e.g. gpt-pro spent
    # its whole 2000/4000-token budget thinking and returned zero text.
    # estimate_cost shares the derivation so the max_run_usd gate prices
    # the same ceiling actually granted here.
    budget = output_budget(entry, capsule_kind, max_output_tokens)
    timeout = entry.get("default_timeout_s", 180)
    # Patience floor (issue #92 hardening): a caller that would rather wait
    # hours than lose a deep model raises every per-spec timeout to at
    # least this. The registry defaults are sized for interactive panels;
    # research sub-runs pass a floor instead of editing the registry.
    if timeout_floor_s is not None:
        timeout = max(timeout, timeout_floor_s)
    provider = entry.get("provider", "")
    # mode=responses models (e.g. gpt-pro, gpt-codex) use the OpenAI Responses
    # API, not chat completions — routed via `_aresponses_as_completion`.
    is_responses = entry.get("mode") == "responses"

    extra: dict[str, Any] = {}
    if "reasoning_effort" in entry:
        extra["reasoning_effort"] = entry["reasoning_effort"]
    # Provider-native web search, requested per call and honoured only for
    # `supports_web` entries. Silent no-op (debug log) otherwise, so a
    # mixed panel with one web-capable model doesn't fail the rest.
    if web_search:
        if entry.get("supports_web"):
            apply_web_search(extra, entry)
        else:
            logger.debug("web_search requested but %s lacks supports_web; ignoring", litellm_id)

    await _write_text_async(paths.prompt_for(slug), per_slug_prompt)
    start = time.time()
    status: Status
    finish: str | None = None
    body = ""
    cost: float | None = None
    cost_known: bool = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None

    # Per-provider concurrency gate. The FRICTION log shows OpenAI panellists
    # rate-limiting concurrently on every panel run — the shared key's
    # per-minute bucket is exhausted by N parallel calls. The semaphore caps
    # the in-flight count per provider so e.g. only 2 OpenAI calls run at
    # once, the rest queue up. `provider_sems["default"]` covers raw LiteLLM
    # IDs whose provider isn't enumerated in models.json. `nullcontext()` is
    # async-compatible since Python 3.10 — same shape as a Semaphore, no-op.
    sem: asyncio.Semaphore | None = None
    if provider_sems:
        sem = provider_sems.get(provider) or provider_sems.get("default")

    # Pre-flight context-budget check + auto-trim. Each provider rejects
    # calls that exceed its context window with a hard 400; without this
    # we'd burn latency (and a per-provider rate-limit slot) to get back
    # a raw BadRequestError that surfaces in the manifest as a wall of
    # provider JSON.
    #
    # When the prompt would overflow, trim per_slug_prompt head+tail
    # (preserving the stance preface and the CONFIDENCE/KEY_REASON
    # footer — both live in the head/tail respectively) so the call
    # succeeds with a slightly-reduced view of the source material.
    # Trimming is logged at warning level and noted in the manifest's
    # `error` field even on success ("trimmed N chars..."), so the
    # caller can see that this panellist saw less than the others.
    #
    # Skipped silently when the model's context size is unknown — falls
    # back to the prior "let the provider reject it" behaviour for
    # unfamiliar models.
    trim_note: str | None = None
    # Facade lookup: tests monkeypatch consult.runner._max_input_tokens.
    max_in = _facade._max_input_tokens(litellm_id, entry)
    if max_in is not None:
        original_chars = len(per_slug_prompt)
        per_slug_prompt, dropped_chars = await _fit_prompt_to_context(
            per_slug_prompt,
            paths=paths,
            prior_turns=prior_turns,
            litellm_id=litellm_id,
            max_input_tokens=max_in,
            max_output_tokens=budget,
        )
        # `_fit_prompt_to_context` reports dropped chars explicitly — we
        # used to sniff for a `[TRIMMED` substring, which false-positived
        # on source-code attachments that mention the word. Now the
        # manifest note is both accurate (no false alarms when we didn't
        # actually trim) and quantified (drops and budget surfaced).
        if dropped_chars > 0:
            pct = (dropped_chars / original_chars * 100) if original_chars else 0.0
            trim_note = (
                f"input auto-trimmed: dropped ~{dropped_chars:,} chars "
                f"(~{pct:.0f}% of {original_chars:,}c source) to fit "
                f"{max_in:,}-token context (reserved {budget:,} tok for output)"
            )

    # OpenTelemetry span per panellist call. No-op when OTel isn't
    # installed or the user hasn't set OTEL_EXPORTER_OTLP_ENDPOINT.
    # Follows the gen_ai.* semantic conventions so consult shows up in
    # off-the-shelf AI-observability dashboards without a custom mapping.
    # Wraps the whole try/except so cost/tokens/finish_reason can be set
    # after the response comes back, the exception handlers can record
    # errors on the span, and the span is guaranteed to close.
    otel_span_name = f"gen_ai.chat {litellm_id}"
    otel_attrs = {
        "gen_ai.system": provider or "unknown",
        "gen_ai.request.model": litellm_id,
        "gen_ai.operation.name": "chat",
        "app.consult.slug": slug,
        "app.consult.run_id": paths.run_id,
    }

    def _accrue_usage_and_cost(resp: Any) -> None:
        """Fold one response's tokens + spend into the entry's running
        totals. Every billable call on this panellist (empty-truncation
        retry, truncation continuation) is a real provider charge; the
        manifest reports the sum, not the last attempt.
        """
        nonlocal tokens_in, tokens_out, cost, cost_known
        usage = getattr(resp, "usage", None)
        attempt_in = getattr(usage, "prompt_tokens", None) if usage else None
        attempt_out = getattr(usage, "completion_tokens", None) if usage else None
        if attempt_in is not None:
            tokens_in = (tokens_in or 0) + attempt_in
        if attempt_out is not None:
            tokens_out = (tokens_out or 0) + attempt_out
        if is_responses:
            # The Responses adapter returns a synthetic chat-shaped object
            # that `completion_cost` can't price; compute from token counts.
            try:
                pc, cc = litellm.cost_per_token(
                    model=litellm_id,
                    prompt_tokens=attempt_in or 0,
                    completion_tokens=attempt_out or 0,
                )
                cost = (cost or 0.0) + float(pc or 0.0) + float(cc or 0.0)
            except Exception as ce:  # noqa: BLE001
                logger.warning("responses cost lookup failed for %s: %s", litellm_id, redact_exc(ce))
                cost_known = False
        else:
            try:
                attempt_cost = litellm.completion_cost(completion_response=resp)
                if attempt_cost is None:
                    cost_known = False
                else:
                    cost = (cost or 0.0) + float(attempt_cost)
            except Exception as ce:  # noqa: BLE001
                logger.warning("cost lookup failed for %s: %s", litellm_id, redact_exc(ce))
                cost_known = False

    retry_note: str | None = None
    with telemetry.span(otel_span_name, attributes=otel_attrs) as tspan:
        try:
            # Empty-truncation retry (FRICTION 2026-07-13): a reasoning model
            # can burn the entire output grant thinking and come back
            # finish_reason=length with a zero-byte body — billed, counted as
            # a panellist, zero value (run 20260713-050837: gpt spent 12k
            # output tokens for an empty response). One retry with a doubled
            # budget converts most of those into usable answers; both
            # attempts' spend is accumulated.
            for attempt in (1, 2):
                async with sem if sem is not None else nullcontext():
                    messages = build_messages(per_slug_prompt, provider, prior_turns)
                    if is_responses:
                        # Responses-API models 404 on chat completions; route them
                        # through the adapter. No streaming on this path.
                        resp = await _aresponses_as_completion(
                            timeout=timeout,
                            model=litellm_id,
                            messages=messages,
                            max_completion_tokens=budget,
                            **extra,
                        )
                    elif stream:
                        resp = await _stream_acompletion(
                            timeout=timeout,
                            on_partial=on_partial,
                            start=start,
                            model=litellm_id,
                            messages=messages,
                            max_completion_tokens=budget,
                            **extra,
                        )
                    else:
                        resp = await _acompletion_with_retry(
                            timeout=timeout,
                            model=litellm_id,
                            messages=messages,
                            max_completion_tokens=budget,
                            **extra,
                        )
                # Persist raw response — use model_dump for Pydantic, fall
                # back to dict
                try:
                    raw = resp.model_dump()  # type: ignore[attr-defined]
                except AttributeError:
                    raw = dict(resp) if hasattr(resp, "__iter__") else {"_repr": repr(resp)}
                await _write_text_async(paths.response_raw(slug), json.dumps(raw, indent=2, default=str))

                status, finish, body = classify(resp)
                # Tokens and cost accumulate PER ATTEMPT: the empty first
                # attempt is billed just like the retry that replaces it.
                _accrue_usage_and_cost(resp)

                if attempt == 1 and status is Status.TRUNCATED and not body.strip():
                    budget *= 2
                    retry_note = (
                        f"empty body at finish_reason={finish} with a "
                        f"{budget // 2}-token grant (reasoning burn); "
                        f"retried once at {budget}"
                    )
                    logger.warning("panellist %s: %s", slug, retry_note)
                    if on_activity is not None:
                        # The retry is a fresh provider call; tell the
                        # slow-tail dropout this panellist is working, not hung.
                        on_activity()
                    continue
                break

            # Truncation continuation (FRICTION 2026-07-13): a panellist that
            # hit finish_reason=length mid-document used to lose its tail —
            # 4 of 9 panellists truncated writing a build spec, and the
            # capsule extractor had to mine incomplete bodies. One follow-up
            # call hands the model its own partial output as an assistant
            # turn and asks it to carry on from the cut; the stitched body
            # is what downstream consumers see. Bounded to a single
            # continuation; disable with CONSULT_CONTINUE_ON_TRUNCATION=0.
            if (
                status is Status.TRUNCATED
                and body.strip()
                and os.environ.get("CONSULT_CONTINUE_ON_TRUNCATION", "1") == "1"
            ):
                if on_activity is not None:
                    # Closes the FRICTION gap where a panellist mid-continuation
                    # could still be dropped by the slow-tail policy.
                    on_activity()
                cont_messages = [
                    *messages,
                    {"role": "assistant", "content": body},
                    {"role": "user", "content": _CONTINUATION_NUDGE},
                ]
                cont = None
                try:
                    async with sem if sem is not None else nullcontext():
                        if is_responses:
                            cont = await _aresponses_as_completion(
                                timeout=timeout,
                                model=litellm_id,
                                messages=cont_messages,
                                max_completion_tokens=budget,
                                **extra,
                            )
                        else:
                            # Non-streamed even when the first call streamed:
                            # the partial-progress callback already saw the
                            # first body, and a second stream adds machinery
                            # for no user-visible gain.
                            cont = await _acompletion_with_retry(
                                timeout=timeout,
                                model=litellm_id,
                                messages=cont_messages,
                                max_completion_tokens=budget,
                                **extra,
                            )
                except Exception as ce:  # noqa: BLE001
                    # Keep the truncated body — a failed continuation must not
                    # downgrade a partial answer to nothing. The provider may
                    # have billed the attempt.
                    logger.warning("continuation failed for %s: %s", slug, redact_exc(ce))
                    cost_known = False
                    retry_note = _join_notes(retry_note, "truncation continuation failed; body kept as-is")
                if cont is not None:
                    _accrue_usage_and_cost(cont)
                    try:
                        cont_raw = cont.model_dump()  # type: ignore[attr-defined]
                    except AttributeError:
                        cont_raw = {"_repr": repr(cont)}
                    await _write_text_async(
                        paths.responses / f"{slug}.continuation.json",
                        json.dumps(cont_raw, indent=2, default=str),
                    )
                    cont_status, cont_finish, cont_body = classify(cont)
                    if cont_body.strip():
                        body = body + cont_body
                        finish = cont_finish
                        if cont_status is Status.OK:
                            status = Status.OK
                            retry_note = _join_notes(retry_note, "continued once after output-cap truncation")
                        else:
                            retry_note = _join_notes(
                                retry_note,
                                "continued once after output-cap truncation; "
                                "continuation itself also truncated",
                            )
                    else:
                        retry_note = _join_notes(
                            retry_note, "truncation continuation returned no content; body kept as-is"
                        )

            # Web-grounded panellists (sonar) cite sources as [n] markers
            # whose URL list lives in response metadata, not the content.
            # Fold it into the body here — the one choke point every
            # consumer (extractor, synth, artifacts, refine) reads from
            # (issue #61). Reads `raw` rather than `resp` so LiteLLM's
            # models can't strip the provider-extra fields.
            if body:
                body = citations.append_sources_footer(body, citations.harvest(raw))
            # Populate the OTel span with per-call telemetry (tokens /
            # cost / finish_reason). No-op when the span is None.
            if tokens_in is not None:
                telemetry.set_attribute(tspan, "gen_ai.usage.input_tokens", tokens_in)
            if tokens_out is not None:
                telemetry.set_attribute(tspan, "gen_ai.usage.output_tokens", tokens_out)
            if cost is not None:
                telemetry.set_attribute(tspan, "app.consult.cost_usd", cost)
            telemetry.set_attribute(tspan, "app.consult.cost_known", cost_known)
            if finish:
                telemetry.set_attribute(tspan, "gen_ai.response.finish_reasons", [finish])
            telemetry.set_attribute(tspan, "app.consult.status", status.value)

        except TimeoutError as te:
            status, finish, body = Status.TIMEOUT, None, ""
            error = f"timeout after {timeout}s"
            # Conservative: a TimeoutError from `asyncio.wait_for` means
            # we gave up waiting, NOT that the HTTP request never landed.
            # The provider may have processed and billed us;
            # cost_known=False surfaces that uncertainty in the ledger
            # as a lower bound rather than silently understating spend.
            # Mirrors slow-tail dropout.
            cost = None
            cost_known = False
            telemetry.record_exception(tspan, te)
            telemetry.set_attribute(tspan, "app.consult.status", status.value)
        except Exception as e:  # noqa: BLE001 — LiteLLM raises many concrete types
            # Scrub the object first: `telemetry.record_exception` hands it
            # to the OTel exporter, which formats message + traceback
            # outside consult's string-redaction boundaries (issue #39).
            scrub_exception_attrs(e)
            status, finish, body = classify(None, exception=e)
            error = _format_error_message(e)
            # Same logic: most provider exceptions imply no billable
            # call, but we can't be sure for every case (e.g. 502
            # mid-stream may have billed). cost_known=False is the
            # conservative encoding.
            cost = None
            cost_known = False
            telemetry.record_exception(tspan, e)
            telemetry.set_attribute(tspan, "app.consult.status", status.value)

    await _write_text_async(paths.response_text(slug), body)
    latency_ms = int((time.time() - start) * 1000)

    # Per-panellist completion line. Visible in MCP server logs
    # (~/.claude/logs/mcp-logs-consult/) — gives mid-run observability so
    # a slow panellist's status is knowable without waiting for the full
    # manifest. Use info-level so production stays quiet by default.
    logger.info("panellist %s: %s in %dms", slug, status.value, latency_ms)

    # Per-run progress log — always on, tailable as JSONL even if the MCP
    # client didn't subscribe to notifications/progress. The typed event
    # makes programmatic consumers easy; (done, total) here are placeholders
    # since `_call_one` doesn't know the panel size — the wrapper in `fanout`
    # constructs the real progress event for the callback path.
    append_progress_log(
        paths.root,
        PanellistCompleted(
            done=0,
            total=0,
            slug=slug,
            status=status.value,
            latency_ms=latency_ms,
        ),
    )

    persona_label = spec.stance if spec.stance else None

    # Status.ERROR/TIMEOUT require a non-empty error per the model invariant.
    # Defensive: if classify() returns ERROR with no exception path taken (e.g.
    # malformed empty response), synthesise a placeholder so construction
    # doesn't blow up — the underlying classifier already logged the shape.
    if status in (Status.ERROR, Status.TIMEOUT) and not error:
        error = f"{status.value}: no provider exception captured"

    # Trim/retry notes go to `note` (info annotation on a successful call),
    # NOT `error` (failure reason). Conflating them made the viewer render
    # the trim message in red error styling on a green-OK card, which read
    # like a contradiction.
    note = _join_notes(trim_note, retry_note)
    return ManifestEntry(
        slug=slug,
        model_id=litellm_id,
        persona=persona_label,
        status=status,
        finish_reason=finish,
        resource_uri=paths.resource_uri(slug),
        body_path=str(paths.response_text(slug)),
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        cost_known=cost_known,
        error=error,
        note=note,
        confidence=None,  # populated by capsule extractor
        capsule=None,
    )


@dataclass
class _PanelProgress:
    """Live counters shared by the per-panellist workers, the heartbeat, and
    the tail-dropout gatherer. Everything runs in one event loop, so mutation
    between awaits is atomic and no lock is needed.
    """

    total: int
    done: int = 0
    started: int = 0
    completed_entries: list[ManifestEntry] = field(default_factory=list)
    pending_slugs: set[str] = field(default_factory=set)

    def record_completed(self, entry: ManifestEntry) -> None:
        self.done += 1
        self.completed_entries.append(entry)
        self.pending_slugs.discard(entry.slug)


def _dropout_entry(
    slug: str, spec: ModelSpec, paths: artifacts.RunPaths, latency_ms: int, tail_dropout_s: float
) -> ManifestEntry:
    """The TIMEOUT manifest entry for a panellist cancelled by slow-tail dropout.

    cost_known=False: cancel() can race an in-flight HTTP request the provider
    still bills, so treating the cost as a known zero would understate the run.
    """
    if not paths.response_text(slug).exists():
        paths.response_text(slug).write_text("")
    return ManifestEntry(
        slug=slug,
        model_id=None,
        persona=spec.stance if spec.stance else None,
        status=Status.TIMEOUT,
        finish_reason=None,
        resource_uri=paths.resource_uri(slug),
        body_path=str(paths.response_text(slug)),
        latency_ms=latency_ms,
        cost_usd=None,
        cost_known=False,
        error=f"slow-tail dropout after {tail_dropout_s}s",
        confidence=None,
        capsule=None,
    )


async def _gather_with_tail_dropout(
    *,
    run_one: Callable[[ModelSpec, str, str], Coroutine[Any, Any, ManifestEntry]],
    specs: list[ModelSpec],
    panel_slugs: list[str],
    per_prompts: list[str],
    state: _PanelProgress,
    paths: artifacts.RunPaths,
    start: float,
    safe_notify: Callable[[ProgressEvent], Awaitable[None]],
    task_activity: dict[str, float] | None = None,
    activity_window_s: float = 0.0,
    tail_dropout_s: float,
    tail_k_frac: float,
) -> list[ManifestEntry]:
    """Run every panellist concurrently and assemble the manifest in spec order.

    Slow-tail dropout: once `total - k` panellists return, the slowest
    stragglers get up to `tail_dropout_s` more, then are cancelled and recorded
    as TIMEOUT entries (FRICTION saw a 5x fast/slow spread; the wall-time win
    is releasing 1-2 laggards). Disabled for panels < 4 or when the env knobs
    fall out of range, in which case this is a plain gather and the per-spec
    timeout still bounds the worst case.

    (`panel_slugs` is named to avoid shadowing the `slugs` module imported at
    the top of this file.)
    """
    total = state.total
    enable_dropout = total >= 4 and tail_dropout_s > 0 and 0 < tail_k_frac < 1.0

    if not enable_dropout:
        coros = [
            run_one(spec, slug, per_prompt)
            for spec, slug, per_prompt in zip(specs, panel_slugs, per_prompts, strict=True)
        ]
        return list(await asyncio.gather(*coros))

    task_list: list[asyncio.Task[ManifestEntry]] = []
    task_meta: dict[asyncio.Task[ManifestEntry], tuple[str, ModelSpec]] = {}
    for spec, slug, per_prompt in zip(specs, panel_slugs, per_prompts, strict=True):
        t = asyncio.create_task(run_one(spec, slug, per_prompt))
        task_list.append(t)
        task_meta[t] = (slug, spec)

    try:
        completed_tasks: set[asyncio.Task[ManifestEntry]] = set()
        pending: set[asyncio.Task[ManifestEntry]] = set(task_list)
        k = max(1, math.ceil(total * tail_k_frac))
        trigger = max(1, total - k)

        while len(completed_tasks) < trigger and pending:
            done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            completed_tasks.update(done_set)

        if pending:
            logger.info(
                "slow-tail dropout: %d/%d complete, waiting up to %.1fs for %d stragglers",
                len(completed_tasks),
                total,
                tail_dropout_s,
                len(pending),
            )
            done_set, pending = await asyncio.wait(pending, timeout=tail_dropout_s)
            completed_tasks.update(done_set)

        # Progress-aware extension: while any straggler shows a REAL sign of
        # life within the window (stream chunk, retry/continuation call
        # starting — task start never counts), keep waiting in short slices
        # instead of cancelling a working panellist. Stragglers with no
        # signals fall straight through, so non-streaming panels behave
        # exactly as before. Per-spec timeouts bound every task, so this
        # loop always terminates.
        while pending and activity_window_s > 0 and task_activity is not None:
            now = time.monotonic()
            fresh = [
                t
                for t in pending
                if now - task_activity.get(task_meta[t][0], float("-inf")) < activity_window_s
            ]
            if not fresh:
                break
            logger.info(
                "slow-tail dropout: extending for %d active straggler(s)",
                len(fresh),
            )
            done_set, pending = await asyncio.wait(pending, timeout=min(activity_window_s, 15.0))
            completed_tasks.update(done_set)

        drop_entries: dict[asyncio.Task[ManifestEntry], ManifestEntry] = {}
        for t in pending:
            t.cancel()
        for t in pending:
            slug, spec = task_meta[t]
            try:
                # A task may complete in the race between asyncio.wait returning
                # and t.cancel(); keep its real result.
                await t
                continue
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001 — _run_one isn't supposed to raise
                # Redact before logging: a LiteLLM exception here can carry the
                # provider auth header, and `logger.exception` would render the
                # raw repr through a formatter we don't own. Format and redact
                # the traceback ourselves, then log it as a plain message.
                logger.error(
                    "panellist task for %s raised; treating as dropout\n%s",
                    slug,
                    redact_traceback(e),
                )
            # The cancel can land AFTER _run_one ran `record_completed` but
            # before its final progress await (only reachable when on_progress
            # is set — the suite's on_progress=None path has no suspension point
            # there). The panellist actually succeeded and its entry is already
            # in completed_entries; keep that instead of overwriting a success
            # with a synthetic TIMEOUT, and re-emit the close the cancel ate.
            recovered = next((e for e in state.completed_entries if e.slug == slug), None)
            if recovered is not None:
                drop_entries[t] = recovered
                await safe_notify(
                    PanellistCompleted(
                        done=state.done,
                        total=total,
                        slug=slug,
                        status=recovered.status.value,
                        latency_ms=recovered.latency_ms or 0,
                    )
                )
                continue
            latency_ms = int((time.time() - start) * 1000)
            entry = _dropout_entry(slug, spec, paths, latency_ms, tail_dropout_s)
            drop_entries[t] = entry
            state.record_completed(entry)
            await safe_notify(
                PanellistCompleted(
                    done=state.done,
                    total=total,
                    slug=slug,
                    status=Status.TIMEOUT.value,
                    latency_ms=latency_ms,
                )
            )
            append_progress_log(
                paths.root,
                PanellistCompleted(
                    done=0,
                    total=0,
                    slug=slug,
                    status=Status.TIMEOUT.value,
                    latency_ms=latency_ms,
                ),
            )

        manifest: list[ManifestEntry] = []
        for t in task_list:
            if t in drop_entries:
                manifest.append(drop_entries[t])
            else:
                manifest.append(t.result())
        return manifest
    except BaseException:
        # Parent unwind (MCP client / task_store cancel, or any error): cancel
        # and drain the panellist tasks so none is left running — and billing —
        # after we're gone. The plain-gather branch above propagates cancellation
        # into its children automatically; the explicit-task branch must do it.
        for t in task_list:
            t.cancel()
        # Drain to completion before propagating, looping over shield so even a
        # second cancel during cleanup can't abandon it — no child left running
        # (and billing). The children were just cancelled, so the gather settles
        # promptly and the loop can't spin.
        drain = asyncio.gather(*task_list, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                continue
        raise


async def fanout(
    prompt: str,
    specs: list[ModelSpec],
    *,
    blinded: bool = False,
    dry_run: bool = False,
    max_run_usd: float | None = None,
    existing_paths: artifacts.RunPaths | None = None,
    on_progress: ProgressCallback | None = None,
    stream: bool = False,
    capsule_kind: str = "decision",
    prior_turns: list[dict[str, Any]] | None = None,
    prior_turns_by_slug: dict[str, list[dict[str, Any]]] | None = None,
    max_concurrency: int | None = None,
    max_output_tokens: int | None = None,
    web_search: bool = False,
    timeout_floor_s: float | None = None,
    tail_dropout_s: float | None = None,
) -> RunHandle:
    """Parallel fan-out. Creates a fresh run by default. Pass `existing_paths`
    to write into an existing run dir (used by `refine` to keep all rounds
    under one run_id with round-suffixed slugs).

    Registers any models.json `pricing` blocks with LiteLLM up front so
    per-panellist cost accounting (`completion_cost` / `cost_per_token`
    below) can price models newer than the shipped tables.

    `prior_turns`, when set, is a sequence of `{role, content}` dicts
    prepended to the messages array for every panellist call. Used by
    `refine` with a `continuation_id` to expose the prior consultation as
    a proper user/assistant exchange. The text is included in cost
    estimation so the cap check stays accurate.

    `prior_turns_by_slug`, when set, is a per-slug override of `prior_turns`.
    A slug present in the dict uses its dict value; a slug absent falls
    back to `prior_turns`. Used by `refine` round-2+ to give each
    panellist its OWN conversation history (its prior question +
    answer) — round-1 question + answer become a stable prefix that
    Anthropic's prompt cache can reuse across rounds, instead of the
    monolithic refinement prompt that changes every round.

    `web_search=True` requests provider-native web search for every
    panellist whose registry entry carries `supports_web`; other
    panellists run unchanged. Search fees are provider-billed on top of
    tokens and are not part of `estimate_cost` (estimates stay floors).

    Patience knobs (issue #92 hardening): `timeout_floor_s` raises every
    panellist's per-spec timeout to at least the floor, and
    `tail_dropout_s` overrides the CONSULT_TAIL_DROPOUT_S env default —
    `0` disables slow-tail dropout entirely, so no straggler is ever
    cancelled and the (floored) per-spec timeouts are the only bound.
    Research sub-runs use both so a deep model can take hours without
    being dropped.

    `max_concurrency` (or `CONSULT_MAX_CONCURRENCY` env var) caps the
    *total* number of panellists in flight at once. The existing
    per-provider semaphores in `_get_provider_sems()` cap concurrency
    *per provider*; on the deep tier (~14 models across ~6 providers)
    they don't bound the aggregate, so all 14 calls launch simultaneously
    and 14 inbound HTTP connections + 14 token-counter cache misses fire
    at once. A global cap (default uncapped, recommended ~5-8 for wide
    panels) smooths this without changing per-provider behaviour.

    If `on_progress` is set, it's called once per panellist as it completes
    with `(done, total, message)`. Failures inside the callback are logged
    and swallowed — progress is best-effort, not load-bearing.
    """
    pricing.ensure_registered()
    # Resolve `model:N` sugar BEFORE estimate_cost so the cap reflects the
    # real panel size, not the pre-expansion request count.
    specs = expand_specs(specs)
    # An empty panel would create a bizarre run dir, fan out to zero
    # panellists, and return a manifest=[] handle that downstream tools
    # treat as "all panellists rate-limited". The MCP schema enforces
    # `minItems: 1` for callers going through the wire, but library
    # callers (sequence, refine, custom drivers) need their own gate.
    if not specs:
        raise ValueError("fanout requires at least one model spec")
    # Env-var override for streaming — lets a user enable streaming across
    # every fanout (incl. the ones nested inside consult/refine/sequence)
    # without changing the tool-call surface.
    if not stream and os.environ.get("CONSULT_STREAM", "0") == "1":
        stream = True
    # Apply LiteLLM tweaks lazily — covers callers that bypass __main__.cli
    # (tests, library use, the `consult-view`/`consult-ledger` CLIs that
    # import this module).
    configure_litellm()
    if existing_paths is None:
        paths = artifacts.create_run()
        # Run-init writes are threaded: the prompt (and its scrubbed copy
        # in the context bundle) can be megabytes once attachments are
        # inlined, and these sync writes ran on the event loop.
        await _write_text_async(paths.prompt_txt, prompt)
        # Snapshot the registry so replays are stable
        await _write_text_async(paths.registry_snapshot, json.dumps(registry.models_config(), indent=2))
        # Single immutable per-run context bundle. Downstream stages
        # (synth, capsule, arbiter) load this rather than re-receiving
        # the prompt — keeps the blinding scrub centralised and avoids
        # silent prompt-prompt skew across stages. `capsule_kind` is
        # persisted here so a `continuation_id` can inherit the prior
        # run's shape without the caller having to specify it again.
        # build() runs the brand-scrub regex over the whole prompt, so it
        # goes to the thread too.
        await asyncio.to_thread(
            lambda: context.write(
                paths,
                context.build(prompt, blinded=blinded, capsule_kind=capsule_kind),
            )
        )
        # Split inlined attachments out to `paths.attachments/<name>` so
        # (a) the per-panellist trim stub can reference a resolvable
        # resource URI and (b) the report and any tool-using downstream
        # model can still read the original source. Best-effort: a parse
        # failure leaves the panellist call unaffected — they still see
        # the inlined blocks in the prompt.
        try:
            await asyncio.to_thread(attachments_mod.persist_inlined_attachments, paths, prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("persist_inlined_attachments failed: %s", e)
    else:
        paths = existing_paths

    # Estimate cost up front. When `prior_turns` is set (continuation),
    # include their text in the token count so the cap check sees the real
    # input size.
    cost_input = prompt
    if prior_turns:
        cost_input = concat_turn_text(prior_turns) + "\n" + prompt
    # `max_output_tokens` is forwarded conditionally so existing test
    # monkeypatches of `estimate_cost` (plain lambdas without the kwarg)
    # keep working when the override isn't in play.
    est_kwargs: dict[str, Any] = {"capsule_kind": capsule_kind}
    if max_output_tokens is not None:
        est_kwargs["max_output_tokens"] = max_output_tokens
    estimate, all_known = await _facade.aestimate_cost(specs, cost_input, **est_kwargs)
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()

    # Don't clobber an existing manifest with the empty-manifest early-return
    # payload. Refine drives multiple rounds through the same `paths`; an
    # over-cap or dry-run rejection on round N+1 would otherwise wipe out
    # round N's successful manifest. New run dirs (existing_paths is None)
    # still get the manifest so downstream tools — `consult-view`,
    # `consult-ledger`, `synth.synthesise` — never face FileNotFoundError on
    # a dry-run / cap-rejected dir.
    def _persist_partial_handle(handle: RunHandle) -> None:
        if existing_paths is None or not paths.manifest_json.exists():
            artifacts.write_manifest(paths, handle.model_dump())

    # dry_run is checked BEFORE the cap gate: a dry_run answers "what would
    # this cost", and the cap guards real spend — a dry_run never spends, so
    # an over-cap estimate must still surface as an estimate, not as a cap
    # rejection. (When it does exceed the cap, the message says so, so the
    # caller still learns a real run would be refused.) This keeps fanout's
    # dry_run consistent with refine/sequence, which return their estimate
    # before any cap enforcement. The cap gate below only fires for real runs.
    if dry_run:
        suffix = "" if all_known else " (some prices unknown — actual cost may differ)"
        cap_note = f" (exceeds cap ${cap:.2f}; a real run would be rejected)" if estimate > cap else ""
        handle = RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=f"dry_run: estimated cost ${estimate:.4f}{suffix}{cap_note}",
            blinded=blinded,
        )
        _persist_partial_handle(handle)
        return handle

    if estimate > cap:
        # When some prices are unknown, `estimate` is only the known-priced
        # portion; the actual run could cost more. Surface that so the cap
        # message isn't misleading low. Mirrors the dry_run branch above.
        suffix = "" if all_known else " (known-priced portion only; some unknown)"
        # Name the panellists driving the estimate so the rejection is
        # actionable (raise the cap, or drop the named models) rather than
        # a bare number. Best-effort: enrichment must never turn a clean
        # rejection into a crash.
        drivers_note = ""
        try:
            drivers = await asyncio.to_thread(
                lambda: _facade.estimate_drivers(specs, cost_input, **est_kwargs)
            )
            if drivers:
                named = ", ".join(f"{m} ~${c:.2f}" for m, c in drivers)
                drivers_note = f"; top estimate drivers: {named}"
        except Exception as e:  # noqa: BLE001
            logger.debug("estimate_drivers enrichment failed: %s", e)
        handle = RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=(f"estimated cost ${estimate:.2f}{suffix} exceeds cap ${cap:.2f}{drivers_note}"),
            blinded=blinded,
        )
        _persist_partial_handle(handle)
        return handle

    # Committed to a real run now. If a cap was set but the estimate isn't
    # fully priced, the `estimate > cap` gate above could not enforce it (the
    # estimate covers only known-priced models), so the run may exceed the cap
    # silently. Warn rather than block — refusing every unpriced panel would be
    # too aggressive for openrouter-routed models, most of which are unpriced.
    if max_run_usd is not None and math.isfinite(max_run_usd) and not all_known:
        logger.warning(
            "max_run_usd=$%.2f set but at least one panellist has unknown pricing; "
            "the $%.2f estimate covers only known-priced models, so the cap cannot be "
            "fully enforced for this run.",
            cap,
            estimate,
        )

    # Build slugs + prompts. (`panel_slugs`, not `slugs` — that would
    # shadow the `slugs` module imported at the top of this file.)
    panel_slugs = _make_slugs(specs, blinded)
    # Duplicate slugs ⇒ multiple panellists racing to write to the same
    # `responses/<slug>.txt`; the second writer silently overwrites the
    # first. `expand_specs` fixes the model:N case but a caller passing
    # two literal `{slug: "foo"}` specs falls through. Fail fast here
    # before the artifact dance starts.
    if len(set(panel_slugs)) != len(panel_slugs):
        seen: dict[str, int] = {}
        for s in panel_slugs:
            seen[s] = seen.get(s, 0) + 1
        dupes = sorted(slug for slug, n in seen.items() if n > 1)
        raise ValueError(
            f"duplicate panel slugs would race the artifact dir: {dupes}. "
            "Disambiguate by giving each spec a distinct `slug` (or omitting it)."
        )
    per_prompts = [_build_per_slug_prompt(prompt, registry.resolve_stance(s.stance)) for s in specs]

    # Process-level provider semaphores (shared across concurrent fanouts
    # in the same event loop). See `_get_provider_sems` for the rationale.
    provider_sems = _get_provider_sems()

    # Total in-flight cap across the whole fanout. None = uncapped (rely on
    # per-provider sems alone). Resolved at call time so a test that sets
    # the env var per case works.
    if max_concurrency is None:
        env_cap = os.environ.get("CONSULT_MAX_CONCURRENCY", "").strip()
        if env_cap:
            try:
                parsed = int(env_cap)
                if parsed >= 1:
                    max_concurrency = parsed
            except ValueError:
                logger.warning(
                    "CONSULT_MAX_CONCURRENCY=%r is not a positive int; ignoring",
                    env_cap,
                )
    fanout_sem: asyncio.Semaphore | None = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    start = time.time()
    total = len(specs)
    # Live counters shared by `_run_one`, the heartbeat, and the gatherer.
    # Single event loop, so atomicity between awaits is enough; no lock.
    state = _PanelProgress(total=total, pending_slugs=set(panel_slugs))

    async def _safe_notify(event: ProgressEvent) -> None:
        """Wrapper that swallows callback exceptions. Progress is best-effort:
        a notification failure (closed session, slow client, raising user
        callback) must never tear down the real work.
        """
        if on_progress is None:
            return
        try:
            await on_progress(event)
        except Exception as e:  # noqa: BLE001
            logger.debug("on_progress callback failed (%s): %s", event.kind, e)

    # PhaseStarted("fanout"): emitted before any panellist begins, so the
    # parent sees fanout starting rather than receiving silence until the
    # first panellist completes.
    phase_event = PhaseStarted(done=0, total=total, phase="fanout")
    append_progress_log(paths.root, phase_event)
    await _safe_notify(phase_event)

    # Heartbeat task: periodic liveness pulse. Set CONSULT_HEARTBEAT_INTERVAL_S=0
    # to disable (used in tests that mock _call_one to instant returns).
    hb_interval = env_float("CONSULT_HEARTBEAT_INTERVAL_S", 5.0)
    heartbeat_task: asyncio.Task[None] | None = None
    if hb_interval > 0:

        async def _heartbeat_loop() -> None:
            while True:
                await asyncio.sleep(hb_interval)
                elapsed_ms = int((time.time() - start) * 1000)
                cost_so_far = sum((e.cost_usd or 0.0) for e in state.completed_entries)
                cost_known = all(e.cost_known for e in state.completed_entries)
                pending = sorted(state.pending_slugs)
                event = Heartbeat(
                    done=state.done,
                    total=total,
                    elapsed_ms=elapsed_ms,
                    cost_so_far_usd=cost_so_far,
                    cost_known=cost_known,
                    pending_count=len(pending),
                    pending_slugs=pending,
                )
                append_progress_log(paths.root, event)
                await _safe_notify(event)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

    # slug → monotonic timestamp of the last REAL sign of life (stream chunk,
    # retry or continuation call starting). Read by the slow-tail dropout to
    # tell working stragglers from hung ones.
    task_activity: dict[str, float] = {}

    async def _run_one(spec: ModelSpec, slug: str, per_prompt: str) -> ManifestEntry:
        # Acquire the global fanout slot BEFORE emitting PanellistStarted so
        # the started_count reflects "doing real work", not "queued". The
        # per-provider semaphores inside `_call_one` are still acquired below
        # — this gate is additive, never replacing them. `nullcontext()` is
        # async-compatible (Python 3.10+), same shape as Semaphore.
        async with fanout_sem if fanout_sem is not None else nullcontext():
            # PanellistStarted: fires BEFORE the LiteLLM call so the parent
            # sees which slugs are in flight, not just which have completed.
            state.started += 1
            started_event = PanellistStarted(
                done=state.done,
                total=total,
                slug=slug,
                started_count=state.started,
            )
            append_progress_log(paths.root, started_event)
            await _safe_notify(started_event)

            # When streaming is enabled, wire each panellist's mid-stream chunk
            # callback to emit `PanellistPartial` events. Throttled to
            # ~1 chunk/sec by `_STREAM_PARTIAL_INTERVAL_S` so the progress
            # channel doesn't drown in micro-updates. The callback ALWAYS
            # records slow-tail activity (task_activity feeds the
            # progress-aware dropout); the progress event is emitted only
            # when a listener is attached.
            on_partial: Callable[[int, int], Awaitable[None]] | None = None
            if stream:

                async def _emit_partial(chars: int, elapsed_ms: int) -> None:
                    task_activity[slug] = time.monotonic()
                    if on_progress is None:
                        return
                    await _safe_notify(
                        PanellistPartial(
                            done=state.done,
                            total=total,
                            slug=slug,
                            chars_so_far=chars,
                            elapsed_ms=elapsed_ms,
                        )
                    )

                on_partial = _emit_partial

            def _record_activity() -> None:
                # Real signs of life only (stream chunks, retry/continuation
                # calls starting) — never task start, or every straggler
                # would earn a free extension window.
                task_activity[slug] = time.monotonic()

            # Per-slug history takes precedence when set (refine round-2+
            # passes each panellist its own conversation). Falls back to
            # the global `prior_turns` (set by `_apply_continuation` for
            # cross-run continuations) when the slug isn't in the dict.
            pt = prior_turns_by_slug.get(slug) if prior_turns_by_slug else None
            if pt is None:
                pt = prior_turns
            # Facade lookup so monkeypatch.setattr(runner, "_call_one", ...)
            # still intercepts the call.
            entry = await _facade._call_one(
                spec,
                slug,
                per_prompt,
                paths,
                provider_sems,
                stream=stream,
                on_partial=on_partial,
                on_activity=_record_activity,
                prior_turns=pt,
                capsule_kind=capsule_kind,
                max_output_tokens=max_output_tokens,
                web_search=web_search,
                timeout_floor_s=timeout_floor_s,
            )
            state.record_completed(entry)
            await _safe_notify(
                PanellistCompleted(
                    done=state.done,
                    total=total,
                    slug=slug,
                    status=entry.status.value,
                    latency_ms=entry.latency_ms or 0,
                )
            )
            return entry

    # Slow-tail dropout knobs (see `_gather_with_tail_dropout`). Default 300s:
    # long-context reviews produce useful capsules from slower models (kimi,
    # qwen, deepseek often take 60-180s on ~200K input), so the prior 30s was
    # dropping real signal on wide panels. 180s still cancelled kimi at 292s
    # on the 2026-07-13 review run while it was inside its own 360s per-spec
    # budget and likely a minute from done; 300s of grace keeps the dropout
    # as a hang guard rather than a working-straggler killer (per-spec
    # timeouts still bound the true worst case).
    # Caller override wins over the env default; 0 disables the dropout
    # (`_gather_with_tail_dropout` treats non-positive grace as plain gather).
    if tail_dropout_s is None:
        tail_dropout_s = env_float("CONSULT_TAIL_DROPOUT_S", 300.0)
    tail_k_frac = env_float("CONSULT_TAIL_K_FRAC", 0.2)
    # Progress-aware extension: a straggler whose stream produced chunks (or
    # whose truncation continuation started) within this window is WORKING,
    # not hung, and survives past the grace; per-spec timeouts still bound
    # everything. 0 disables. Only real signals count, so non-streaming runs
    # without continuations behave exactly as before.
    tail_activity_window_s = env_float("CONSULT_TAIL_ACTIVITY_WINDOW_S", 120.0)

    try:
        manifest = await _gather_with_tail_dropout(
            run_one=_run_one,
            specs=specs,
            panel_slugs=panel_slugs,
            per_prompts=per_prompts,
            state=state,
            paths=paths,
            start=start,
            safe_notify=_safe_notify,
            task_activity=task_activity,
            activity_window_s=tail_activity_window_s,
            tail_dropout_s=tail_dropout_s,
            tail_k_frac=tail_k_frac,
        )
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

    wall_ms = int((time.time() - start) * 1000)

    # `blinded=True` controls what panellists see of each other DURING the
    # run (the brand-scrub in `context.build` and the greek-letter slugs in
    # `_make_slug`); the synth's `anonymised` switch additionally selects
    # the brand-scrubbed prompt for the synthesiser's view of the original
    # question. Panellist labels are blinded to the synth unconditionally
    # via blind labels (Alpha/Beta/...). The manifest itself keeps real
    # model_ids so the final report (viewer, ledger) can surface them to
    # the human reader. Earlier code scrubbed manifest model_ids here,
    # which leaked the blinding past its useful boundary.

    cost_total = sum((m.cost_usd or 0.0) for m in manifest)
    all_known = all(m.cost_known for m in manifest)
    # Zero-usable-panel guard. When every panellist times out, errors, or is
    # rate-limited, callers (refine arbiter, sequence) must not proceed —
    # there is no signal to evaluate and the downstream spend (arbiter,
    # synthesis) would burn for nothing. partial=True here lets the caller
    # short-circuit; a non-empty manifest of failure entries is still
    # returned for diagnosis.
    usable_count = sum(1 for m in manifest if m.status in (Status.OK, Status.TRUNCATED))
    if manifest and usable_count == 0:
        statuses = sorted({m.status.value for m in manifest})
        partial = True
        partial_reason: str | None = (
            f"zero usable panellists ({len(manifest)} returned: {', '.join(statuses)})"
        )
    else:
        partial = False
        partial_reason = None

    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=manifest,
        cost_usd=cost_total,
        cost_known=all_known,
        wall_ms=wall_ms,
        partial=partial,
        partial_reason=partial_reason,
        blinded=blinded,
    )
    artifacts.write_manifest(paths, handle.model_dump())
    return handle
