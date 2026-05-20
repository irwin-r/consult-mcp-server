"""Typed progress events.

A Pydantic discriminated union representing every progress signal the
server can emit during a tool call. Used in three places:

1. The `ProgressCallback` (runner.py) — instead of `(done, total, str_msg)`,
   handlers now receive a typed `ProgressEvent` they can introspect via
   `isinstance` or the `kind` discriminator.
2. The JSONL fallback log at `<run>/_progress.log` — each line is one
   `event.model_dump_json()` so programmatic consumers can tail it
   without parsing free-text strings.
3. The MCP `notifications/progress` adapter (server.py) — derives the
   wire-format `(progress, total, message)` from the event via
   `event_message()`, keeping the wire shape backwards-compatible.

`done` and `total` ride on the base class so any consumer reading the
event has access to monotonic progress numbers regardless of kind.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _BaseProgressEvent(BaseModel):
    """Common shape: every event carries the monotonic `(done, total)` pair."""

    # Forbid extras so a typo in an event-emitter kwarg fails loudly rather
    # than silently dropping the value (see types._STRICT for the same
    # rationale; this base class propagates the policy to every event kind).
    model_config = ConfigDict(extra="forbid")

    done: int = Field(..., ge=0)
    total: int = Field(..., ge=0)


class PanellistStarted(_BaseProgressEvent):
    """A panellist call has been initiated.

    Emitted before `_call_one` enters the LiteLLM call so the parent
    agent can see which models are currently in flight, not just which
    have completed. `done` carries the completion count at the moment
    the start fires (it does NOT advance on a start — that would conflict
    with `PanellistCompleted`'s monotonic semantics). `started_count`
    tracks how many panellists have begun work so far.
    """

    kind: Literal["panellist_started"] = "panellist_started"
    slug: str
    started_count: int = Field(..., ge=0)


class PanellistCompleted(_BaseProgressEvent):
    """One panellist in a fanout finished (OK, error, rate-limited, etc.)."""

    kind: Literal["panellist_completed"] = "panellist_completed"
    slug: str
    status: str  # Status enum value as string (decoupled to avoid import cycle)
    latency_ms: int = Field(..., ge=0)


class PanellistPartial(_BaseProgressEvent):
    """Mid-stream chunk indicator from a panellist.

    Emitted only when `fanout(stream=True)` is requested. Lets MCP clients
    show liveness during long panel runs without waiting for the full
    response. Carries cumulative chars (not the chunk content itself —
    sending raw chunks would balloon the progress channel).
    """

    kind: Literal["panellist_partial"] = "panellist_partial"
    slug: str
    chars_so_far: int = Field(..., ge=0)
    elapsed_ms: int = Field(..., ge=0)


class CapsuleExtracted(_BaseProgressEvent):
    """The cheap extractor produced (or failed to produce) a Capsule for one slug."""

    kind: Literal["capsule_extracted"] = "capsule_extracted"
    slug: str


class ArbiterScored(_BaseProgressEvent):
    """The refine arbiter has scored a round's sufficiency."""

    kind: Literal["arbiter_scored"] = "arbiter_scored"
    round: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)


class SynthStarted(_BaseProgressEvent):
    """The synthesiser pass is starting."""

    kind: Literal["synth_started"] = "synth_started"


class SynthCompleted(_BaseProgressEvent):
    """The synthesiser pass has completed."""

    kind: Literal["synth_completed"] = "synth_completed"


class SequenceStepStarted(_BaseProgressEvent):
    """A new step in a `sequence` invocation is starting."""

    kind: Literal["sequence_step_started"] = "sequence_step_started"
    step: int = Field(..., ge=1)


class SequenceStepCompleted(_BaseProgressEvent):
    """A step in a `sequence` invocation has finished (synth done)."""

    kind: Literal["sequence_step_completed"] = "sequence_step_completed"
    step: int = Field(..., ge=1)


class PhaseStarted(_BaseProgressEvent):
    """A new phase in a multi-phase tool is starting.

    Phases are "fanout" → "capsules" → "synth". `SynthStarted` /
    `SynthCompleted` predate this event and remain in the union for
    backwards compat; `PhaseStarted(phase="synth")` may be emitted
    alongside `SynthStarted` for clients that prefer the uniform shape.
    """

    kind: Literal["phase_started"] = "phase_started"
    phase: Literal["fanout", "capsules", "synth"]


class Heartbeat(_BaseProgressEvent):
    """Periodic liveness pulse during a long fanout.

    Fires every ~`CONSULT_HEARTBEAT_INTERVAL_S` seconds (default 5)
    while panellists are in flight. The client gets a "still working"
    signal between completion events, plus a snapshot of elapsed,
    cost-so-far, and the slugs still pending. Especially useful when
    the slowest panellist gates the rest and no completions fire for
    tens of seconds.

    `cost_so_far_usd` is the sum of `cost_usd` from panellists that
    have already returned; `cost_known` is false if any of those had a
    pricing-table miss (treat the displayed total as a lower bound).
    """

    kind: Literal["heartbeat"] = "heartbeat"
    elapsed_ms: int = Field(..., ge=0)
    cost_so_far_usd: float = Field(..., ge=0.0)
    cost_known: bool = True
    pending_count: int = Field(..., ge=0)
    pending_slugs: list[str] = Field(default_factory=list)


ProgressEvent = Annotated[
    PanellistStarted | PanellistCompleted | PanellistPartial | CapsuleExtracted | ArbiterScored | SynthStarted | SynthCompleted | SequenceStepStarted | SequenceStepCompleted | PhaseStarted | Heartbeat,
    Field(discriminator="kind"),
]


def event_message(event: ProgressEvent) -> str:
    """Human-readable one-liner for the MCP `notifications/progress.message`
    field. Kept stable for backwards compat with clients that just render
    the string.
    """
    if isinstance(event, PanellistStarted):
        return f"{event.slug}: in flight ({event.started_count}/{event.total} started)"
    if isinstance(event, PanellistCompleted):
        return f"{event.slug}: {event.status}"
    if isinstance(event, PanellistPartial):
        return f"{event.slug}: streaming ({event.chars_so_far} chars / {event.elapsed_ms}ms)"
    if isinstance(event, CapsuleExtracted):
        return f"capsule {event.slug}"
    if isinstance(event, ArbiterScored):
        return f"r{event.round} arbiter score={event.score:.2f}"
    if isinstance(event, SynthStarted):
        return "synthesising"
    if isinstance(event, SynthCompleted):
        return "synthesis complete"
    if isinstance(event, SequenceStepStarted):
        return f"step {event.step} starting"
    if isinstance(event, SequenceStepCompleted):
        return f"step {event.step} complete"
    if isinstance(event, PhaseStarted):
        return f"phase: {event.phase}"
    if isinstance(event, Heartbeat):
        # ≥ prefix marks the running total as a lower bound when any
        # completed panellist had a pricing-table miss — same convention
        # the ledger uses.
        cost_prefix = "" if event.cost_known else "≥"
        elapsed_s = event.elapsed_ms / 1000.0
        if event.pending_slugs:
            shown = event.pending_slugs[:3]
            more = f" +{len(event.pending_slugs) - 3}" if len(event.pending_slugs) > 3 else ""
            pending = ", ".join(shown) + more
        else:
            pending = "—"
        return (
            f"working {elapsed_s:.0f}s, "
            f"{cost_prefix}${event.cost_so_far_usd:.4f}, "
            f"{event.pending_count} pending: {pending}"
        )
    # Unreachable — Pydantic validation on the union prevents other kinds.
    return ""
