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


class PanellistCompleted(_BaseProgressEvent):
    """One panellist in a fanout finished (OK, error, rate-limited, etc.)."""

    kind: Literal["panellist_completed"] = "panellist_completed"
    slug: str
    status: str  # Status enum value as string (decoupled to avoid import cycle)
    latency_ms: int = Field(..., ge=0)


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


ProgressEvent = Annotated[
    PanellistCompleted | CapsuleExtracted | ArbiterScored | SynthStarted | SynthCompleted | SequenceStepStarted | SequenceStepCompleted,
    Field(discriminator="kind"),
]


def event_message(event: ProgressEvent) -> str:
    """Human-readable one-liner for the MCP `notifications/progress.message`
    field. Kept stable for backwards compat with clients that just render
    the string.
    """
    if isinstance(event, PanellistCompleted):
        return f"{event.slug}: {event.status}"
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
    # Unreachable — Pydantic validation on the union prevents other kinds.
    return ""
