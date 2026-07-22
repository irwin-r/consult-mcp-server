"""MCP server entry point.

Wires the MCP protocol surface (tool listing, dispatch, resource access,
stdio main loop) to the per-tool handlers in `consult/mcp/handlers.py` and
the schemas in `consult/mcp/schemas.py`. Kept deliberately small —
orchestration lives in the engine (`consult.orchestrate`, `consult.runner`,
`consult.refine`, `consult.sequence`, `consult.synth`) so this file doesn't
drift when engine flows change.

Owns the MCP-specific glue that used to leak into handlers:
- builds the `progressToken`-aware progress callback per request and passes
  it into the handler (removes the lazy `from .server import server` cycle
  the handler used to do)
- wraps `synthesise`'s markdown output in `TextContent` (every other tool
  returns a dict for structured-content compatibility)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from datetime import datetime

    from mcp.types import TaskStatus

import dotenv
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import (
    INVALID_PARAMS,
    CancelTaskRequest,
    CancelTaskResult,
    CreateTaskResult,
    ErrorData,
    GetTaskPayloadRequest,
    GetTaskPayloadResult,
    GetTaskRequest,
    GetTaskResult,
    Resource,
    ServerResult,
    Task,
    TextContent,
    Tool,
    ToolAnnotations,
)
from pydantic import AnyUrl

from .. import __version__, artifacts, task_store
from ..envutil import env_bool
from ..progress import ProgressEvent, event_message
from ..redact import install_redaction_filter, redact_exc, redact_traceback, scrub_exception_attrs
from . import errors, handlers, schemas

logger = logging.getLogger("consult")

# Load .env from the working directory, the package directory, and the user's home.
# `Path(__file__).parents[2]` resolves to the repo root (consult/mcp/server.py →
# consult/mcp → consult → repo root) which is where the dev-time .env sits.
for p in (Path.cwd() / ".env", Path(__file__).parents[2] / ".env", Path.home() / ".consult" / ".env"):
    if p.exists():
        dotenv.load_dotenv(p, override=False)

server: Server = Server("consult")


# ---- Tool listing -----------------------------------------------------------


# Tool annotations are *hints* to clients (Claude Code, Cursor, ChatGPT
# dev-mode) for auto-approval / confirmation UX. They are NOT security
# boundaries — the spec is explicit that an untrusted server's annotations
# cannot be trusted. They reflect the *consult* engine's behaviour:
# - every tool here calls external LLMs (openWorldHint=True) — costs money,
#   the parent agent should usually surface a confirmation
# - none of them destroy anything on the local filesystem outside the
#   `~/.consult/runs/<id>/` artifact dir (destructiveHint=False)
# - `synthesise(run_id)` is idempotent: same inputs replay the same on-disk
#   artifacts (LLM stochasticity aside; the manifest contract is stable).
# - the four panel/consult/refine/sequence tools each create a fresh run_id
#   on every call, so idempotentHint=False.
_PANEL_ANN = ToolAnnotations(
    title="Multi-model panel",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
_SYNTH_ANN = ToolAnnotations(
    title="Synthesise existing run",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)
_CONSULT_ANN = ToolAnnotations(
    title="Consult multi-model panel",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
_REFINE_ANN = ToolAnnotations(
    title="Iterative refine loop",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
_SEQUENCE_ANN = ToolAnnotations(
    title="Chained multi-step consultation",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
_RESEARCH_ANN = ToolAnnotations(
    title="Director-driven research loop",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    # Tool descriptions are prompts read by the calling agent, not docs
    # for humans. Verb-first, explicit "Use when…" / "Don't use for…", and
    # disambiguation against the other four tools so the agent reliably
    # picks the right one. The shape "<one-line action>. Use when: <X>.
    # Don't use for: <Y>. Returns <Z>." is consistent across all five.
    tools = [
        Tool(
            name="consult",
            description=(
                "Get a synthesised second opinion from a panel of LLMs in parallel. "
                "Use when: you want one consolidated answer to a single question and "
                "don't want to manage the panel yourself. "
                "Don't use for: code/PR review with line-anchored findings (use `consult` "
                'with `capsule_kind="review"` and `rubric="code_review"`), iterative '
                "back-and-forth (use `refine`), or chained multi-step research (use `sequence`). "
                "Returns: {run_id, synthesis (markdown), manifest, cost_usd, synthesiser}."
            ),
            inputSchema=schemas.consult_schema(),
            annotations=_CONSULT_ANN,
        ),
        Tool(
            name="panel",
            description=(
                "Run a parallel panel and return the raw manifest WITHOUT server-side "
                "synthesis. Use when: you (the calling agent) want to do the synthesis "
                "yourself, e.g. to interleave panel evidence with your own reasoning. "
                "Don't use for: 'just give me the answer' — use `consult` instead, which "
                "is faster and cheaper to drive. "
                "Returns: {run_id, manifest with ~200-token capsules + resource URIs for "
                "full bodies, cost_usd}."
            ),
            inputSchema=schemas.PANEL_SCHEMA,
            annotations=_PANEL_ANN,
        ),
        Tool(
            name="refine",
            description=(
                "Consortium-style iterative consultation: fan out, arbiter scores "
                "sufficiency, run another round if below threshold (hard cap 3 rounds). "
                "Use when: a single panel pass isn't enough — high-stakes decisions, "
                "disagreement-heavy questions, or when the prior panel left obvious gaps. "
                "Don't use for: cheap one-shot answers (use `consult`), or when you "
                "already have a satisfactory panel and just want a fresh synthesis "
                "(use `synthesise`). Pass `continuation_id` to thread a follow-up onto "
                "a prior refine run. "
                "Returns: {run_id, rounds_completed, verdicts per round, final_manifest, "
                "synthesis, converged}."
            ),
            inputSchema=schemas.REFINE_SCHEMA,
            annotations=_REFINE_ANN,
        ),
        Tool(
            name="synthesise",
            description=(
                "Re-synthesise an existing run via a flagship model under a (possibly "
                "different) rubric. "
                "Use when: you already ran `panel`/`consult`/`refine` and want a fresh "
                "synthesis — a different rubric, a different synthesiser model, or "
                "`anonymised=true` to brand-scrub the question shown to the synthesiser "
                "(panellist identities are always blinded to the synth regardless). "
                "Don't use for: fresh questions (use `consult`) or when you don't have a "
                "prior `run_id` to feed in. "
                "Returns: markdown text (not a structured dict)."
            ),
            inputSchema=schemas.SYNTH_SCHEMA,
            annotations=_SYNTH_ANN,
        ),
    ]
    # `sequence` is demoted off the default tool surface (issue #59): the panel
    # review judged the always-on multi-step chain low-value relative to its
    # schema cost. The engine and library API stay, and the handler stays
    # registered (so an explicit call still works); it just isn't advertised
    # unless CONSULT_ENABLE_SEQUENCE is set, so it can be revisited with usage
    # data without re-implementing anything.
    # `research` ships env-gated (issue #92, same rollout as sequence):
    # dogfood it behind CONSULT_ENABLE_RESEARCH, promote with usage data.
    if env_bool("CONSULT_ENABLE_RESEARCH"):
        tools.append(
            Tool(
                name="research",
                description=(
                    "Run a director-driven research loop on an open-ended goal: a "
                    "director model freezes a brief (explicit assumptions, deliverable "
                    "sections, per-section acceptance bars), then repeatedly plans work "
                    "items, executes each as a panel/consult sub-run, assembles a "
                    "dossier, and judges it against the brief until accepted, stalled, "
                    "over budget, or out of rounds. "
                    "Use when: the goal is too big for one panel — strategy dossiers, "
                    "multi-part research deliverables ('plan an e-commerce brand'). "
                    "The director can plan `evidence` work items that gather cited live "
                    "web facts via provider-native search (no agentic browsing). "
                    "Don't use for: a single question (use `consult`) or iterating on "
                    "one contested question (use `refine`). "
                    "Long-running BY DESIGN: sub-runs are patient — slow-tail dropout "
                    "is off and per-model timeouts are floored at two hours, so deep "
                    "models are never dropped for being slow and a run can take hours. "
                    "Use task mode (`params.task`) and poll tasks/get. If the server "
                    "restarts mid-run, tasks/get forgets the task — re-invoke with "
                    "`continuation_id` set to the run_id to resume the crashed run "
                    "from its journal. Costs real money per round; "
                    "`max_run_usd` defaults to $25 and explicit null opts into UNCAPPED "
                    "spend (stall detection stays on either way). "
                    "Returns: {run_id, summary, brief, verdicts, open_gaps, stop_reason, "
                    "converged, cost_usd, dossier_uri (full dossier as an MCP resource)}."
                ),
                inputSchema=schemas.RESEARCH_SCHEMA,
                annotations=_RESEARCH_ANN,
            )
        )
    if env_bool("CONSULT_ENABLE_SEQUENCE"):
        tools.append(
            Tool(
                name="sequence",
                description=(
                    "Run an ordered list of prompts where every prior step's synthesis is "
                    "prepended as context for the next step. "
                    "Use when: a question is too large for a single prompt (decompose → "
                    "per-subquestion → meta-synth), or for plan-then-execute workflows where "
                    "step N depends on step N-1's conclusion. "
                    "Don't use for: parallel diverse opinions on the same question (use "
                    "`consult`/`refine`) — sequence is for chained reasoning, not breadth. "
                    "Returns: {step_run_ids, final synthesis, total cost_usd}."
                ),
                inputSchema=schemas.SEQUENCE_SCHEMA,
                annotations=_SEQUENCE_ANN,
            )
        )
    return tools


# ---- Tool dispatch ----------------------------------------------------------


_HANDLERS = {
    "panel": handlers.panel,
    "synthesise": handlers.synthesise,
    "consult": handlers.consult,
    "refine": handlers.refine,
    "sequence": handlers.sequence,
    # Dispatch-map entry is all research needs for SEP-1686 task mode —
    # the background-task plumbing in handle_call_tool is tool-agnostic.
    "research": handlers.research,
}

# Tools whose result is a markdown blob rather than a structured dict. The
# dispatcher wraps these in `TextContent` so clients render them directly;
# every other tool returns a dict so MCP also surfaces `structuredContent`.
_TEXT_RESULT_TOOLS = {"synthesise"}


def _build_progress_callback() -> Callable[[ProgressEvent], Awaitable[None]] | None:
    """Build a `progressToken`-aware MCP progress callback for the current request.

    Returns None if the client didn't send a `progressToken` — silent for
    non-subscribers. The wire-format message string is derived from the
    event via `progress.event_message()`; the event's `(done, total)`
    populate the wire `progress` / `total` fields. The token is opaque to
    us; we echo what the client supplied.

    Previously the handlers reached back into `server.request_context` via
    a lazy import (`from .server import server`) — an upside-down dependency
    that made the engine handlers MCP-aware. Building the callback here and
    passing it as a kwarg keeps the handlers pure orchestration.
    """
    try:
        ctx = server.request_context
    except LookupError:
        return None
    token = ctx.meta.progressToken if ctx.meta else None
    if token is None:
        return None
    session = ctx.session

    async def notify(event: ProgressEvent) -> None:
        await session.send_progress_notification(
            progress_token=token,
            progress=float(event.done),
            total=float(event.total),
            message=event_message(event),
        )

    return notify


async def _run_handler_with_envelopes(
    name: str,
    arguments: dict[str, Any],
    on_progress: Callable | None,
) -> dict[str, Any] | list[TextContent]:
    """Execute a tool handler, mapping exceptions into ErrorEnvelopes.

    Extracted from `handle_call_tool` so the same engine flow runs in
    foreground (synchronous response) and Task mode (background asyncio
    Task storing into `task_store`). Both paths return the same wire
    shape so a client polling `tasks/get` sees identical data to a
    client receiving the synchronous response.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return errors.envelope(errors.ErrorCode.INVALID_INPUT, f"Unknown tool: {name}")
    try:
        result = await handler(arguments, on_progress=on_progress)
    except ValueError as e:
        return errors.envelope(errors.ErrorCode.INVALID_INPUT, str(e))
    except KeyError as e:
        # str() on a KeyError is the quoted repr of its message
        # ("'Unknown model: x'"); unwrap args[0] so the envelope carries
        # the message itself.
        message = str(e.args[0]) if e.args else str(e)
        return errors.envelope(errors.ErrorCode.UNKNOWN_MODEL, message)
    except FileNotFoundError as e:
        return errors.envelope(errors.ErrorCode.RUN_NOT_FOUND, str(e))
    except Exception as e:  # noqa: BLE001
        # Redact: a provider exception can carry the auth header, and both the
        # log traceback and the returned envelope reach outside the process.
        scrub_exception_attrs(e)
        logger.error("unhandled exception in tool %s\n%s", name, redact_traceback(e))
        return errors.envelope(errors.ErrorCode.INTERNAL_ERROR, redact_exc(e))
    if name in _TEXT_RESULT_TOOLS and isinstance(result, str):
        return [TextContent(type="text", text=result)]
    return result


def _is_task_request() -> int | None:
    """Return the request's `task.ttl` (or None if non-task mode).

    SEP-1686 task augmentation: when the client sends
    `params.task: {ttl: N}` on a tools/call request, the server returns
    a `CreateTaskResult` immediately and runs the work async. Returns
    the ttl integer when task mode is active (None ttl means "default"
    per the spec — we treat that as 0 = no expiry).

    The SDK surfaces task metadata via `ctx.experimental.task_metadata`
    (parsed from `params.task` by the lowlevel server). `ctx.request`
    only gets populated under SSE transport; reading it for stdio
    would always return None.
    """
    try:
        ctx = server.request_context
    except LookupError:
        return None
    experimental = getattr(ctx, "experimental", None)
    task_meta = getattr(experimental, "task_metadata", None) if experimental else None
    if task_meta is None:
        return None
    return getattr(task_meta, "ttl", None) or 0


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> dict[str, Any] | list[TextContent] | CreateTaskResult:
    """Dispatch a tool call. Two modes:

    - **Foreground** (`params.task` absent or null): runs the handler
      to completion, returns the result inline. Exceptions land on the
      structured `ErrorEnvelope` shape.

    - **Task** (`params.task: {ttl}` set): kicks off the handler as a
      background asyncio Task, returns `CreateTaskResult` immediately
      with a freshly-minted `taskId`. The client polls `tasks/get`
      until terminal. Per the SEP-1686 spec, `taskSupport: "optional"`
      on every consult tool — clients can use either mode.
    """
    on_progress = _build_progress_callback()
    ttl = _is_task_request()

    if ttl is None:
        # Foreground path — original behaviour.
        return await _run_handler_with_envelopes(name, arguments, on_progress)

    # Task path: register, spawn, return immediately.
    rec = task_store.create(ttl_ms=ttl or None)

    # Background tasks cannot use `server.request_context` (it's tied
    # to the originating request which is about to end). Pass a None
    # callback — clients polling `tasks/get` see status changes via the
    # status field; per-panellist progress is still tailable on disk
    # via `~/.consult/runs/<id>/_progress.log`.
    async def _bg() -> None:
        try:
            result = await _run_handler_with_envelopes(name, arguments, None)
            # If the handler returned a dict containing run_id, attach it
            # for forensic correlation.
            if isinstance(result, dict) and isinstance(result.get("run_id"), str):
                task_store.attach_run_id(rec.task_id, result["run_id"])
            task_store.complete(rec.task_id, result)
        except asyncio.CancelledError:
            # Already marked CANCELLED by task_store.cancel(); just exit.
            raise
        except Exception as e:  # noqa: BLE001
            # Same redaction as the foreground path: the traceback goes to the
            # log and the error string is surfaced to a polling client.
            logger.error("background task %s failed\n%s", rec.task_id, redact_traceback(e))
            task_store.fail(rec.task_id, redact_exc(e))

    bg = asyncio.create_task(_bg())
    task_store.attach_bg_task(rec.task_id, bg)

    return CreateTaskResult(
        task=Task(
            taskId=rec.task_id,
            # task_store keeps plain strings so the engine never imports
            # mcp.types; the values are the TaskStatus literals.
            status=cast("TaskStatus", rec.status),
            createdAt=_ts(rec.created_at),
            lastUpdatedAt=_ts(rec.last_updated_at),
            ttl=rec.ttl_ms,
            pollInterval=rec.poll_interval_ms,
        ),
    )


def _ts(epoch_seconds: float) -> datetime:
    """UTC datetime from an `epoch_seconds` float. The SDK's Task fields are
    datetimes; handing them a real datetime beats relying on string coercion."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch_seconds, tz=UTC)


def _require_task(task_id: str) -> task_store.TaskRecord:
    """Look up a task or raise the JSON-RPC invalid-params error.

    `GetTaskResult` / `CancelTaskResult` have no None-task sentinel — their
    taskId/status fields are required — and the spec doesn't reserve a
    dedicated error code for an unknown id. Clients should treat the error
    as "resubmit", per SEP-1686.
    """
    rec = task_store.get(task_id)
    if rec is None:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown taskId: {task_id}"))
    return rec


def _task_snapshot_kwargs(rec: task_store.TaskRecord) -> dict[str, Any]:
    """The Task wire fields shared by tasks/get and tasks/cancel responses."""
    return {
        "taskId": rec.task_id,
        "status": rec.status,  # already one of the TaskStatus literals
        "statusMessage": rec.status_message,
        "createdAt": _ts(rec.created_at),
        "lastUpdatedAt": _ts(rec.last_updated_at),
        "ttl": rec.ttl_ms,
        "pollInterval": rec.poll_interval_ms,
    }


async def _handle_get_task(req: GetTaskRequest) -> ServerResult:
    """Respond to a `tasks/get` poll with the current task snapshot."""
    rec = _require_task(req.params.taskId)
    return ServerResult(GetTaskResult(**_task_snapshot_kwargs(rec)))


def _tool_result_fields(result: Any) -> dict[str, Any]:
    """Convert a stored handler result into CallToolResult-shaped fields.

    Mirrors the SDK's own `call_tool` conversion: a dict becomes
    `structuredContent` plus a JSON-text content block; a list of content
    blocks (the `synthesise` tool's TextContent wrapping) passes through
    as `content`. Kept in lockstep so a client sees the identical payload
    whether the call ran foreground or as a task.
    """
    import json as _json

    if isinstance(result, dict):
        return {
            "content": [TextContent(type="text", text=_json.dumps(result, indent=2))],
            "structuredContent": result,
            "isError": False,
        }
    return {"content": list(result), "structuredContent": None, "isError": False}


async def _handle_get_task_payload(req: GetTaskPayloadRequest) -> ServerResult:
    """Respond to `tasks/result`: the completed task's tool result.

    Per SEP-1686 the payload matches the original request's result type —
    for a tools/call task, the CallToolResult shape. Non-terminal tasks
    error (poll `tasks/get` until terminal); cancelled tasks have no
    result to return; failed tasks return the same ErrorEnvelope a
    foreground call would have produced, so client error-handling code
    is identical for both modes.
    """
    rec = _require_task(req.params.taskId)
    if rec.status == task_store.STATUS_WORKING:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"Task {rec.task_id} is still working; poll tasks/get until terminal",
            )
        )
    if rec.status == task_store.STATUS_CANCELLED:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Task {rec.task_id} was cancelled; no result"))
    if rec.status == task_store.STATUS_FAILED:
        payload = errors.envelope(errors.ErrorCode.INTERNAL_ERROR, rec.error or "task failed")
        return ServerResult(GetTaskPayloadResult(**_tool_result_fields(payload)))
    return ServerResult(GetTaskPayloadResult(**_tool_result_fields(rec.result)))


async def _handle_cancel_task(req: CancelTaskRequest) -> ServerResult:
    """Respond to `tasks/cancel`: cancel the background work and return the
    updated snapshot. Cancelling an already-terminal task is an error per
    SEP-1686 (there is nothing left to cancel).
    """
    rec = _require_task(req.params.taskId)
    if rec.status in (
        task_store.STATUS_COMPLETED,
        task_store.STATUS_FAILED,
        task_store.STATUS_CANCELLED,
    ):
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"Task {rec.task_id} is already {rec.status}; nothing to cancel",
            )
        )
    task_store.cancel(rec.task_id)
    return ServerResult(CancelTaskResult(**_task_snapshot_kwargs(rec)))


# Wire the task handlers. The standard `Server` class doesn't expose
# decorators for these (they're not part of the "core" surface) so we
# register them directly on `request_handlers`.
server.request_handlers[GetTaskRequest] = _handle_get_task
server.request_handlers[GetTaskPayloadRequest] = _handle_get_task_payload
server.request_handlers[CancelTaskRequest] = _handle_cancel_task


# ---- Resources --------------------------------------------------------------


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List the most recent N runs as resource roots. The parent typically
    addresses specific responses by URI, but listing helps for discovery.
    """
    runs_dir = artifacts.runs_root()
    out: list[Resource] = []
    runs = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:20]
    for run in runs:
        responses = run / "responses"
        if not responses.exists():
            continue
        for resp in responses.glob("*.txt"):
            slug = resp.stem
            out.append(
                Resource(
                    uri=AnyUrl(f"consult://runs/{run.name}/responses/{slug}"),
                    name=f"{run.name}/{slug}",
                    mimeType="text/plain",
                    description=f"Panellist body from run {run.name}",
                )
            )
    return out


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    run_id, kind, name = artifacts.parse_resource_uri(str(uri))
    paths = artifacts.load_run(run_id)
    if kind == "responses":
        f = paths.response_text(name)
        if not f.exists():
            raise FileNotFoundError(f"Body not found: {uri}")
        return f.read_text()
    if kind == "attachments":
        f = paths.attachment_path(name)
        if not f.exists():
            raise FileNotFoundError(f"Attachment not found: {uri}")
        return f.read_text()
    if kind == "dossier":
        # A research run's assembled document. Exactly one file per run,
        # so the name segment is fixed rather than caller-chosen.
        f = paths.root / "dossier.md"
        if name != "dossier.md" or not f.exists():
            raise FileNotFoundError(f"Dossier not found: {uri}")
        return f.read_text()
    raise ValueError(f"Unsupported resource kind: {kind}")


# ---- Main loop --------------------------------------------------------------


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("CONSULT_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # Root-handler redaction: every record that propagates here — the whole
    # consult.* tree and any chatty dependency — gets key-shaped tokens
    # scrubbed before the formatter renders them (issue #40).
    install_redaction_filter("")
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="consult",
                server_version=__version__,
                # Derive capabilities from the registered @list_tools /
                # @list_resources / @read_resource handlers so the initialize
                # response actually advertises tools+resources to the client.
                # An empty ServerCapabilities() tells spec-compliant clients
                # the server has neither, which suppresses tools/list polls.
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )
