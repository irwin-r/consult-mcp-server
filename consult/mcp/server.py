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
from typing import Any

import dotenv
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    AnyUrl,
    CreateTaskResult,
    GetTaskRequest,
    GetTaskResult,
    Resource,
    ServerResult,
    Task,
    TextContent,
    Tool,
    ToolAnnotations,
)

from .. import __version__, artifacts, task_store
from ..progress import ProgressEvent, event_message
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


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    # Tool descriptions are prompts read by the calling agent, not docs
    # for humans. Verb-first, explicit "Use when…" / "Don't use for…", and
    # disambiguation against the other four tools so the agent reliably
    # picks the right one. The shape "<one-line action>. Use when: <X>.
    # Don't use for: <Y>. Returns <Z>." is consistent across all five.
    return [
        Tool(
            name="consult",
            description=(
                "Get a synthesised second opinion from a panel of LLMs in parallel. "
                "Use when: you want one consolidated answer to a single question and "
                "don't want to manage the panel yourself. "
                "Don't use for: code/PR review with line-anchored findings (use `consult` "
                "with `capsule_kind=\"review\"` and `rubric=\"code_review\"`), iterative "
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
            name="sequence",
            description=(
                "Run an ordered list of prompts where each step's synthesis is prepended "
                "as context for the next step. "
                "Use when: a question is too large for a single prompt (decompose → "
                "per-subquestion → meta-synth), or for plan-then-execute workflows where "
                "step N depends on step N-1's conclusion. "
                "Don't use for: parallel diverse opinions on the same question (use "
                "`consult`/`refine`) — sequence is for chained reasoning, not breadth. "
                "Returns: {step_run_ids, final synthesis, total cost_usd}."
            ),
            inputSchema=schemas.SEQUENCE_SCHEMA,
            annotations=_SEQUENCE_ANN,
        ),
        Tool(
            name="synthesise",
            description=(
                "Re-synthesise an existing run via a flagship model under a (possibly "
                "different) rubric. "
                "Use when: you already ran `panel`/`consult`/`refine` and want a fresh "
                "synthesis — different rubric, different synthesiser model, or to "
                "anonymise the model IDs. "
                "Don't use for: fresh questions (use `consult`) or when you don't have a "
                "prior `run_id` to feed in. "
                "Returns: markdown text (not a structured dict)."
            ),
            inputSchema=schemas.SYNTH_SCHEMA,
            annotations=_SYNTH_ANN,
        ),
    ]


# ---- Tool dispatch ----------------------------------------------------------


_HANDLERS = {
    "panel": handlers.panel,
    "synthesise": handlers.synthesise,
    "consult": handlers.consult,
    "refine": handlers.refine,
    "sequence": handlers.sequence,
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
    name: str, arguments: dict[str, Any], on_progress: Callable | None,
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
        return errors.envelope(errors.ErrorCode.UNKNOWN_MODEL, str(e))
    except FileNotFoundError as e:
        return errors.envelope(errors.ErrorCode.RUN_NOT_FOUND, str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("unhandled exception in tool %s", name)
        return errors.envelope(
            errors.ErrorCode.INTERNAL_ERROR, f"{type(e).__name__}: {e}"
        )
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
            logger.exception("background task %s failed", rec.task_id)
            task_store.fail(rec.task_id, f"{type(e).__name__}: {e}")

    bg = asyncio.create_task(_bg())
    task_store.attach_bg_task(rec.task_id, bg)

    return CreateTaskResult(
        task=Task(
            taskId=rec.task_id,
            status=rec.status,  # already one of the TaskStatus literals
            createdAt=_iso(rec.created_at),
            lastUpdatedAt=_iso(rec.last_updated_at),
            ttl=rec.ttl_ms,
            pollInterval=rec.poll_interval_ms,
        ),
    )


def _iso(epoch_seconds: float) -> str:
    """ISO-8601 timestamp from an `epoch_seconds` float, UTC."""
    from datetime import UTC, datetime
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat()


async def _handle_get_task(req: GetTaskRequest) -> ServerResult:
    """Respond to a `tasks/get` poll.

    Returns the current task snapshot for the requested taskId. Unknown
    taskIds resolve via the JSON-RPC error path (the SDK has no
    "task=None" sentinel — `GetTaskResult` is flat, with required
    taskId/status fields).
    """
    task_id = req.params.taskId
    rec = task_store.get(task_id)
    if rec is None:
        # `GetTaskResult` has no None-task sentinel — its taskId/status
        # fields are required. The JSON-RPC invalid-params code is the
        # closest standard match; the spec doesn't reserve a code for
        # this case. Clients should treat it as "resubmit", per SEP-1686.
        from mcp.shared.exceptions import McpError
        from mcp.types import INVALID_PARAMS, ErrorData
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message=f"Unknown taskId: {task_id}")
        )
    return ServerResult(
        GetTaskResult(
            taskId=rec.task_id,
            status=rec.status,  # already one of the TaskStatus literals
            statusMessage=rec.status_message,
            createdAt=_iso(rec.created_at),
            lastUpdatedAt=_iso(rec.last_updated_at),
            ttl=rec.ttl_ms,
            pollInterval=rec.poll_interval_ms,
        )
    )


# Wire the tasks/get handler. The standard `Server` class doesn't
# expose a decorator for this (it's not part of the "core" surface)
# so we register it directly on `request_handlers`.
server.request_handlers[GetTaskRequest] = _handle_get_task


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
    raise ValueError(f"Unsupported resource kind: {kind}")


# ---- Main loop --------------------------------------------------------------


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("CONSULT_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
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
