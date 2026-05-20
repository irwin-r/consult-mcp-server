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
    Resource,
    TextContent,
    Tool,
)

from .. import artifacts
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


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="panel",
            description=(
                "Fan a prompt out to multiple models in parallel. Returns a manifest "
                "with structured capsules (~200 tokens each) and resource URIs for full "
                "bodies. Use when the parent agent wants to synthesise itself."
            ),
            inputSchema=schemas.PANEL_SCHEMA,
        ),
        Tool(
            name="synthesise",
            description=(
                "Synthesise an existing run via a flagship model. Reads the run's "
                "manifest + bodies and returns markdown under a consensus rubric."
            ),
            inputSchema=schemas.SYNTH_SCHEMA,
        ),
        Tool(
            name="consult",
            description=(
                "Hero tool: parallel panel + server-side synthesis. Returns synthesis "
                "+ manifest. Use for 'just give me the answer' workflows."
            ),
            inputSchema=schemas.consult_schema(),
        ),
        Tool(
            name="refine",
            description=(
                "Consortium-style iterative consultation. Fans out, asks an arbiter "
                "to score sufficiency, refines with another round if below threshold. "
                "Hard cap at 3 rounds. Per-round transcripts available as MCP resources."
            ),
            inputSchema=schemas.REFINE_SCHEMA,
        ),
        Tool(
            name="sequence",
            description=(
                "Run an ordered list of prompts where each step's synthesis is "
                "prepended as context for the next step. Use for multi-stage "
                "research (e.g. break-down → per-subquestion → meta-synth) or "
                "any plan-then-execute workflow. Returns per-step run_ids + "
                "the final synthesis."
            ),
            inputSchema=schemas.SEQUENCE_SCHEMA,
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


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> dict[str, Any] | list[TextContent]:
    # All failures funnel into the structured `ErrorEnvelope` shape so the
    # agent never has to parse free-text. Map known exception types to stable
    # error codes; anything unhandled becomes INTERNAL_ERROR (and we log the
    # traceback so the maintainer can find the bug).
    handler = _HANDLERS.get(name)
    if handler is None:
        return errors.envelope(errors.ErrorCode.INVALID_INPUT, f"Unknown tool: {name}")
    on_progress = _build_progress_callback()
    try:
        result = await handler(arguments, on_progress=on_progress)
    except ValueError as e:
        # Caller-side problems: out-of-range params, bad continuation_id,
        # empty prompt list, missing required fields, etc. Raised
        # synchronously by the handler / library code before any model call.
        return errors.envelope(errors.ErrorCode.INVALID_INPUT, str(e))
    except KeyError as e:
        # `registry.resolve_model` raises KeyError on unknown alias — relevant
        # for `synthesise.by_model` and explicit `arbiter`/`synthesiser`
        # overrides that don't go through `_call_one`'s per-spec ERROR path.
        return errors.envelope(errors.ErrorCode.UNKNOWN_MODEL, str(e))
    except FileNotFoundError as e:
        # `artifacts.load_run` raises this when a run_id doesn't exist on
        # disk. Relevant for `synthesise(run_id=...)` and any `continuation_id`
        # that bypasses `_apply_continuation`'s wrapping.
        return errors.envelope(errors.ErrorCode.RUN_NOT_FOUND, str(e))
    except Exception as e:  # noqa: BLE001 — last-resort envelope
        logger.exception("unhandled exception in tool %s", name)
        return errors.envelope(
            errors.ErrorCode.INTERNAL_ERROR, f"{type(e).__name__}: {e}"
        )
    # Wire-shape adaptation for synthesise: a markdown blob is more usefully
    # delivered as `TextContent` so clients render it directly rather than
    # forcing them to unwrap a dict.
    if name in _TEXT_RESULT_TOOLS and isinstance(result, str):
        return [TextContent(type="text", text=result)]
    return result


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
    run_id, slug = artifacts.parse_resource_uri(str(uri))
    paths = artifacts.load_run(run_id)
    body_file = paths.response_text(slug)
    if not body_file.exists():
        raise FileNotFoundError(f"Body not found: {uri}")
    return body_file.read_text()


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
                server_version="0.1.0",
                # Derive capabilities from the registered @list_tools /
                # @list_resources / @read_resource handlers so the initialize
                # response actually advertises tools+resources to the client.
                # An empty ServerCapabilities() tells spec-compliant clients
                # the server has neither, which suppresses tools/list polls.
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )
