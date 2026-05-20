"""MCP server entry point.

Wires the MCP protocol surface (tool listing, dispatch, resource access,
stdio main loop) to the per-tool handlers in `consult/handlers.py` and the
schemas in `consult/schemas.py`. Kept deliberately small — orchestration
lives elsewhere so the wiring file doesn't drift when handlers change.
"""

from __future__ import annotations

import logging
import os
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

from . import (
    artifacts,
    errors,
    handlers,
    schemas,
)

logger = logging.getLogger("consult")

# Load .env from the working directory, the package directory, and the user's home.
for p in (Path.cwd() / ".env", Path(__file__).parent.parent / ".env", Path.home() / ".consult" / ".env"):
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
    try:
        return await handler(arguments)
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
