"""MCP server entry point.

Registers four tools — `panel`, `synthesise`, `consult`, `refine` — and a
resource handler for `consult://runs/<id>/responses/<slug>` URIs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import dotenv
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    AnyUrl,
    Resource,
    ServerCapabilities,
    TextContent,
    Tool,
)

from . import (
    artifacts,
    capsule,
    refine as refine_mod,
    registry,
    runner,
    sequence as sequence_mod,
    synth,
)
from .types import ModelSpec, RunResult

logger = logging.getLogger("consult")

# Load .env from the working directory, the package directory, and the user's home
for p in (Path.cwd() / ".env", Path(__file__).parent.parent / ".env", Path.home() / ".consult" / ".env"):
    if p.exists():
        dotenv.load_dotenv(p, override=False)

server: Server = Server("consult")


# ---- Tool schemas -----------------------------------------------------------

_PANEL_SCHEMA = {
    "type": "object",
    "required": ["prompt", "models"],
    "properties": {
        "prompt": {"type": "string", "description": "The question / task for the panel."},
        "models": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["model"],
                "properties": {
                    "model": {"type": "string", "description": "Registry alias or LiteLLM ID"},
                    "stance": {
                        "type": "string",
                        "description": "Stance key (e.g. 'security') or a literal stance prompt.",
                    },
                    "slug": {"type": "string", "description": "Optional explicit slug override."},
                },
            },
            "description": "Panellists. Each entry: {model, stance?, slug?}.",
        },
        "blinded": {
            "type": "boolean",
            "default": False,
            "description": "Anonymise slugs to panelist-alpha/beta/... and strip model_id from manifest.",
        },
        "attachments": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Absolute file paths to inline into the prompt.",
        },
        "dry_run": {
            "type": "boolean",
            "default": False,
            "description": "Estimate cost without calling any model.",
        },
        "max_run_usd": {
            "type": "number",
            "description": "Per-run cost cap. Defaults to CONSULT_MAX_RUN_USD.",
        },
        "extract_capsules": {
            "type": "boolean",
            "default": True,
            "description": "Run the capsule extractor after fanout. Disable for raw output.",
        },
    },
}

_SYNTH_SCHEMA = {
    "type": "object",
    "required": ["run_id"],
    "properties": {
        "run_id": {"type": "string", "description": "A run_id returned by `panel` or `consult`."},
        "by_model": {
            "type": "string",
            "description": "Synthesiser model (alias or LiteLLM ID). Defaults to the configured default synthesiser (see models.json → defaults.synthesiser).",
        },
        "rubric": {"type": "string", "description": "Custom rubric. Defaults to the consensus rubric."},
        "anonymised": {
            "type": "boolean",
            "default": False,
            "description": "Hide real model IDs from the synthesiser input.",
        },
    },
}

_REFINE_SCHEMA = {
    "type": "object",
    "required": ["prompt", "models"],
    "properties": {
        "prompt": {"type": "string"},
        "models": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["model"],
                "properties": {
                    "model": {"type": "string"},
                    "stance": {"type": "string"},
                    "slug": {"type": "string"},
                },
            },
        },
        "arbiter": {
            "type": "string",
            "description": "Arbiter model alias. Defaults to the default synthesiser.",
        },
        "threshold": {
            "type": "number",
            "default": 0.85,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Sufficiency score (0..1) above which the loop stops early.",
        },
        "max_rounds": {
            "type": "integer",
            "default": 3,
            "minimum": 1,
            "maximum": 3,
            "description": "Hard cap on rounds. Cannot exceed 3.",
        },
        "blinded": {"type": "boolean", "default": False},
        "attachments": {"type": "array", "items": {"type": "string"}},
        "max_run_usd": {"type": "number"},
        "synthesiser": {
            "type": "string",
            "description": "Final synthesis model. Defaults to the arbiter.",
        },
        "continuation_id": {
            "type": "string",
            "description": (
                "Optional run_id of a prior refine to continue. The earlier "
                "synthesis.md is prepended to this prompt as 'Prior consultation "
                "summary' before the new round runs. Unknown IDs raise an error."
            ),
        },
    },
}

_SEQUENCE_SCHEMA = {
    "type": "object",
    "required": ["prompts", "models"],
    "properties": {
        "prompts": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                "Ordered list of prompts. Each step's synthesis is prepended "
                "to the next step's prompt as 'prior synthesis' context."
            ),
        },
        "models": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["model"],
                "properties": {
                    "model": {"type": "string"},
                    "stance": {"type": "string"},
                    "slug": {"type": "string"},
                },
            },
        },
        "synthesiser": {"type": "string", "description": "Per-step synth model."},
        "blinded": {"type": "boolean", "default": False},
        "attachments": {"type": "array", "items": {"type": "string"}},
        "max_run_usd": {
            "type": "number",
            "description": "Cap across the whole sequence (cumulative, not per-step).",
        },
    },
}


_CONSULT_SCHEMA = {
    "type": "object",
    "required": ["prompt"],
    "properties": {
        "prompt": {"type": "string"},
        "tier": {
            "type": "string",
            "enum": ["quick", "standard", "deep"],
            "default": "standard",
        },
        "roles": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Map model alias → stance key. Defaults to neutral.",
        },
        "attachments": {"type": "array", "items": {"type": "string"}},
        "synthesiser": {"type": "string", "description": "Override synth model."},
        "blinded": {"type": "boolean", "default": False},
        "max_run_usd": {"type": "number"},
    },
}


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
            inputSchema=_PANEL_SCHEMA,
        ),
        Tool(
            name="synthesise",
            description=(
                "Synthesise an existing run via a flagship model. Reads the run's "
                "manifest + bodies and returns markdown under a consensus rubric."
            ),
            inputSchema=_SYNTH_SCHEMA,
        ),
        Tool(
            name="consult",
            description=(
                "Hero tool: parallel panel + server-side synthesis. Returns synthesis "
                "+ manifest. Use for 'just give me the answer' workflows."
            ),
            inputSchema=_CONSULT_SCHEMA,
        ),
        Tool(
            name="refine",
            description=(
                "Consortium-style iterative consultation. Fans out, asks an arbiter "
                "to score sufficiency, refines with another round if below threshold. "
                "Hard cap at 3 rounds. Per-round transcripts available as MCP resources."
            ),
            inputSchema=_REFINE_SCHEMA,
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
            inputSchema=_SEQUENCE_SCHEMA,
        ),
    ]


# ---- Helpers ----------------------------------------------------------------


def _inline_attachments(prompt: str, attachments: list[str] | None) -> str:
    if not attachments:
        return prompt
    parts = [prompt, "\n\n--- ATTACHMENTS ---\n"]
    for path in attachments:
        try:
            content = Path(path).read_text()
            parts.append(f"\n# {path}\n```\n{content}\n```\n")
        except OSError as e:
            parts.append(f"\n# {path}\n[ERROR: {e}]\n")
    return "".join(parts)


def _specs_from_args(models_arg: list[dict[str, Any]]) -> list[ModelSpec]:
    return [ModelSpec(**m) for m in models_arg]


def _text_result(payload: dict | str) -> list[TextContent]:
    if isinstance(payload, str):
        return [TextContent(type="text", text=payload)]
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


def _progress_callback():
    """Build a callback that forwards `(done, total, msg)` tuples as MCP
    `notifications/progress`. Returns None if the client didn't send a
    `progressToken` (so nothing is sent at all — silent for non-subscribers).

    The token is opaque to us; we echo whatever the client supplied. Any
    notification failure (e.g. closed session) is caught upstream by the
    per-callsite try/except so it never aborts the underlying tool call.
    """
    try:
        ctx = server.request_context
    except LookupError:
        return None
    token = ctx.meta.progressToken if ctx.meta else None
    if token is None:
        return None
    session = ctx.session

    async def notify(done: int, total: int, message: str) -> None:
        await session.send_progress_notification(
            progress_token=token,
            progress=float(done),
            total=float(total),
            message=message,
        )

    return notify


# ---- Tool dispatch ----------------------------------------------------------


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "panel":
        return await _handle_panel(arguments)
    if name == "synthesise":
        return await _handle_synth(arguments)
    if name == "consult":
        return await _handle_consult(arguments)
    if name == "refine":
        return await _handle_refine(arguments)
    if name == "sequence":
        return await _handle_sequence(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def _handle_panel(args: dict[str, Any]) -> list[TextContent]:
    prompt = _inline_attachments(args["prompt"], args.get("attachments"))
    specs = _specs_from_args(args["models"])
    progress = _progress_callback()
    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        dry_run=args.get("dry_run", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=progress,
    )
    if args.get("extract_capsules", True) and not handle.partial and handle.manifest:
        handle = await capsule.annotate(handle, on_progress=progress)
    return _text_result(handle.model_dump())


async def _handle_synth(args: dict[str, Any]) -> list[TextContent]:
    text = await synth.synthesise(
        args["run_id"],
        by_model=args.get("by_model"),
        rubric=args.get("rubric"),
        anonymised=args.get("anonymised", False),
    )
    return _text_result(text)


async def _handle_consult(args: dict[str, Any]) -> list[TextContent]:
    prompt = _inline_attachments(args["prompt"], args.get("attachments"))
    tier = args.get("tier", "standard")
    tier_models = registry.resolve_tier(tier)
    roles = args.get("roles") or {}
    synth_alias = args.get("synthesiser") or registry.default_synthesiser()

    # Exclude the synthesiser from the panel to avoid self-inclusion bias
    panel_aliases = [m for m in tier_models if m != synth_alias]
    specs = [ModelSpec(model=m, stance=roles.get(m)) for m in panel_aliases]

    # consult has three phases (fanout → capsules → synth). MCP progress
    # is monotonic, so wrap each phase callback with an offset into a
    # single growing total.
    base = _progress_callback()
    overall_total = len(specs) * 2 + 1  # fanout + capsules + synth
    offset = 0

    def phase(label: str):
        if base is None:
            return None

        async def cb(done: int, _local_total: int, msg: str) -> None:
            await base(offset + done, overall_total, f"{label}: {msg}")

        return cb

    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=phase("fanout"),
    )
    if handle.partial or not handle.manifest:
        return _text_result(
            {"partial": True, "reason": handle.partial_reason, "manifest": []}
        )
    offset = len(specs)
    handle = await capsule.annotate(handle, on_progress=phase("capsules"))
    offset = len(specs) * 2
    if base is not None:
        await base(offset, overall_total, "synthesising")
    synthesis = await synth.synthesise(
        handle.run_id, by_model=synth_alias, anonymised=args.get("blinded", False)
    )
    if base is not None:
        await base(overall_total, overall_total, "synthesis complete")
    result = RunResult(
        run_id=handle.run_id,
        synthesis=synthesis,
        manifest=handle.manifest,
        cost_usd=handle.cost_usd,
        wall_ms=handle.wall_ms,
        partial=False,
        synthesiser=synth_alias,
    )
    return _text_result(result.model_dump())


async def _handle_sequence(args: dict[str, Any]) -> list[TextContent]:
    # Attachments — if supplied — are inlined into every step's prompt, since
    # a sequence is one logical consultation with shared context. Per-step
    # attachment overrides are a v2 feature.
    raw_prompts = args["prompts"]
    attachments = args.get("attachments")
    prompts = [_inline_attachments(p, attachments) for p in raw_prompts]
    specs = _specs_from_args(args["models"])
    result = await sequence_mod.sequence(
        prompts,
        specs,
        synthesiser=args.get("synthesiser"),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=_progress_callback(),
    )
    return _text_result(result.model_dump())


async def _handle_refine(args: dict[str, Any]) -> list[TextContent]:
    prompt = _inline_attachments(args["prompt"], args.get("attachments"))
    specs = _specs_from_args(args["models"])
    result = await refine_mod.refine(
        prompt,
        specs,
        arbiter=args.get("arbiter"),
        threshold=args.get("threshold", 0.85),
        max_rounds=args.get("max_rounds", 3),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        synthesiser=args.get("synthesiser"),
        continuation_id=args.get("continuation_id"),
        on_progress=_progress_callback(),
    )
    return _text_result(result.model_dump())


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
                capabilities=ServerCapabilities(),
            ),
        )
