"""MCP server entry point.

Registers four tools — `panel`, `synthesise`, `consult`, `refine` — and a
resource handler for `consult://runs/<id>/responses/<slug>` URIs.
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
    capsule,
    errors,
    progress,
    registry,
    runner,
    sources,
    synth,
)
from . import (
    refine as refine_mod,
)
from . import (
    sequence as sequence_mod,
)
from .types import ModelSpec, RunResult

logger = logging.getLogger("consult")

# Load .env from the working directory, the package directory, and the user's home
for p in (Path.cwd() / ".env", Path(__file__).parent.parent / ".env", Path.home() / ".consult" / ".env"):
    if p.exists():
        dotenv.load_dotenv(p, override=False)

server: Server = Server("consult")


# ---- Tool schemas -----------------------------------------------------------

# Used by every tool's `attachments` array. Bare strings stay supported for
# backwards compat; labelled objects let callers tag a file with a section
# heading ("DESIGN_DOC", "AUTH_MODULE"); the `git_diff` source form makes
# the server resolve the diff itself (parent never holds the diff in its
# own context).
_ATTACHMENT_SCHEMA_ITEMS = {
    "anyOf": [
        {"type": "string", "description": "Absolute file path."},
        {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "Absolute file path."},
                "label": {
                    "type": "string",
                    "description": "Human-readable label rendered as a section heading.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["text", "diff", "design_doc", "source", "data"],
                    "description": "Hint for fence language and presentation.",
                },
            },
        },
        {
            "type": "object",
            "required": ["source", "base", "head"],
            "properties": {
                "source": {"type": "string", "enum": ["git_diff"]},
                "base": {"type": "string", "description": "Base ref (e.g. 'main')."},
                "head": {"type": "string", "description": "Head ref (e.g. 'HEAD')."},
                "repo_path": {
                    "type": "string",
                    "description": (
                        "Repo dir. Must resolve under CONSULT_TRUSTED_REPO_ROOTS "
                        "(defaults to cwd)."
                    ),
                },
                "label": {"type": "string"},
            },
        },
    ],
}

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
            "items": _ATTACHMENT_SCHEMA_ITEMS,
            "description": (
                "Each entry is either an absolute file path (string), a labelled "
                "file `{path, label?, kind?}`, or a server-resolved source "
                "`{source: \"git_diff\", base, head, repo_path?, label?}`."
            ),
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
        "capsule_kind": {
            "type": "string",
            "enum": ["decision", "review", "research"],
            "default": "decision",
            "description": (
                "Shape of the extracted capsule. 'decision' = position/recommendation "
                "(general-purpose). 'review' = line-anchored Finding[] for code/PR "
                "review. 'research' = claims/evidence/uncertainties for research."
            ),
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
            "maximum": 5,
            "description": "Hard cap on rounds. 1-5; default 3.",
        },
        "blinded": {"type": "boolean", "default": False},
        "attachments": {"type": "array", "items": {"type": "string"}},
        "max_run_usd": {"type": "number"},
        "synthesiser": {
            "type": "string",
            "description": "Final synthesis model. Defaults to the arbiter.",
        },
        "rubric": {
            "type": "string",
            "description": (
                "Rubric name (e.g. 'consensus', 'code_review', 'research_brief', "
                "'critique') or a literal rubric string. Defaults to 'consensus'."
            ),
        },
        "capsule_kind": {
            "type": "string",
            "enum": ["decision", "review", "research"],
            "default": "decision",
            "description": (
                "Shape of the extracted capsule. 'decision' (default), 'review' for "
                "code reviews, 'research' for evidence-gathering panels."
            ),
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
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "required": ["prompt"],
                        "properties": {
                            "prompt": {"type": "string"},
                            "attachments": {
                                "type": "array",
                                "items": _ATTACHMENT_SCHEMA_ITEMS,
                                "description": (
                                    "Per-step attachments. When set, override the "
                                    "top-level `attachments` for this step. Lets step "
                                    "N attach files step N-1 didn't need (closes the "
                                    "'step N reasons over English summary of step N-1's "
                                    "code' trap)."
                                ),
                            },
                        },
                    },
                ],
            },
            "description": (
                "Ordered list of prompts. Each entry is either a string (uses "
                "the top-level `attachments`) or an object `{prompt, attachments?}` "
                "with per-step attachments. Each step's synthesis is prepended to "
                "the next step's prompt as 'prior synthesis' context."
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


_TIER_NAMES = list(registry.models_config().get("tiers", {}).keys())

_CONSULT_SCHEMA = {
    "type": "object",
    "required": ["prompt"],
    "properties": {
        "prompt": {"type": "string"},
        "tier": {
            "type": "string",
            "enum": _TIER_NAMES or ["quick", "standard", "deep"],
            "default": "standard" if "standard" in _TIER_NAMES else (_TIER_NAMES[0] if _TIER_NAMES else "standard"),
        },
        "roles": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Map model alias → stance key. Defaults to neutral.",
        },
        "attachments": {"type": "array", "items": {"type": "string"}},
        "synthesiser": {"type": "string", "description": "Override synth model."},
        "rubric": {
            "type": "string",
            "description": (
                "Rubric name (e.g. 'consensus', 'code_review', 'research_brief', "
                "'critique') or a literal rubric string. Defaults to 'consensus'."
            ),
        },
        "capsule_kind": {
            "type": "string",
            "enum": ["decision", "review", "research"],
            "default": "decision",
            "description": (
                "Shape of the extracted capsule. 'decision' (default), 'review' for "
                "code reviews, 'research' for evidence-gathering panels."
            ),
        },
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


_KIND_FENCE_LANG = {
    "diff": "diff",
    "design_doc": "markdown",
    "source": "",
    "data": "",
    "text": "",
    "git_diff": "diff",
}


def _render_attachment(item: Any) -> str:
    """Render a single attachment spec into a markdown block.

    Three input shapes are accepted (see `_attachments_schema_items`):
    - bare string → absolute file path
    - `{path, label?, kind?}` → labelled file attachment
    - `{source: "git_diff", base, head, repo_path?, label?}` → server-side
      resolved git diff
    """
    # Bare string → file path
    if isinstance(item, str):
        path = item
        label = None
        kind = None
        try:
            content = Path(path).read_text()
        except OSError as e:
            return f"\n# {path}\n[ERROR: {e}]\n"
    elif isinstance(item, dict) and item.get("source") == "git_diff":
        # Source resolver
        base = item.get("base")
        head = item.get("head")
        repo_path = item.get("repo_path")
        label = item.get("label") or f"git_diff[{base}..{head}]"
        kind = "git_diff"
        path = f"git_diff:{base}..{head}"
        try:
            content = sources.resolve_git_diff(base, head, repo_path)
        except (ValueError, RuntimeError) as e:
            return f"\n## {label}\n[ERROR: {e}]\n"
    elif isinstance(item, dict) and item.get("path"):
        path = item["path"]
        label = item.get("label")
        kind = item.get("kind")
        try:
            content = Path(path).read_text()
        except OSError as e:
            return f"\n# {path}\n[ERROR: {e}]\n"
    else:
        return f"\n[ERROR: malformed attachment spec: {item!r}]\n"

    fence_lang = _KIND_FENCE_LANG.get(kind or "", "")
    header = f"## {label}: {path}" if label else f"# {path}"
    return f"\n{header}\n```{fence_lang}\n{content}\n```\n"


def _inline_attachments(prompt: str, attachments: list | None) -> str:
    if not attachments:
        return prompt
    parts = [prompt, "\n\n--- ATTACHMENTS ---\n"]
    for item in attachments:
        parts.append(_render_attachment(item))
    return "".join(parts)


def _specs_from_args(models_arg: list[dict[str, Any]]) -> list[ModelSpec]:
    return [ModelSpec(**m) for m in models_arg]


def _error_result(
    code: errors.ErrorCode,
    message: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Wrap a failure in the standard `{"ok": false, "error": {...}}` envelope.

    Returned by the top-level `handle_call_tool` try/except so every failure
    mode an agent sees has the same shape — they can branch on `error.code`
    instead of regex-matching free-text.

    Returns a `dict` so the MCP SDK populates `structuredContent` alongside
    the JSON text fallback — clients can branch on `error.code` directly
    without parsing the text body.
    """
    envelope = errors.ErrorEnvelope(
        error=errors.ConsultError(code=code, message=message, run_id=run_id),
    )
    return envelope.model_dump()


def _progress_callback():
    """Build a callback that converts a typed `ProgressEvent` into an MCP
    `notifications/progress` send. Returns None if the client didn't send a
    `progressToken` (so nothing is sent at all — silent for non-subscribers).

    The wire-format message string is derived from the event via
    `progress.event_message()`; the event's `done` and `total` populate the
    `progress` / `total` fields. The token is opaque to us; we echo what the
    client supplied.

    Any notification failure (e.g. closed session) is caught upstream by the
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

    async def notify(event: progress.ProgressEvent) -> None:
        await session.send_progress_notification(
            progress_token=token,
            progress=float(event.done),
            total=float(event.total),
            message=progress.event_message(event),
        )

    return notify


# ---- Tool dispatch ----------------------------------------------------------


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> dict[str, Any] | list[TextContent]:
    # All failures are funnelled into the structured `ErrorEnvelope` shape so
    # the agent never has to parse free-text. Map known exception types to
    # stable error codes; anything unhandled becomes INTERNAL_ERROR (and we
    # log the traceback so the maintainer can find the bug).
    try:
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
        return _error_result(errors.ErrorCode.INVALID_INPUT, f"Unknown tool: {name}")
    except ValueError as e:
        # Caller-side problems: out-of-range params, bad continuation_id,
        # empty prompt list, missing required fields, etc. All raised
        # synchronously by the handler / library code before any model call.
        return _error_result(errors.ErrorCode.INVALID_INPUT, str(e))
    except KeyError as e:
        # `registry.resolve_model` raises KeyError on unknown alias — relevant
        # for `synthesise.by_model` and explicit `arbiter`/`synthesiser`
        # overrides that don't go through `_call_one`'s per-spec ERROR path.
        return _error_result(errors.ErrorCode.UNKNOWN_MODEL, str(e))
    except FileNotFoundError as e:
        # `artifacts.load_run` raises this when a run_id doesn't exist on
        # disk. Relevant for `synthesise(run_id=...)` and any
        # `continuation_id` that bypasses `_apply_continuation`'s wrapping.
        return _error_result(errors.ErrorCode.RUN_NOT_FOUND, str(e))
    except Exception as e:  # noqa: BLE001 — last-resort envelope
        logger.exception("unhandled exception in tool %s", name)
        return _error_result(
            errors.ErrorCode.INTERNAL_ERROR,
            f"{type(e).__name__}: {e}",
        )


async def _handle_panel(args: dict[str, Any]) -> dict[str, Any]:
    prompt = _inline_attachments(args["prompt"], args.get("attachments"))
    specs = _specs_from_args(args["models"])
    progress_cb = _progress_callback()
    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        dry_run=args.get("dry_run", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=progress_cb,
    )
    if args.get("extract_capsules", True) and not handle.partial and handle.manifest:
        handle = await capsule.annotate(
            handle, on_progress=progress_cb, kind=args.get("capsule_kind", "decision")
        )
    return handle.model_dump()


async def _handle_synth(args: dict[str, Any]) -> list[TextContent]:
    # Synth's output is a markdown blob — `TextContent` is the natural shape
    # since a structured-content dict would force clients to unwrap the text
    # before rendering. (Every other tool returns a dict so MCP also surfaces
    # `structuredContent` for programmatic callers.)
    text = await synth.synthesise(
        args["run_id"],
        by_model=args.get("by_model"),
        rubric=args.get("rubric"),
        anonymised=args.get("anonymised", False),
    )
    return [TextContent(type="text", text=text)]


async def _handle_consult(args: dict[str, Any]) -> dict[str, Any]:
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
    # single growing total. Events keep their identity (PanellistCompleted,
    # CapsuleExtracted) — only `done`/`total` get shifted into the outer
    # consult-wide bucket.
    base = _progress_callback()
    overall_total = len(specs) * 2 + 1  # fanout + capsules + synth
    offset = 0

    def phase():
        if base is None:
            return None

        async def cb(event: progress.ProgressEvent) -> None:
            shifted = event.model_copy(
                update={"done": offset + event.done, "total": overall_total},
            )
            await base(shifted)

        return cb

    handle = await runner.fanout(
        prompt,
        specs,
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=phase(),
    )
    if handle.partial or not handle.manifest:
        # Return a real `RunResult` so the partial response has the same shape
        # as the success path — clients can rely on a single dict schema and
        # branch on `partial` / `partial_reason` rather than two layouts.
        partial = RunResult(
            run_id=handle.run_id,
            synthesis="",
            manifest=handle.manifest,
            cost_usd=handle.cost_usd,
            cost_known=handle.cost_known,
            wall_ms=handle.wall_ms,
            partial=True,
            partial_reason=(
                handle.partial_reason or "no panellists returned usable responses"
            ),
            synthesiser=synth_alias,
        )
        return partial.model_dump()
    offset = len(specs)
    handle = await capsule.annotate(
        handle, on_progress=phase(), kind=args.get("capsule_kind", "decision")
    )
    offset = len(specs) * 2
    if base is not None:
        await base(progress.SynthStarted(done=offset, total=overall_total))
    synthesis = await synth.synthesise(
        handle.run_id,
        by_model=synth_alias,
        anonymised=args.get("blinded", False),
        rubric=args.get("rubric"),
    )
    if base is not None:
        await base(progress.SynthCompleted(done=overall_total, total=overall_total))
    # Persist the synthesiser choice on disk so `consult-view` can badge it
    # in the header. `RunResult` carries it on the wire, but the manifest
    # written by `runner.fanout` was assembled before synth ran.
    artifacts.augment_manifest(artifacts.load_run(handle.run_id), synthesiser=synth_alias)
    result = RunResult(
        run_id=handle.run_id,
        synthesis=synthesis,
        manifest=handle.manifest,
        cost_usd=handle.cost_usd,
        wall_ms=handle.wall_ms,
        partial=False,
        synthesiser=synth_alias,
    )
    return result.model_dump()


async def _handle_sequence(args: dict[str, Any]) -> dict[str, Any]:
    # Each step gets its own inlined-attachments prompt. The top-level
    # `attachments` is the default for every step; a step that's an
    # object can supply its own `attachments` to override (per-step
    # source material — closes the "naïve sequence drops step N's code"
    # trap from the v2 audit).
    raw_prompts = args["prompts"]
    default_attachments = args.get("attachments")
    prompts: list[str] = []
    for item in raw_prompts:
        if isinstance(item, str):
            prompts.append(_inline_attachments(item, default_attachments))
        else:
            step_atts = item.get("attachments")
            effective_atts = step_atts if step_atts is not None else default_attachments
            prompts.append(_inline_attachments(item["prompt"], effective_atts))
    specs = _specs_from_args(args["models"])
    result = await sequence_mod.sequence(
        prompts,
        specs,
        synthesiser=args.get("synthesiser"),
        blinded=args.get("blinded", False),
        max_run_usd=args.get("max_run_usd"),
        on_progress=_progress_callback(),
    )
    return result.model_dump()


async def _handle_refine(args: dict[str, Any]) -> dict[str, Any]:
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
        rubric=args.get("rubric"),
        capsule_kind=args.get("capsule_kind", "decision"),
        on_progress=_progress_callback(),
    )
    return result.model_dump()


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
