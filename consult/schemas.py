"""JSON Schemas for every MCP tool exposed by the server.

Schemas were originally inlined in server.py (the file grew past 800 lines
doing four jobs at once: MCP wiring, attachment rendering, schema
definitions, and per-tool orchestration). Splitting them out:

- keeps server.py focused on tool registration and dispatch;
- makes schemas testable in isolation (the agent-discovered
  `dropout_marks_cost_unknown` bug had a similar shape — extra fields
  silently dropped because nothing was validating against the schema);
- gives operators one place to look when they want to know what every
  tool's contract is.

`ATTACHMENT_SCHEMA_ITEMS` lives in attachments.py — it's both a schema
fragment and a renderer concern, and putting it next to `render_attachment`
keeps the two from drifting.

`consult_schema()` is a function (not a constant) because its `tier` enum
is derived from the live registry — adding a tier to models.json must take
effect on the next call without a server restart.
"""

from __future__ import annotations

from typing import Any

from . import registry
from .attachments import ATTACHMENT_SCHEMA_ITEMS

# --- Shared sub-schemas -------------------------------------------------------

_ATTACHMENTS_FIELD_DESC = (
    "Each entry is either an absolute file path (string), a labelled "
    'file `{path, label?, kind?}`, or a server-resolved source '
    '`{source: "git_diff", base, head, repo_path?, label?}`.'
)

_MODEL_SPEC_ITEM = {
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
}

_CAPSULE_KIND_FIELD = {
    "type": "string",
    "enum": ["decision", "review", "research"],
    "default": "decision",
    "description": (
        "Shape of the extracted capsule. 'decision' (default), 'review' for "
        "code reviews, 'research' for evidence-gathering panels."
    ),
}

_RUBRIC_FIELD = {
    "type": "string",
    "description": (
        "Rubric name (e.g. 'consensus', 'code_review', 'research_brief', "
        "'critique') or a literal rubric string. Defaults to 'consensus'."
    ),
}


# --- Tool schemas -------------------------------------------------------------

PANEL_SCHEMA = {
    "type": "object",
    "required": ["prompt", "models"],
    "properties": {
        "prompt": {"type": "string", "description": "The question / task for the panel."},
        "models": {
            "type": "array",
            "minItems": 1,
            "items": _MODEL_SPEC_ITEM,
            "description": "Panellists. Each entry: {model, stance?, slug?}.",
        },
        "blinded": {
            "type": "boolean",
            "default": False,
            "description": (
                "Anonymise slugs to panelist-alpha/beta/... and strip model_id "
                "from manifest."
            ),
        },
        "attachments": {
            "type": "array",
            "items": ATTACHMENT_SCHEMA_ITEMS,
            "description": _ATTACHMENTS_FIELD_DESC,
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
            **_CAPSULE_KIND_FIELD,
            "description": (
                "Shape of the extracted capsule. 'decision' = position/recommendation "
                "(general-purpose). 'review' = line-anchored Finding[] for code/PR "
                "review. 'research' = claims/evidence/uncertainties for research."
            ),
        },
    },
}


SYNTH_SCHEMA = {
    "type": "object",
    "required": ["run_id"],
    "properties": {
        "run_id": {
            "type": "string",
            "description": "A run_id returned by `panel` or `consult`.",
        },
        "by_model": {
            "type": "string",
            "description": (
                "Synthesiser model (alias or LiteLLM ID). Defaults to the "
                "configured default synthesiser (see models.json → "
                "defaults.synthesiser)."
            ),
        },
        "rubric": {
            "type": "string",
            "description": "Custom rubric. Defaults to the consensus rubric.",
        },
        "anonymised": {
            "type": "boolean",
            "default": False,
            "description": "Hide real model IDs from the synthesiser input.",
        },
    },
}


REFINE_SCHEMA = {
    "type": "object",
    "required": ["prompt", "models"],
    "properties": {
        "prompt": {"type": "string"},
        "models": {
            "type": "array",
            "minItems": 1,
            "items": _MODEL_SPEC_ITEM,
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
        "attachments": {
            "type": "array",
            "items": ATTACHMENT_SCHEMA_ITEMS,
            "description": _ATTACHMENTS_FIELD_DESC,
        },
        "max_run_usd": {"type": "number"},
        "synthesiser": {
            "type": "string",
            "description": "Final synthesis model. Defaults to the arbiter.",
        },
        "rubric": _RUBRIC_FIELD,
        "capsule_kind": _CAPSULE_KIND_FIELD,
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


SEQUENCE_SCHEMA = {
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
                                "items": ATTACHMENT_SCHEMA_ITEMS,
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
            "items": _MODEL_SPEC_ITEM,
        },
        "synthesiser": {"type": "string", "description": "Per-step synth model."},
        "blinded": {"type": "boolean", "default": False},
        "attachments": {
            "type": "array",
            "items": ATTACHMENT_SCHEMA_ITEMS,
            "description": (
                "Default attachments for every step. Each entry is a string path, "
                'a labelled file `{path, label?, kind?}`, or a git_diff source '
                '`{source: "git_diff", base, head, repo_path?, label?}`. '
                "Steps with object form may override these per-step."
            ),
        },
        "max_run_usd": {
            "type": "number",
            "description": "Cap across the whole sequence (cumulative, not per-step).",
        },
        "rubric": {
            **_RUBRIC_FIELD,
            "description": (
                "Rubric name (e.g. 'consensus', 'code_review', 'research_brief', "
                "'critique') or a literal rubric string. Applies to every step's "
                "synthesis. Defaults to 'consensus'."
            ),
        },
        "capsule_kind": {
            **_CAPSULE_KIND_FIELD,
            "description": (
                "Capsule shape for every step. 'decision' (default), 'review', "
                "or 'research'."
            ),
        },
    },
}


def _tier_names() -> list[str]:
    """Read tier names at call time so test monkeypatching of the registry
    config (and any future live-reload of models.json) is honoured. The
    registry already caches the config via `lru_cache`, so this is cheap.
    """
    return list(registry.models_config().get("tiers", {}).keys())


def consult_schema() -> dict[str, Any]:
    """Build the `consult` schema at call time.

    The `tier` enum is derived from the live registry so adding a tier to
    models.json doesn't require a restart to expose it through MCP. Kept
    as a function (rather than a constant) for that reason.
    """
    tier_names = _tier_names()
    default_tier = (
        "standard"
        if "standard" in tier_names
        else (tier_names[0] if tier_names else "standard")
    )
    return {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "tier": {
                "type": "string",
                "enum": tier_names or ["quick", "standard", "deep"],
                "default": default_tier,
            },
            "roles": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Map model alias → stance key. Defaults to neutral.",
            },
            "attachments": {
                "type": "array",
                "items": ATTACHMENT_SCHEMA_ITEMS,
                "description": _ATTACHMENTS_FIELD_DESC,
            },
            "synthesiser": {"type": "string", "description": "Override synth model."},
            "rubric": _RUBRIC_FIELD,
            "capsule_kind": _CAPSULE_KIND_FIELD,
            "extract_capsules": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Run the capsule extractor after fanout. The synthesiser does NOT "
                    "use capsules (it reads raw bodies), so set False when you only "
                    "need the synthesis and want to skip the extractor's latency + "
                    "cost. The returned manifest will have capsule=null per entry."
                ),
            },
            "blinded": {"type": "boolean", "default": False},
            "max_run_usd": {"type": "number"},
        },
    }
