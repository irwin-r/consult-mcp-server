"""ContextBundle — the single immutable per-run context artefact.

Written once at run-init by `runner.fanout` / `refine.refine`. Downstream
stages (synth, capsule extractor, arbiter) load by `run_id` rather than
receiving the prompt as a string parameter — so all the fidelity-layer
thread-the-prompt-everywhere plumbing lives behind one type.

Legacy runs without `context.json` are still readable — `load_or_none()`
returns `None` and callers fall back to their pre-existing behaviour.

The bundle is also where `blinded`-mode scrubbing happens: the prompt is
masked once at run-init (so anonymisation cannot leak via downstream
stages receiving the raw prompt), and the masked variant is cached on
disk as `prompt_scrubbed`.
"""

from __future__ import annotations

import logging
import os
import re

from pydantic import BaseModel, ConfigDict, Field

from . import artifacts

logger = logging.getLogger(__name__)

_CONTEXT_SCHEMA_VERSION = 1
_CONTEXT_FILENAME = "context.json"

_STRICT = ConfigDict(extra="forbid")

# Brand-name scrubber for blinded mode. Word-bounded matches; replacement
# token is `[MODEL]` so the prompt shape stays readable. Update the regex
# additively as new models join the registry — removing a brand from the
# list would un-scrub prompts in legacy bundles, so prefer adding.
_BRAND_RE = re.compile(
    r"\b(claude(?:-[\w.]+)?|gpt(?:-[\w.]+)?|gemini(?:-[\w.]+)?|grok(?:-[\w.]+)?|"
    r"qwen(?:-[\w.]+)?|kimi(?:-[\w.]+)?|glm(?:-[\w.]+)?|llama(?:-[\w.]+)?|"
    r"mistral(?:-[\w.]+)?|deepseek(?:-[\w.]+)?|mimo(?:-[\w.]+)?|sonar(?:-[\w.]+)?|"
    r"anthropic|openai|google|openrouter|perplexity|moonshot(?:ai)?|xiaomi)\b",
    re.IGNORECASE,
)
# Provider-prefixed IDs like `x-ai/grok-4.3` survive the \b boundary in
# `_BRAND_RE` because of the embedded hyphen — handle them in a separate
# pass that anchors to the namespace prefix.
_PROVIDER_PREFIX_RE = re.compile(
    r"\b(x-ai|z-ai|meta-llama|anthropic|openai|google|openrouter)/[\w.-]+",
    re.IGNORECASE,
)


def scrub_brands(text: str) -> str:
    """Mask model and provider brand names for blinded contexts.

    Idempotent — running twice produces the same output. Conservative
    on word boundaries to avoid mangling unrelated tokens.
    """
    out = _PROVIDER_PREFIX_RE.sub("[MODEL]", text)
    out = _BRAND_RE.sub("[MODEL]", out)
    return out


class ContextBundle(BaseModel):
    """Immutable per-run context written once at run-init.

    Downstream stages (synth, capsule, arbiter) load this rather than
    receiving the prompt as a string parameter — it is the single source
    of truth for "what did the user ask?".
    """

    model_config = _STRICT

    schema_version: int = Field(_CONTEXT_SCHEMA_VERSION, ge=1)
    prompt: str = Field(..., description="The base prompt as the caller sent it.")
    prompt_scrubbed: str = Field(
        ...,
        description=(
            "Brand-scrubbed variant for blinded mode. Identical to prompt "
            "when scrubbing is a no-op."
        ),
    )
    blinded: bool = Field(
        False, description="Whether downstream stages should default to the scrubbed prompt."
    )

    def prompt_for_downstream(self, *, anonymised: bool | None = None) -> str:
        """Return the prompt downstream stages should use.

        Defaults to `prompt_scrubbed` when `blinded=True`. Pass
        `anonymised` to override (e.g. when a synth call explicitly
        requests anonymisation independent of the bundle's default).
        """
        use_scrubbed = self.blinded if anonymised is None else anonymised
        return self.prompt_scrubbed if use_scrubbed else self.prompt


def build(prompt: str, *, blinded: bool) -> ContextBundle:
    return ContextBundle(
        prompt=prompt,
        prompt_scrubbed=scrub_brands(prompt) if blinded else prompt,
        blinded=blinded,
    )


def write(paths: artifacts.RunPaths, bundle: ContextBundle) -> None:
    """Persist the bundle alongside prompt.txt."""
    (paths.root / _CONTEXT_FILENAME).write_text(bundle.model_dump_json(indent=2))


# Char-based budgets (4 chars ≈ 1 token; counting chars is exact and cheap).
# These are *input* budgets — i.e. how much we'll send to the downstream
# stage's model — not response budgets (those live in models.json as
# `default_budget_tokens`). The defaults are sized to fit comfortably in
# 200K-token contexts (Claude, Gemini Pro) with headroom for the response.
_DEFAULT_SYNTH_INPUT_BUDGET = int(
    os.environ.get("CONSULT_SYNTH_BUDGET_CHARS", 600_000)
)
_DEFAULT_CAPSULE_BODY_BUDGET = int(
    os.environ.get("CONSULT_CAPSULE_BODY_BUDGET_CHARS", 150_000)
)


def trim_text(text: str, max_chars: int, *, label: str = "TEXT") -> str:
    """Head+tail truncate `text` to fit within `max_chars`.

    Preserves the first ~75% and last ~25% of the budget with a marker
    indicating how much was dropped. No-op when already under budget.

    Logs at warning level when a trim happens — silent truncation here
    would be the exact "model didn't see what we sent" failure mode the
    Phase-1 audit was set up to prevent.
    """
    if len(text) <= max_chars:
        return text
    head_budget = (max_chars * 3 // 4) - 50
    tail_budget = (max_chars - head_budget) - 50
    if head_budget <= 0 or tail_budget <= 0:
        # Budget too tight for head+tail; degrade to a head-only cut.
        head_budget = max(0, max_chars - 50)
        tail_budget = 0
    dropped = len(text) - (head_budget + tail_budget)
    logger.warning(
        "trim_text: %s exceeds %d chars (was %d, dropping %d from middle)",
        label,
        max_chars,
        len(text),
        dropped,
    )
    marker = f"\n\n... [TRIMMED {dropped} chars from middle of {label}] ...\n\n"
    if tail_budget == 0:
        return text[:head_budget] + marker
    return text[:head_budget] + marker + text[-tail_budget:]


def trim_synth_input(
    *,
    original_prompt: str | None,
    bodies: dict[str, str],
    overall_budget: int | None = None,
) -> tuple[str | None, dict[str, str]]:
    """Returns (maybe-trimmed prompt, maybe-trimmed bodies).

    Strategy when total chars exceed `overall_budget`:
    1. Trim the longest bodies first (head+tail). Keep at least 5000 chars
       per body so each panellist still has signal.
    2. If still over budget, trim the original prompt last — it's the
       source material we most want to preserve.

    Returns the inputs unchanged when already within budget.
    """
    budget = overall_budget if overall_budget is not None else _DEFAULT_SYNTH_INPUT_BUDGET
    prompt_len = len(original_prompt) if original_prompt else 0
    total = prompt_len + sum(len(b) for b in bodies.values())
    if total <= budget:
        return original_prompt, bodies

    over = total - budget
    new_bodies = dict(bodies)
    # Trim from the largest body first — that's where we'll recover the
    # most slack with the least signal loss per body.
    for slug, body in sorted(new_bodies.items(), key=lambda kv: len(kv[1]), reverse=True):
        if over <= 0:
            break
        cur = len(body)
        target = max(5000, cur - over)
        if target >= cur:
            continue
        trimmed = trim_text(body, target, label=f"body[{slug}]")
        recovered = cur - len(trimmed)
        over -= recovered
        new_bodies[slug] = trimmed

    new_prompt = original_prompt
    if over > 0 and new_prompt:
        target = max(2000, len(new_prompt) - over)
        new_prompt = trim_text(new_prompt, target, label="original_prompt")

    return new_prompt, new_bodies


def trim_capsule_body(body: str, max_chars: int | None = None) -> str:
    """Head+tail trim a panellist body before capsule extraction.

    The extractor (cheap model) only needs the body's structure, not every
    word. Trimming overlong bodies here keeps the extractor's input bounded
    so a single 500KB panellist response can't OOM the cheap extractor.
    """
    cap = max_chars if max_chars is not None else _DEFAULT_CAPSULE_BODY_BUDGET
    return trim_text(body, cap, label="capsule_body")


def load_or_none(paths: artifacts.RunPaths) -> ContextBundle | None:
    """Load the bundle for a run, returning None for legacy runs.

    Returns None — not an exception — when:
    - `context.json` does not exist (pre-Phase-1 run)
    - the file is present but unreadable (corrupt, schema mismatch, etc.)

    Logs a warning in the second case. The caller's responsibility is to
    fall back to its pre-bundle behaviour when None is returned.
    """
    path = paths.root / _CONTEXT_FILENAME
    if not path.exists():
        return None
    try:
        return ContextBundle.model_validate_json(path.read_text())
    except Exception as e:  # noqa: BLE001 — any parse/validate error is "legacy"
        logger.warning(
            "context.json present but unreadable for run %s: %s", paths.run_id, e
        )
        return None
