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

import functools
import logging
import os
import re

from pydantic import Field

from . import artifacts, registry
from .types import StrictModel

logger = logging.getLogger(__name__)

_CONTEXT_SCHEMA_VERSION = 2
_CONTEXT_FILENAME = "context.json"

# Words that are alpha-only, ≥3 chars, but too common to scrub safely.
# (Adding "the" would mangle ordinary English. Adding nothing here means
# we'd scrub e.g. "max" out of "qwen-max" — which is actually fine, but
# also out of "max(a, b)" in code, which is not.) Kept narrow on purpose.
_COMMON_NON_BRAND_WORDS: frozenset[str] = frozenset({
    "max", "pro", "mini", "nano", "flash", "lite", "preview", "latest",
    "ultra", "turbo", "instruct", "chat", "base",
})


@functools.cache
def _brand_patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Build the brand-scrub regexes from the live model registry.

    Returns `(brand_re, provider_prefix_re)`. Cached because the registry
    config itself is cached (`registry.models_config` LRU), so this is a
    one-shot cost per process.

    Words harvested from:
    - Each model alias (split on `-`, alpha-only parts ≥3 chars)
    - Each model's `family` field
    - Each model's `litellm_id`: provider segments (everything before the
      last `/`) → provider regex; model-name segments (after the last `/`)
      → brand regex

    Common tier suffixes ("max", "pro", "mini", …) are filtered to avoid
    scrubbing every casual occurrence of "max" or "mini" out of normal
    English. New models added to `models.json` automatically extend
    coverage on the next process start.
    """
    cfg = registry.models_config()
    brand_words: set[str] = set()
    provider_words: set[str] = set()

    def _harvest_brand_parts(s: str) -> None:
        for part in re.split(r"[-/]", s):
            if part.isalpha() and len(part) >= 3 and part.lower() not in _COMMON_NON_BRAND_WORDS:
                brand_words.add(part.lower())

    for alias, entry in cfg.get("models", {}).items():
        _harvest_brand_parts(alias)
        family = entry.get("family", "")
        if family:
            _harvest_brand_parts(family)
        lid = entry.get("litellm_id", "")
        if "/" in lid:
            prefix, name = lid.rsplit("/", 1)
            # Provider prefix(es) — these go in the provider-prefix regex
            for seg in prefix.split("/"):
                seg_low = seg.lower()
                if seg_low and re.fullmatch(r"[a-z0-9._-]+", seg_low):
                    provider_words.add(seg_low)
            _harvest_brand_parts(name)
        else:
            _harvest_brand_parts(lid)

    # Always include the known lab/provider names even if not in the
    # registry yet — anchors a stable baseline of coverage.
    for w in ("anthropic", "openai", "google", "openrouter", "perplexity",
              "deepmind", "moonshotai", "xiaomi"):
        provider_words.add(w)

    # Provider words appear in BOTH regexes:
    # - `provider_re` matches `<provider>/<id-suffix>` so raw LiteLLM IDs
    #   scrub as a whole token (avoiding leaking the `id-suffix` part).
    # - `brand_re` matches standalone provider mentions like "Anthropic" or
    #   "OpenAI" without a `/` after.
    brand_words.update(provider_words)

    if not brand_words:
        brand_words.add("__no_brands__")  # never matches; keeps regex valid
    if not provider_words:
        provider_words.add("__no_providers__")

    brand_pat = "|".join(sorted(re.escape(w) for w in brand_words))
    provider_pat = "|".join(sorted(re.escape(w) for w in provider_words))

    # Match a brand word, optionally followed by `-<version-or-name>` so
    # "claude-opus-4-7" scrubs as a whole token. Word boundary on both sides.
    brand_re = re.compile(
        rf"\b({brand_pat})(?:-[\w.]+)*\b",
        re.IGNORECASE,
    )
    # Match `<provider>/<rest-of-id>` so raw LiteLLM IDs scrub fully.
    provider_re = re.compile(
        rf"\b({provider_pat})/[\w./-]+",
        re.IGNORECASE,
    )
    return brand_re, provider_re


def scrub_brands(text: str) -> str:
    """Mask model and provider brand names for blinded contexts.

    Idempotent — running twice produces the same output. Conservative
    on word boundaries to avoid mangling unrelated tokens.
    """
    brand_re, provider_re = _brand_patterns()
    out = provider_re.sub("[MODEL]", text)
    out = brand_re.sub("[MODEL]", out)
    return out


class ContextBundle(StrictModel):
    """Immutable per-run context written once at run-init.

    Downstream stages (synth, capsule, arbiter) load this rather than
    receiving the prompt as a string parameter — it is the single source
    of truth for "what did the user ask?".

    `capsule_kind` records the extraction shape the caller asked for
    (decision / review / research). Persisted so a `continuation_id`
    can inherit the prior run's shape without the caller having to
    specify it again.
    """

    # schema_version=2 added `capsule_kind`. v1 bundles are still loaded
    # (the field has a default), and the validator below upgrades them.
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
    capsule_kind: str = Field(
        "decision",
        description=(
            "Capsule shape the caller requested: 'decision' | 'review' | "
            "'research'. Inherited by continuation runs if the caller "
            "doesn't override."
        ),
    )

    def prompt_for_downstream(self, *, anonymised: bool | None = None) -> str:
        """Return the prompt downstream stages should use.

        Defaults to `prompt_scrubbed` when `blinded=True`. Pass
        `anonymised` to override (e.g. when a synth call explicitly
        requests anonymisation independent of the bundle's default).
        """
        use_scrubbed = self.blinded if anonymised is None else anonymised
        return self.prompt_scrubbed if use_scrubbed else self.prompt


def build(prompt: str, *, blinded: bool, capsule_kind: str = "decision") -> ContextBundle:
    return ContextBundle(
        prompt=prompt,
        prompt_scrubbed=scrub_brands(prompt) if blinded else prompt,
        blinded=blinded,
        capsule_kind=capsule_kind,
    )


def write(paths: artifacts.RunPaths, bundle: ContextBundle) -> None:
    """Persist the bundle alongside prompt.txt."""
    (paths.root / _CONTEXT_FILENAME).write_text(bundle.model_dump_json(indent=2))


# Char-based budgets (4 chars ≈ 1 token; counting chars is exact and cheap).
# These are *input* budgets — i.e. how much we'll send to the downstream
# stage's model — not response budgets (those live in models.json as
# `default_budget_tokens`). The defaults are sized to fit comfortably in
# 200K-token contexts (Claude, Gemini Pro) with headroom for the response.
#
# Read at call time (not import time) so test monkeypatching of the env
# vars works and so a long-lived MCP process picks up live env changes.
_SYNTH_BUDGET_DEFAULT = 600_000
_CAPSULE_BODY_BUDGET_DEFAULT = 150_000


def _synth_input_budget() -> int:
    return int(os.environ.get("CONSULT_SYNTH_BUDGET_CHARS", _SYNTH_BUDGET_DEFAULT))


def _capsule_body_budget() -> int:
    return int(os.environ.get("CONSULT_CAPSULE_BODY_BUDGET_CHARS", _CAPSULE_BODY_BUDGET_DEFAULT))


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
    2. If still over budget, trim the original prompt (down to 2000 chars
       floor) — it's the source material we most want to preserve.
    3. If STILL over budget (large panels where N × 5000 > budget), do a
       final pass that ignores the per-body floor and shrinks bodies
       proportionally to fit. Better to give every panellist a small
       slice than to silently OOM the synth model.

    Returns the inputs unchanged when already within budget.
    """
    budget = overall_budget if overall_budget is not None else _synth_input_budget()
    prompt_len = len(original_prompt) if original_prompt else 0
    total = prompt_len + sum(len(b) for b in bodies.values())
    if total <= budget:
        return original_prompt, bodies

    over = total - budget
    new_bodies = dict(bodies)
    # Trim from the largest body first — that's where we'll recover the
    # most slack with the least signal loss per body. Track whether any
    # body hit the per-body floor so we know whether the hard-trim pass
    # below is necessary (the floor is the only reason pass 1 can leave
    # `over > 0` for a non-trivial overage; marker overhead is harmless).
    floor_hit = False
    for slug, body in sorted(new_bodies.items(), key=lambda kv: len(kv[1]), reverse=True):
        if over <= 0:
            break
        cur = len(body)
        target = max(5000, cur - over)
        if target == 5000 and cur > 5000:
            floor_hit = True
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
        over = (
            (len(new_prompt) if new_prompt else 0)
            + sum(len(b) for b in new_bodies.values())
        ) - budget

    # Final hard pass — only triggers when pass-1 hit the per-body floor
    # on at least one body, i.e. the budget genuinely can't accommodate
    # N panellists at the minimum. The trim_text marker (~80 chars/body)
    # can leave a tiny `over > 0` after pass 1 even when the budget was
    # achievable; gating on `floor_hit` avoids hard-trimming for that case.
    if floor_hit and over > 0 and new_bodies:
        new_prompt_len = len(new_prompt) if new_prompt else 0
        remaining_for_bodies = max(0, budget - new_prompt_len)
        current_total = sum(len(b) for b in new_bodies.values())
        if current_total > 0 and remaining_for_bodies < current_total:
            scale = remaining_for_bodies / current_total
            for slug in list(new_bodies):
                cur = len(new_bodies[slug])
                target = max(500, int(cur * scale))
                if target < cur:
                    new_bodies[slug] = trim_text(
                        new_bodies[slug], target, label=f"body[{slug}]/hardtrim",
                    )

    return new_prompt, new_bodies


def trim_capsule_body(body: str, max_chars: int | None = None) -> str:
    """Head+tail trim a panellist body before capsule extraction.

    The extractor (cheap model) only needs the body's structure, not every
    word. Trimming overlong bodies here keeps the extractor's input bounded
    so a single 500KB panellist response can't OOM the cheap extractor.
    """
    cap = max_chars if max_chars is not None else _capsule_body_budget()
    return trim_text(body, cap, label="capsule_body")


def load_or_none(paths: artifacts.RunPaths) -> ContextBundle | None:
    """Load the bundle for a run, returning None for legacy runs.

    Returns None — not an exception — when:
    - `context.json` does not exist (pre-Phase-1 run)
    - the file is present but unreadable (corrupt, unknown schema, etc.)

    Logs a warning in the second case. The caller's responsibility is to
    fall back to its pre-bundle behaviour when None is returned.

    v1 bundles (no `capsule_kind` field) are upgraded transparently —
    the model defaults to `capsule_kind="decision"` on load.
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
