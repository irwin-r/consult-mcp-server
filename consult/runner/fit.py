"""Context fitting: token budgets, attachment-aware shrink, head+tail trim."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

from .. import artifacts, context
from .. import attachments as attachments_mod

logger = logging.getLogger(__name__)


def concat_turn_text(turns: list[dict[str, Any]]) -> str:
    """Flatten a list of `{role, content}` turns into a single text blob
    for token counting. `content` may be a string or a list of content
    parts (Anthropic-style blocks); we only count text. Other block kinds
    (images, tool_use) are skipped — none flow through `prior_turns` today.
    """
    parts: list[str] = []
    for turn in turns:
        content = turn.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _max_input_tokens(litellm_id: str, entry: dict[str, Any]) -> int | None:
    """Return the model's maximum input-token budget, or None if unknown.

    Lookup order:
    1. `entry["max_input_tokens"]` — explicit registry override. Use when
       LiteLLM's table is wrong or stale for a model.
    2. `litellm.get_model_info(litellm_id)["max_input_tokens"]` — LiteLLM
       maintains this for most known models. Returns None when missing.
    3. None — unknown context size; the pre-flight check below skips
       gracefully (provider will reject the call if it's too large, same
       as the prior behaviour).

    Kept narrow on purpose: we don't want a broad provider-capability layer
    here, just enough to surface "your prompt is too big" as a manifest
    Status.ERROR rather than as a raw provider BadRequestError that the
    user has to decode.
    """
    override = entry.get("max_input_tokens")
    if override is not None:
        try:
            return int(override)
        except (TypeError, ValueError):
            logger.warning(
                "registry max_input_tokens for %s is not an int: %r",
                litellm_id,
                override,
            )
    try:
        info = litellm.get_model_info(litellm_id)
    except Exception:  # noqa: BLE001 — get_model_info raises on unknown IDs
        return None
    if isinstance(info, dict):
        v = info.get("max_input_tokens")
        if isinstance(v, int) and v > 0:
            return v
    return None


def _replace_attachments_with_stubs(
    prompt: str,
    paths: artifacts.RunPaths,
    available_chars: int,
) -> str:
    """Drop oversized inlined attachment blocks largest-first, replacing
    each with a short stub that references the persisted resource URI.

    Stops as soon as the prompt fits `available_chars`. Returns the
    prompt unchanged when there are no parseable attachment blocks or
    when no single drop would help (a block smaller than its stub isn't
    worth dropping). The caller still re-counts tokens after this —
    char→token is approximate, so this is a fast pre-filter, not the
    final budget check.

    Why drop whole blocks instead of head+tail-slicing through them:
    half a source file with `[TRIMMED 50000 chars from middle]` in the
    middle is worse than useless for a code reviewer — they can't trust
    any claim about the body. A clean "this file was too big, here's
    where to read it" stub lets the panellist reason about what it
    can't see rather than pretending the partial view is complete.
    """
    blocks = attachments_mod.extract_inlined_blocks(prompt)
    if not blocks:
        return prompt
    used: set[str] = set()
    named = [(b, attachments_mod.safe_attachment_name(b, used)) for b in blocks]

    def _stub(block: attachments_mod.InlinedBlock, name: str) -> str:
        uri = paths.attachment_resource_uri(name)
        line_count = block.content.count("\n") + 1
        return (
            f"{block.header}\n"
            f"[Attachment dropped to fit context: {line_count:,} lines, "
            f"{len(block.content):,} chars. Full source at {uri}]"
        )

    # Largest-first drop priority. We commit each drop only if it
    # actually shrinks the prompt — pathological case: a 60-char block
    # whose stub is 200 chars is not worth dropping.
    drop_priority = sorted(
        range(len(named)),
        key=lambda i: -(named[i][0].end - named[i][0].start),
    )
    dropped: set[int] = set()
    current_chars = len(prompt)
    for idx in drop_priority:
        if current_chars <= available_chars:
            break
        block, name = named[idx]
        block_len = block.end - block.start
        stub_len = len(_stub(block, name))
        if stub_len >= block_len:
            continue  # would grow the prompt
        dropped.add(idx)
        current_chars -= block_len - stub_len

    if not dropped:
        return prompt

    parts: list[str] = []
    last_end = 0
    for i, (block, name) in enumerate(named):
        parts.append(prompt[last_end : block.start])
        parts.append(_stub(block, name) if i in dropped else prompt[block.start : block.end])
        last_end = block.end
    parts.append(prompt[last_end:])
    logger.info(
        "attachment-aware trim: dropped %d/%d blocks (%d chars → ~%d chars)",
        len(dropped),
        len(named),
        len(prompt),
        current_chars,
    )
    return "".join(parts)


async def _fit_prompt_to_context(
    per_slug_prompt: str,
    *,
    paths: artifacts.RunPaths | None = None,
    prior_turns: list[dict[str, Any]] | None,
    litellm_id: str,
    max_input_tokens: int,
    max_output_tokens: int,
) -> tuple[str, int]:
    """Trim `per_slug_prompt` so input+output fits the model's context.

    Returns `(maybe_trimmed_prompt, dropped_chars)`. `dropped_chars == 0`
    means no trim happened (prompt already fit, or budget made trimming
    impossible). Callers use the explicit count to surface a quantified
    trim note in the manifest — sniffing the prompt for a marker substring
    false-positives on source-code attachments that contain the word.

    Trim strategy: head+tail truncates the prompt via `context.trim_text`
    (preserves the stance preface at the head and the CONFIDENCE/KEY_REASON
    footer at the tail) and re-counts; if still over budget after one pass
    (rare — usually means `prior_turns` alone exceed the budget), shrinks
    further via a smaller char target.

    The `prior_turns` text is included in the token count but never
    trimmed — those are the prior consultation's role-separated exchange
    in a `refine` continuation, and trimming them would corrupt the
    user/assistant boundary the model relies on. If they alone exceed the
    budget the caller should re-prompt without continuation.
    """
    prior_text = concat_turn_text(prior_turns) if prior_turns else ""
    target_input = max_input_tokens - max_output_tokens
    if target_input <= 0:
        # Defensive: the registry's `default_budget_tokens` shouldn't ever
        # be larger than the model's whole context, but if it is, return
        # the prompt as-is and let the provider reject — the caller's
        # config is the real bug.
        return per_slug_prompt, 0

    async def _count(text: str) -> int:
        # token_counter is sync + CPU-bound; offload so we don't block the
        # event loop during the pre-flight check.
        try:
            return int(
                await asyncio.to_thread(
                    litellm.token_counter,
                    model=litellm_id,
                    text=text,
                )
            )
        except Exception:  # noqa: BLE001
            return -1  # unknown — caller treats as "skip the check"

    prior_tokens = await _count(prior_text) if prior_text else 0
    if prior_tokens < 0:
        return per_slug_prompt, 0  # token_counter is broken; let provider decide
    available_for_prompt = target_input - prior_tokens
    if available_for_prompt <= 0:
        # prior_turns alone exceed the budget. We don't trim prior_turns
        # (would corrupt role boundaries); log and pass through so the
        # caller sees the provider's rejection with the real reason.
        logger.warning(
            "fit_prompt: prior_turns alone (%d tokens) exceed available input "
            "budget (%d). Returning prompt untrimmed; provider will reject.",
            prior_tokens,
            target_input,
        )
        return per_slug_prompt, 0

    prompt_tokens = await _count(per_slug_prompt)
    if prompt_tokens < 0:
        return per_slug_prompt, 0
    if prompt_tokens <= available_for_prompt:
        return per_slug_prompt, 0  # already fits, no-op

    original_len = len(per_slug_prompt)
    # Attachment-aware shrink first (when we have a run dir to anchor
    # resource URIs to). Drops whole attachment blocks largest-first,
    # replaces each with a stub pointing at the persisted resource.
    # Better signal than head+tail slicing through code files. The 4x
    # char-per-token rule of thumb sets the char target — final budget
    # check is the re-count below.
    if paths is not None:
        avail_chars = available_for_prompt * 4
        shrunk = _replace_attachments_with_stubs(
            per_slug_prompt,
            paths,
            avail_chars,
        )
        if shrunk is not per_slug_prompt:
            per_slug_prompt = shrunk
            prompt_tokens = await _count(per_slug_prompt)
            if prompt_tokens < 0:
                return per_slug_prompt, original_len - len(per_slug_prompt)
            if prompt_tokens <= available_for_prompt:
                return per_slug_prompt, original_len - len(per_slug_prompt)

    # Still over budget — fall back to head+tail trim. token_counter
    # <-> char-count is approximate; aim for 90% of the available
    # budget so a recount comes in under cleanly. The 500-char floor
    # prevents pathological "shrink to nothing" outcomes on tiny-context
    # models. Cap iterations at 3 — usually one pass suffices.
    trimmed = per_slug_prompt
    for _attempt in range(3):
        ratio = (available_for_prompt * 0.9) / prompt_tokens
        target_chars = max(500, int(len(trimmed) * ratio))
        if target_chars >= len(trimmed):
            # Can't shrink further without breaking the floor — return what
            # we have and let the provider reject (or accept) the call.
            break
        trimmed = context.trim_text(
            trimmed,
            target_chars,
            label="panellist prompt",
        )
        prompt_tokens = await _count(trimmed)
        if prompt_tokens < 0 or prompt_tokens <= available_for_prompt:
            break
    dropped = max(0, original_len - len(trimmed))
    return trimmed, dropped
