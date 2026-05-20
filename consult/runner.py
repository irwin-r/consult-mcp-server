"""Async fan-out runner. Calls LiteLLM in parallel, classifies responses,
writes artifacts, and assembles a RunHandle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm

from . import artifacts, registry
from .status import classify
from .types import ManifestEntry, ModelSpec, RunHandle, Status

logger = logging.getLogger(__name__)

# Async progress callback signature: (completed, total, message). Wrapped at
# each call site in a try/except so a notification failure never aborts the
# real work (best-effort observability, not a hard contract).
ProgressCallback = Callable[[int, int, str], Awaitable[None]]


def _append_progress_log(run_root: Path, entry: dict[str, Any]) -> None:
    """Append a JSONL line to `<run>/_progress.log` for client-less tailing.

    Always on — gives mid-run observability via `tail -f` even when the MCP
    client didn't ask for `notifications/progress`. A write error here is
    logged at debug and swallowed; the progress log is best-effort.
    """
    entry = {"ts": datetime.now(UTC).isoformat(), **entry}
    try:
        with (run_root / "_progress.log").open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as e:  # pragma: no cover — log-write failure is benign
        logger.debug("progress log write failed: %s", e)

# Drop unsupported params per provider so e.g. `reasoning_effort` on a
# non-reasoning model is silently ignored rather than failing the panel.
litellm.drop_params = True
# LiteLLM's default error path prints an ANSI-coloured "Give Feedback /
# Get Help" footer + "debug this error" hint to stderr on every caught
# exception. We already log a clean .warning() at the catch site; this
# silences the noisy trailer so a single pricing-table miss doesn't
# drown the rest of the log.
litellm.suppress_debug_info = True
# LiteLLM attaches its own coloured handler to the "LiteLLM" logger AND
# lets it propagate to root. When a caller (test harness, dogfood
# script, MCP host) configures the root logger at INFO or below, every
# LiteLLM line prints twice — once via the coloured handler, once via
# root. Stop the propagation so callers see exactly one copy.
logging.getLogger("LiteLLM").propagate = False

# CONTRACT: capsule.py:_CONFIDENCE and the capsule extractor prompt depend on
# these exact line prefixes (`CONFIDENCE:` and `KEY_REASON:`). Don't rename
# either without updating both.
_FOOTER = """\
---
End your response with EXACTLY these two lines (after your main answer):
CONFIDENCE: <number between 0.0 and 1.0 reflecting how sure you are>
KEY_REASON: <one sentence — the single most important reason for your view>"""


def _build_per_slug_prompt(base_prompt: str, stance_prompt: str) -> str:
    head = f"{stance_prompt}\n\n" if stance_prompt else ""
    return f"{head}{base_prompt}\n\n{_FOOTER}"


_MODEL_COUNT_SUFFIX = re.compile(r"^(.+):(\d+)$")


def expand_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    """Expand `model:N` syntax into N copies of the spec.

    A trailing `:<positive int>` on `spec.model` requests N instances of the
    same model in the panel — useful for stochastic averaging (run the same
    prompt N times and compare), or to grow a panel without adding new
    aliases. The slug-disambiguation in `_make_slug` already appends an
    index suffix when the same base name repeats, so no extra work needed
    downstream.

    Idempotent: passing already-expanded specs (no `:N` suffix on any
    `model`) returns them unchanged. Bare model strings with internal
    colons that aren't followed by digits (rare, but possible for raw
    LiteLLM IDs) are left alone — the regex anchors to the end.

    Raises `ValueError` if N < 1 (zero copies is almost certainly a typo).
    """
    out: list[ModelSpec] = []
    for spec in specs:
        m = _MODEL_COUNT_SUFFIX.match(spec.model)
        if not m:
            out.append(spec)
            continue
        base, count_str = m.group(1), m.group(2)
        count = int(count_str)
        if count < 1:
            raise ValueError(
                f"model:count must be ≥1 (got {spec.model!r})"
            )
        for _ in range(count):
            out.append(ModelSpec(model=base, stance=spec.stance, slug=spec.slug))
    return out


def _build_messages(prompt: str, provider: str) -> list[dict[str, Any]]:
    # Anthropic-only: mark the user prompt as a cache breakpoint. Repeat
    # panellists in the same fanout share the bulk of their prefix (base
    # prompt + footer; stance varies). Without cache_control LiteLLM
    # serialises a plain string and no caching is requested.
    if provider == "anthropic":
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
            ],
        }]
    return [{"role": "user", "content": prompt}]


def _make_slug(spec: ModelSpec, idx: int, blinded: bool) -> str:
    if blinded:
        # alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu
        greek = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
        ]
        return f"panelist-{greek[idx]}" if idx < len(greek) else f"panelist-{idx}"
    if spec.slug:
        return spec.slug
    # Derive a stable, readable slug from the alias/id
    base = spec.model.split("/")[-1].lower()
    return f"{base}-{idx}" if idx > 0 else base


async def _call_one(
    spec: ModelSpec,
    slug: str,
    per_slug_prompt: str,
    paths: artifacts.RunPaths,
) -> ManifestEntry:
    # An unknown alias must fail this single panellist, not the whole panel.
    # `asyncio.gather` without return_exceptions=True would otherwise cancel
    # every sibling call when the KeyError propagates out.
    try:
        entry = registry.resolve_model(spec.model)
    except KeyError as e:
        paths.response_text(slug).write_text("")
        return ManifestEntry(
            slug=slug,
            model_id=None,
            persona=spec.stance if spec.stance else None,
            status=Status.ERROR,
            finish_reason=None,
            resource_uri=paths.resource_uri(slug),
            body_path=str(paths.response_text(slug)),
            latency_ms=0,
            tokens_in=None,
            tokens_out=None,
            cost_usd=None,
            cost_known=True,
            error=str(e),
            confidence=None,
            capsule=None,
        )
    litellm_id = entry["litellm_id"]
    budget = entry.get("default_budget_tokens", 8000)
    timeout = entry.get("default_timeout_s", 180)
    provider = entry.get("provider", "")

    extra: dict[str, Any] = {}
    if "reasoning_effort" in entry:
        extra["reasoning_effort"] = entry["reasoning_effort"]

    paths.prompt_for(slug).write_text(per_slug_prompt)
    start = time.time()
    status: Status
    finish: str | None = None
    body = ""
    cost: float | None = None
    cost_known: bool = True
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None

    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=litellm_id,
                messages=_build_messages(per_slug_prompt, provider),
                max_tokens=budget,
                **extra,
            ),
            timeout=timeout,
        )
        # Persist raw response — use model_dump for Pydantic, fall back to dict
        try:
            raw = resp.model_dump()  # type: ignore[attr-defined]
        except AttributeError:
            raw = dict(resp) if hasattr(resp, "__iter__") else {"_repr": repr(resp)}
        paths.response_raw(slug).write_text(json.dumps(raw, indent=2, default=str))

        status, finish, body = classify(resp)
        usage = getattr(resp, "usage", None)
        if usage:
            tokens_in = getattr(usage, "prompt_tokens", None)
            tokens_out = getattr(usage, "completion_tokens", None)
        try:
            cost = litellm.completion_cost(completion_response=resp)
            cost_known = cost is not None
        except Exception as ce:  # noqa: BLE001
            logger.warning("cost lookup failed for %s: %s", litellm_id, ce)
            cost = None
            cost_known = False

    except TimeoutError:
        status, finish, body = Status.TIMEOUT, None, ""
        error = f"timeout after {timeout}s"
        cost_known = True  # no call was billable
    except Exception as e:  # noqa: BLE001 — LiteLLM raises many concrete types
        status, finish, body = classify(None, exception=e)
        error = str(e)[:500] or f"{type(e).__name__}"
        cost_known = True  # no call was billable

    paths.response_text(slug).write_text(body)
    latency_ms = int((time.time() - start) * 1000)

    # Per-panellist completion line. Visible in MCP server logs
    # (~/.claude/logs/mcp-logs-consult/) — gives mid-run observability so
    # a slow panellist's status is knowable without waiting for the full
    # manifest. Use info-level so production stays quiet by default.
    logger.info("panellist %s: %s in %dms", slug, status.value, latency_ms)

    # Per-run progress log (kind=panellist) — always on, tailable as
    # JSONL even if the MCP client didn't subscribe to notifications/progress.
    # See pass #2's panel synthesis: "A + D" — progress notifications PLUS
    # a local log file as the fallback when the in-band path fails.
    _append_progress_log(
        paths.root,
        {
            "kind": "panellist",
            "slug": slug,
            "status": status.value,
            "latency_ms": latency_ms,
        },
    )

    persona_label = spec.stance if spec.stance else None

    # Status.ERROR/TIMEOUT require a non-empty error per the model invariant.
    # Defensive: if classify() returns ERROR with no exception path taken (e.g.
    # malformed empty response), synthesise a placeholder so construction
    # doesn't blow up — the underlying classifier already logged the shape.
    if status in (Status.ERROR, Status.TIMEOUT) and not error:
        error = f"{status.value}: no provider exception captured"

    return ManifestEntry(
        slug=slug,
        model_id=litellm_id,
        persona=persona_label,
        status=status,
        finish_reason=finish,
        resource_uri=paths.resource_uri(slug),
        body_path=str(paths.response_text(slug)),
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        cost_known=cost_known,
        error=error,
        confidence=None,  # populated by capsule extractor
        capsule=None,
    )


def estimate_cost(specs: list[ModelSpec], prompt: str) -> tuple[float, bool]:
    """Returns (total_estimate, all_known).

    Uses LiteLLM's per-token price tables via `cost_per_token()`. Models that
    don't have pricing data set `all_known=False`; caller must treat unknown
    costs conservatively (a panel with even one unknown-cost spec cannot be
    validated against `max_run_usd`).

    Unknown-alias specs are treated as cost-unknown (rather than raising) so
    a single typo can't abort the panel here; `_call_one` surfaces the alias
    as a per-spec Status.ERROR.
    """
    total = 0.0
    all_known = True
    for spec in specs:
        try:
            entry = registry.resolve_model(spec.model)
        except KeyError:
            all_known = False
            continue
        litellm_id = entry["litellm_id"]
        try:
            tin = litellm.token_counter(model=litellm_id, text=prompt)
            tout = entry.get("default_budget_tokens", 8000)
            in_per_tok, out_per_tok = litellm.cost_per_token(
                model=litellm_id, prompt_tokens=tin, completion_tokens=tout
            )
            if in_per_tok is None or out_per_tok is None:
                all_known = False
                continue
            total += in_per_tok + out_per_tok
        except Exception as e:  # noqa: BLE001
            logger.warning("estimate_cost: no price for %s (%s)", litellm_id, e)
            all_known = False
            continue
    return total, all_known


async def fanout(
    prompt: str,
    specs: list[ModelSpec],
    *,
    blinded: bool = False,
    dry_run: bool = False,
    max_run_usd: float | None = None,
    existing_paths: artifacts.RunPaths | None = None,
    on_progress: ProgressCallback | None = None,
) -> RunHandle:
    """Parallel fan-out. Creates a fresh run by default. Pass `existing_paths`
    to write into an existing run dir (used by `refine` to keep all rounds
    under one run_id with round-suffixed slugs).

    If `on_progress` is set, it's called once per panellist as it completes
    with `(done, total, message)`. Failures inside the callback are logged
    and swallowed — progress is best-effort, not load-bearing.
    """
    # Resolve `model:N` sugar BEFORE estimate_cost so the cap reflects the
    # real panel size, not the pre-expansion request count.
    specs = expand_specs(specs)
    if existing_paths is None:
        paths = artifacts.create_run()
        paths.prompt_txt.write_text(prompt)
        # Snapshot the registry so replays are stable
        paths.registry_snapshot.write_text(json.dumps(registry.models_config(), indent=2))
    else:
        paths = existing_paths

    # Estimate cost up front; if dry_run, return immediately with empty manifest
    estimate, all_known = estimate_cost(specs, prompt)
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()
    if estimate > cap:
        # When some prices are unknown, `estimate` is only the known-priced
        # portion; the actual run could cost more. Surface that so the cap
        # message isn't misleading low. Mirrors the dry_run branch below.
        suffix = "" if all_known else " (known-priced portion only; some unknown)"
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=(
                f"estimated cost ${estimate:.2f}{suffix} exceeds cap ${cap:.2f}"
            ),
            blinded=blinded,
        )
    if dry_run:
        suffix = "" if all_known else " (some prices unknown — actual cost may differ)"
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            cost_known=all_known,
            wall_ms=0,
            partial=True,
            partial_reason=f"dry_run: estimated cost ${estimate:.4f}{suffix}",
            blinded=blinded,
        )

    # Build slugs + prompts
    slugs = [_make_slug(s, i, blinded) for i, s in enumerate(specs)]
    per_prompts = [
        _build_per_slug_prompt(prompt, registry.resolve_stance(s.stance)) for s in specs
    ]

    start = time.time()
    total = len(specs)
    done = 0

    async def _run_one(spec: ModelSpec, slug: str, per_prompt: str) -> ManifestEntry:
        nonlocal done
        entry = await _call_one(spec, slug, per_prompt, paths)
        done += 1
        if on_progress is not None:
            try:
                await on_progress(done, total, f"{slug}: {entry.status.value}")
            except Exception as e:  # noqa: BLE001 — notification is best-effort
                logger.debug("on_progress callback failed: %s", e)
        return entry

    tasks = [
        _run_one(spec, slug, per_prompt)
        for spec, slug, per_prompt in zip(specs, slugs, per_prompts, strict=True)
    ]
    manifest = await asyncio.gather(*tasks)
    wall_ms = int((time.time() - start) * 1000)

    # If blinded, scrub model_id from the manifest (kept in registry_snapshot for audit)
    if blinded:
        for m in manifest:
            m.model_id = None

    cost_total = sum((m.cost_usd or 0.0) for m in manifest)
    all_known = all(m.cost_known for m in manifest)
    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=manifest,
        cost_usd=cost_total,
        cost_known=all_known,
        wall_ms=wall_ms,
        partial=False,
        blinded=blinded,
    )
    artifacts.write_manifest(paths, handle.model_dump())
    return handle
