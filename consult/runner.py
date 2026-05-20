"""Async fan-out runner. Calls LiteLLM in parallel, classifies responses,
writes artifacts, and assembles a RunHandle.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import litellm

from . import artifacts, registry
from .status import classify
from .types import ManifestEntry, ModelSpec, RunHandle, Status

litellm.drop_params = True  # silently drop unsupported params per provider

_FOOTER = """\
---
End your response with EXACTLY these two lines (after your main answer):
CONFIDENCE: <number between 0.0 and 1.0 reflecting how sure you are>
KEY_REASON: <one sentence — the single most important reason for your view>"""


def _build_per_slug_prompt(base_prompt: str, stance_prompt: str) -> str:
    head = f"{stance_prompt}\n\n" if stance_prompt else ""
    return f"{head}{base_prompt}\n\n{_FOOTER}"


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
    entry = registry.resolve_model(spec.model)
    litellm_id = entry["litellm_id"]
    budget = entry.get("default_budget_tokens", 8000)
    timeout = entry.get("default_timeout_s", 180)

    extra: dict[str, Any] = {}
    if "reasoning_effort" in entry:
        extra["reasoning_effort"] = entry["reasoning_effort"]

    paths.prompt_for(slug).write_text(per_slug_prompt)
    start = time.time()
    status: Status
    finish: str | None = None
    body = ""
    cost: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None

    try:
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=litellm_id,
                messages=[{"role": "user", "content": per_slug_prompt}],
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
        except Exception:
            cost = None

    except TimeoutError:
        status, finish, body = Status.TIMEOUT, None, ""
        error = f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001 — LiteLLM raises many concrete types
        status, finish, body = classify(None, exception=e)
        error = str(e)[:500]

    paths.response_text(slug).write_text(body)
    latency_ms = int((time.time() - start) * 1000)

    persona = registry.resolve_stance(spec.stance) if spec.stance else None
    persona_label = spec.stance if spec.stance else None

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
        error=error,
        confidence=None,  # populated by capsule extractor
        capsule=None,
    )


def estimate_cost(specs: list[ModelSpec], prompt: str) -> float:
    """Rough cost estimate using LiteLLM's token counter and registered prices.

    Best-effort; LiteLLM does not have prices for every OR model. Missing prices
    contribute 0 to the estimate (caller may treat 0 as "unknown").
    """
    total = 0.0
    for spec in specs:
        entry = registry.resolve_model(spec.model)
        litellm_id = entry["litellm_id"]
        try:
            tin = litellm.token_counter(model=litellm_id, text=prompt)
            tout = entry.get("default_budget_tokens", 8000)
            cost = litellm.completion_cost(
                model=litellm_id, prompt_tokens=tin, completion_tokens=tout
            )
            total += cost or 0.0
        except Exception:
            continue
    return total


async def fanout(
    prompt: str,
    specs: list[ModelSpec],
    *,
    blinded: bool = False,
    dry_run: bool = False,
    max_run_usd: float | None = None,
    existing_paths: artifacts.RunPaths | None = None,
) -> RunHandle:
    """Parallel fan-out. Creates a fresh run by default. Pass `existing_paths`
    to write into an existing run dir (used by `refine` to keep all rounds
    under one run_id with round-suffixed slugs).
    """
    if existing_paths is None:
        paths = artifacts.create_run()
        paths.prompt_txt.write_text(prompt)
        # Snapshot the registry so replays are stable
        paths.registry_snapshot.write_text(json.dumps(registry.models_config(), indent=2))
    else:
        paths = existing_paths

    # Estimate cost up front; if dry_run, return immediately with empty manifest
    estimate = estimate_cost(specs, prompt)
    cap = max_run_usd if max_run_usd is not None else registry.default_max_run_usd()
    if estimate > cap:
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            wall_ms=0,
            partial=True,
            partial_reason=f"estimated cost ${estimate:.2f} exceeds cap ${cap:.2f}",
            blinded=blinded,
        )
    if dry_run:
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=[],
            cost_usd=0.0,
            wall_ms=0,
            partial=True,
            partial_reason=f"dry_run: estimated cost ${estimate:.4f}",
            blinded=blinded,
        )

    # Build slugs + prompts
    slugs = [_make_slug(s, i, blinded) for i, s in enumerate(specs)]
    per_prompts = [
        _build_per_slug_prompt(prompt, registry.resolve_stance(s.stance)) for s in specs
    ]

    start = time.time()
    tasks = [
        _call_one(spec, slug, per_prompt, paths)
        for spec, slug, per_prompt in zip(specs, slugs, per_prompts, strict=True)
    ]
    manifest = await asyncio.gather(*tasks)
    wall_ms = int((time.time() - start) * 1000)

    # If blinded, scrub model_id from the manifest (kept in registry_snapshot for audit)
    if blinded:
        for m in manifest:
            m.model_id = None

    cost_total = sum((m.cost_usd or 0.0) for m in manifest)
    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=manifest,
        cost_usd=cost_total,
        wall_ms=wall_ms,
        partial=False,
        blinded=blinded,
    )
    artifacts.write_manifest(paths, handle.model_dump())
    return handle
