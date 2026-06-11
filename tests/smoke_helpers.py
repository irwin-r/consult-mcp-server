"""Shared builders for the per-module test files split out of the old test_smoke.py."""

from __future__ import annotations

import json
import os

from consult import artifacts
from consult.types import Capsule, ManifestEntry, RunHandle, Status

HAVE_KEYS = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))


def _make_run_dir(
    tmp_path,
    run_id: str,
    *,
    entries: list[dict],
    bodies: dict[str, str] | None = None,
    synth: str | None = None,
    arbiters: list[dict] | None = None,
    progress_lines: list[dict] | None = None,
    prompt: str = "test prompt",
    extras: dict | None = None,
):
    """Materialise a minimal-but-realistic run dir on disk for viewer tests.

    Shared across the render_run cases so each test stays focused on one
    invariant rather than rebuilding the artifact tree.
    """
    root = tmp_path / run_id
    (root / "responses").mkdir(parents=True)
    (root / "capsules").mkdir()
    (root / "arbiters").mkdir()
    (root / "prompts").mkdir()
    (root / "prompt.txt").write_text(prompt)
    manifest = {
        "run_id": run_id,
        "artifacts_dir": str(root),
        "manifest": entries,
        "cost_usd": sum((e.get("cost_usd") or 0) for e in entries),
        "cost_known": all(e.get("cost_known", True) for e in entries),
        "wall_ms": 1234,
        "partial": False,
        "blinded": False,
        **(extras or {}),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for slug, body in (bodies or {}).items():
        (root / "responses" / f"{slug}.txt").write_text(body)
    if synth is not None:
        (root / "synthesis.md").write_text(synth)
    for v in arbiters or []:
        (root / "arbiters" / f"round-{v['round']}.json").write_text(json.dumps(v))
    if progress_lines:
        (root / "_progress.log").write_text("\n".join(json.dumps(p) for p in progress_lines) + "\n")
    return root


def _refine_fake_fanout(fanout_calls, *, cost=0.001):
    """A minimal `runner.fanout` stand-in for refine loop tests: records the
    prompt, writes a body file per slug, and returns an all-OK manifest."""

    async def fake_fanout(prompt, specs, **kwargs):
        fanout_calls.append(prompt)
        paths = kwargs.get("existing_paths") or artifacts.create_run()
        manifest = []
        for spec in specs:
            slug = spec.slug or spec.model
            bp = paths.root / "responses" / f"{slug}.txt"
            bp.parent.mkdir(parents=True, exist_ok=True)
            bp.write_text(f"{slug} body")
            manifest.append(
                ManifestEntry(
                    slug=slug,
                    model_id="x/a",
                    status=Status.OK,
                    resource_uri=f"consult://x/{slug}",
                    body_path=str(bp),
                    latency_ms=10,
                    cost_usd=cost,
                    cost_known=True,
                    capsule=Capsule(position=f"{slug} pos"),
                )
            )
        return RunHandle(
            run_id=paths.run_id,
            artifacts_dir=str(paths.root),
            manifest=manifest,
            cost_usd=cost * len(specs),
            cost_known=True,
            wall_ms=10,
        )

    return fake_fanout


async def _refine_fake_synth(*args, **kwargs):
    from consult import synth as _synth_mod

    return _synth_mod.SynthResult(text="final synth")


async def _refine_noop_annotate(handle, **kwargs):
    return handle
