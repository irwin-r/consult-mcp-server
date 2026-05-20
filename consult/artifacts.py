"""On-disk run layout + MCP resource URI conventions.

Directory tree under ~/.consult/runs/<run_id>/:
  prompt.txt               # the base prompt sent to every panellist
  manifest.json            # the full RunHandle serialised
  registry_snapshot.json   # frozen registry at run time
  prompts/<slug>.txt       # per-slug prompt (with stance prefix)
  responses/<slug>.json    # raw provider response (LiteLLM ModelResponse dump)
  responses/<slug>.txt     # extracted body text (for resource serving)
  capsules/<slug>.json     # extracted capsule
  arbiters/round-<n>.json  # refine: per-round ArbiterVerdict (refine runs only)
  synth_input.txt          # synth: full input sent to the synthesiser
  synthesis.md             # synth: synthesis output (or error sentinel)

Resource URI scheme: consult://runs/<id>/responses/<slug>

Note: refine round-suffixes slugs as `<base>.r<n>`, so the URI grammar is
unchanged but slugs may contain dots and an `.r<digit>` suffix.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path


def runs_root() -> Path:
    env = os.environ.get("CONSULT_RUNS_DIR")
    base = Path(os.path.expanduser(env)) if env else Path.home() / ".consult" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{random.randint(1000, 99999)}"


@dataclass
class RunPaths:
    run_id: str
    root: Path

    @property
    def prompts(self) -> Path:
        return self.root / "prompts"

    @property
    def responses(self) -> Path:
        return self.root / "responses"

    @property
    def capsules(self) -> Path:
        return self.root / "capsules"

    @property
    def arbiters(self) -> Path:
        return self.root / "arbiters"

    @property
    def manifest_json(self) -> Path:
        return self.root / "manifest.json"

    @property
    def prompt_txt(self) -> Path:
        return self.root / "prompt.txt"

    @property
    def registry_snapshot(self) -> Path:
        return self.root / "registry_snapshot.json"

    def response_text(self, slug: str) -> Path:
        return self.responses / f"{slug}.txt"

    def response_raw(self, slug: str) -> Path:
        return self.responses / f"{slug}.json"

    def prompt_for(self, slug: str) -> Path:
        return self.prompts / f"{slug}.txt"

    def capsule_for(self, slug: str) -> Path:
        return self.capsules / f"{slug}.json"

    def resource_uri(self, slug: str) -> str:
        return f"consult://runs/{self.run_id}/responses/{slug}"

    def arbiter_for(self, round_num: int) -> Path:
        return self.arbiters / f"round-{round_num}.json"


def create_run() -> RunPaths:
    rid = new_run_id()
    root = runs_root() / rid
    root.mkdir(parents=True, exist_ok=False)
    paths = RunPaths(run_id=rid, root=root)
    paths.prompts.mkdir()
    paths.responses.mkdir()
    paths.capsules.mkdir()
    paths.arbiters.mkdir()
    return paths


def load_run(run_id: str) -> RunPaths:
    root = runs_root() / run_id
    if not root.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return RunPaths(run_id=run_id, root=root)


def write_manifest(paths: RunPaths, payload: dict) -> None:
    paths.manifest_json.write_text(json.dumps(payload, indent=2, default=str))


def augment_manifest(paths: RunPaths, **fields: object) -> None:
    """Merge fields into the existing manifest.json.

    Used after synth/refine to persist metadata that wasn't available at
    fanout time — `synthesiser` is the canonical example: the consult
    handler picks the synth model after the panel returns, but the manifest
    is written during `runner.fanout` before that decision exists.

    Single-writer per run dir (each run_id is unique), so no locking. Best
    effort: if the manifest is missing or malformed, no-op rather than
    raise — augmentation is a UX nicety for downstream viewers, not a
    correctness boundary.
    """
    if not paths.manifest_json.exists():
        return
    try:
        current = json.loads(paths.manifest_json.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(current, dict):
        return
    current.update(fields)
    paths.manifest_json.write_text(json.dumps(current, indent=2, default=str))


def parse_resource_uri(uri: str) -> tuple[str, str]:
    """Parse `consult://runs/<id>/responses/<slug>` → (run_id, slug)."""
    if not uri.startswith("consult://runs/"):
        raise ValueError(f"Not a consult resource URI: {uri}")
    rest = uri[len("consult://runs/") :]
    parts = rest.split("/")
    if len(parts) != 3 or parts[1] != "responses":
        raise ValueError(f"Malformed consult URI: {uri}")
    return parts[0], parts[2]
