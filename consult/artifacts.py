"""On-disk run layout + MCP resource URI conventions.

Directory tree under ~/.consult/runs/<run_id>/:
  prompt.txt                # the base prompt sent to every panellist
  manifest.json             # the full RunHandle serialised
  registry_snapshot.json    # frozen registry at run time
  prompts/<slug>.txt        # per-slug prompt (with stance prefix)
  responses/<slug>.json     # raw provider response (LiteLLM ModelResponse dump)
  responses/<slug>.txt      # extracted body text (for resource serving)
  capsules/<slug>.json      # extracted capsule
  arbiters/round-<n>.json   # refine: per-round ArbiterVerdict (refine runs only)
  attachments/<name>        # original inlined attachment text (split out so
                            # a panellist's trim-stub can reference the
                            # full source via consult://...)
  synth_input.txt           # synth: full input sent to the synthesiser
  synthesis.md              # synth: synthesis output (or error sentinel)

Resource URI scheme:
  consult://runs/<id>/responses/<slug>       # panellist body
  consult://runs/<id>/attachments/<name>     # original attachment source

Note: refine round-suffixes slugs as `<base>.r<n>`, so the URI grammar is
unchanged but slugs may contain dots and an `.r<digit>` suffix.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

# Injectable URI formatter. The default produces the `consult://` scheme
# that the MCP adapter dereferences via `read_resource`. Library or HTTP
# consumers can swap this for `http(s)://...` or a relative form via
# `set_resource_uri_formatter()`; the manifest then carries whatever URI
# their downstream tooling knows how to fetch.
#
# Stored in a `ContextVar` rather than a module-level binding so that two
# concurrent fanouts in the same process (e.g. a multi-tenant HTTP server,
# each request scoped to its own URI scheme) can each install their own
# formatter without corrupting the other. Async tasks inherit the
# formatter from their spawning context — call `set_resource_uri_formatter`
# once at request entry and it scopes to the awaitable that follows.
ResourceUriFormatter = Callable[[str, str], str]


def _default_resource_uri(run_id: str, slug: str) -> str:
    return f"consult://runs/{run_id}/responses/{slug}"


_resource_uri_formatter: ContextVar[ResourceUriFormatter] = ContextVar(
    "consult_resource_uri_formatter", default=_default_resource_uri
)


def set_resource_uri_formatter(fn: ResourceUriFormatter) -> None:
    """Override the URI formatter used by every new manifest entry in the
    current async context.

    Scoped to the calling `contextvars.Context`: tasks spawned after this
    call inherit the override; sibling contexts (other concurrent HTTP
    requests, other test cases) are unaffected. Existing on-disk manifests
    are NOT rewritten; `parse_resource_uri()` below only knows the default
    `consult://` scheme, so a custom formatter implies the consumer owns
    its own parse path too.
    """
    _resource_uri_formatter.set(fn)


def reset_resource_uri_formatter() -> None:
    """Restore the default `consult://` formatter in the current context.

    Useful for tests; production code that wants per-request scope should
    just install a new formatter at request entry and let context isolation
    do the work."""
    _resource_uri_formatter.set(_default_resource_uri)


def runs_root() -> Path:
    env = os.environ.get("CONSULT_RUNS_DIR")
    base = Path(os.path.expanduser(env)) if env else Path.home() / ".consult" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{random.randint(1000, 99999)}"


# Slugs and run IDs are interpolated directly into filesystem paths and
# resource URIs. A value like "../manifest" or "/etc/passwd" would escape
# the per-run artifact directory; refine's round suffix needs the dot, so
# we allow `.` but reject path separators and any non-printable chars. The
# leading alphanumeric anchor rejects pure-punctuation slugs like `..` that
# match the body class but still traverse. Every legitimate identifier we
# generate (`<base>-<idx>`, `<base>-<idx>.r<n>`, `YYYYMMDD-HHMMSS-<rand>`)
# starts with an alphanumeric character.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_id(value: str, kind: str) -> str:
    """Reject identifiers that could traverse out of the artifact root.

    `kind` is "slug" or "run_id" — appears in the error message so a bad
    request surfaces the offending field name. Empty strings are rejected
    along with separator/special characters.
    """
    if not isinstance(value, str) or not value or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"invalid {kind} {value!r}: must match {_SAFE_ID_RE.pattern}"
        )
    return value


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
    def attachments(self) -> Path:
        return self.root / "attachments"

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
        return self.responses / f"{_validate_id(slug, 'slug')}.txt"

    def response_raw(self, slug: str) -> Path:
        return self.responses / f"{_validate_id(slug, 'slug')}.json"

    def prompt_for(self, slug: str) -> Path:
        return self.prompts / f"{_validate_id(slug, 'slug')}.txt"

    def capsule_for(self, slug: str) -> Path:
        return self.capsules / f"{_validate_id(slug, 'slug')}.json"

    def resource_uri(self, slug: str) -> str:
        return _resource_uri_formatter.get()(self.run_id, _validate_id(slug, "slug"))

    def arbiter_for(self, round_num: int) -> Path:
        return self.arbiters / f"round-{round_num}.json"

    def attachment_path(self, name: str) -> Path:
        return self.attachments / _validate_id(name, "attachment name")

    def attachment_resource_uri(self, name: str) -> str:
        """URI for an inlined-attachment resource. Bypasses the injectable
        formatter — attachments are an internal trim-stub concern; the
        formatter override is only meaningful for panellist response URIs
        embedded in the wire-shape manifest.
        """
        return f"consult://runs/{self.run_id}/attachments/{_validate_id(name, 'attachment name')}"


def create_run() -> RunPaths:
    rid = new_run_id()
    root = runs_root() / rid
    root.mkdir(parents=True, exist_ok=False)
    paths = RunPaths(run_id=rid, root=root)
    paths.prompts.mkdir()
    paths.responses.mkdir()
    paths.capsules.mkdir()
    paths.arbiters.mkdir()
    paths.attachments.mkdir()
    return paths


def load_run(run_id: str) -> RunPaths:
    # `_validate_id` rejects path-traversal characters; a containment check
    # via `.resolve()` is the belt-and-braces guard against case-insensitive
    # FS resolution or pre-validation symlink swaps under `runs_root()`.
    _validate_id(run_id, "run_id")
    base = runs_root().resolve()
    root = (base / run_id).resolve()
    try:
        root.relative_to(base)
    except ValueError as e:
        raise ValueError(f"run_id {run_id!r} escapes runs_root") from e
    if not root.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return RunPaths(run_id=run_id, root=root)


def write_manifest(paths: RunPaths, payload: dict) -> None:
    paths.manifest_json.write_text(json.dumps(payload, indent=2, default=str))


def augment_manifest(paths: RunPaths, **fields: object) -> None:
    """Merge fields into the existing manifest.json (synchronous).

    Used after synth/refine to persist metadata that wasn't available at
    fanout time — `synthesiser` is the canonical example: the consult
    handler picks the synth model after the panel returns, but the manifest
    is written during `runner.fanout` before that decision exists.

    Single-writer per run dir (each run_id is unique), so no locking. Best
    effort: if the manifest is missing or malformed, no-op rather than
    raise — augmentation is a UX nicety for downstream viewers, not a
    correctness boundary.

    Sync by design: easy to call from non-async contexts (CLI tools,
    tests, the viewer). Async fan-out paths call `aaugment_manifest`
    below so the disk write doesn't block the event loop during
    multi-round refine or large sequence runs.
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


async def aaugment_manifest(paths: RunPaths, **fields: object) -> None:
    """Async wrapper around `augment_manifest`.

    Offloads the JSON read + write to a thread so the event loop stays
    responsive during heavy refine/sequence orchestration where
    augment_manifest is called repeatedly. Test monkeypatches still bind
    to the sync `augment_manifest` symbol — this picks up whatever's
    currently bound there, so test setup is unchanged.
    """
    import asyncio
    await asyncio.to_thread(augment_manifest, paths, **fields)


_RESOURCE_KINDS: tuple[str, ...] = ("responses", "attachments")


def parse_resource_uri(uri: str) -> tuple[str, str, str]:
    """Parse `consult://runs/<id>/<kind>/<name>` → (run_id, kind, name).

    `kind` is one of `responses` (panellist body) or `attachments`
    (original inlined attachment source). The kind makes the read path
    explicit at parse time so the MCP resource handler dispatches without
    re-validating.
    """
    if not uri.startswith("consult://runs/"):
        raise ValueError(f"Not a consult resource URI: {uri}")
    rest = uri[len("consult://runs/") :]
    parts = rest.split("/")
    if len(parts) != 3 or parts[1] not in _RESOURCE_KINDS:
        raise ValueError(f"Malformed consult URI: {uri}")
    # Validate before returning so a malicious URI can't reach disk via
    # downstream callers that forget to call `_validate_id` themselves.
    _validate_id(parts[0], "run_id")
    _validate_id(parts[2], parts[1].rstrip("s"))  # "slug" or "attachment"
    return parts[0], parts[1], parts[2]
