"""Attachment rendering — turn the schema's `attachments` array into a
single markdown block to splice onto the user prompt.

Three input shapes (see `ATTACHMENT_SCHEMA_ITEMS`):
- bare string → absolute file path
- `{path, label?, kind?}` → labelled file attachment
- `{source: "git_diff", base, head, repo_path?, label?}` → server-side
  resolved git diff

Extracted from server.py so the rendering can be tested in isolation and
so server.py stays focused on MCP wiring + tool orchestration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import sources
from .envutil import env_int

logger = logging.getLogger(__name__)

# Hard cap on a single attachment's size. A panel call against a multi-GB
# log file would burn token budget and may also OOM the server before
# LiteLLM even rejects the payload. The cap is generous enough for whole
# small codebases (~5MB ≈ 1M tokens at 5 bytes/token) but small enough to
# fail fast on operator typos like attaching `/var/log/system.log`.
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def _max_bytes() -> int:
    """Read the size cap at call time so test monkeypatching works."""
    return env_int("CONSULT_ATTACHMENT_MAX_BYTES", _DEFAULT_MAX_BYTES)


def _read_text_safely(path: Path) -> str:
    """Read a file as text, surfacing every failure mode inline.

    Path containment is enforced via `sources.validate_under_trusted_roots`:
    the file must live under `CONSULT_TRUSTED_REPO_ROOTS` (or, by default,
    the process's CWD). A symlink that points outside the trusted set
    fails closed because we resolve the path before checking.

    Catches:
    - containment failure: raises `ValueError` from the validator
    - `OSError`: missing / permission denied / non-regular file
    - `UnicodeDecodeError`: binary blob attached by mistake — without this,
      it bubbles up as INTERNAL_ERROR and the whole tool call fails. The
      whole point of the inline `[ERROR: ...]` shape is that a broken
      attachment shouldn't take down the panel.
    - size cap: stat() first so we don't slurp gigabytes into RAM.

    Returns the file content or raises `ValueError` with a one-line reason
    the caller turns into an `[ERROR: ...]` block.
    """
    resolved = sources.validate_under_trusted_roots(path)
    try:
        size = resolved.stat().st_size
    except OSError as e:
        raise ValueError(f"{type(e).__name__}: {e}") from e
    cap = _max_bytes()
    if size > cap:
        raise ValueError(f"attachment {size} bytes exceeds CONSULT_ATTACHMENT_MAX_BYTES={cap}")
    try:
        return resolved.read_text()
    except UnicodeDecodeError as e:
        raise ValueError(f"not a text file (UnicodeDecodeError at byte {e.start})") from e
    except OSError as e:
        raise ValueError(f"{type(e).__name__}: {e}") from e


# JSON Schema fragment for one attachment entry. Used by every tool that
# accepts attachments (panel, refine, consult, sequence) — keeping the
# definition in one place prevents drift across schemas.
ATTACHMENT_SCHEMA_ITEMS = {
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
                        "Repo dir. Must resolve under CONSULT_TRUSTED_REPO_ROOTS (defaults to cwd)."
                    ),
                },
                "label": {"type": "string"},
            },
        },
    ],
}

_KIND_FENCE_LANG = {
    "diff": "diff",
    "design_doc": "markdown",
    "source": "",
    "data": "",
    "text": "",
    "git_diff": "diff",
}


def render_attachment(item: Any) -> str:
    """Render a single attachment spec into a markdown block.

    Failure modes are surfaced inline (e.g. unreadable file produces
    `[ERROR: ...]`) rather than raising — a single broken attachment must
    not abort the whole tool call. The caller still sees the partial
    prompt and the panel can reason about whatever did make it through.
    """
    if isinstance(item, str):
        path = item
        label = None
        kind = None
        try:
            content = _read_text_safely(Path(path))
        except ValueError as e:
            return f"\n# {path}\n[ERROR: {e}]\n"
    elif isinstance(item, dict) and item.get("source") == "git_diff":
        base = item.get("base")
        head = item.get("head")
        # The MCP schema marks base/head required, but library callers can
        # hand us anything; without this guard a missing ref reached the
        # subprocess resolver as None.
        if not isinstance(base, str) or not isinstance(head, str):
            return f"\n[ERROR: malformed git_diff spec (base/head must be strings): {item!r}]\n"
        repo_path = item.get("repo_path")
        label = item.get("label") or f"git_diff[{base}..{head}]"
        kind = "git_diff"
        path = f"git_diff:{base}..{head}"
        try:
            content = sources.resolve_git_diff(base, head, repo_path)
        except (ValueError, RuntimeError) as e:
            return f"\n## {label}\n[ERROR: {e}]\n"
        # git_diff bypasses `_read_text_safely` because it doesn't come from
        # a file path — apply the same size cap here so a thousand-commit
        # diff can't OOM the server or blow the LLM's token budget. Encoded
        # length matches the byte-level convention used in `_read_text_safely`.
        cap = _max_bytes()
        if len(content.encode("utf-8", errors="ignore")) > cap:
            return (
                f"\n## {label}\n"
                f"[ERROR: diff {len(content)} chars exceeds "
                f"CONSULT_ATTACHMENT_MAX_BYTES={cap}]\n"
            )
        # An empty diff (base..head resolve to no changes — e.g. a review run
        # whose branch has no commit yet, so HEAD == base) would otherwise
        # render as a blank ```diff fence. Reviewers then can't tell "nothing
        # changed" from "the prompt forgot the diff" and rubber-stamp a verdict
        # off the surrounding prose (issue #76). Surface it as a directive
        # marker instead. The check runs AFTER the size cap so a whitespace-only
        # blob is already bounded before `.strip()` touches it. No fence, so the
        # block parser/trimmer ignores it, same as the [ERROR: ...] cases above.
        # Scoped to git_diff; empty file attachments are tracked separately.
        if not content.strip():
            return (
                f"\n## {label}\n"
                f"[WARNING: git diff {base}..{head} produced no output. There are "
                f"no changes between these refs to review. Do not infer a review "
                f"or verdict from the surrounding prompt text; report that the diff "
                f"is empty and there is nothing to review.]\n"
            )
    elif isinstance(item, dict) and item.get("path"):
        path = item["path"]
        label = item.get("label")
        kind = item.get("kind")
        try:
            content = _read_text_safely(Path(path))
        except ValueError as e:
            return f"\n# {path}\n[ERROR: {e}]\n"
    else:
        return f"\n[ERROR: malformed attachment spec: {item!r}]\n"

    header = f"## {label}: {path}" if label else f"# {path}"
    # An empty or whitespace-only file renders a directive marker, not a blank
    # ```fence, the same rubber-stamp footgun the git_diff branch above guards
    # against (#76), now generalized to file attachments (#79). The git_diff
    # branch already returned on empty content above, so anything empty here came
    # from a file branch; a bare check with no `kind` guard means a {path} dict a
    # non-schema caller tags kind="git_diff" still warns rather than rendering a
    # blank ```diff fence. The wording stays file-specific: an empty file is a
    # valid artifact state (e.g. __init__.py), unlike an empty diff, so it must
    # not say "nothing to review". Unfenced, so the block parser and persister
    # skip it, same as the [ERROR: ...] markers. `_read_text_safely` size-capped
    # `content` before this `.strip()`, so a whitespace-only blob is bounded.
    if not content.strip():
        return (
            f"\n{header}\n"
            f"[WARNING: attachment {path} is empty or contains only whitespace. "
            f"Report that it is empty; do not infer its contents from the "
            f"surrounding prompt text.]\n"
        )
    fence_lang = _KIND_FENCE_LANG.get(kind or "", "")
    return f"\n{header}\n```{fence_lang}\n{content}\n```\n"


def inline_attachments(prompt: str, attachments: list | None) -> str:
    """Splice rendered attachments onto the user prompt under a separator.

    No-op when `attachments` is falsy — callers don't have to branch on
    presence at every call site.
    """
    if not attachments:
        return prompt
    parts = [prompt, ATTACHMENT_SEPARATOR]
    for item in attachments:
        parts.append(render_attachment(item))
    return "".join(parts)


# --------------------------------------------------------------------------
# Parsing & persistence — used by runner to (a) write each inlined block to
# disk as a resource and (b) trim by dropping whole blocks (replaced with a
# stub that references the resource URI) instead of head+tail-slicing through
# the middle of a code file.
# --------------------------------------------------------------------------

ATTACHMENT_SEPARATOR = "\n\n--- ATTACHMENTS ---\n"

# Block shape produced by `render_attachment`:
#   \n# <path>\n```<lang>\n<content>\n```\n
#   \n## <label>: <path>\n```<lang>\n<content>\n```\n
#   \n## <label-only>\n```<lang>\n<content>\n```\n   (git_diff)
# The non-greedy `(.*?)` between fences terminates at the first closing
# triple-backtick. Source files rarely contain literal triple-backticks;
# attached markdown could trip this, but a slightly-short block is a
# softer failure mode than mis-parsing the whole prompt.
#
# The `[ERROR: ...]` and `[WARNING: ...]` markers (unreadable file, empty
# diff, empty file) carry no fence on purpose, so this regex skips them and
# `persist_inlined_attachments` writes no artifact for them. That's intended:
# a failed or empty attachment has no content worth persisting, and a reader
# learns its state from the marker in the prompt, not from a 0-byte file.
_BLOCK_RE = re.compile(
    r"(?P<header>^#{1,2} [^\n]+)\n```(?P<lang>[^\n]*)\n(?P<content>.*?)\n```",
    re.DOTALL | re.MULTILINE,
)
_HEADER_LABEL_PATH_RE = re.compile(r"^## (?P<label>[^:]+): (?P<path>.+)$")
_HEADER_PATH_ONLY_RE = re.compile(r"^# (?P<path>.+)$")
_HEADER_LABEL_ONLY_RE = re.compile(r"^## (?P<label>.+)$")
_NAME_SANITISE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class InlinedBlock:
    """A parsed attachment block in an inlined prompt.

    `start`/`end` are character offsets into the prompt covering the
    *whole* block (from the leading header line through the closing
    fence). Replacing prompt[start:end] with a stub is how the trimmer
    drops a block without disturbing the surrounding prose.
    """

    header: str
    label: str | None
    path: str | None
    lang: str
    content: str
    start: int
    end: int


def _parse_header(header: str) -> tuple[str | None, str | None]:
    """Decompose a block header into (label, path).

    Three shapes from `render_attachment`:
      `# <path>`              → (None, path)
      `## <label>: <path>`    → (label, path)
      `## <label-only>`       → (label, None)   (git_diff)
    """
    m = _HEADER_LABEL_PATH_RE.match(header)
    if m:
        return m.group("label").strip(), m.group("path").strip()
    m = _HEADER_PATH_ONLY_RE.match(header)
    if m:
        return None, m.group("path").strip()
    m = _HEADER_LABEL_ONLY_RE.match(header)
    if m:
        return m.group("label").strip(), None
    return None, None


def extract_inlined_blocks(prompt: str) -> list[InlinedBlock]:
    """Find every attachment block that `inline_attachments` would have
    produced. Returns an empty list when the `--- ATTACHMENTS ---`
    separator isn't present (the prompt has no inlined attachments to
    enumerate).

    Blocks before the separator are ignored — those would be triple-fenced
    code in the user's prose, not attachments. Without this guard, a
    user pasting ```py blocks in their question would mis-classify them
    as attachments and the trimmer could drop genuine question content.
    """
    sep_idx = prompt.find(ATTACHMENT_SEPARATOR)
    if sep_idx < 0:
        return []
    region_start = sep_idx + len(ATTACHMENT_SEPARATOR)
    blocks: list[InlinedBlock] = []
    for m in _BLOCK_RE.finditer(prompt, region_start):
        label, path = _parse_header(m.group("header"))
        blocks.append(
            InlinedBlock(
                header=m.group("header"),
                label=label,
                path=path,
                lang=m.group("lang"),
                content=m.group("content"),
                start=m.start(),
                end=m.end(),
            )
        )
    return blocks


def safe_attachment_name(block: InlinedBlock, used: set[str]) -> str:
    """Filename-safe identifier for an inlined block. Prefers the file's
    basename; falls back to a sanitised label (git_diff case). On
    collision within `used`, appends `-N` until unique. Always passes
    `artifacts._SAFE_ID_RE` so the on-disk write can't traverse.
    """
    raw: str
    if block.path:
        raw = Path(block.path).name or block.path
    elif block.label:
        raw = block.label
    else:
        raw = "attachment"
    cleaned = _NAME_SANITISE_RE.sub("-", raw).strip("-.")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"x-{cleaned}" if cleaned else "attachment"
    name = cleaned
    n = 1
    while name in used:
        name = f"{cleaned}-{n}"
        n += 1
    used.add(name)
    return name


def persist_inlined_attachments(paths: Any, prompt: str) -> dict[int, str]:
    """Write each inlined attachment's content to `paths.attachments/<name>`
    and return `{block_start_offset: resource_uri}`.

    Idempotent enough: re-writes files if called twice with the same
    prompt. Used by the runner at run-init so panellist trim stubs can
    reference a resource URI that actually resolves, and so the human
    reader of the report can read the original source even after the
    trim stub replaced it in the panellist prompt.

    `paths` is duck-typed for `artifacts.RunPaths` to avoid a
    `attachments` → `artifacts` cycle. It must expose `.attachments`
    (dir Path) and `.attachment_resource_uri(name) -> str`.
    """
    blocks = extract_inlined_blocks(prompt)
    if not blocks:
        return {}
    paths.attachments.mkdir(exist_ok=True)
    used: set[str] = set()
    out: dict[int, str] = {}
    for block in blocks:
        name = safe_attachment_name(block, used)
        target = paths.attachments / name
        try:
            target.write_text(block.content)
        except OSError as e:
            logger.warning(
                "could not persist attachment block %r → %s: %s",
                block.path or block.label,
                target,
                e,
            )
            continue
        out[block.start] = paths.attachment_resource_uri(name)
    return out
