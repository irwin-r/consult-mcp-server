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

import os
from pathlib import Path
from typing import Any

from . import sources

# Hard cap on a single attachment's size. A panel call against a multi-GB
# log file would burn token budget and may also OOM the server before
# LiteLLM even rejects the payload. The cap is generous enough for whole
# small codebases (~5MB ≈ 1M tokens at 5 bytes/token) but small enough to
# fail fast on operator typos like attaching `/var/log/system.log`.
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def _max_bytes() -> int:
    """Read the size cap at call time so test monkeypatching works."""
    return int(os.environ.get("CONSULT_ATTACHMENT_MAX_BYTES", _DEFAULT_MAX_BYTES))


def _read_text_safely(path: Path) -> str:
    """Read a file as text, surfacing every failure mode inline.

    Catches:
    - `OSError`: missing / permission denied / non-regular file
    - `UnicodeDecodeError`: binary blob attached by mistake — without this,
      it bubbles up as INTERNAL_ERROR and the whole tool call fails. The
      whole point of the inline `[ERROR: ...]` shape is that a broken
      attachment shouldn't take down the panel.
    - size cap: stat() first so we don't slurp gigabytes into RAM.

    Returns the file content or raises `ValueError` with a one-line reason
    the caller turns into an `[ERROR: ...]` block.
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ValueError(f"{type(e).__name__}: {e}") from e
    cap = _max_bytes()
    if size > cap:
        raise ValueError(
            f"attachment {size} bytes exceeds CONSULT_ATTACHMENT_MAX_BYTES={cap}"
        )
    try:
        return path.read_text()
    except UnicodeDecodeError as e:
        raise ValueError(
            f"not a text file (UnicodeDecodeError at byte {e.start})"
        ) from e
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
                        "Repo dir. Must resolve under CONSULT_TRUSTED_REPO_ROOTS "
                        "(defaults to cwd)."
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

    fence_lang = _KIND_FENCE_LANG.get(kind or "", "")
    header = f"## {label}: {path}" if label else f"# {path}"
    return f"\n{header}\n```{fence_lang}\n{content}\n```\n"


def inline_attachments(prompt: str, attachments: list | None) -> str:
    """Splice rendered attachments onto the user prompt under a separator.

    No-op when `attachments` is falsy — callers don't have to branch on
    presence at every call site.
    """
    if not attachments:
        return prompt
    parts = [prompt, "\n\n--- ATTACHMENTS ---\n"]
    for item in attachments:
        parts.append(render_attachment(item))
    return "".join(parts)
