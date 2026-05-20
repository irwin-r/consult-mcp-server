"""Source resolvers for non-file attachments.

Currently supports git diff resolution: given `{source: "git_diff", base,
head, repo_path?}`, the server runs `git diff base..head` itself and
inlines the result. Means the parent agent never has to materialise the
diff into its own context window to pass it across the MCP boundary.

# Security model

Untrusted strings (refs, paths) flow from the MCP caller to a subprocess
exec. Three layers of defence:

1. **Ref validation** — refs must match the `_REF_RE` regex. This blocks
   every shell metacharacter (semicolon, pipe, ampersand, dollar, backtick,
   newline, etc) AND rejects a leading `-` so a ref cannot smuggle a git
   option through as a positional arg (e.g. `base="--no-index"`).
   Well-aligned with `git check-ref-format` (which is even stricter; we
   defer the final word to git itself).
2. **Repo-path containment** — the requested `repo_path` must resolve
   under one of the directories named in `CONSULT_TRUSTED_REPO_ROOTS`
   (colon-separated). Default: the current working directory only.
3. **`subprocess.run([...], shell=False)`** with an arg list — even if
   a metachar got through, it would be passed as a single git arg and
   rejected by git's own ref parsing.

This is intentionally conservative. If a user wants to diff a repo
outside CWD, they opt in by setting the env var. Better to fail loudly
than to silently exec on arbitrary paths.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Allows `~` and `^` for relative refs like `HEAD~1` / `HEAD^`. Neither is
# a shell metacharacter when subprocess.run uses shell=False + arg list, so
# they're safe to permit. The leading character cannot be `-` — otherwise
# a "ref" like `--no-index` becomes `git diff --no-index..HEAD`, smuggling
# a git option through as a positional arg.
_REF_RE = re.compile(r"^[A-Za-z0-9._/+~^][A-Za-z0-9._/+~^-]*$")
_DEFAULT_GIT_TIMEOUT_S = float(os.environ.get("CONSULT_GIT_TIMEOUT_S", 30.0))


def _trusted_roots() -> list[Path]:
    env = os.environ.get("CONSULT_TRUSTED_REPO_ROOTS")
    if env:
        return [Path(p).expanduser().resolve() for p in env.split(":") if p.strip()]
    # Default to CWD only — keeps drive-by diffs of unrelated repos off
    # the table unless the user opts in.
    return [Path.cwd().resolve()]


def _validate_ref(ref: str, *, field: str) -> None:
    if not _REF_RE.fullmatch(ref):
        raise ValueError(
            f"invalid {field} {ref!r}: must match {_REF_RE.pattern} "
            "(no shell metacharacters; leading '-' rejected to block git "
            "option injection)"
        )


def _validate_repo_path(repo_path: str | None) -> Path:
    if repo_path is None:
        return Path.cwd().resolve()
    resolved = Path(repo_path).expanduser().resolve()
    for trusted in _trusted_roots():
        try:
            resolved.relative_to(trusted)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"repo_path {repo_path!r} is not under any CONSULT_TRUSTED_REPO_ROOTS entry "
        f"(trusted: {[str(p) for p in _trusted_roots()]})"
    )


def resolve_git_diff(
    base: str, head: str, repo_path: str | None = None
) -> str:
    """Run `git diff base..head` in the repo and return the diff text.

    Raises `ValueError` for invalid refs or untrusted repo paths.
    Raises `RuntimeError` if git fails or times out.
    """
    _validate_ref(base, field="base")
    _validate_ref(head, field="head")
    repo = _validate_repo_path(repo_path)
    git = shutil.which("git") or "git"
    try:
        # `--` after the diff range forces git to stop interpreting any
        # subsequent arg as an option. Belt-and-braces with the ref regex's
        # leading-dash rejection: even if a future regex change re-admits
        # a `-`, the `--` separator keeps the ref from being interpreted
        # as a git option.
        result = subprocess.run(
            [git, "diff", f"{base}..{head}", "--"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_GIT_TIMEOUT_S,
            check=True,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"git diff timed out after {_DEFAULT_GIT_TIMEOUT_S}s for {base}..{head}"
        ) from None
    except subprocess.CalledProcessError as e:
        # stderr length capped — a buggy git wrapper can spew megabytes.
        raise RuntimeError(
            f"git diff failed (exit {e.returncode}): {e.stderr.strip()[:1024]}"
        ) from e
    return result.stdout
