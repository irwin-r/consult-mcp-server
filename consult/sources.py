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

from .envutil import env_float
from .exceptions import PathTrustError

logger = logging.getLogger(__name__)

# Allows `~` and `^` for relative refs like `HEAD~1` / `HEAD^`. Neither is
# a shell metacharacter when subprocess.run uses shell=False + arg list, so
# they're safe to permit. The leading character cannot be `-` — otherwise
# a "ref" like `--no-index` becomes `git diff --no-index..HEAD`, smuggling
# a git option through as a positional arg.
_REF_RE = re.compile(r"^[A-Za-z0-9._/+~^][A-Za-z0-9._/+~^-]*$")
_GIT_TIMEOUT_DEFAULT_S = 30.0


def _git_timeout_s() -> float:
    """Read at call time: the old module-level `float(os.environ[...])`
    crashed the whole import on a malformed value."""
    return env_float("CONSULT_GIT_TIMEOUT_S", _GIT_TIMEOUT_DEFAULT_S)


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


def validate_under_trusted_roots(path: str | Path, *, strict: bool = False) -> Path:
    """Resolve `path` and confirm it sits under one of the trusted roots.

    Uses `Path.resolve()` (symlinks resolved) and `os.path.commonpath()`
    against each trusted root so a symlink pointing outside the root fails
    closed.

    Three modes:

    * `strict=True` — always enforce containment. Caller is `git_diff`,
      which then spawns a subprocess; allowing arbitrary paths there is
      a higher-impact threat than for read-only file attachments.

    * `strict=False` and `CONSULT_TRUSTED_REPO_ROOTS` is set — enforce
      containment. The operator has explicitly opted into restricted mode
      for file attachments; honour it.

    * `strict=False` and `CONSULT_TRUSTED_REPO_ROOTS` is unset — only
      verify the path exists and is readable, then return the resolved
      path. Rationale: the calling agent already has full filesystem
      access via its own tools; refusing to read files the agent
      explicitly attached is friction without much added security.

    Raises `PathTrustError` (a `ValueError` subclass) on containment
    failure, `ValueError` on a missing/unreadable path.
    """
    p = Path(path).expanduser()
    try:
        resolved = p.resolve(strict=True)
    except (OSError, FileNotFoundError) as e:
        raise ValueError(f"path {str(path)!r} does not exist or is unreadable: {e}") from e
    env_set = bool(os.environ.get("CONSULT_TRUSTED_REPO_ROOTS"))
    if not strict and not env_set:
        return resolved
    roots = _trusted_roots()
    resolved_str = str(resolved)
    for trusted in roots:
        try:
            common = os.path.commonpath([resolved_str, str(trusted)])
        except ValueError:
            # Different drives on Windows — definitively not under this root.
            continue
        if common == str(trusted):
            return resolved
    raise PathTrustError(
        f"path {str(path)!r} is not under any CONSULT_TRUSTED_REPO_ROOTS entry "
        f"(trusted: {[str(p) for p in roots]})"
    )


def _validate_repo_path(repo_path: str | None) -> Path:
    # git_diff containment policy:
    # - repo_path unset → use CWD (the agent is operating from its own repo).
    # - repo_path set → must resolve under the trusted roots. By default
    #   the only trusted root is CWD, so a malicious prompt asking to
    #   diff `/etc` still gets rejected unless the operator opted in by
    #   setting CONSULT_TRUSTED_REPO_ROOTS. strict=True is forced here
    #   because git diff spawns a subprocess — higher impact than a
    #   read-only file attachment.
    if repo_path is None:
        return Path.cwd().resolve()
    return validate_under_trusted_roots(repo_path, strict=True)


def resolve_git_diff(base: str, head: str, repo_path: str | None = None) -> str:
    """Run `git diff base..head` in the repo and return the diff text.

    Raises `ValueError` for invalid refs or untrusted repo paths.
    Raises `RuntimeError` if git fails or times out.
    """
    _validate_ref(base, field="base")
    _validate_ref(head, field="head")
    repo = _validate_repo_path(repo_path)
    git = shutil.which("git") or "git"
    # Neutralise global / system git config files. A malicious repo could
    # define a `.gitattributes` filter or a `core.fsmonitor` hook that
    # executes on `git diff`; pointing GIT_CONFIG_GLOBAL/SYSTEM at /dev/null
    # blocks the user-level and system-level configs from injecting hooks.
    # GIT_TERMINAL_PROMPT=0 stops git from blocking on a credential prompt
    # when a ref accidentally references a remote.
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    timeout_s = _git_timeout_s()
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
            timeout=timeout_s,
            check=True,
            shell=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git diff timed out after {timeout_s}s for {base}..{head}") from None
    except subprocess.CalledProcessError as e:
        # stderr length capped — a buggy git wrapper can spew megabytes.
        raise RuntimeError(f"git diff failed (exit {e.returncode}): {e.stderr.strip()[:1024]}") from e
    return result.stdout
