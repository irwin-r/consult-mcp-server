"""CLI-as-panellist transport.

A registry entry with `provider: "cli"` and a `cli_command` array lets
consult include an interactive CLI (e.g. `gemini`, `codex`, `claude`) as
a panel member alongside API panellists. The motivation: agentic CLIs
have native tool access, 1M-token context windows (Gemini CLI), and use
the user's existing browser-auth session — features that are awkward or
expensive to replicate via the API layer.

Inspired by:
- pal-mcp-server's `clink` tool (CLI-to-CLI bridge)
- multi_mcp's `CLIExecutor` (CLI dispatch alongside API dispatch)

Security note: invoking a CLI runs arbitrary executable code with the
parent process's permissions. This module passes the registry-configured
`cli_command` array directly to `asyncio.create_subprocess_exec` — no
shell interpolation, no argv injection. The PROMPT goes to stdin (also
not argv), so no quoting concerns. But the *cli_command* itself comes
from the registry; trusting a registry override (`~/.consult/models.json`)
is equivalent to trusting any local code on the user's machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


async def call_cli(
    cli_command: list[str],
    prompt: str,
    *,
    timeout: float,
    extra_env: dict[str, str] | None = None,
) -> Any:
    """Run a CLI panellist via subprocess; return a LiteLLM-shape response.

    Contract: the CLI must accept the prompt on STDIN and emit the
    response on STDOUT. Exit code != 0 raises RuntimeError with stderr
    snippet attached so `_call_one`'s manifest-friendly error formatter
    can carry it cleanly.

    Returns a `SimpleNamespace` with the subset of LiteLLM's
    `ModelResponse` shape that `runner.classify()` and downstream code
    actually read:

      resp.choices[0].message.content   — the body text
      resp.choices[0].finish_reason     — "stop"
      resp.usage.prompt_tokens          — None (CLIs don't report)
      resp.usage.completion_tokens      — None

    `cost_usd` is the caller's responsibility (CLI panellists are
    free at the per-call level — the user-CLI auth covers usage).
    """
    if not cli_command:
        raise ValueError("CLI panellist invoked with empty cli_command")

    proc = await asyncio.create_subprocess_exec(
        *cli_command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_compose_env(extra_env),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=timeout,
        )
    except TimeoutError:
        # Kill the child and let the caller raise — `_call_one`'s outer
        # `except TimeoutError` already handles the status update.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except TimeoutError:  # pragma: no cover — pathological
            logger.warning("cli %r did not exit within 1s of SIGKILL", cli_command[0])
        raise

    if proc.returncode != 0:
        # Surface stderr (clipped) so the manifest error is debuggable.
        # The full stderr lives in the per-panellist body via the caller
        # (we pass through the empty stdout) — but for the error string
        # the head is the most useful.
        err_snippet = stderr.decode("utf-8", "replace")[:500]
        raise RuntimeError(f"CLI {cli_command[0]!r} exited {proc.returncode}: {err_snippet}")

    body = stdout.decode("utf-8", "replace")
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=body),
                finish_reason="stop",
            ),
        ],
        usage=SimpleNamespace(
            prompt_tokens=None,
            completion_tokens=None,
        ),
    )


def _compose_env(extra: dict[str, str] | None) -> dict[str, str] | None:
    """Merge process env with caller-supplied overrides.

    Returning `None` lets `create_subprocess_exec` inherit the parent's
    environment (the usual case). When overrides are provided we merge
    on top of `os.environ` so the CLI still finds PATH, HOME, etc.
    """
    if not extra:
        return None
    import os as _os  # late import — keeps the module fast for the no-extra path

    merged = dict(_os.environ)
    merged.update(extra)
    return merged
