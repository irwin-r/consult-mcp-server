"""`consult-doctor` — environment + provider self-check.

Designed for the "Claude Desktop just shows red" failure mode. The MCP
client gives almost no signal back to the user when stdio startup goes
wrong; `consult-doctor` lets the user run a one-shot diagnostic from
their own shell that prints exactly what consult sees, what's missing,
and (with `--ping`) whether each configured provider key actually works.

Three modes:

  consult-doctor              # offline: config, paths, perms, key presence
  consult-doctor --ping       # also fire one 1-token call per provider
  consult-doctor --config     # print copy-paste-ready Claude Desktop JSON

Exit codes: 0 if everything looks healthy, 1 if any check failed (so
this can be wired into a smoke-test step or `consult-mcp --check`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__, artifacts, registry

# Provider env-var → human label mapping. Each key MUST be present in the
# environment for that provider's panellists to function. Aggregator keys
# (OpenRouter) unlock multiple providers at once; the absence message
# explains the impact in terms of which registry aliases will fail.
_PROVIDER_KEYS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "Anthropic (claude-* aliases)",
    "OPENAI_API_KEY": "OpenAI (gpt-* aliases)",
    "GEMINI_API_KEY": "Google Gemini (gemini-* aliases)",
    "OPENROUTER_API_KEY": "OpenRouter (grok / kimi / qwen / deepseek / llama / mistral / glm / mimo / sonar-pro)",
}


def _ok(msg: str) -> str:
    return f"  [OK] {msg}"


def _warn(msg: str) -> str:
    return f"  [WARN] {msg}"


def _fail(msg: str) -> str:
    return f"  [FAIL] {msg}"


def _check_paths() -> tuple[list[str], int]:
    """Verify the runs directory exists, is owned by us, and is 0o700."""
    lines: list[str] = []
    fails = 0
    try:
        root = artifacts.runs_root()
    except Exception as e:  # noqa: BLE001
        lines.append(_fail(f"runs root: {type(e).__name__}: {e}"))
        return lines, 1
    lines.append(_ok(f"runs root: {root}"))
    try:
        mode = root.stat().st_mode & 0o777
        if mode == 0o700:
            lines.append(_ok(f"runs root mode: 0o{mode:o}"))
        else:
            lines.append(
                _warn(
                    f"runs root mode: 0o{mode:o} (expected 0o700 — per-run prompts may be "
                    "world-readable; chmod 700 ~/.consult/runs)"
                )
            )
    except OSError as e:
        lines.append(_warn(f"could not stat runs root: {e}"))
    return lines, fails


def _check_keys() -> tuple[list[str], int]:
    """Report which provider keys are set. No key is necessarily a fail —
    the user might be using a subset of providers — but having none is a
    fail because nothing can run.
    """
    lines: list[str] = []
    present: list[str] = []
    for env_var, label in _PROVIDER_KEYS.items():
        if os.environ.get(env_var):
            present.append(env_var)
            lines.append(_ok(f"{env_var} present → {label}"))
        else:
            lines.append(_warn(f"{env_var} absent → {label} unavailable"))
    fails = 0
    if not present:
        lines.append(_fail("no provider keys present — no panellist can run"))
        fails = 1
    return lines, fails


def _check_registry() -> tuple[list[str], int]:
    """Validate that the model registry loads and tiers resolve."""
    lines: list[str] = []
    fails = 0
    try:
        cfg = registry.models_config()
    except Exception as e:  # noqa: BLE001
        lines.append(_fail(f"models.json failed to load: {type(e).__name__}: {e}"))
        return lines, 1
    n_models = len(cfg.get("models", {}))
    n_tiers = len(cfg.get("tiers", {}))
    lines.append(_ok(f"registry loaded: {n_models} models, {n_tiers} tiers"))
    try:
        synth = registry.default_synthesiser()
        lines.append(_ok(f"default synthesiser: {synth}"))
    except Exception as e:  # noqa: BLE001
        lines.append(_warn(f"default synthesiser missing: {e}"))
    return lines, fails


def _check_trusted_roots() -> tuple[list[str], int]:
    """Show the configured trusted roots so users can see what file paths
    and git_diff specs will be accepted.
    """
    lines: list[str] = []
    env = os.environ.get("CONSULT_TRUSTED_REPO_ROOTS")
    if env:
        roots = [p.strip() for p in env.split(":") if p.strip()]
        lines.append(_ok(f"CONSULT_TRUSTED_REPO_ROOTS: {roots}"))
        for r in roots:
            if not Path(r).expanduser().exists():
                lines.append(_warn(f"  trusted root {r!r} does not exist — paths under it will be rejected"))
    else:
        lines.append(
            _warn(
                "CONSULT_TRUSTED_REPO_ROOTS unset — file attachments and git_diff "
                f"restricted to CWD ({Path.cwd()}) only. Set this env var to a "
                "colon-separated list of directories to allow attachments from elsewhere."
            )
        )
    return lines, 0


def _check_git() -> tuple[list[str], int]:
    lines: list[str] = []
    git = shutil.which("git")
    if git:
        lines.append(_ok(f"git: {git}"))
    else:
        lines.append(_warn("git not on PATH — git_diff attachments will fail"))
    return lines, 0


async def _ping_provider(env_var: str, alias: str) -> tuple[bool, str]:
    """Fire one 1-token completion to verify the key actually works.

    Imports `litellm` lazily so `consult-doctor` (without --ping) stays
    fast and never touches the network.
    """
    if not os.environ.get(env_var):
        return False, "key not set"
    try:
        import litellm

        info = registry.resolve_model(alias)
        litellm_id = info["litellm_id"]
        await litellm.acompletion(
            model=litellm_id,
            messages=[{"role": "user", "content": "."}],
            max_completion_tokens=1,
            timeout=10,
        )
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        # Redact secrets before surfacing — same risk as runner.py.
        from .runner import _redact_secrets

        msg = _redact_secrets(f"{type(e).__name__}: {e}")
        return False, msg[:200]


async def _check_pings() -> tuple[list[str], int]:
    """Live provider ping. ~$0.0001 per provider, ~5s total."""
    lines: list[str] = []
    fails = 0
    targets = [
        ("ANTHROPIC_API_KEY", "claude-haiku"),
        ("OPENAI_API_KEY", "gpt-nano"),
        ("GEMINI_API_KEY", "gemini-flash"),
        ("OPENROUTER_API_KEY", "grok"),
    ]
    for env_var, alias in targets:
        ok, msg = await _ping_provider(env_var, alias)
        prefix = _ok if ok else (_warn if msg == "key not set" else _fail)
        lines.append(prefix(f"ping {alias}: {msg}"))
        if not ok and msg != "key not set":
            fails += 1
    return lines, fails


def _claude_desktop_config() -> dict[str, Any]:
    """Build a copy-paste-ready claude_desktop_config.json snippet.

    Uses the absolute path to the currently-running `consult-mcp` binary
    so Claude Desktop (which does NOT inherit shell PATH) finds it.
    Populates the `env` block with whichever provider keys are present in
    the current shell — Claude Desktop also does not inherit shell env,
    so each key must be set explicitly inside the config.
    """
    bin_path = shutil.which("consult-mcp") or "consult-mcp"
    env: dict[str, str] = {}
    for env_var in _PROVIDER_KEYS:
        val = os.environ.get(env_var)
        if val:
            env[env_var] = val
    if not env:
        env = {k: f"<your {k} here>" for k in _PROVIDER_KEYS}
    return {
        "mcpServers": {
            "consult": {
                "command": bin_path,
                "env": env,
            }
        }
    }


def quick_check() -> int:
    """Offline self-check. Returns 0/1 for sys.exit.

    Exposed so `consult-mcp --check` reuses the same logic. Prints to
    stdout — callers that want silent checks should redirect.
    """
    print(f"consult-doctor {__version__}")
    print()
    total_fails = 0
    for section, fn in [
        ("Paths", _check_paths),
        ("Registry", _check_registry),
        ("Trusted roots", _check_trusted_roots),
        ("Git", _check_git),
        ("Provider keys", _check_keys),
    ]:
        print(f"{section}:")
        lines, fails = fn()
        for line in lines:
            print(line)
        total_fails += fails
        print()
    if total_fails:
        print(f"FAIL: {total_fails} blocking issue(s). Fix above and rerun.")
        return 1
    print("OK: ready to serve. For a live provider ping run `consult-doctor --ping`.")
    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="consult-doctor",
        description="Diagnose a consult-mcp-server install.",
    )
    parser.add_argument("--version", action="version", version=f"consult-doctor {__version__}")
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Fire one 1-token completion per configured provider to verify keys work. "
        "Costs roughly $0.0001 in total. Implies the offline check first.",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Print a copy-paste-ready Claude Desktop / Cursor JSON snippet and exit.",
    )
    args = parser.parse_args()

    if args.config:
        print(json.dumps(_claude_desktop_config(), indent=2))
        return

    exit_code = quick_check()
    if args.ping:
        # Force LiteLLM to use the bundled price table even for the ping
        # so we don't accidentally make a metadata fetch the user didn't
        # opt into.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        print("Live provider pings (1 token each, ~$0.0001 total):")
        lines, fails = asyncio.run(_check_pings())
        for line in lines:
            print(line)
        if fails:
            exit_code = max(exit_code, 1)
            print(f"\nFAIL: {fails} provider ping(s) failed.")
        else:
            print("\nOK: all configured providers responding.")

    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
