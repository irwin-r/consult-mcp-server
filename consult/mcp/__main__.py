"""Entry point: `consult-mcp` starts the stdio MCP server.

Lives under `consult.mcp` because it imports the MCP SDK transitively via
`.server`. The engine package (`consult.*`) has no `__main__` — library
consumers drive the engine directly via `runner.fanout`, `refine.refine`,
`sequence.sequence`, `synth.synthesise`, or `orchestrate.consult`.

Startup is intentionally I/O-free: no network calls happen until a tool
is actually invoked. This matters for the Smithery registry scanner,
which spins up the server briefly to enumerate tools — any blocking
network at startup causes the scan to time out and the server to be
marked broken.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Force LiteLLM to use its bundled model_prices table at import time. Without
# this, recent LiteLLM versions may attempt a GitHub fetch to refresh the
# price table when `import litellm` runs — that's a network call during
# startup that breaks Smithery's sandboxed tool-enumeration scan and adds
# latency to every CLI invocation. The bundled table is good enough for
# cost lookup; users wanting fresh prices can set the env var to "False"
# explicitly to opt back in.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from .. import __version__  # noqa: E402
from ..runner import configure_litellm  # noqa: E402
from .server import main  # noqa: E402


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="consult-mcp",
        description="Multi-model panel consultation MCP server (stdio transport).",
    )
    parser.add_argument("--version", action="version", version=f"consult-mcp {__version__}")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a self-check (config + provider key presence) and exit. "
        "Does not call any model. For a full diagnostic with live provider "
        "pings, use `consult-doctor`.",
    )
    args = parser.parse_args()

    if args.check:
        from ..doctor import quick_check

        sys.exit(quick_check())

    configure_litellm()
    asyncio.run(main())


if __name__ == "__main__":
    cli()
