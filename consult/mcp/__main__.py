"""Entry point: `consult-mcp` starts the stdio MCP server.

Lives under `consult.mcp` because it imports the MCP SDK transitively via
`.server`. The engine package (`consult.*`) has no `__main__` — library
consumers drive the engine directly via `runner.fanout`, `refine.refine`,
`sequence.sequence`, `synth.synthesise`, or `orchestrate.consult`.
"""

from __future__ import annotations

import asyncio

from ..runner import configure_litellm
from .server import main


def cli() -> None:
    configure_litellm()
    asyncio.run(main())


if __name__ == "__main__":
    cli()
