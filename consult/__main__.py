"""Entry point: `consult-mcp` starts the stdio MCP server."""

from __future__ import annotations

import asyncio

from .runner import configure_litellm
from .server import main


def cli() -> None:
    configure_litellm()
    asyncio.run(main())


if __name__ == "__main__":
    cli()
