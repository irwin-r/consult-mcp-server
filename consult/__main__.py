"""Entry point: `consult-mcp` starts the stdio MCP server."""

from __future__ import annotations

import asyncio

from .server import main


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
