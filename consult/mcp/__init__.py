"""MCP adapter for the consult engine.

Everything that imports the `mcp` SDK lives under this subpackage. The
engine (`consult.runner`, `consult.refine`, `consult.sequence`,
`consult.synth`, `consult.orchestrate`) is MCP-free and can be driven
directly from Python without installing the `mcp` extra.

`pip install consult-mcp-server[mcp]` installs the MCP SDK; the
`consult-mcp` console script enters `consult.mcp.__main__:cli`.
"""
