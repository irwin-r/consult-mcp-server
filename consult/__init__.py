"""consult — multi-model panel orchestration.

The engine (`consult.*`) is MCP-free and directly usable as a library:

    from consult import orchestrate, runner, refine, sequence, synth
    result = await orchestrate.consult("question?", tier="standard")

The MCP adapter lives at `consult.mcp.*` and is an optional install
(`pip install consult-mcp-server[mcp]`) that exposes the same engine
over the MCP stdio protocol via the `consult-mcp` console script.
"""

__version__ = "0.1.0"
