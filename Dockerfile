# consult-mcp-server — multi-model panel orchestration MCP server.
#
# stdio transport. The container expects API keys via env (ANTHROPIC_API_KEY,
# OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY). Mount a volume at
# /root/.consult to persist runs across container starts.
FROM python:3.12-slim AS build

WORKDIR /app

# Copy the package metadata + sources. We keep tests + workflows out of the
# image — they're not needed at runtime and only pad the layer.
COPY pyproject.toml README.md ./
COPY consult/ ./consult/

# Install the engine + MCP adapter. `--no-cache-dir` shrinks the layer.
RUN pip install --no-cache-dir ".[mcp]"

FROM python:3.12-slim AS runtime
WORKDIR /app

# Copy the installed site-packages and console scripts from the build stage.
# This keeps the runtime image free of pip's caches and intermediate state.
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/consult-mcp /usr/local/bin/consult-mcp
COPY --from=build /usr/local/bin/consult-ledger /usr/local/bin/consult-ledger
COPY --from=build /usr/local/bin/consult-view /usr/local/bin/consult-view

# Run directory: per-run artifacts land under /root/.consult/runs/. Mount a
# host volume here if you want runs to survive container restarts.
VOLUME ["/root/.consult"]

ENTRYPOINT ["consult-mcp"]
