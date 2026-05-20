# consult-mcp-server

A multi-model panel-orchestration MCP server. Built for agentic CLIs (Claude Code, Codex, etc.) that want a structured second opinion from a panel of LLMs without bloating the parent's context window.

## Why it exists

Existing options each fall short:

- **PAL `consensus`** serialises model calls — sum of latencies.
- **`multi_mcp`** parallelises but bakes server-side synthesis with no escape hatch.
- **`ai-council-mcp`** has clever blinded anonymisation but small surface.
- **`llm-consortium`** has a refinement loop but isn't MCP.
- **Skill-only fan-outs** assembled by the LLM via bash are brittle (token budget traps, endpoint drift, key handling).

`consult` keeps the parallel fan-out, kills the brittleness, and exposes responses as MCP Resources so panel breadth doesn't cost parent context.

## Surface

Four tools:

| Tool | When |
|---|---|
| `panel(prompt, models, blinded?, ...)` → manifest | Parent wants to synthesise itself from rich capsules |
| `synthesise(run_id, by_model?)` → markdown | Collapse a prior run via a flagship |
| `consult(prompt, tier?, roles?)` → synthesis + manifest | "Just give me the answer" |
| `refine(prompt, models, arbiter, threshold, max_rounds)` → result + verdicts | Iterate to consensus; arbiter scores sufficiency per round |

Plus MCP resources at `consult://runs/<id>/responses/<slug>` for direct body access.

## The manifest capsule

Each panellist returns ~200 structured tokens, extracted by a cheap model (`claude-haiku` by default):

```json
{
  "slug": "panelist-alpha",
  "model_id": "anthropic/claude-opus-4-7",
  "status": "OK",
  "persona": "contrarian",
  "capsule": {
    "position": "supports B with caveats",
    "recommendation": "Use approach B with fallback to A",
    "key_points": ["...", "..."],
    "unique_claims": ["Only model to flag cold-start regression"],
    "caveats": ["Assumes >100 RPS steady-state"],
    "confidence": 0.85
  },
  "resource_uri": "consult://runs/abc/responses/alpha",
  "latency_ms": 3420,
  "cost_usd": 0.04
}
```

The parent can synthesise from this alone in most cases. Bodies are fetched via MCP resource read only when depth is needed.

## Install

```bash
git clone <this repo> ~/Projects/personal/consult-mcp-server
cd ~/Projects/personal/consult-mcp-server
uv venv
uv pip install -e ".[dev]"
cp .env.example .env  # then fill in your provider keys
```

Provider keys (set whichever you'll use):

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
```

## Claude Code config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (or the equivalent for your client):

```json
{
  "mcpServers": {
    "consult": {
      "command": "/Users/you/Projects/personal/consult-mcp-server/.venv/bin/consult-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "OPENROUTER_API_KEY": "...",
        "GEMINI_API_KEY": "...",
        "OPENAI_API_KEY": "..."
      }
    }
  }
}
```

A copy lives in `claude_config_example.json`.

## Quick start

```text
> consult: tier=standard, prompt="Is async asyncio fanout the right move here?"
```

The hero tool runs ~8 panellists in parallel, drops the synthesiser from the panel, extracts capsules, and synthesises via Gemini 3.1 Pro.

```text
> panel: models=[{model:"gpt-5-pro"},{model:"claude-opus",stance:"contrarian"},{model:"gemini-pro"}], prompt="..."
```

Lower-level — returns the manifest, you synthesise yourself.

## What's in scope for v1

- 4 tools: `panel`, `synthesise`, `consult`, `refine`
- LiteLLM provider layer (no custom HTTP)
- Rich manifest capsules
- MCP Resources for bodies
- Anonymous Alpha/Beta blinding
- Per-run cost cap + `dry_run`
- Stance/persona injection
- Parametric `usable()` viability check
- Status enum: OK / TRUNCATED / MALFORMED / EMPTY / REFUSED / CONTENT_FILTERED / RATE_LIMITED / TIMEOUT / ERROR

## Cut from v1 (defer to v2)

- `sequence` (chained models)
- Auto-retry on token exhaustion (cost bomb — return TRUNCATED, let the parent decide)
- Streaming progress notifications
- Daily cost ledger
- Continuation IDs for per-panellist follow-up
- `model:count` multi-instance
- Startup registry validation against provider `/models`

## Layout

```
consult/
  server.py        # MCP wiring
  __main__.py      # entry point
  runner.py        # asyncio.gather + LiteLLM fanout
  capsule.py       # post-fanout structured extraction
  synth.py         # flagship synthesiser pass
  refine.py        # iterative arbiter-driven loop (max 3 rounds)
  registry.py      # models.json + stances.json loader
  artifacts.py     # ~/.consult/runs/<id>/ layout + resource URIs
  status.py        # LiteLLM response → Status
  types.py         # Pydantic models
config/
  models.json      # default model registry
  stances.json     # default persona prompts
tests/
  test_smoke.py    # offline + live tests
```

## Testing

```bash
uv run pytest -v
```

Live tests are gated on API keys; they skip cleanly when absent.

## License

MIT.
