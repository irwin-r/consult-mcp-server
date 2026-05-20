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

Five tools:

| Tool | When |
|---|---|
| `panel(prompt, models, blinded?, ...)` → manifest | Parent wants to synthesise itself from rich capsules |
| `synthesise(run_id, by_model?)` → markdown | Collapse a prior run via a flagship |
| `consult(prompt, tier?, roles?)` → synthesis + manifest | "Just give me the answer" |
| `refine(prompt, models, arbiter, threshold, max_rounds, continuation_id?)` → result + verdicts | Iterate to consensus; arbiter scores sufficiency per round. `continuation_id` chains a follow-up onto a prior run. |
| `sequence(prompts, models)` → per-step results + final synth | Chain a list of prompts where each step's synthesis is prepended to the next. Multi-stage research, plan-then-execute. |

Plus MCP resources at `consult://runs/<id>/responses/<slug>` for direct body access, and live progress via `notifications/progress` (when the client sends a `progressToken`) or a tailable JSONL log at `<run>/_progress.log`.

Multi-instance panellists via `model:N` syntax — e.g. `claude-haiku:3` requests 3 parallel instances of the same model for stochastic averaging.

## Models & tiers

Aliases are `<family>-<tier>` — version-neutral. The registry maps each alias to the current best model in that slot; the exact resolved ID is captured per run in `registry_snapshot.json` for reproducibility. Pass a raw LiteLLM ID (e.g. `openai/gpt-5.5-pro`) to bypass the registry.

| Tier | Models | Use |
|---|---|---|
| `nano` (3) | `claude-haiku`, `gemini-flash`, `gpt-nano` | sub-$0.05 panels for smoke tests / trivia |
| `quick` (5) | `claude-haiku`, `gemini-pro`, `grok`, `qwen-max`, `kimi` | ~30s snap second opinions |
| `standard` (10) | `claude-opus`, `claude-sonnet`, `gpt-pro`, `gpt`, `gemini-pro`, `grok`, `qwen-max`, `kimi`, `glm`, `llama` | normal decisions |
| `deep` (14) | standard + `mistral`, `deepseek`, `mimo`, `sonar-pro` | high-stakes; includes Perplexity for web search |
| `code` (5) | `claude-opus`, `gpt-codex`, `gpt-mini`, `gemini-pro`, `deepseek` | code-heavy questions |

Specialist single-model aliases also available: `gpt-codex` (latest OpenAI codex), `sonar-pro` (web search), `gpt-mini`/`gpt-nano` (fast/ultra-cheap general-purpose).

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
uv pip install -e ".[dev]"   # full install: engine + mcp adapter + dev deps
# or, for library-only use (no MCP SDK pulled in):
# uv pip install -e .
# or, engine + MCP adapter without dev deps:
# uv pip install -e ".[mcp]"
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
> panel: models=[{model:"gpt-pro"},{model:"claude-opus",stance:"contrarian"},{model:"gemini-pro"}], prompt="..."
```

Lower-level — returns the manifest, you synthesise yourself.

## End-to-end walkthrough

A complete tour through the v1 surface. Assumes the install above is done and at least one provider key is in `.env`.

### 1. Smoke-test the install (no API spend)

A dry-run verifies config + cost-estimation without any model calls:

```bash
.venv/bin/python -c "
import asyncio
from consult.runner import fanout
from consult.types import ModelSpec
async def go():
    h = await fanout('hello', [ModelSpec(model='claude-haiku')], dry_run=True)
    print('partial:', h.partial, '| reason:', h.partial_reason)
asyncio.run(go())
"
# partial: True | reason: dry_run: estimated cost $0.0001
```

### 2. First real consult (~$0.10–0.20 on the `code` tier)

From any MCP client connected to consult:

```text
> consult: prompt="Polars vs DuckDB for 10GB Parquet timeseries?", tier=code
```

Runs the panel in parallel, extracts ~200-token capsules, and synthesises. Returns `{run_id, synthesis, manifest, cost_usd, synthesiser, ...}`.

### 3. Inspect a panellist's full body

The manifest has resource URIs for every panellist:

```text
> read resource: consult://runs/<run_id>/responses/claude-opus
```

### 4. Tail progress in real time

While a long refine runs, in a second terminal:

```bash
tail -f ~/.consult/runs/<run_id>/_progress.log
# {"ts":"...","kind":"panellist","slug":"claude-opus","status":"OK","latency_ms":35420}
# {"ts":"...","kind":"capsule","slug":"claude-opus"}
```

MCP clients that send a `progressToken` get the same events as `notifications/progress` — no log-tailing required.

### 5. Follow up via `continuation_id`

```text
> refine: prompt="OK now what about Iceberg vs Delta on top of that?", continuation_id="<prior run_id>", models=[{model:"claude-opus"},{model:"deepseek"}]
```

The prior run's synthesis is prepended as "Prior consultation summary" context so the next panel knows where the discussion has been.

### 6. Multi-step research with `sequence`

```text
> sequence:
    prompts=[
      "Decompose 'how should we scale our event pipeline?' into 4 sub-questions",
      "Answer sub-question 1: throughput requirements",
      "Answer sub-question 2: ordering guarantees",
      "Synthesise the final recommendation across the prior steps"
    ],
    models=[{model:"claude-opus"},{model:"gpt-pro"}]
```

Each step's synthesis feeds the next step's prompt. Returns per-step run_ids + the final synthesis.

### 7. Stochastic averaging with `model:N`

```text
> panel: models=[{model:"claude-haiku:3"},{model:"gpt-mini:3"}], prompt="..."
```

Six panellists total — three runs each of two cheap models. Useful for measuring response variance on prompts where temperature matters.

### 8. Check today's spend

```bash
.venv/bin/consult-ledger today
# {"date":"2026-05-20","total_usd":2.36,"total_known":false,"runs":[...]}
```

`total_known: false` means at least one panellist had pricing missing from the LiteLLM table — the displayed total is a lower bound. Pass any `YYYY-MM-DD` to ledger past days.

### 9. View a run as a rich HTML page

```bash
.venv/bin/consult-view <run_id>          # writes ~/.consult/runs/<run_id>/feed.html, prints the path
.venv/bin/consult-view <run_id> --open   # also opens it in the default browser
```

Renders the entire run as one self-contained HTML file — header with cost / wall-time / status pills, prompt, synthesis (markdown), per-round arbiter verdicts (for `refine`), per-panellist cards with capsule + full body, and a chronological timeline derived from `_progress.log`. No external assets, no JavaScript, light/dark via `prefers-color-scheme`. Regenerable: `feed.html` is a pure derivation of the on-disk artifacts.

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

## Landed since v1

- `sequence` tool (chained multi-step consultations with shared context)
- MCP `notifications/progress` + JSONL `_progress.log` fallback
- Daily cost ledger (`consult-ledger` CLI + `consult/ledger.py`)
- Static HTML run viewer (`consult-view <run_id>` → `feed.html`)
- Continuation IDs for `refine` (chain a follow-up onto a prior run)
- `model:N` multi-instance syntax
- GitHub Actions CI on Python 3.11/3.12/3.13
- Anthropic prompt caching (`cache_control: ephemeral`) on fanout + synth
- Schema-enforced capsule extraction via `response_format=Capsule`
- LiteLLM stderr-noise suppression (`suppress_debug_info=True`)
- `RunResult.synthesiser`, `RefineResult.partial_reason` / `continuation_of` surfaced to callers
- Per-panellist `Status.ERROR` isolation when a single alias is unknown

## Still deferred

- Auto-retry on token exhaustion (cost bomb — return TRUNCATED, let the parent decide)
- Startup registry validation against provider `/models`
- PyPI publish (needs explicit user authorisation)

## Layout

The package splits engine (`consult.*`) from MCP adapter (`consult.mcp.*`).
Only the adapter imports the `mcp` SDK; everything under `consult/` is
pure async-Python + Pydantic and is directly callable from any consumer
(CLI, HTTP, library use, tests). `pip install consult-mcp-server` gives
you the engine; `pip install consult-mcp-server[mcp]` adds the adapter
and the `consult-mcp` stdio server.

```
consult/                # ENGINE — no mcp.* imports
  runner.py             # asyncio.gather + LiteLLM fanout + progress log
  capsule.py            # post-fanout structured extraction
  synth.py              # flagship synthesiser pass
  refine.py             # iterative arbiter-driven loop (max 3 rounds) + continuation
  sequence.py           # chained multi-step consultations
  orchestrate.py        # consult() hero: fanout → capsule → synth, typed
  ledger.py             # daily cost ledger (consult-ledger entry point)
  viewer.py             # static HTML run renderer (consult-view entry point)
  registry.py           # models.json + stances.json loader
  artifacts.py          # ~/.consult/runs/<id>/ layout + injectable URI formatter
  attachments.py        # file/diff inlining
  context.py            # per-run context bundle + blinding scrub
  progress.py           # typed ProgressEvent union (consumer-agnostic)
  status.py             # LiteLLM response → Status
  types.py              # Pydantic models
  mcp/                  # MCP ADAPTER — only thing that imports mcp.*
    server.py           # MCP wiring (panel, synthesise, consult, refine, sequence)
    handlers.py         # MCP args-dict adapter — calls engine via typed kwargs
    schemas.py          # MCP tool JSON Schemas
    errors.py           # MCP error-envelope wire shape
    __main__.py         # consult-mcp entry point
  config/
    models.json         # default model registry
    stances.json        # default persona prompts
tests/
  test_smoke.py         # offline + live tests
.github/
  workflows/
    tests.yml           # CI: pytest on 3.11 / 3.12 / 3.13
FRICTION.md             # dogfooding log
```

### Driving the engine directly

```python
from consult import orchestrate
result = await orchestrate.consult("question?", tier="standard")
print(result.synthesis, result.cost_usd)

# Or compose primitives directly:
from consult import runner, capsule, synth
handle = await runner.fanout(prompt, specs)
handle = await capsule.annotate(handle)
synth_result = await synth.synthesise(handle.run_id)
```

A non-MCP consumer can swap the URI scheme on the manifest:

```python
from consult import artifacts
artifacts.set_resource_uri_formatter(
    lambda run_id, slug: f"https://api.example.com/runs/{run_id}/{slug}"
)
```

## Testing

```bash
uv run pytest -v
```

Live tests are gated on API keys; they skip cleanly when absent.

## License

MIT.
