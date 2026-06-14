# consult-mcp-server

> **Get a second opinion from a parallel panel of LLMs — without bloating your agent's context window.**

[![CI](https://github.com/irwin-r/consult-mcp-server/actions/workflows/tests.yml/badge.svg)](https://github.com/irwin-r/consult-mcp-server/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/consult-mcp-server.svg)](https://pypi.org/project/consult-mcp-server/)
[![Python](https://img.shields.io/pypi/pyversions/consult-mcp-server.svg)](https://pypi.org/project/consult-mcp-server/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Smithery](https://smithery.ai/badge/consult-mcp-server)](https://smithery.ai/server/consult-mcp-server)

`consult` is an MCP server that lets your agent (Claude Desktop, Cursor,
Claude Code, etc.) fan a single prompt out to many LLMs in parallel, then
return either the synthesised answer or a manifest of structured
~200-token capsules — so panel breadth doesn't cost parent-context tokens.

```
┌────────────┐    consult tool call     ┌──────────────────┐    parallel    ┌──────────┐
│ Your agent │ ───────────────────────▶ │  consult-mcp     │ ─────────────▶ │ Claude   │
│ (Claude    │   "what's your take?"    │  (this server)   │                │ GPT      │
│  Desktop / │ ◀─────────────────────── │                  │ ◀───────────── │ Gemini   │
│  Cursor /  │  synthesis + manifest    │  capsules ~200t  │   capsules     │ Grok     │
│  …)        │                          │  + resources     │                │ DeepSeek │
└────────────┘                          └──────────────────┘                │ …        │
                                                                            └──────────┘
```

## Why this exists

If your agent already calls `claude` once, you might wonder why you'd want to
ask 8 more models the same question. Three reasons:

1. **One pass, many perspectives.** Different families catch different things.
   Anthropic finds different bugs than OpenAI; Gemini calls out different
   risks; DeepSeek often surfaces the contrarian take.
2. **Cheap structured second opinion.** The manifest's per-panellist capsule
   is ~200 tokens — your agent can synthesise it in-band without paying for
   another flagship round-trip.
3. **No context-window bloat.** Full panellist bodies live as MCP resources
   at `consult://runs/<id>/responses/<slug>`; your agent only fetches them
   when it needs depth.

Alternatives fall short: PAL `consensus` serialises calls (sum of latencies);
`multi_mcp` parallelises but no escape hatch from server-side synth;
skill-only fan-outs assembled by the LLM via bash are brittle (token traps,
key handling, endpoint drift).

---

## Install

### Claude Desktop

> Claude Desktop does **not** inherit your shell's `PATH` or environment
> variables — you must give it the absolute path to `consult-mcp` and
> declare API keys inside the `env` block.

Tip: run `consult-doctor --config` after install to print a ready-to-paste
JSON block populated with the absolute binary path and whichever keys are
present in your shell environment.

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "consult": {
      "command": "/Users/you/.local/bin/uvx",
      "args": ["--from", "consult-mcp-server[mcp]", "consult-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-…",
        "OPENAI_API_KEY": "sk-…",
        "GEMINI_API_KEY": "AIza…",
        "OPENROUTER_API_KEY": "sk-or-…"
      }
    }
  }
}
```

Restart Claude Desktop, then ask: *"use the consult tool to ask 3 models
which Python package manager I should use."*

### Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "consult": {
      "command": "/Users/you/.local/bin/uvx",
      "args": ["--from", "consult-mcp-server[mcp]", "consult-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-…",
        "OPENAI_API_KEY": "sk-…"
      }
    }
  }
}
```

Same caveat as Claude Desktop: absolute path to `uvx`, env keys in the block.

### Claude Code CLI

```sh
claude mcp add consult -- uvx --from "consult-mcp-server[mcp]" consult-mcp
```

The CLI inherits your shell env, so the keys you already have in `.env` /
your shell rc will be visible.

### Docker

```sh
docker run -i --rm \
  -e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e GEMINI_API_KEY -e OPENROUTER_API_KEY \
  -v ~/.consult:/home/consult/.consult \
  ghcr.io/irwin-r/consult-mcp-server:latest
```

Stdio in / stdio out, just like the local binary. Image published per release
to GHCR (multi-stage Python 3.12-slim base, ~150MB).

### Smithery

```
https://smithery.ai/server/consult-mcp-server
```

Smithery's hosted UI prompts for keys; the same `smithery.yaml` config-schema
applies.

### From source (development)

```sh
git clone https://github.com/irwin-r/consult-mcp-server
cd consult-mcp-server
uv venv
uv pip install -e ".[dev]"
cp .env.example .env   # fill in keys
uv run pytest -v
```

### Verify the install

```sh
consult-doctor          # offline: config + paths + key presence
consult-doctor --ping   # also fires a 1-token call per provider (~$0.0001)
consult-doctor --config # print copy-paste-ready MCP client JSON
```

---

## The tools

| Tool | What it does | Use when |
|---|---|---|
| **`consult`** | Parallel panel + server-side synthesis. Hero. | "Just give me the answer." |
| **`panel`** | Parallel panel, returns raw manifest (no synth). | You want to synthesise yourself. |
| **`refine`** | Iterative consortium with arbiter scoring (≤3 rounds). | High-stakes; disagreement-heavy. |
| **`synthesise`** | Re-collapse an existing run via a flagship model. | Different rubric/synthesiser on a prior `run_id`. |
| **`sequence`** | Chained multi-step where step N depends on N-1. | Decompose-then-answer; plan-then-execute. Opt-in: set `CONSULT_ENABLE_SEQUENCE=1`. |

Tool descriptions are intentionally written as **prompts for the calling
agent** (verb-first, explicit "use when…/don't use for…") so the agent
reliably picks the right one without you having to spell it out.

---

## Tiers & cost

Aliases are `<family>-<tier>` — version-neutral. The registry maps each alias
to the current best model; the resolved LiteLLM ID is captured per run in
`registry_snapshot.json` for reproducibility.

| Tier | Models | Typical run cost | Use |
|---|---|---|---|
| `nano` (3) | claude-haiku, gemini-flash, gpt-nano | < $0.01 | smoke tests / trivia |
| `quick` (5) | claude-haiku, gemini-pro, grok, qwen-max, kimi | ~$0.05 | snap second opinions |
| `standard` (9) | opus, sonnet, gpt, gemini-pro, grok, qwen-max, kimi, glm, llama | $0.30–0.60 | normal decisions |
| `deep` (14) | standard + gpt-pro, mistral, deepseek, mimo, sonar-pro | $0.50–1.00 | high-stakes, includes web search |
| `code` (5) | opus, gpt-codex, gpt-mini, gemini-pro, deepseek | $0.20–0.40 | code-heavy questions |
| `review` (6) | opus, gpt-codex, gpt-pro, gemini-pro, deepseek, grok | $0.30–0.60 | PR / code review |

A per-run cap (`max_run_usd`, default `$5.00`) refuses panels whose estimated
cost exceeds the limit before any provider is called.

### Stance diversity & calibration

By default `consult` rotates a spread of decision-useful stances across the
panel (staff engineer, contrarian, security, product, long-term, cost) so
identical prompts stop producing correlated errors. Pass explicit `roles`, or
`diverse_stances=false`, for an all-neutral panel. Every result also carries a
`calibration` block: the blinding/shuffle disclosure, the disagreement score,
panel health with per-status spend, and the family / privacy-tier / stance
spread. When disagreement runs high, the synthesis is held to a two-sided
account rather than a single recommendation.

### Custom models and overrides

Drop a `~/.consult/models.json` (and/or `stances.json`) containing only what
differs; it deep-merges over the packaged config. Add a model by declaring
just its entry, override a single field of a packaged model by naming only
that field, or remove a packaged entry by setting it to JSON `null`:

```json
{
  "models": {
    "my-local": { "litellm_id": "ollama/llama3", "provider": "ollama" },
    "deepseek": null
  }
}
```

Config is cached for the process lifetime; restart the server after edits.

---

## The manifest capsule

Each panellist returns a ~200-token structured extract (decision shape shown
below; `review` and `research` kinds also supported):

```json
{
  "slug": "claude-opus-1",
  "model_id": "anthropic/claude-opus-4-7",
  "status": "OK",
  "capsule": {
    "kind": "decision",
    "position": "supports B with caveats",
    "recommendation": "Use B with fallback to A",
    "key_points": ["…"],
    "unique_claims": ["Only model to flag cold-start regression"],
    "caveats": ["Assumes >100 RPS steady-state"],
    "confidence": 0.85
  },
  "resource_uri": "consult://runs/abc/responses/claude-opus-1",
  "latency_ms": 3420,
  "cost_usd": 0.04
}
```

Your agent can synthesise from this alone in most cases. Read the full body
via the resource URI only when depth is needed.

---

## Quickstart

After installing, from any connected agent:

```text
> consult: prompt="Polars vs DuckDB for a 10GB Parquet timeseries?", tier="code"
```

Returns `{run_id, synthesis, manifest, cost_usd, synthesiser}`. The synthesis
is markdown, ready to drop into your conversation.

For iterative consensus:

```text
> refine:
    prompt="Should we migrate from REST to gRPC for the internal mesh?",
    models=[{model:"claude-opus"},{model:"gpt-pro"},{model:"gemini-pro"},{model:"deepseek"}],
    threshold=0.85
```

For chained reasoning:

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

---

## End-to-end walkthrough

A full tour. Assumes the install above and at least one provider key in
`.env`.

### 1. Smoke-test the install (no API spend)

```sh
.venv/bin/python -c "
import asyncio
from consult import panel, ModelSpec
async def go():
    h = await panel('hello', [ModelSpec(model='claude-haiku')], dry_run=True)
    print('partial:', h.partial, '| reason:', h.partial_reason)
asyncio.run(go())
"
# partial: True | reason: dry_run: estimated cost $0.0001
```

### 2. First real consult (~$0.20 on the `code` tier)

```text
> consult: prompt="Polars vs DuckDB for 10GB Parquet timeseries?", tier=code
```

### 3. Inspect a panellist's full body

```text
> read resource: consult://runs/<run_id>/responses/claude-opus-1
```

### 4. Tail progress in real time

```sh
tail -f ~/.consult/runs/<run_id>/_progress.log
```

Agents that send a `progressToken` get the same events as
`notifications/progress`.

### 5. Follow-up via `continuation_id`

```text
> refine: prompt="OK now what about Iceberg vs Delta on top of that?",
          continuation_id="<prior run_id>",
          models=[{model:"claude-opus"},{model:"deepseek"}]
```

The prior run's synthesis is prepended as "Prior consultation summary".

### 6. Stochastic averaging with `model:N`

```text
> panel: models=[{model:"claude-haiku:3"},{model:"gpt-mini:3"}], prompt="…"
```

Six panellists total — three runs each of two cheap models.

A single panel is capped at 64 panellists (the `:N` counts plus any bare
specs, summed). Set `CONSULT_MAX_PANEL_SIZE` to raise or lower it; a request
over the cap is rejected before the run starts. The default leaves room for
the largest built-in tier (`deep`, 14) and generous averaging.

### 7. Check today's spend

```sh
consult-ledger today
# {"date":"2026-05-21","total_usd":2.36,"total_known":false,"runs":[…]}
```

`total_known: false` means at least one panellist had pricing missing from
the LiteLLM table.

### 8. View a run as a rich HTML page

```sh
consult-view <run_id>          # writes ~/.consult/runs/<run_id>/feed.html
consult-view <run_id> --open   # also opens in default browser
```

Self-contained HTML — header pills, prompt, synthesis (markdown), per-round
arbiter verdicts (refine), per-panellist cards with capsule + full body, and
a chronological timeline from `_progress.log`. No external assets, no JS.

---

## Driving the engine without MCP

The engine package (`consult.*`) is MCP-free and reusable as a library:

```python
from consult import consult, panel, refine, ModelSpec

# Hero tool
result = await consult("question?", tier="standard")
print(result.synthesis, result.cost_usd)

# Lower-level
handle = await panel("question?", [ModelSpec(model="claude-opus"), ModelSpec(model="gpt-pro")])

# Iterative
verdict = await refine(
    "tough decision?",
    [ModelSpec(model="claude-opus"), ModelSpec(model="deepseek")],
    threshold=0.85,
)
```

Swap the URI scheme for a non-MCP transport:

```python
from consult import artifacts
artifacts.set_resource_uri_formatter(
    lambda run_id, slug: f"https://api.example.com/runs/{run_id}/{slug}"
)
```

---

## Telemetry

Install the `otel` extra and point it at a collector to emit `gen_ai.*`
OpenTelemetry spans, one per panellist call:

```bash
pip install "consult-mcp-server[otel]"
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Without the extra (or without the endpoint set) the telemetry helpers are
no-ops at zero overhead.

## Retention

Run artefacts under the runs directory accumulate indefinitely. `consult-gc`
prunes them by age and/or count:

```bash
consult-gc --max-age-days 30           # drop runs older than 30 days
consult-gc --max-count 500             # keep only the newest 500 runs
consult-gc --dry-run --max-count 500   # show what would go, delete nothing
```

Defaults can be set with `CONSULT_RUNS_RETENTION_DAYS` and `CONSULT_RUNS_MAX`;
the CLI flags override them.

## Security

Read [`SECURITY.md`](SECURITY.md) for the full threat model. Short version:

- **`git_diff` attachments** must resolve under `CONSULT_TRUSTED_REPO_ROOTS`
  (defaults to the server's CWD). Symlinks are resolved before the check;
  escape attempts fail closed.
- **File attachments** are containment-checked only when
  `CONSULT_TRUSTED_REPO_ROOTS` is set. When it's unset, any path the server
  process can read is accepted, on the reasoning that the calling agent
  already has filesystem access of its own. If the server runs with broader
  filesystem access than the calling agent (Docker, a shared host, a
  remote deployment), set `CONSULT_TRUSTED_REPO_ROOTS` so attachments are
  confined to the directories you intend.
- **Run artefacts are `chmod 0o700`** — per-run prompts (often containing
  pasted credentials or code) are not world-readable on shared hosts.
- **`git diff` runs with global/system git config neutralised** so a
  malicious `.gitattributes` filter can't execute.
- **LiteLLM exception strings are scrubbed** for `sk-…`, `AIza…`,
  `Bearer …`, `x-api-key:` and similar before they reach the manifest or
  the progress log. This covers provider *error* text only. The raw prompt
  and panellist responses are stored unredacted inside the `0o700` run
  directory, so treat that directory as sensitive.

### Privacy note

The model registry tags each entry with a `privacy_tier`:

- `first_party` — direct API to Anthropic / OpenAI / Google.
- `aggregator` — routed via OpenRouter (Grok, Kimi, Qwen, DeepSeek, Llama,
  Mistral, GLM, MiMo, Sonar-Pro).

Mixing tiers in one panel broadcasts the **same prompt** to providers with
**different data-retention policies**. For prompts containing sensitive
material, prefer `tier="standard"` (mostly first-party) over `tier="deep"`
(heavily aggregator-routed).

---

## Repo layout

```
consult/                # ENGINE — no mcp.* imports
  runner/               # async fan-out package (facade in __init__)
    transport.py        #   LiteLLM retry/streaming/Responses adapter
    fit.py              #   context-budget fitting + attachment trims
    specs.py            #   model:N expansion + slug grammar
    costs.py            #   panel cost estimation
    fanout.py           #   _call_one, slow-tail dropout, fanout()
  capsule.py            # post-fanout structured extraction
  synth.py              # flagship synthesiser
  refine.py             # arbiter-driven loop (max 3 rounds) + continuation
  sequence.py           # chained multi-step
  orchestrate.py        # consult() hero
  ledger.py             # daily cost ledger (consult-ledger)
  viewer.py             # static HTML run renderer (consult-view)
  doctor.py             # diagnostic CLI (consult-doctor)
  registry.py           # models.json + stances.json loader
  artifacts.py          # ~/.consult/runs/<id>/ layout + URI formatter
  attachments.py        # file/diff inlining + trusted-roots enforcement
  sources.py            # git_diff resolver (hardened subprocess)
  context.py            # per-run bundle + blinding
  progress.py           # typed ProgressEvent union
  status.py             # LiteLLM response → Status
  types.py              # Pydantic models (StrictModel base)
  jsonparse.py          # tolerant JSON extraction from model output
  cost.py               # CostMeter — spend roll-up with unknown propagation
  envutil.py            # tolerant numeric env parsing
  exceptions.py         # typed exception taxonomy
  provider_caps.py      # per-provider capability flags (temperature, etc.)
  telemetry.py          # optional OpenTelemetry spans (otel extra)
  peer_rank.py          # peer-ranking aggregation (opt-in, off by default)
  voting.py             # medoid selection over model:N capsules (opt-in)
  strategies.py         # pluggable refine round-to-round hook (default only)
  task_store.py         # in-process registry for long-running task mode
  mcp/                  # MCP ADAPTER — only thing that imports mcp.*
    server.py, handlers.py, schemas.py, errors.py, __main__.py
  config/
    models.json         # registry with privacy_tier annotations
    stances.json        # persona prompts
    rubrics/            # synthesis rubric markdown (consensus, code_review, ...)
  schemas/
    capsule_v2.json     # JSON Schema for the capsule wire shape (reference)
tests/                  # pytest (offline + live, gated on keys)
.github/workflows/      # CI: ruff + pytest on Py 3.11/3.12/3.13
FRICTION.md             # internal dogfooding log (kept for transparency)
SECURITY.md             # threat model + disclosure path
CONTRIBUTING.md         # dev setup + style
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Issues and PRs welcome; please open
an issue first for non-trivial changes so we can agree on shape.

## License

MIT. See [`LICENSE`](LICENSE).
