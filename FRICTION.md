# Friction log

Things that snag during real use of consult-mcp-server. One line per item,
dated. Add as you hit them. Severity tags: [bug] [ux] [env] [meta]. "env"
= upstream tool, not ours.

## 2026-05-20 — first dogfood pass

Ran `consult` (code tier, ~$0.21, 4 panellists) on the question "highest-leverage
improvement before adding features" with the 8 core modules attached.

- [env] **Claude Code doesn't surface MCP tools on session start** despite
  `claude mcp list` showing the server "Connected". The tools (`panel`, `synthesise`,
  `consult`, `refine`) advertise correctly when probed directly (`handle_list_tools()`
  returns 4 tools). Likely race between session-start tool enumeration and server
  spawn. Workaround so far: full session restart.
- [env] **`/mcp reconnect consult` reconnects transport but doesn't re-enumerate
  tools** into Claude Code's registry. Same symptom as above persists post-reconnect.
- [ux] ~~**Noisy LiteLLM pricing-table miss**~~ — RESOLVED: set
  `litellm.suppress_debug_info = True` next to `drop_params` (runner.py). Kills
  the ANSI feedback-link footer LiteLLM prints on every caught exception.
- [ux] ~~**`code` tier shrinks silently when its members overlap with the default
  synthesiser**~~ — PARTIAL: `RunResult` now carries `synthesiser` (types.py:148,
  populated in `_handle_consult`). Panel-size shrinkage is now inferable from the
  response. Still no proactive warning when this happens — open question whether
  that's worth the noise.
- [ux] **Wide latency variance with no per-call timeout intelligence**. In this run
  deepseek took 136s while claude-opus finished in 39s. PARTIAL FIX: `_call_one`
  now logs each panellist completion at INFO (`runner.py`) so MCP server logs
  (`~/.claude/logs/mcp-logs-consult/`) carry a mid-run signal. The full
  user-facing fix is MCP progress notifications, deferred from v1 by the refine
  panel verdict.
- [bug] **First real bug found via dogfooding (panel surfaced it)**: an unknown model
  alias (typo, stale config) raises `KeyError` from `registry.resolve_model` inside
  `estimate_cost`, propagates out of `fanout`, and crashes the whole call before any
  artifacts are written. One bad spec kills the entire panel. Fix in flight.

## 2026-05-20 — second dogfood pass (refine on MCP progress design)

Ran `refine` (code tier, max_rounds=2, threshold=0.85, cap $2.50, ~$0.07) asking
the panel which mid-run progress mechanism to use (notifications/progress vs
resource subscribe vs polling vs file-tail). Refine STOPPED at round 1 with
score 0.20 and no convergence — and the script had no idea why.

- [bug] **`RefineResult` silently dropped `partial`, `partial_reason`, `wall_ms`**.
  `refine()` (refine.py:355) was passing these as kwargs, but the fields didn't
  exist on the `RefineResult` schema — Pydantic v2 default behaviour ignores extras,
  so the values were dropped without an error. End result: when refine refused
  round 2 because of unknown deepseek pricing (refine.py:308), the caller had no
  way to learn why. RESOLVED: added the three fields + partial-coupling validator
  to `RefineResult` (types.py), matching the existing `RunHandle` pattern. Now the
  refusal reason actually round-trips to the caller.
- [ux] **Gemini-3 emits a temperature warning** on every capsule-extractor call:
  *"Setting temperature < 1.0 for Gemini 3 models can cause infinite loops,
  degraded reasoning performance, and failure on complex tasks."* We pass
  `temperature=0.0` in `capsule._extract_one` (capsule.py:101) for determinism.
  Gemini-3 specifically dislikes that. Open: provider-aware temperature override,
  or drop the explicit `temperature=0.0` and trust the model.
- [ux] **Double-logging when client enables basicConfig(level=INFO)**. LiteLLM has
  its own coloured logger AND propagates to the Python root logger, so every
  `LiteLLM completion()` info line and warning prints twice when the calling code
  configures the root logger. Not our bug, but pollutes any caller-side script.
  Workaround: `logging.getLogger("LiteLLM").propagate = False`.
- [env] **Repeated OpenAI rate-limits on `gpt-codex` + `gpt-mini`** across runs —
  same shared OpenAI bucket in `.env`. Two of five panellists fail every time
  on `code` tier. Either rotate keys or accept that the panel is effectively 3/5
  on the code tier from this machine.
- [meta] **The panel was actually useful**. Even with only 2/4 panellists returning
  (claude-opus + deepseek), the synthesis surfaced a substantive disagreement
  about whether MCP stdio multiplexes notifications during a blocking tool call,
  steel-manned the dissent, and recommended A (notifications/progress) + D (log
  file fallback). That's exactly the answer worth keeping for when the
  deferred-from-v1 progress feature comes back online.

## 2026-05-20 — third dogfood pass (consult on Polars vs DuckDB)

Ran `consult` (standard tier, 9 panellists, cap $7, ~$0.11) on an external
analytics question. 7/9 OK, 2 OpenAI panellists rate-limited (third run in a
row hitting the same shared bucket). Panel unanimous on DuckDB — no bug claims
made about the codebase. Observations:

- [ux] ~~**Cost cap message is misleading when pricing is partial-unknown**~~ —
  RESOLVED: cap-exceeded message in `runner.fanout` (runner.py:283) now appends
  `(known-priced portion only; some unknown)` when `all_known=False`. Mirrors
  the dry_run branch which already had this disambiguation. Test added.
- [env] **OpenAI rate-limits hit on every run, both shared keys**. Pass 1 (code
  tier): gpt-codex + gpt-mini. Pass 2 (code tier): same. Pass 3 (standard tier):
  gpt-pro + gpt. Pattern: ALL OpenAI panellists fail concurrently on every run
  from this machine. Per-minute request limit on the OpenAI key. Workarounds:
  rotate keys, throttle per-provider concurrency, or accept the partial panel.
- [ux] **Latency variance widens with bigger panels**. Standard-tier run had
  claude-opus at 33s, kimi at 156s. Wall time is bounded by the slowest. Bigger
  panels mean longer tails. Per-model timeouts in config already cap at 180-240s
  so worst case is bounded, but a 5× spread between fastest and slowest in the
  same panel makes the "wait" feel uneven.
- [meta] **No bug claims this pass.** The panel engaged with the external
  question, not the codebase, as intended.
