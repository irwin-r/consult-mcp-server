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
- [ux] **Noisy LiteLLM pricing-table miss for `openrouter/deepseek/deepseek-v4-pro`**.
  Our cost lookup catches the exception and logs a clean `.warning()`, but LiteLLM
  prints the underlying error block to stderr first (3 lines of error + HTML
  feedback links per panellist on each lookup). Drowns the rest of the log. Could
  filter stderr or suppress at the LiteLLM logger.
- [ux] **`code` tier shrinks silently when its members overlap with the default
  synthesiser**. `code` has 5 aliases; `gemini-pro` is the default synthesiser and
  gets filtered out of the panel by `_handle_consult`, leaving 4. No warning. Either
  surface the effective panel size in the result, or pick non-overlapping synthesiser
  by default.
- [ux] **Wide latency variance with no per-call timeout intelligence**. In this run
  deepseek took 136s while claude-opus finished in 39s. No mid-run signal to user
  that one panellist is dragging; only the final manifest reveals it. (Per-model
  timeouts exist in config, but a 240s ceiling means a single laggard pegs wall time.)
- [bug] **First real bug found via dogfooding (panel surfaced it)**: an unknown model
  alias (typo, stale config) raises `KeyError` from `registry.resolve_model` inside
  `estimate_cost`, propagates out of `fanout`, and crashes the whole call before any
  artifacts are written. One bad spec kills the entire panel. Fix in flight.
