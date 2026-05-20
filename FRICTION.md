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
