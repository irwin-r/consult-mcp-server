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
- [ux] ~~**Gemini-3 emits a temperature warning** on every capsule-extractor call~~ —
  RESOLVED: `capsule._extract_one` now omits `temperature` entirely when the
  extractor's `litellm_id` contains "gemini" (substring match catches both
  direct `gemini/...` and `openrouter/google/gemini-...`). Other providers
  still get `temperature=0.0` for deterministic JSON output. Per Google's
  own guidance, Gemini-3 defaults to 1.0 and clamping it lower can produce
  infinite loops on complex tasks.
- [ux] ~~**Double-logging when client enables basicConfig(level=INFO)**~~ —
  RESOLVED: `consult/runner.py` (module-level, runs at first import) sets
  `logging.getLogger("LiteLLM").propagate = False`. LiteLLM's own coloured
  handler still prints; the duplicate via root is suppressed. Effect is
  global to the process — any caller of consult-* picks it up automatically.
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

## 2026-05-20 — fourth dogfood pass (refine blinded on multi-tenant Postgres)

Ran `refine` (code tier, blinded=True, max_rounds=2, ~$0.09) on a Postgres
multi-tenancy question. 2/4 OK (gpt-codex + gpt-mini rate-limited again),
score 0.40, refine stopped at round 1 with `partial_reason` correctly
surfacing the pass-#2 fix:

> "refusing further rounds: per-model pricing unknown for at least one
> panellist, can't validate cap ($0.09 spent / $2.50 cap)"

That's the partial_reason a132810 added, doing its job in real use. The
caller can now see exactly why refine stopped early.

- [meta] **Blinded mode works correctly.** Greek-alphabet slugs assigned
  (`panelist-alpha` … `panelist-delta`), `model_id=None` on every manifest
  entry, synthesis anonymised (refers to "panelist-alpha" etc, never the
  real model). No friction in this path.
- [meta] **No panel-claimed bugs.** Panel engaged with the external
  Postgres question; unanimous on (A) shared-schema + RLS with steel-manned
  dissent for database-per-tenant. Refine-level disagreement was about
  bottleneck mechanics, not about anything in the codebase.

**Counter: 2 consecutive passes with no panel-claimed bugs.** Switching to
deferred-from-v1 feature work next pass.

## 2026-05-20 — pass #14 (live sequence test post-features)

After all deferred-v1 features landed, ran the `sequence` tool live (code
tier, 3 steps) to validate pass #10's work end-to-end. Found a real bug —
caught by dogfooding, not by the integration tests.

- [bug] **Sequence stopped at step 1 because of partial pricing**. sequence.py
  inherited refine's `if not est_known and i > 1: break` check, which made
  step 2+ refuse whenever any panellist had unmapped LiteLLM pricing (almost
  every openrouter model). The reasoning was wrong for sequence's semantics:
  unlike refine (where an arbiter could extend rounds unboundedly), sequence
  iterates a fixed user-supplied step list. The cumulative-cost check at the
  top of every step + the `max_run_usd = cap - cumulative_cost` passed into
  each `runner.fanout` already provided the cap guarantee. The extra
  est_known check just refused legitimate work — refine's protection,
  miscopied. RESOLVED: dropped the est_known short-circuit from sequence.py;
  partial pricing now just propagates `cost_known=False` to the result. New
  test `test_sequence_continues_through_partial_pricing` locks the fixed
  behaviour: 3-step sequence with `est_known=False` from every step now
  completes all three.
- [meta] **The dogfood loop continues to pay**. Pass #10 wrote sequence with
  integration tests that mocked everything; pass #14 was the first live
  panel test and immediately surfaced the inherited-too-much-of-refine bug.
  Worth keeping a live-smoke pass routinely after copying logic between
  similar modules.
