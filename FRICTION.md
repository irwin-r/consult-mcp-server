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

## 2026-06-11 — strategy dogfood pass (9 runs, ~$3.1 known spend)

Exercised all five tools live for the strategy review in
docs/strategy/2026-06-dogfood-review.md: consult (standard + deep/research),
refine (blinded, stances), sequence (3 steps, per-step attachments), panel
(peer_rank), synthesise (re-run under critique rubric), plus two dry_runs.

- [bug] **synthesise() clobbers the run's recorded cost**. Re-synthesising
  20260611-042818-67174 rewrote the manifest's top-level cost_usd from $1.26
  to $0.087 (the re-synth call alone) and flipped cost_known false->true.
  synth.py wrote its own cost instead of accumulating, and the ledger reads
  the manifest, so one re-synth erased the panel's spend from the books.
  RESOLVED in this PR: accumulate cost, AND the known flags, regression test.
- [bug] **Test suite was polluting the real runs dir and ledger**. 1,243 run
  dirs had accumulated; ~130 created today alone in bursts of five (prompt
  "p", "estimated cost exceeds cap" rejections) by pytest runs. consult-ledger
  counted each as a $0 run, drowning the day's real spend. No conftest set
  CONSULT_RUNS_DIR. RESOLVED in this PR: autouse fixture isolates every test
  into tmp_path. Full suite now adds zero dirs.
- [bug] **refine's arbiter died parsing its own verdict**. Run
  20260611-043229-95424 round 1: the arbiter (gemini-pro) returned non-JSON,
  score 0.0, `json_parse_failed`, loop aborted. refine degraded to a
  single-round panel at full cost. partial_reason surfaced it well, but
  there is no retry and no structured-output request on the arbiter call.
  Second arbiter-fragility incident (see 2026-05-20 pass two). Issue filed.
- [ux] **Truncation waste dominated every flagship-bearing run, again**.
  6/9 TRUNCATED on the $1.26 standard run (042818); 6/13 on the $1.23 deep
  run (044542); gpt-pro burned $1.45 across the two for zero usable capsules
  (reasoning burn inside max_completion_tokens). Subtler: the consensus
  rubric tells the synth to down-weight TRUNCATED, so terse cheap models
  ended up steering the strategy verdicts while truncated flagships were
  discounted. The cap distorts synthesis weighting, not just cost. Issue filed.
- [ux] **cost_known=false on effectively every standard/deep run**: all
  OpenRouter panellists (5/9 standard, 8/13 deep) return null pricing, so
  caps and the ledger run half-blind on exactly the most diverse tiers.
- [ux] **peer_rank: silent ranker dropout, invisible spend**. deepseek
  returned no usable ranking and was skipped per design, but the result
  carries only an empty per_ranker list, no reason, and PeerRanking's
  cost fields aren't serialised into the result. Feature verdict: earns
  its keep (clean Borda separation for ~$0.04, surfaced the strongest
  answer). Issue filed for the two gaps.
- [ux] **sequence gives no per-step health rollup**. Step syntheses
  mentioned a truncated panellist, but the result JSON has no per-step
  run_summary/no_value, so step-level waste is invisible without reading
  run dirs. Wall time 493s for 3 steps x 3 models is fine for research
  use, but only if progress reaches the host.
- [ux] **synthesise(anonymised=true) output cites real slugs**, which read
  as a blinding bug until reading synth.py: the synth input is always
  blinded and shuffled, output deliberately de-blinded, and `anonymised`
  only scrubs brand names from the prompt bundle. The tool description
  ("hide real model IDs from the synthesiser input") invites the wrong
  read, and nothing in the result says blinding happened. Fold into the
  calibration-report work: disclose the bias controls per run.
- [ux] **blinded refine now returns real model_id per final_manifest
  entry** (043229), where the 2026-05-20 pass four recorded model_id=None
  on blinded manifests. Either an intentional post-hoc reveal or a
  regression from the 0.4.0 identity work; needs a decided, documented
  answer. Issue filed.
- [ux] **research capsules lose the web panellist's sources**. sonar-pro's
  sources_cited came back as opaque indices (["[3]", "[4]"]) that only
  resolve inside the full body. The one web-grounded panellist's citations
  are unusable from the capsule. Kind verdict: claims/evidence/uncertainties
  shape clearly beat decision-shape for the competitive scan; worth keeping
  with the sources fix and a rethink of the 4000 cap (3/13 zero-value).
- [meta] **Panellists asserted stale host facts with confidence**: "agent
  hosts time out MCP tool calls over 15s" (glm, echoed by the synth) while
  this very session ran 271s and 493s tool calls in Claude Code without
  incident. Only 1 of 14 deep panellists had web access. Good calibration
  case for the research tier and for weighting web-grounded panellists.

## 2026-06-11 — issue #55 fix pass (5 runs, ~$2.4 known spend)

Shipped the truncation-economics fix with live before/after probes
(baselines 052420/052638, design check 053100, after-fix 054539/054753).

- [bug] **Research-kind extractor retry could never adopt its result**.
  The empty-extraction retry gate checked `findings` for research bodies,
  a field ResearchCapsule does not have, so the re-ask fired on every
  substantial research body and the recovered capsule was discarded
  unconditionally. One wasted haiku call per research panellist since the
  retry shipped. Found by code-reading during #55; fixed in this PR with
  a kind-aware substance check on both the fire and adopt sides.
- [meta] **The design-check panel demonstrated the bug class it was
  convened to fix**. Run 053100 (code tier) returned two no_value entries
  on OK bodies: the extractor stochastically missed claude-opus and
  gpt-codex decision capsules, and decision kind had no retry. The same
  PR extends the retry to decision capsules.
- [env] **OpenAI quota died mid-session and split by model class**.
  gpt-5.5-pro and gpt-5.5 both returned "exceeded your current quota" on
  the P1 rerun at 05:45; gpt-5.5 recovered by the P2 rerun two minutes
  later while gpt-5.5-pro stayed blocked through a direct bounded probe.
  The baselines' own gpt-pro burn (about a dollar across two probes for
  zero usable text) plausibly drained the shared key. Consequence: the
  worst truncation offender could not be validated live; the fix's claim
  for gpt-pro rests on budget arithmetic (8000 floor vs observed 4000
  full-reasoning burn) plus the validated rescue of gpt-5.5, kimi, glm.
- [env] **gpt-5.5-pro rejects reasoning_effort=low**: the API answers
  400 with "Supported values are: 'medium', 'high', and 'xhigh'". Fix
  variant (d) from the design debate was dead on arrival for the one
  model it targeted, and nothing in the registry validates per-model
  effort values against the live API. Worth folding into #57's
  registry-canary scope.
- [meta] **Weighting now tracks substance, not cap luck**. Baseline P1
  synthesis: "Responses glm, gpt-pro, and kimi were heavily down-weighted
  due to being TRUNCATED". After-fix P1: zero truncations, kimi authors
  the minority report it previously could not enter, and the only
  down-weight is grok for low confidence and brevity.
- [meta] **Arbiter hardening validated live** (run 20260611-055844-23863,
  3-model panel, 2 rounds, $0.11). Both verdicts parsed first try through
  the gemini-pro thinking arbiter at its new 16000 budget with JSON mode
  (scores 0.60 then 0.95, converged). The same run exposed that refine
  results carried no run_summary at all — the rollup keyed on `manifest`
  while refine returns `final_manifest`. Fixed in this PR.

## 2026-06-14 — dogfood pass (in-session, ~$0.10 known spend)

Dogfooded the live MCP tools from inside the repo to chase the reported
"errors / token-limit" pain. The reproduction came for free: a `dry_run`
guard call surfaced a real bug.

- [bug] **`refine` and `sequence` silently dropped `dry_run`**. Passing
  `dry_run: true` to either tool ran the full thing at full cost: the MCP
  schema carried no `dry_run` property, the handler never forwarded it, and
  the engine functions had no such parameter, so the JSON field was dropped
  with no error (same "extra fields silently dropped" class as the
  2026-05-20 RefineResult and dropout-cost passes). Worst case on the two
  most expensive tools (a 3-round flagship refine or a multi-step
  sequence) could spend dollars on what the caller thought was a free
  estimate. I tripped it myself: a `dry_run` refine I expected to be free
  ran 2 rounds for $0.10 (run 20260614-064338-75435). RESOLVED in this PR:
  `dry_run` is now threaded end to end (schema + handler + engine). Refine
  prices one round (panel + arbiter) and reports the per-round estimate
  plus the worst case across `max_rounds`; sequence sums each step's panel
  estimate (a floor, since per-step synth and prior-step context growth
  aren't priced). Both return early with `partial=true`, no run dir, no
  spend. Five tests added, including in-process MCP-adapter tests that
  drive the real server object end to end (the layer where the flag was
  actually being dropped).
- [meta] **Arbiter hardening still holds.** The accidental refine
  (gemini-pro arbiter, 2 rounds, scores 0.7 then 0.8) parsed both verdicts
  first try with no `json_parse_failed`. Second clean live arbiter run
  since the #57 hardening.
- [env] **Flagship truncation is expensive to regenerate.** Standard tier
  dry-runs at $2.12; a 3-flagship gpt-pro panel at $1.81 worst case. Rather
  than pay to re-trigger reasoning-burn truncation, note that the evidence
  already sits on disk: 1,282 run dirs under `~/.consult/runs`, e.g. the
  $1.26 standard run 20260611-042818-67174 carries 12 truncated/length
  panellists. Live truncation repro should reuse these, not re-buy them.
- [meta] **The MCP server runs a snapshot of the code at spawn time.** A
  working-tree edit (this fix) is invisible to the already-connected
  `mcp__consult__*` tools until the server restarts, so the fix was
  validated through the in-process adapter test rather than a live tool
  call. Worth remembering before "I fixed it, let me re-run the tool live."
- [bug] **Live re-test (fresh server) surfaced a dry_run/cap ordering
  inconsistency.** The validation agent ran the fixed tools against a
  restarted server (refine/sequence dry_run both returned `dry_run:`
  estimates, zero spend, confirmed loaded). Its consult regression check,
  though, hit `estimated cost $0.04 exceeds cap $0.01` instead of a
  `dry_run:` reason: in `runner/fanout.py` the `estimate > cap` gate sat
  ABOVE the `if dry_run` gate, so an over-cap dry_run reported the cap
  rejection and the estimate never surfaced. consult still refused to
  spend (cost_usd 0), so no money was at risk, but the behaviour diverged
  from the just-fixed refine/sequence, which return the estimate before any
  cap enforcement. RESOLVED in this PR: dry_run now precedes the cap gate in
  fanout, and when the estimate exceeds the cap the message appends "(exceeds
  cap $X; a real run would be rejected)" so the cap signal isn't lost. Two
  tests lock it: over-cap dry_run yields the estimate; over-cap real run is
  still refused. The cap-first ordering was pre-existing, not introduced by
  the dry_run work — the live re-test is what exposed it.

## 2026-07-13 — third dogfood pass (two production refine runs on an e-commerce architecture question)

Two real `refine` runs (9-model standard panel, ~$1.5 and ~$1.1) plus one
zero-panellist failure. Every item below observed live; fixes landed in
this pass.

- [ux] ~~**Stale example model names in the Claude Code skill doc killed a
  run.**~~ RESOLVED. `~/.claude/skills/consult/SKILL.md` showed
  `gpt-5` / `gemini-2.5-pro` / `opus-4.7` in its refine example. Passing
  them: `opus-4.7` raised Unknown model per-panellist, `gemini-2.5-pro`
  fell through raw-ID routing to Vertex (missing SDK), `gpt-5` went to
  OpenAI raw and timed out at 180s. Net: 3 minutes, $0, zero panellists.
  Skill doc now lists the registry aliases and points at tiers.
- [ux] ~~**Unknown aliases only failed per-panellist, mid-run.**~~ RESOLVED:
  `panel`/`refine`/`sequence` handlers now resolve every alias up front
  and return the `unknown_model` envelope with close-match hints, the full
  alias list, and the tier names before any provider call.
- [ux] ~~**Skill docs promised `tier` on panel/refine; only consult had
  it.**~~ RESOLVED: `panel` and `refine` accept `tier` as an alternative
  to `models` (anyOf in the schema, expansion in the handler).
- [bug] ~~**Synthesis had no fallback: one Gemini 503 left "Synthesis
  unavailable" on a converged 9-model run** (20260713-045311).~~ RESOLVED:
  `defaults.synthesiser_fallbacks` (packaged: `["claude-sonnet"]`) is
  tried in order after the primary fails/empties; all attempts billed and
  itemised in the sentinel when everything fails.
- [bug] ~~**Arbiter non-JSON (twice) aborted the refine loop at round 1**
  (20260713-050837: same overloaded Gemini as arbiter).~~ RESOLVED: after
  the parse-retry on the primary, the arbiter falls back through
  `synthesiser_fallbacks` before scoring the round 0.
- [bug] ~~**Reasoning burn: gpt spent 12k output tokens, $0.37, and
  returned a zero-byte TRUNCATED body** (20260713-050837, gpt-2.r1).~~
  RESOLVED: `_call_one` retries an empty-body length-truncation once with
  a doubled budget, sums both attempts' spend/tokens, and notes the retry
  in the manifest entry.
- [ux] **KeyError envelopes carried the quoted repr** (`'Unknown model:
  opus-4.7'`). RESOLVED: server unwraps `args[0]`.
- [meta] ~~**Long-form prompts overwhelm per-model output budgets.**~~ The
  second run asked for an agent-executable build spec; 4 of 9 panellists
  hit finish_reason=length (opus/sonnet at 8k, gpt at 12k, kimi at 12k)
  and glm hit slow-tail dropout at 180s while still writing. The capsule
  extractor recovered most of them, but tier budgets are sized for
  decision capsules, not 10-page deliverables. RESOLVED two ways:
  (1) `max_output_tokens` on panel/refine/consult/sequence raises every
  panellist's grant (single `output_budget()` derivation shared with the
  cost estimators so the cap gate prices the real ceiling); (2) a
  truncated-with-body response is auto-continued once — the model gets
  its partial answer back as an assistant turn and carries on from the
  cut, bodies stitched, both calls billed, noted in the manifest
  (`CONSULT_CONTINUE_ON_TRUNCATION=0` to disable). `run_summary` now
  carries `truncation_advice` naming the knob. STILL OPEN: the 180s
  slow-tail dropout doesn't know about continuations — a panellist mid-
  continuation can still be dropped if it's in the last 20% of the panel;
  raise CONSULT_TAIL_DROPOUT_S for long-form runs.

## 2026-07-13 — fourth dogfood pass (9-model review refine on the ecommerce codebase, run 20260713-091722-76480)

$4.40, 2 rounds, converged 0.85 — but 5 of 9 round-1 capsules and grok's
round-2 capsule came back EMPTY despite 15-20KB of well-structured findings
in every response body, and two panellists timed out. Diagnosed from the run
artifacts; fixes landed in this pass.

- [bug] ~~**Tool-call envelope nesting silently emptied capsules.**~~
  RESOLVED. claude-haiku via Anthropic tool-use (`response_format`
  emulation) sometimes returns `{"parameter": {...capsule...}}`. The
  known-fields filter in `_extract_one` dropped the single unknown key and
  built a VALID empty capsule — no error, no log, findings gone (confidence
  still backfilled from the body, which made it look like a deliberate
  abstain). The empty-extraction retry then failed the same way, so the
  round-1 arbiter scored coverage 0.5 and blamed the panellists ("failed to
  extract parseable findings"), forcing a second round that roughly doubled
  run cost. Reproduced live against the archived grok-4.r1 body (9.4KB
  reply, wrapped, findings=0 after filter). Fix: `_unwrap_capsule_data`
  descends through known envelope keys and single-key dict wrappers
  (bounded) on both the first attempt and the retry; two regression tests
  cover the exact live shape.
- [config] ~~**claude-sonnet's 180s timeout is too short for big review
  prompts.**~~ RESOLVED: 180 → 300 (parity with gpt/gemini flagships).
  Non-streaming calls are all-or-nothing, and sonnet-5's 8000-token
  thinking budget alone can burn most of 180s on a ~100KB prompt — it
  returned 0 bytes in BOTH rounds while every 240s+ peer finished.
- [config] ~~**Slow-tail dropout cancelled a working straggler inside its
  own budget.**~~ RESOLVED: default grace 180 → 300. kimi (360s per-spec
  timeout) was cancelled at 292s while presumably still generating; the
  dropout should catch hangs, not slow-but-working panellists. STILL OPEN:
  a progress-aware dropout (only cancel stragglers whose stream has
  stalled) would be strictly better, but needs streaming on by default;
  today `CONSULT_STREAM=1` is opt-in.
- [bug] ~~**All-or-nothing capsule validation dropped whole capsules over
  one bad enum.**~~ RESOLVED (found on the fifth dogfood run,
  20260713-095511-17732: even with envelope unwrapping in place, 3 of 8
  capsules extracted empty). claude-haiku emitted `category:
  "architecture"` on one finding out of six; the strict Pydantic build
  discarded all six. Fix: `_salvage_review_findings` validates PER
  FINDING, coerces common severity/category aliases
  (critical→blocker, architecture→maintainability, compliance→correctness,
  ...), parses string line ranges ("80-92"), normalises the verdict, and
  drops only the individually hopeless entries with a warning log.
- [ux] ~~**Slow-tail dropout was blind to progress and to per-model
  budgets.**~~ RESOLVED three ways. (1) kimi and kimi-code per-spec
  timeouts raised 360 -> 600 (kimi hit its own ceiling on the fifth run
  after surviving the dropout). (2) The dropout is now progress-aware:
  stream chunks, the empty-body retry, and a truncation continuation all
  record per-slug activity, and stragglers with a signal inside
  CONSULT_TAIL_ACTIVITY_WINDOW_S (default 120s, 0 disables) keep running
  while silent ones are cancelled on the old schedule; per-spec timeouts
  still bound everything. Task start deliberately does not count, so
  non-streaming panels without continuations behave exactly as before.
  (3) This also closes the earlier STILL OPEN note about panellists being
  dropped mid-continuation. Practical upshot: CONSULT_STREAM=1 now buys
  dropout immunity for any panellist that is actually producing tokens.

## 2026-07-22 — registry refresh smoke (fable-5 / gpt-5.6 / gemini-3.6 / kimi-k3)

Live smoke after adding claude-fable, gpt-terra, kimi-k3, deepseek-flash and
bumping gemini-flash to 3.6. Five of six new IDs worked first try; total spend
~$0.11 across two fanouts plus doctor pings.

- [env] **gpt-5.6 `-pro` variants are gated on the direct OpenAI API.**
  `openai/gpt-5.6-sol-pro` returns `model_not_found` even though OpenRouter
  lists it; this key's `/v1/models` shows sol/terra/luna but no 5.6 pro tier.
  `gpt-pro` stays on `openai/gpt-5.5-pro` ($30/$180) until the pro variants
  appear in the key's model list — recheck with
  `curl -s https://api.openai.com/v1/models` before the next bump.
- [env] **LiteLLM's price table lagged 13 of 24 registry models**, so
  `estimate_cost` returned `all_known=False` (cap gate degraded to
  warn-don't-block) and fanout under-reported actual spend for those
  panellists. FIXED: `pricing.ensure_registered()` gap-fills
  `litellm.register_model` from new per-entry `pricing` blocks in models.json;
  shipped tables still win when they know the model. Keep pricing blocks in
  sync when bumping a litellm_id.
- [bug] ~~**`consult-doctor --ping` false-failed on reasoning models**~~ —
  RESOLVED: the 1-token grant gets burned on reasoning and OpenAI returns an
  output-limit error, which proves the key works; doctor now counts that as
  "ok (output-capped reasoning reply)" (doctor.py `_ping_provider`).
- [ux] **kimi-k3 is the slow panellist of the new set** (24s for a one-line
  answer vs 4-7s for the rest; registry timeout stays 600s like the other
  moonshot entries). Not actionable yet, just expect it to pace wide panels.

## 2026-07-22 — refine dogfood on the deep-research plan (stale registry)

Ran `refine` (review tier, blinded, critique rubric, cap $10, ~$1.02 known
spend) to pressure-test the implementation plan for the proposed `research`
tool. Converged in one round at 0.95 with 6/7 panellists usable; the panel's
revisions are folded into the plan issue. Run 20260722-003510-38368.

- [ux] **A long-lived server served a dead model ID to a paid run.** The
  running process resolved `gpt-pro` to `openai/gpt-5.6-sol-pro`, the exact
  ID the same-day refresh smoke (entry above) had already rejected; the
  panellist burned its slot on `model_not_found`. models.json on disk was
  correct (`openai/gpt-5.5-pro`, confirmed 200 via the models endpoint),
  but the registry is cached for the process lifetime, so the
  editable-install server kept serving a mid-refresh tree state from
  memory. Restarting the server is the documented fix; the sharp edge is
  that nothing warns when the on-disk config has drifted from the loaded
  one. Filed as issue #91 (drift warning + key-gated existence probe).
- [meta] **Detection gap between refreshes**: the weekly canary skips its
  provider ping in CI (no repository keys) and `consult-doctor --ping`
  probes one model per provider, so a retired ID's first detector today is
  a paid panellist. The zero-token existence probe proposed in #91 would
  cover it locally.
