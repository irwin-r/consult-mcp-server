# 2026-06 strategy review (dogfood pass)

Strategic review of consult-mcp-server at v0.4.1, produced by dogfooding the
product on its own strategy questions. The tactical backlog was cleared in
0.4.0/0.4.1; this review asks what to double down on, what to build next,
what to kill, and whether any pivot beats the current course.

## Method and spend

I wrote down eight hypotheses before any model call, grounded in the
README, FRICTION.md, the self-improve journal, the configs, and the run
ledger. Each was then pressure-tested through the consult tools themselves,
so every call doubles as a live product test. Friction found along the way
is logged in FRICTION.md (2026-06-11 entry) and informed the
recommendations below.

| run | tool | shape | cost (known) | panel health |
|---|---|---|---|---|
| 20260611-042818-67174 | consult | standard, 9 panellists, repo docs attached | $1.26 | 3 OK, 6 TRUNCATED |
| 20260611-043229-95424 | refine | blinded, 6 opposing stances, 2-round cap | $0.17 | 4 OK, 2 TRUNCATED; arbiter failed round 1 |
| 20260611-043441-83971 | panel | 5 models, peer_rank=true | $0.09 | 5 OK; 1 ranker dropped |
| 20260611-043652 / -043952 / -044257 | sequence | 3 steps, 3 models, per-step attachments | $0.37 | 1 truncated panellist per later step |
| 20260611-044542-32339 | consult | deep, capsule_kind=research, web panellist | $1.23 | 7 OK, 6 TRUNCATED; panel silently 13/14 |
| (re-run on 042818) | synthesise | critique rubric, anonymised | $0.09 | exposed the ledger-clobber bug |

Two dry runs preceded the expensive calls. Known spend ~$3.21; true spend is
somewhat higher because every OpenRouter panellist returns unknown pricing
(5/9 on standard, 8/13 on deep). Budget was $15.

One external check: the panels flagged their own uncertainty about
zen-mcp-server's current state, so I verified that one claim by web search
instead of taking it from the panel.

## What the dogfooding changed

Two of my eight pre-panel hypotheses did not survive contact:

- **CLI-as-panellist: I leaned develop; the verdict is delete.** A focused
  peer-ranked panel went 5/5 for deletion (run 043441). The web check
  sharpened it: pal-mcp-server's headline feature IS CLI orchestration, so
  an undocumented, zero-user module here contests a competitor's flagship
  instead of reinforcing our own.
- **Eval mode: I leaned build-it-now; the verdict is fix-first, then a
  one-shot report.** The blinded refine (run 043229) went 4 stances to 2:
  benchmarking the panel before the truncation fix would measure a degraded
  system and risk publishing a false negative about the product's core
  premise.

## Recommendations

### R1. Stop paying for empty answers (truncation economics)

Impact: high. Effort: medium. This is the unanimous #1.

Every flagship-bearing run this session wasted a large share of spend on
panellists that truncated at the per-kind output cap before emitting a
usable capsule: 6/9 on the standard run, 6/13 on deep, and gpt-pro alone
burned $1.45 across two runs for zero usable output (reasoning tokens
consume max_completion_tokens before any text is emitted). The self-improve
agent's own runs show the same pattern (7/9 truncated on a $1.89 run this
morning).

The damage is worse than wasted money. The consensus rubric tells the
synthesiser to down-weight TRUNCATED responses, so the strategy verdicts in
this very review were steered by whichever models happened to be terse
(qwen-max, grok, llama) while truncated flagships were discounted. The cap
distorts synthesis weighting, not just cost.

What to build: reasoning-aware per-model output budgets (the per-kind cap
must not bind below a reasoning model's burn), a cost_per_useful_capsule
metric in run_summary, and an opt-in single retry for a panellist that
truncated with an empty capsule. The minority position worth recording:
claude-opus argued continuation-on-truncation may beat a global cap raise,
since raising caps raises baseline cost 30-50% on every decision run.

Evidence: runs 042818, 044542, 043229; ledger runs 001116 and 010408;
panel quotes in runs 042818 ("existential", every responder) and the
blinded refine 043229 (4/6 stances made it the prerequisite for anything
else).

### R2. Ship the calibration block: make the bias machinery visible

Impact: high. Effort: low-medium.

The bias-mitigation stack is the strongest under-marketed asset. The
synthesiser's input is already blinded and shuffled unconditionally
(synth.py docstring); disagreement is already scored on every consult;
peer-rank and medoid voting exist. None of it is visible in a result unless
you read the source. It is so invisible that this review's own agent
misread synthesise(anonymised=true) as a broken blinding feature before
reading the code; nothing in any result says "the synth never saw model
identities".

The shape of it: a per-run `calibration` block in every result, carrying
the blinding and shuffle disclosure, the disagreement score,
usable/truncated counts with per-status spend, family and privacy-tier
diversity, stance coverage, and peer-rank when enabled. Then make it
load-bearing rather than decorative: high disagreement should force the
synthesis to present both sides (claude-opus, run 042818), and panellist
confidence should interact with truncation status rather than the rubric's
blanket down-weight.

One sharpening from the deep scan worth adopting here: glm argued that
prompt/persona diversity decorrelates panel errors better than blinding
does ("asking 3 models the exact same blinded prompt yields homogenous
errors"). The stances system already exists; auto-assigning diverse stances
per tier and disclosing them in the calibration block is cheap and directly
answers that critique.

Evidence: run 042818 capsules (the "deliberation engine, not a commodity
router" framing appears across qwen-max, llama, claude-opus); sequence
step 3 (privacy tiers as a headline capability); deep run 044542 (glm's
persona-partitioning critique); the anonymised-flag confusion in
FRICTION 2026-06-11.

### R3. Make refine trustworthy: it is the moat candidate and the weakest link

Impact: high. Effort: low-medium.

The sequence decomposition (runs 043652/043952/044257) converged on a moat
argument I find more durable than my own context-economy framing: hosts are
structurally unable to bundle cross-vendor adversarial consensus, because no
vendor will route your prompt to a rival to grade its own model. Privacy
tiers and blinding reinforce that neutrality story. If that argument is
right, refine is the headline tool.

And refine is currently the least reliable tool. Its arbiter (gemini-pro)
returned non-JSON on round 1 of this session's blinded refine, scored 0.0,
and aborted the loop; the run degraded to an expensive single-round panel.
Same failure family as the May incident (score 0.20 stop). The arbiter call
uses neither structured outputs nor a retry.

The fix: structured-output (or tool-call) enforcement on the arbiter
and capsule extractor where the provider supports it, one retry on parse
failure, and a salvage path that returns partial dimension scores instead
of 0.0. qwen-max raised exactly this in run 042818 (native structured
outputs end mid-JSON truncation as a class), and it was the minority report
of that synthesis.

Evidence: run 043229 (json_parse_failed verdict, loop abort); sequence
final synthesis (unanimous on refine as the differentiator); FRICTION
2026-05-20 pass 2.

### R4. Prove the premise with a one-shot eval report, not an eval platform

Impact: high (credibility). Effort: medium. Sequenced after R1.

The product's core claim ("a panel beats one flagship") is unproven, and the
deep scan's literature pointers cut both ways: ensemble gains shrink as base
models strengthen (Du 2023, Verga 2024, cited by claude-opus in run
044542). Every panel this session named the same open question: quantified
ROI of the panel versus a single flagship.

The blinded refine's resolution is the right one. Don't build benchmark
infrastructure (Promptfoo and inspect-ai own that space, and an LLM-judged
leaderboard "measures agreement with the judge"). Instead: after R1 lands,
run roughly 50 questions across 5 task types through the existing blinding,
Borda, and cost machinery, panel versus flagship, blind judged, and publish
it as a dated report with pinned registry snapshots. Meanwhile, publish the
dogfooding case studies already sitting in the self-improve journal (the
panel catching the round-1 cost-arithmetic trap, the logging.Filter
correction, this review's own H5 reversal); claude-opus called these "the
strongest case, buried".

A useful by-product: the report doubles as guidance for when NOT to convene
a panel (low-disagreement question types), which the calibration block (R2)
can then enforce per-run via gate_synth_at_agreement.

Evidence: run 043229 (4-2 verdict with the "for" stance's steel-man
recorded in the synthesis); run 042818 (three-way split: build now / defer /
reframe); run 044542 (open-questions section).

### R5. Delete CLI-as-panellist; record an ADR

Impact: medium (focus). Effort: low.

Unanimous 5/5 delete from the peer-ranked panel (run 043441), overturning my
initial lean. The top-ranked answer's framing is worth quoting:
"experimental is not a product state, it is a maintenance liability with a
friendlier label", and "zero users means zero moat". The module executes
arbitrary subprocesses, takes minutes against the panel's seconds, bills
outside the cost model, and no packaged config references it.

Delete cli_executor.py and its fanout branch; write an ADR capturing the
design and explicit reintroduction triggers (e.g. two concrete user
requests, or agentic CLIs becoming the dominant panellist substrate). The
competitive scan strengthens this: CLI orchestration is pal-mcp-server's
headline, and governance is ours; fighting their flagship with an
undocumented module is the wrong battle.

Evidence: run 043441 (peer_ranking: claude-sonnet 9, deepseek 6, gpt-mini
6, gemini-pro 3, grok 0); claude-opus dissent in run 042818 (keep-but-
experimental) recorded and rejected.

### R6. Registry freshness: deterministic canary, agent-authored bumps

Impact: medium. Effort: low.

The registry pins fast-rotting model IDs, and the live half of the rot is
already visible: pricing. Every OpenRouter panellist returned cost_usd=null
this session, which half-blinds the cost caps and the ledger on the most
diverse tiers.

The panel split three ways (run 042818): agent-owned (llama, claude-opus),
deterministic CI only (qwen-max: an LLM managing model rot is
"over-engineered and fragile"), and don't-add-surface (grok). The hybrid
takes the best of each: a weekly deterministic CI job runs
consult-doctor --ping plus a LiteLLM pricing-presence check per registry
entry and opens a plain report issue; the self-improve agent's only job is
authoring the alias-bump PR from that report, gated by the existing warden.

Evidence: run 042818 capsules; cost_known=false on every multi-provider run
this session; models.json pins (gemini-3.1-pro-preview, deepseek-v4-pro)
that will not survive the year.

### R7. Prune the tool surface; demote rather than delete sequence

Impact: medium. Effort: low.

Consensus across runs 042818 and the sequence itself: cut schema bloat. The
specific cuts I endorse: retire the `wide` tier (overlaps standard), retire
the `elimination` refine strategy (unused complexity), and tighten the
synthesise tool description (its `anonymised` flag confused this review's
agent; say plainly that synth input is always blinded).

On `sequence` I am dissenting from the panel's kill verdict, with the
evidence of this session: the sequence run produced the single most useful
strategic artifact (the neutrality-moat argument in steps 2-3), precisely
because each step's panel read the prior step's full synthesis without any
of it transiting the parent context. That is the product's core trick. But
the MCP schema cost is real and the calling agent can chain consults
itself. Resolution: demote sequence from the MCP tool surface to the
library API (or behind an env flag), keep the engine code, and revisit with
usage data. Add the missing per-step health rollup if it stays.

The maximal version (qwen-max: fold all five tools into one consult with an
orchestration_mode enum) is recorded as an option, but it trades tool-level
"use when" guidance, which the README claims is what makes agents pick the
right tool reliably.

Evidence: run 042818 (kill sequence/wide/elimination; keep synthesise per
claude-opus); sequence step 1 dissent (claude-sonnet and grok rank sequence
scarce, qwen-max calls it commodity); this session's own sequence output.

### R8. Transport: hold stdio; spike streamable HTTP only on demand

Impact: low now. Effort: n/a (watch item).

The deep scan split on whether MCP is even the right delivery layer. The
"wrong abstraction" position (glm, qwen-max) leaned on a claim that hosts
time out tool calls over ~15s, which this session directly contradicted:
271s and 493s calls completed fine in Claude Code. With the timeout premise
gone, the case for a remote proxy weakens to multi-tenant ambitions, which
are a different product (auth, key custody, billing, abuse). artifacts.py
already isolates the URI scheme behind a ContextVar, so nothing rots while
waiting. Revisit when a host or user actually asks for remote deployment;
no issue filed.

Evidence: run 044542 (Position A: MCP-native is the distribution advantage,
mistral/llama; Position B: hosted proxy, glm/qwen-max; the live-latency
contradiction in FRICTION 2026-06-11).

## Top three

1. **R1, truncation economics.** Unanimous across every run, and the
   product's pitch ("cheap structured second opinion") is false until it
   lands. Everything else inherits a degraded panel.
2. **R2 + R3, the trust layer.** One theme, two halves: make the existing
   bias machinery visible (calibration block) and make the moat tool
   reliable (arbiter hardening). This is the defensible ground the
   competitive scan found vacant: governed deliberation, not fan-out.
3. **R4, the proof.** One-shot panel-vs-flagship report after R1, case
   studies from the journal now. It converts the core claim from anecdote
   to evidence, or tells us to reposition while it is still cheap to do so.

## Where the panels were wrong or split (calibration notes)

- The "hosts time out at 15s" claim (glm, repeated by the deep synthesis)
  was contradicted by live observation in the same session. Confident,
  stale, and from the panellist class without web access: a concrete
  argument for R2's evidence-kind labelling and for weighting web-grounded
  panellists in research runs.
- qwen-max's "death of the capsule moat" (step 2) directly contradicts the
  step-1 consensus that capsules are the scarcest asset. The review adopts
  the reconciliation: capsules are the current efficiency mechanism,
  neutrality and governance are the durable moat.
- The first panel's synthesis recommended "drop H1" while its own
  highest-substance capsule (claude-opus) said "reframe H1"; the synth had
  down-weighted it for truncation. R1's weighting-distortion finding came
  from noticing exactly this.

## Tool-surface verdicts from live use

- **peer_rank: earns its keep.** ~$0.04 for a clean Borda separation that
  surfaced the best answer; fix the silent ranker dropout and serialise its
  cost (issue filed).
- **capsule_kind=research: earns its keep, with two fixes.** The
  claims/evidence/uncertainties shape beat decision-shape for the
  competitive scan, but sources_cited degrades to opaque indices and the
  4000-token cap zeroed 3/13 panellists.
- **refine: keep, harden.** See R3.
- **sequence: demote to library.** See R7.
- **synthesise: keep.** Cheap and useful (the critique re-synthesis added
  a usable verdict for $0.09), and dogfooding it caught the ledger clobber
  fixed in this PR.
