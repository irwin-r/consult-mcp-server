# Self-improvement journal

One entry per cycle, newest first. Each cycle's agent appends its entry inside the
PR branch, so the journal lands atomically with the work it describes. The warden
(`scripts/self-improve.sh`) and the operating manual (`prompts/self-improver.md`)
describe the loop that writes this.

---

## 2026-06-15 — Disagreement scoring excludes truncated capsules

**Shipped.** `voting.panel_disagreement` scored over both `Status.OK` and
`Status.TRUNCATED` capsules. A truncated capsule is cut at the output-token cap
and often carries only its `position` field. Two such short fragments score high
SequenceMatcher similarity and drag the score toward false agreement, on exactly
the degraded panels where the recovery synth matters most. The score now uses
complete `Status.OK` capsules only. The existing `None` contract (fewer than two
comparable capsules) now fires when fewer than two OK capsules exist, which fails
closed so the flagship synth still runs. `medoid_slugs` and `_feature_string` are
left unchanged, so medoid voting keeps tolerating truncated entries for
best-available routing. One production line, an expanded docstring, four tests.

**How it surfaced.** Four scout lanes (correctness, tests, security, dead code)
plus deep verification. Every agent-flagged "critical" washed out: the task_store
cancel/complete race is already guarded in `complete`/`fail`; the synth cost-cap
unknown-pricing branch is documented intended policy; slow-tail recovery-by-slug
is safe because `state` is per-fanout-call; the "unredacted secrets in
responses/<slug>.json" success path carries a completion object, not the request
auth header, which is the threat model `redact.py` actually covers; the
`(real cost + cost_known=False)` test gap tests a combo no fanout path produces.
The disagreement skew was the one candidate with teeth, and the plan panel
confirmed it is real.

**Validation.** Pure in-process scoring logic, so no provider contract to probe.
A mutation check is the proof: reverting the one-line filter fails the three new
disagreement tests (one reads 0.26, false low-disagreement) and leaves the medoid
test green. Both live panel calls this cycle (8-way plan, 4-way diff) had zero
TRUNCATED entries, so the OK-only path returned the same score the old code would
have (0.91 and 0.86), confirming the change is behaviour-neutral on healthy
panels. Full suite 471 passing, ruff clean.

**Panel — plan (standard/consensus, ~$0.47 known):** disagreement 0.91, a genuine
split. 7 of 7 usable panellists agreed the skew is a real mechanism; the split was
only whether a mature codebase should ship the fix. Three personas (gemini-pro
security, qwen-max future_self, claude-sonnet staff_engineer) converged on the
identical 3-line `Status.OK` filter from different angles. claude-sonnet then
declined to ship it on churn grounds, which is the honest staff-engineer call, but
the security framing (a token-exhaustion attacker can drive disagreement to 0 and
bypass the gate, and the fix fails closed) plus the precondition check tipped it to
ship. The panel set one hard gate: verify every `disagreement` consumer handles
`None`. I did, before building.

**Panel — diff (code/code_review, ~$0.17 known x2):** the first review attached an
EMPTY diff because the work was staged but not committed, so `git_diff base=main
head=HEAD` resolved to nothing. Three of four reviewers reviewed my prose
description and rubber-stamped SHIP; only gemini-pro caught that the attachment was
empty. I committed and re-ran. The real review went 4/4 SHIP, RISK low, MERGE yes,
no blockers. Dismissed findings, each with reason: the double `_feature_string`
call is pre-existing and behaviour-neutral (panel agreed not worth touching); the
`Status.OK` enum-brittleness is already documented in the expanded docstring;
strict float `==` in the exclusion test is the stronger regression signal (3 of 4
reviewers defended it over `pytest.approx`); the downstream `None` grep was done
pre-build.

**Consult behaviour found.** The empty-diff footgun above is a real one: the manual's
step-6 review recipe (`base: main, head: HEAD`) silently reviews nothing when the
branch has no commit yet, and most panellists won't notice. Filed an issue to make
the `git_diff` attachment resolver surface a warning when `base..head` is empty,
rather than handing a reviewer a blank. Also noted: `kimi` slow-tail timed out at
180s on the plan call, already captured in `run_summary.no_value`.

**Panel spend this cycle:** ~$0.81 known-priced (plan $0.47, two diff reviews
$0.17 + $0.17); several OpenRouter panellists unpriced.

**Considered, not done this cycle:**
- gpt's design idea to split `disagreement` from an `evidence_quality` /
  `disagreement_basis_count` signal in the calibration block. The right long-term
  move, too big for this cycle, left as a design note for a future one.
- A walrus rewrite of the `panel_disagreement` comprehension to drop the
  pre-existing double `_feature_string` call. Behaviour-neutral micro-optimisation,
  out of scope for a correctness fix.

---

## 2026-06-12 — Web citation URLs reach bodies and research capsules

**Shipped.** Issue #61. Web-grounded panellists (sonar-pro) cite sources as
bracket markers whose URL list arrives as response metadata, never as message
content, so saved bodies and research capsules carried unusable entries like
`["[3]", "[4]"]`. A probe against `openrouter/perplexity/sonar-pro` settled
where the metadata lives: top-level `citations`/`search_results` arrive null
through OpenRouter, and the URLs survive only as `message.annotations` in
`url_citation` shape, list order matching the 1-indexed markers.

The fix is a new `consult/citations.py`. `harvest` reads the raw `model_dump`
dict (LiteLLM's models can strip provider-extra fields) and tolerates all
three shapes. Fanout appends a numbered `Sources:` footer to the body right
after `classify`, the one choke point every consumer reads. The capsule pass
splits the footer off before trimming and re-attaches it, and a deterministic
Python pass rewrites bare-marker `sources_cited` entries against the footer.
The extractor prompt now demands verbatim copying, no marker resolution, on
the panel's argument that cheap extractors hallucinate string lookups.

Validation: 10-way concurrent fanout (5x sonar-pro, 5x claude-haiku), run
twice. All 5 sonar bodies got footers, all 5 research capsules carried full
`Title - URL` entries with zero bare-marker leaks, haiku bodies stayed
byte-identical, no lost results, fanout 11.8s and 14.4s. Suite 459 passing
(22 new tests), ruff clean.

**Panel, plan (standard/consensus, $3.09 known):** 8/8 proceed. Adopted all
five amendments: probe the real metadata shape first (it reshaped `harvest`,
the documented top-level fields are dead on the OpenRouter route), read the
raw dict not the response object, Python-only marker resolution, a stable
footer delimiter with split-trim-reattach, and no `capsule_kind` gating since
an empty harvest already scopes the append.

**Panel, diff (code/code_review, $0.42 known):** 4/4 MERGE yes, RISK low x1
medium x3, verdict DISCUSS. Adopted: structural validation on the footer
split (an organic delimiter in prose no longer misparses), the inline-dedupe
rule tightened to require a numbered source-list line, a comment plus test
pinning bare-digit marker handling, and a drift debug log when the body cites
more markers than were harvested. Dismissed with reasons: a UUID delimiter
(bodies are human-read; validation covers the misparse), punctuation-wrapped
marker resolution (none observed live; pass-through is the safe default),
warning on unresolvable `[99]` (documented pass-through, a per-capsule warn
is noise), refine double-append (append runs once per fresh response, and
the dedupe makes it idempotent anyway), and "classify ordering" (already the
case; the reviewer misread the diff).

**Panel spend this cycle:** ~$3.51 known-priced (plan $3.09, diff $0.42);
live validation ~$0.11 further. Several OpenRouter panellists unpriced.

**Considered, not done this cycle:**
- The arbiter cost-gate input underestimate a scan lane flagged in
  `refine.py`. Same family as the arbiter-reservation idea the 2026-06-03
  entry rejected: bounded estimate-vs-actual variance, no clean win.
- The stance/rubric "prompt injection" a security lane flagged. The MCP
  caller composes their own panel prompts by design; no boundary is crossed.
- gpt-pro truncated at its 8000-token budget on the plan call and burned
  $1.98 for an empty capsule, after the issue 55 budget fix. Filed.
- Streamed web panellists get no footer (`stream_chunk_builder` drops
  annotations). Documented in the module docstring and filed.

---

## 2026-06-12 — Peer-rank forensics: drop reasons, spend, and a prompt bug

**Shipped.** Issue #58. This cycle resumed an interrupted one: the branch
`fix/si-58-peer-rank-forensics` existed with the whole change sitting
uncommitted on main, no PR, no journal entry. The work was finished, verified,
and shipped rather than redone.

A failing peer ranker used to vanish into an empty `per_ranker` pair list with
no reason, and the rank pass's spend never appeared in the panel result. Now
every ranker exit returns a `RankerOutcome` carrying the drop reason (truncated
to 200 chars) and the billed spend, the MCP handler serialises those as
objects, and the `peer_ranking` block itemises its own `cost_usd`/`cost_known`
before the roll-up. The serialised `per_ranker` shape changes from arrays to
objects; the old shape shipped in v0.4.1, so the commit carries the breaking
marker for the 0.5.0 notes.

Validating against reality found a second bug and the fix earned its keep
immediately: the rank prompt's JSON example hardcoded three labels, and on a
two-peer ranking 7 of 30 nano-tier rankers copied it verbatim and invented
"Gamma". Each was dropped, visible for the first time through the new reason
field. The example is now built from the ranker's real labels. The same
10-pass concurrent load (about 30 simultaneous ranker calls) went from 7
dropped to 0, wall time unchanged at 1.8s, all costs known.

**Panel — plan:** not run as a separate call. The build was inherited
complete, so the plan question (right scope, cheaper alternative) was folded
into the diff review. All four reviewers agreed the outcome shape was the
right long-term model over a handler-only patch or a dual-emit migration.

**Panel — diff (code/code_review, $0.36):** 4/4 OK, verdict SHIP, RISK low,
MERGE yes. Adopted: breaking-change marker plus release-notes visibility,
softer `cost_known` docstring, a comment on the lossy gather-exception branch,
a comment pinning the example's label-order invariant. Dismissed: a
double-count regression test (the exact panel+rank sum is already asserted in
`test_panel_peer_rank_attaches_ranking_and_cost`); explicitly setting
`cost_known=True` in `_attach_peer_ranking` (the panel result always carries
the key; filed nothing, it is one line if it ever bites); the
unresolvable-model path now reading as a failure outcome (nothing consumes
empty pairs as neutral, and the field shape is new in this same PR).

**Panel spend this cycle:** ~$0.37 known-priced (diff review $0.36, deepseek
unpriced; real-API probes ~$0.01).

**Considered, not done this cycle:**
- #61 research capsules losing the web panellist's source URLs: next in line
  once the inherited work was cleared.
- #59 tool-surface pruning and #60 blinded-refine manifest ids: both need a
  human call on surface changes, left for a cycle with room.

---

## 2026-06-03 — Cover refine's cost-gate refusal branches

**Shipped.** Three tests in `test_smoke.py` for cost-control branches in
`refine.py` that had no effective coverage. The 80%-of-cap safety valve in
`_round_cost_gate` (refine.py:606, refuse the next round when a panellist's
pricing is unknown and spend is already past 80% of cap) had zero tests; the only
cap test exercised the hard `cumulative + estimate > cap` branch. The partial-
fanout cost roll-up (refine.py:830-832) was reached by an existing test that
passed `cost_usd=0.0, cost_known=True`, so the accumulation line and the
`cost_known` flip never ran with effect. The arbiter unknown-pricing propagation
(refine.py:909-910, an unmapped-price arbiter must drag the result's `cost_known`
False) was guarded only by the `RefineResult` pydantic invariant, never end to
end through the loop. No behaviour change.

**Validation.** Tests-only, so no provider contract is touched; the live
validation this cycle was the plan panel call itself (a real fan-out). Coverage
confirms 606, 830-832, and 909-910 are now hit and the hard-cap branch at 597
stays uncovered, proving the valve test trips only the 80% branch. Mutation checks
on the valve: flipping `cap * 0.8` to `cap * 0.99` and `not est_known` to
`est_known` each make the test fail at the `fanout_calls == 1` assertion, so it is
sensitive to both the threshold and the boolean. Full suite 335 passing, ruff
clean.

**Panel — plan (standard/consensus, ~$1.24):** endorsed proceeding (3 OK
responders, qwen-max 0.95 / llama 0.9 / grok 0.85, plus the synth; 6 of 9
TRUNCATED). It caught a real trap in my round-1 mock arithmetic: round-1 cumulative
is fanout + capsule + arbiter, not fanout alone, so mocking only fanout at 0.82
risked pushing cumulative over the cap and tripping the hard-cap branch instead of
the valve. Adopted by controlling every cost source (noop annotate so no extractor
cost, zero-cost known arbiter), leaving round-1 cumulative provably 0.82. It also
endorsed rejecting the arbiter-budget-reservation alternative as architecturally
correct (LLM APIs are post-paid, so the estimate-vs-actual gap is inherent and
bounded by the arbiter's hardcoded 2000-token cap). claude-sonnet's freebie, the
arbiter `cost_known` propagation, became the third test.

**Panel — diff (code/code_review, $0.24):** 4/4 MERGE yes, verdict SHIP, no
blockers. Reviewers pinned off the Claude family that wrote the tests (gpt-codex,
gpt-mini, gemini-pro, deepseek); three RISK low, deepseek RISK medium for the
`est_n` call-counter coupling but still merge-yes. Took its three strengthenings:
assert round-1 `cost_usd == 0.82` to lock the valve's precondition against a hidden
cost leak, relax test 2's brittle dict-equality to individual call-counter asserts
(keeps the arbiter/synth short-circuit guarantee without coupling to the full set),
and assert test 3's `cost_usd == 0.001` so a regression that nukes the total when
the arbiter price is None is caught. Added a comment noting `est_n` encodes the
gate's call order.

**Panel spend this cycle:** ~$1.48 (known-priced portion; several OpenRouter and
qwen panellists were unpriced on the plan call).

**Considered, not done this cycle:**
- Reserving the arbiter's estimated cost in the fanout's per-round cap (`cap -
  cumulative - arbiter_est` rather than `cap - cumulative`). Rejected: the residual
  overspend is the inherent estimate-vs-actual variance, bounded by the arbiter's
  2000-token cap, and reserving wouldn't give a real guarantee since the arbiter
  actual can also exceed its estimate. A behaviour change with heavy validation and
  no clean correctness win.
- The slow-tail dropout race-recovery path in `runner.py` (a panellist that
  completes just before a cancel lands) only runs with `on_progress` set, and the
  suite drives it with `on_progress=None`, so that recovery branch is untested.
  Filed as a deferred issue.
- Smaller test gaps: medoid lexicographic tiebreaker in `voting.py` and the
  all-rankers-fail path in `peer_rank.py`. Filed as a deferred issue.

**Consult behaviour seen this cycle.** 6 of 9 panellists TRUNCATED at the decision
token cap on the standard-tier plan call (claude-sonnet, gpt-pro, gpt, gemini-pro,
kimi, glm), and the one extra OK responder (qwen-max) returned an empty capsule.
`run_summary.no_value` flagged all 7, so the #38 surfacing works. This is the same
truncation-at-cap pattern already tracked around #34/#37, so not re-filing.

---

## 2026-06-02 — Redact secrets at every boundary, not just the manifest

**Shipped.** The secret-shaped-token redactor (`_redact_secrets` + the key-shape
patterns) was applied at exactly one place: the manifest error field
(`_format_error_message`). The same provider exception flowed UNREDACTED into every
other place it can leave the process or hit disk: synth's failure text (returned as
`SynthResult.text` AND written to `synthesis.md`), the refine arbiter verdict
(`ArbiterVerdict.error`, returned in `RefineResult.verdicts` and persisted to
`arbiters/round-N.json`), the MCP error envelope and task-failure string returned to
the client, and three `logger.exception` sites that dump the full traceback (with the
exception repr) to the server log. The code's own threat-model comment names disk and
"a parent agent's transcript" as the things redaction must protect, so the control
was right but inconsistently wired.

The patterns and redactor moved into a new dependency-free `consult/redact.py` shared
by every boundary (this also removed `doctor.py`'s reach into a private name in
`runner.py`). Added `redact_exc` (redacts BEFORE truncating, so a key straddling the
cut point can't survive as a sub-20-char fragment the pattern misses) and
`redact_traceback` (formats the full chained traceback and redacts it, logged as a
plain message rather than raw `exc_info`, which a handler's formatter would otherwise
re-render unredacted). The redaction control had zero tests; this adds 15, including
boundary tests that plant a fake key in a provider exception and assert it reaches
none of `SynthResult.text`, `synthesis.md`, `ArbiterVerdict.error`, the MCP envelope,
or the captured log.

**Validation against reality.** Forced a real 401 from OpenAI, Anthropic, and
OpenRouter with clearly-fake keys (no billing on auth failure). All three mask or omit
the key provider-side (OpenAI prints `sk-…****…FFFF`, the others return a bare
"invalid key" body), so the worst case (a full key verbatim in the exception) does NOT
reproduce on plain auth errors for the current provider set. So this is honestly
consistency + defense-in-depth hardening, not the closing of an actively-bleeding
leak. It still matters: the redactor masks the verbatim `Authorization: Bearer <key>`
form that litellm debug mode or a header-echoing proxy produces (unit-tested), and the
real bug, the same exception being safe at one boundary and unsafe at four others, is
fixed regardless of today's provider behaviour.

**Panel — plan (standard/consensus, $2.14):** endorsed proceeding (5/6 non-truncated
panellists; none said rethink). Adopted its amendments: `ArbiterVerdict.error` was a
leak site I'd missed (flagged by claude-sonnet + glm); redact-before-truncate
(the `{e!s:.300}` slice would leave a short fragment); confirmed the header patterns
already carry `(?i)`. Deferred qwen-max's shift-left exception-attribute mutation
(`e.body`/`e.response.text`) and the `LITELLM_LOG=DEBUG` concern to follow-up issues,
since no current site reads those attributes and the mutation is fragile against
litellm/httpx drift. The synth (claude-opus) asserted, with confidence, that a filter
on a parent logger sees records propagated from child loggers ("This is correct
Python") and that my reason for rejecting a logging.Filter was wrong. I checked it
empirically: only the HANDLER filter ran, not the parent-logger filter. The synth
over-trusted the highest-confidence panellist (qwen-max 0.95) on a verifiable fact.
My original reasoning held, so the plan was unchanged.

**Panel — diff (code/code_review, $0.29):** 4/4 RISK low, MERGE yes, verdict SHIP, no
blockers. Reviewers pinned off the Claude family that wrote the code (gpt-codex,
gpt-mini, gemini-pro, deepseek). Took its one real nit: guard `redact_exc` against a
non-positive `limit` so a zero/negative cap can't produce a `text[:-1]` slice (no
caller hits it, but cheap to enforce). Did the repo-wide egress grep it asked for: no
remaining `logger.exception`/`exc_info` in source, and the one other
`ArbiterVerdict.error` path (`no_dimensions_or_score`) now goes through the redactor
too, so that field is uniformly safe. Dismissed the "redacting non-provider shape
errors is over-broad" nit: three reviewers called it harmless, and the consistency is
worth more than saving one regex pass.

**Panel spend this cycle:** ~$2.43 (known-priced portion; several OpenRouter and qwen
panellists were unpriced on the plan call).

**Considered, not done this cycle:**
- Shift-left mutation of `e.args`/`e.body`/`e.response.text` at the litellm call sites
  (secure-by-construction). Deferred: fragile against SDK attribute drift, and no
  current site reads those attributes. Filed as a follow-up issue.
- A `RedactingFilter` / `LITELLM_LOG` default so litellm's own loggers (outside the
  `consult` tree) can't echo headers under debug. Filed as a follow-up issue.
- The unredacted `attachments.py` `ValueError(f"{type(e).__name__}: {e}")` paths: left
  alone, those are file-read/parse errors, not provider exceptions.

---

## 2026-06-02 — Surface truncated/empty panellists in run_summary

**Shipped.** `_summarise_manifest` (handlers.py) builds `run_summary.no_value`, the
list the invoking agent reads to see which panellists returned nothing usable. Its
emptiness check keyed on the presence of a `findings` field, which only review-kind
capsules carry. So a decision-kind or research-kind panellist that truncated at the
output-token cap before emitting anything looked like a healthy contributor, and its
wasted spend was invisible.

The fix adds a `_capsule_is_empty` helper keyed on `kind` (position/recommendation/
key_points for decision, findings for review, claims/evidence for research) and uses
it for the empty check across all kinds. The caller still gates on a present capsule
so `extract_capsules=false` runs (every capsule null) aren't all flagged. Reason
strings are now kind-agnostic. The rollup had zero tests; this adds ten.

This was a dogfooding find from the plan review (closes #34). The plan-review call I
made this cycle reproduced it live: 9 panellists, 6 TRUNCATED, 4 with empty capsules
(one cost $0.61), and `no_value` came back `[]`. Running the fixed summariser over
that real on-disk manifest flags exactly those 4 and leaves the 2 truncated-but-
useful and 3 OK panellists alone.

**Panel — plan (standard/consensus, $1.08):** endorsed detect-and-surface and talked
me out of over-reaching. Convergence from the non-truncated responders (qwen-max
0.95, claude-sonnet 0.92, llama 0.85, grok 0.85). Adopted its refinements: `.strip()`
the decision string fields for whitespace-only extractions, default missing `kind` to
decision, keep the helper defensive against None/non-dict, and add two edge tests
(capsule=None lands via hard_fail not empty; truncated-but-non-empty stays unflagged).
claude-sonnet's second-order point (truncated-empty entries inflate the `usable()`
denominator) became issue #37, deferred until `no_value` data accumulates.

**Panel — diff (code/code_review, $0.25):** 4/4 RISK low, MERGE yes, verdict SHIP, no
blockers. Reviewers pinned off-family (gpt-codex, gpt-mini, gemini-pro, deepseek).
Took its one cheap test suggestion (pin the research `and` so claims-only or
evidence-only isn't flagged). Dismissed the rest: the uncertainties-only / verdict-
only "policy" question is intentional and documented in the helper docstring (those
are partial extractions, worth flagging); the non-dict guard is dead from the rollup
caller but exercised directly and kept as defensive; `findings_total` keying on the
field rather than the kind is equivalent in practice.

**Panel spend this cycle:** ~$1.33 (known-priced portion; several OpenRouter
panellists were unpriced).

**Considered, not done this cycle:**
- Raising `MAX_TOKENS_BY_KIND["decision"]` (2000) to stop verbose models truncating.
  Rejected: raises cost for every decision run to fix a minority case, and wouldn't
  surface the waste when it still happens.
- A continuation/retry of a truncated-but-promising answer. Bigger and riskier;
  revisit once `no_value` data shows how often it'd pay off.
- Excluding truncated-empty entries from synth's usable set or the `usable()`
  denominator. Filed as #37; gather data before changing the heuristic.

---

## 2026-06-02 — Bound panel expansion against an OOM DoS

**Shipped.** `expand_specs` turned caller-supplied `model:N` sugar into N specs
with no upper bound. A request like `model:1000000000` built a billion `ModelSpec`
objects and ran the process out of memory before the cost gate (which runs after
`expand_specs`) could reject it. `expand_specs` is the one choke point `fanout`,
`sequence`, and `refine` all share, including the untrusted MCP `models` array
(which has `minItems:1` but no `maxItems`).

The fix caps the expanded panel at `CONSULT_MAX_PANEL_SIZE` (default 64): reject a
per-spec count over the cap before `range()` allocates, strip leading zeros then
guard the digit length before `int()` (CVE-2020-10735), and re-check the running
total after each spec so many small specs can't sum past the cap. The schema
`maxItems` would only guard the wire, so the guard lives in the engine instead; a
follow-up issue tracks adding `maxItems` as wire-level defense-in-depth.

Validation: pathological `model:1000000000` now rejected in ~30µs with ~1.6KB
allocated (was an OOM); the real `fanout` dry-run path rejects an over-cap spec
before token-counting, and a legitimate dry-run still returns its estimate. Full
suite 307 passing (9 new tests), ruff clean.

**Panel — plan (standard/consensus, $0.67):** endorsed the choke-point approach
unanimously and sharpened it with three amendments I adopted — move the total-len
check inside the loop (not post-loop, or 1000 specs each at the per-spec cap build
64k objects first), add the pre-`int()` digit guard, and validate the env override
with a safe fallback. I diverged on one point: the panel suggested reading the env
at import; I read per-call to match the repo idiom and keep it testable.

**Panel — diff (code/code_review, $0.27):** 4/4 RISK low, MERGE yes, verdict SHIP,
no blockers. Took two of its optional nits — make the digit guard magnitude-aware
(strip leading zeros so a padded legal count isn't mis-rejected) and add the
aggregate-at-cap boundary test. Dismissed the rest: the env `int()` path is already
safe (repo is Python ≥3.11, so the 4300-digit limit applies and any ValueError
falls back to default); the README "narrative" note and a greedy-regex comment are
non-issues.

**Panel spend this cycle:** ~$0.94 (known-priced portion; several OpenRouter
panellists were unpriced).

**Considered, not done this cycle:**
- A run-mode/cancel race in `task_store` terminal-state guards is covered in code
  but untested (filed as a deferred test-coverage issue).
- Several flagship panellists TRUNCATED on the standard-tier plan review — one
  produced an empty capsule yet still cost ~$0.39. Filed as a consult-behavior
  issue.
- The unpriced-model `$0` cost path that can't be enforced against `max_run_usd` is
  a known, documented tradeoff (warns rather than blocks), so left alone.

---

## Bootstrap

The harness was set up by hand, not by a cycle:

- `prompts/self-improver.md` — the agent's operating manual (worker role).
- `scripts/self-improve.sh` — the warden: lock, daily spend cap, repo preflight,
  the merge gate, and post-merge auto-revert. The agent never merges; it opens a
  PR and writes a verdict, and the warden decides.
- A launchd job runs the warden every six hours.

The manual was itself reviewed by a standard consult panel before going live. The
panel's verdict (MAJOR_REVISIONS) is why the warden enforces budget, concurrency,
and the merge gate in code rather than trusting the model to self-police.
