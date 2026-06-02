# Self-improvement journal

One entry per cycle, newest first. Each cycle's agent appends its entry inside the
PR branch, so the journal lands atomically with the work it describes. The warden
(`scripts/self-improve.sh`) and the operating manual (`prompts/self-improver.md`)
describe the loop that writes this.

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
