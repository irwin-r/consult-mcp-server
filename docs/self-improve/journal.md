# Self-improvement journal

One entry per cycle, newest first. Each cycle's agent appends its entry inside the
PR branch, so the journal lands atomically with the work it describes. The warden
(`scripts/self-improve.sh`) and the operating manual (`prompts/self-improver.md`)
describe the loop that writes this.

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
