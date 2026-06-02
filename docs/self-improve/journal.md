# Self-improvement journal

One entry per cycle, newest first. Each cycle's agent appends its entry inside the
PR branch, so the journal lands atomically with the work it describes. The warden
(`scripts/self-improve.sh`) and the operating manual (`prompts/self-improver.md`)
describe the loop that writes this.

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
