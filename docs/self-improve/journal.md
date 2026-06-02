# Self-improvement journal

One entry per cycle, newest first. Each cycle's agent appends its entry inside the
PR branch, so the journal lands atomically with the work it describes. The warden
(`scripts/self-improve.sh`) and the operating manual (`prompts/self-improver.md`)
describe the loop that writes this.

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
