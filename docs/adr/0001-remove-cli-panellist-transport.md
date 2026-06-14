# 1. Remove the CLI-as-panellist transport

Date: 2026-06-14

## Status

Accepted. Supersedes the experimental CLI transport shipped in `cli_executor.py`.

## Context

`cli_executor.py` let a registry entry with `provider: "cli"` and a
`cli_command` array run a local executable as a panellist: spawn the process,
write the prompt to stdin, read stdout, and adapt the result into the
chat-completions shape the rest of the pipeline consumes. It was fully wired
into `runner.fanout` and the cost estimator.

In practice it carried cost without earning it:

- No packaged config used it, it was undocumented, and it had zero known users.
- It executed arbitrary subprocesses, which is a different and larger trust and
  sandboxing surface than the HTTP calls every other transport makes.
- It ran in minutes against the panel's seconds, and billed outside the cost
  model (cost was hard-coded to zero, so caps and the ledger could not see it).
- CLI orchestration is `pal-mcp-server`'s headline feature. This project's
  distinct ground is governed deliberation (blinding, capsules, arbiters, cost
  caps), and the CLI path pulled toward someone else's strength.

A focused peer-ranked panel went 5 for 5 in favour of deletion
(run `20260611-043441-83971`; Borda: claude-sonnet 9, deepseek 6, gpt-mini 6,
gemini-pro 3, grok 0). The top-ranked framing was that "experimental" is a
maintenance liability with a friendlier label, and that zero users means zero
moat to defend.

## Decision

Remove the CLI transport: delete `cli_executor.py`, the `provider == "cli"`
branch in `runner.fanout`, the matching `cli` skips in the cost estimator, and
the `cli_command` / `cli_env` fields on `ModelEntry`. The Responses-API adapter
keeps its own SimpleNamespace adaptation; nothing else depended on the CLI
module.

## Consequences

- The fan-out path has one fewer branch and one less subprocess-execution
  surface to reason about. Cost accounting no longer has a "free" panellist
  class that the cap and ledger were blind to.
- A user who set `provider: "cli"` in their own `~/.consult/models.json` will
  now get an `UnknownModelError`-style failure for that entry rather than a
  silent subprocess call. No packaged config used it, so the default surface is
  unaffected.

## Reintroduction triggers

Bring the transport back only if one of these holds:

1. Two or more concrete user requests for running a local CLI as a panellist.
2. Agentic CLIs become the dominant substrate panellists are expected to run
   through, such that HTTP-only transport is the limiting factor.

If reintroduced, it should bill into the cost model rather than hard-coding zero,
and run inside the same trust boundary the rest of the engine assumes.

## Rejected alternative

claude-opus argued to keep the module behind an "experimental" label
(run `20260611-042818-67174`). Rejected: an experimental feature with no users
is the maintenance liability this ADR removes, and the reintroduction triggers
above are a cheaper way to keep the option open than carrying live code.
