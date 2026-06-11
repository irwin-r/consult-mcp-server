# Security Policy

## Supported Versions

`consult-mcp-server` is pre-1.0. The latest minor release on the `main` branch
receives security fixes; older versions do not.

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

Please report security issues privately via
[GitHub Security Advisories](https://github.com/irwin-r/consult-mcp-server/security/advisories/new).
Do **not** open a public issue for security-impacting bugs.

You can expect:

- An acknowledgement within 5 working days.
- A coordinated disclosure window of up to 90 days before public release of
  the fix and an advisory. If a fix lands sooner we'll cut a release sooner.
- Credit in the advisory (unless you ask to remain anonymous).

## Threat model

`consult-mcp-server` runs as a local process invoked over MCP stdio by the
user's own agent (Claude Desktop, Cursor, Claude Code CLI, etc). It accepts
prompts and file/diff attachments from the agent, sends them to one or more
third-party LLM providers via [LiteLLM](https://github.com/BerriAI/litellm),
and writes per-run artefacts to disk.

**In scope:**

- Path-traversal / arbitrary file read via attachments or git_diff.
- Shell-injection in the git_diff subprocess.
- API-key leakage via exception messages or on-disk artefacts.
- World-readable run artefacts containing prompt data.
- Tool-description prompt injection that could mislead a calling agent.

**Out of scope:**

- Privacy of prompts at third-party LLM providers — that's governed by your
  agreement with the provider. Mixing first-party (Anthropic, OpenAI,
  Google) and aggregator (OpenRouter) panellists broadcasts your prompt to
  providers with different data-retention policies; the `privacy_tier` field
  in `consult/config/models.json` documents this per model, but enforcement
  is the operator's responsibility.
- Supply-chain attacks against pinned dependencies — we pin `litellm` and
  `mcp` to tight minor ranges and audit each bump, but transitive chains
  are large; we recommend running `pip-audit` periodically.
- Denial-of-service against the local process. The MCP transport is stdio,
  so the threat surface is the calling agent.

## Hardening defaults

- All run artefacts under `~/.consult/runs/<id>/` (or the XDG-spec location)
  are created with mode `0o700`.
- `git_diff` attachments always enforce trusted-roots containment. A
  `repo_path` must resolve under one of the `CONSULT_TRUSTED_REPO_ROOTS`
  directories; when that variable is unset, the only trusted root is the
  server's CWD, so any repo path outside CWD is rejected. This is the
  highest-impact attack surface because `git diff` spawns a subprocess.
- File attachments (`{path}` and bare-string forms) verify the path exists
  and is readable. When `CONSULT_TRUSTED_REPO_ROOTS` is set, they
  additionally enforce containment under one of those roots. When it's
  unset, file attachments are accepted as-is — the calling agent already
  has full filesystem access via its own tools, and refusing to read
  files it explicitly attached is friction without much added security.
  Operators handling sensitive corpora **should** set
  `CONSULT_TRUSTED_REPO_ROOTS` to opt into strict mode.
- The `git diff` subprocess runs with `shell=False`, neutralised global
  git config (`GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_SYSTEM=/dev/null`,
  `GIT_TERMINAL_PROMPT=0`), and ref-pattern validation that rejects shell
  metacharacters and leading dashes.
- API-key-shaped tokens (`sk-…`, `AIza…`, `Bearer …`, `x-api-key:`) in
  LiteLLM exception strings are redacted before hitting disk or the
  manifest.

Run `consult-doctor` for a per-install audit.
