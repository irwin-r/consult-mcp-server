# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the capsule schema and tool surface may change in minor releases.

## [Unreleased]

## [0.2.0] - 2026-05-21

First public release.

### Added

- **`consult-doctor` CLI** — diagnostics for local installs: validates the
  config, paths and permissions, checks provider key presence, and (with
  `--ping`) fires a 1-token call against each configured provider to verify
  keys work. Also `consult-doctor --config` to print a copy-paste-ready
  Claude Desktop / Cursor JSON snippet.
- **`--version` / `--check` flags** on every console script
  (`consult-mcp`, `consult-ledger`, `consult-view`, `consult-doctor`).
- **Privacy-tier annotations** on every model in the registry
  (`first_party` vs `aggregator`) so callers can reason about which
  providers receive their prompts.
- **Public exception taxonomy** (`ConsultError`, `UnknownModelError`,
  `BudgetExceededError`, `PathTrustError`, `ProviderError`,
  `CapsuleParseError`) — library consumers can now catch typed errors
  instead of branching on string matches.
- **Public API re-exports** from `consult/__init__.py` (`consult`, `panel`,
  `refine`, `sequence`, `synthesise`, plus the Pydantic result types).
- **XDG state-dir support** — runs default to
  `$XDG_STATE_HOME/consult/runs` on fresh installs; existing
  `~/.consult/runs` continues to work.
- Repo boilerplate: `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, issue templates, PR template.

### Security

- **Path-traversal hardening**: file attachments and `git_diff` repo paths
  must resolve (with symlinks followed, `strict=True`) under
  `CONSULT_TRUSTED_REPO_ROOTS`. Defaults to CWD only; `git_diff` with an
  explicit `repo_path` requires the env var to be set.
- **Run artefacts are mode 0o700** — per-run prompts (which may contain
  pasted secrets) are no longer world-readable on shared hosts.
- **`git diff` subprocess lockdown**: `GIT_CONFIG_GLOBAL=/dev/null`,
  `GIT_CONFIG_SYSTEM=/dev/null`, `GIT_TERMINAL_PROMPT=0` so a malicious
  `.gitattributes` filter or system git hook can't execute during diff.
- **Secret redaction in LiteLLM exception strings** — `sk-…`, `AIza…`,
  `Bearer …`, `x-api-key:` and similar patterns are scrubbed before
  exceptions hit disk or the manifest.
- **Pinned dependencies**: `litellm>=1.55,<1.57`, `mcp>=1.2,<2`,
  `pydantic>=2.6,<3` — we audit each minor bump manually.
- **Lazy startup**: `LITELLM_LOCAL_MODEL_COST_MAP=True` defaulted so
  `consult-mcp` start-up has no network I/O. Required for Smithery
  scanner compatibility.

### Changed

- Tool descriptions rewritten as agent-facing prompts with explicit
  "use when…/don't use for…" guidance.
- Server version reported in `InitializationOptions` now tracks
  `consult.__version__` instead of being hard-coded.

## [0.1.0] - prior to public release

Internal iterations (iter1..iter11). See git history.

[Unreleased]: https://github.com/irwin-r/consult-mcp-server/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/irwin-r/consult-mcp-server/releases/tag/v0.2.0
[0.1.0]: https://github.com/irwin-r/consult-mcp-server/releases/tag/v0.1.0
