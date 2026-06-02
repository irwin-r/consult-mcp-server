# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the capsule schema and tool surface may change in minor releases.

## [0.3.0](https://github.com/irwin-r/consult-mcp-server/compare/v0.2.0...v0.3.0) (2026-06-02)


### Features

* add --watch streaming mode to the self-improve warden ([#36](https://github.com/irwin-r/consult-mcp-server/issues/36)) ([a2e2792](https://github.com/irwin-r/consult-mcp-server/commit/a2e2792751ac182e2db91df71ae31d804691ece0))
* add consult-gc to prune old run artifacts ([31a0126](https://github.com/irwin-r/consult-mcp-server/commit/31a0126cd6fda7371a1cc0ad1959c510cdba36a6))
* add self-improvement agent harness ([#31](https://github.com/irwin-r/consult-mcp-server/issues/31)) ([aed691d](https://github.com/irwin-r/consult-mcp-server/commit/aed691d0b03431e945945b84dee4bc097fb7c159))
* route responses-API models through litellm aresponses ([#30](https://github.com/irwin-r/consult-mcp-server/issues/30)) ([ce40a71](https://github.com/irwin-r/consult-mcp-server/commit/ce40a7198183d3987c221d01f3624bef320721a9))


### Bug Fixes

* address iteration-1 panel-review findings (tiers, dropout, prune, docs) ([6ebdde0](https://github.com/irwin-r/consult-mcp-server/commit/6ebdde03f79ffb7fb78fedad3c2b24ba02e09e6a))
* cap panel size so a huge model count can't OOM the process ([#35](https://github.com/irwin-r/consult-mcp-server/issues/35)) ([ca1ec55](https://github.com/irwin-r/consult-mcp-server/commit/ca1ec55f62d0ba92d6b3021ee6cd8f8303da73ef))
* capsule-retry cost, synth-failure surfacing, and cancelled-task status ([#18](https://github.com/irwin-r/consult-mcp-server/issues/18)) ([7bfdbc6](https://github.com/irwin-r/consult-mcp-server/commit/7bfdbc6abacf908c57be1655fe50864e33706212))
* **deps:** bump litellm onto the line that carries the proxy-auth fixes ([33d7fb0](https://github.com/irwin-r/consult-mcp-server/commit/33d7fb0f6c9edac7c4084aed72286dc3da7df252))
* make consult-doctor --config resolve without an installed binary ([06e5581](https://github.com/irwin-r/consult-mcp-server/commit/06e55815f47d24462c27a5fbf8f422ea3b8b256f))
* shield the fanout drain, drop dead code, add a parent-cancel drain test ([272213f](https://github.com/irwin-r/consult-mcp-server/commit/272213f200de24453edfa150c4758513c3b26c60))
* surface truncated/empty decision and research panellists in run_summary ([#38](https://github.com/irwin-r/consult-mcp-server/issues/38)) ([6418cf6](https://github.com/irwin-r/consult-mcp-server/commit/6418cf6eb29b7fa97adec8a5a10026df9706b8f4))
* warn when max_run_usd can't be enforced on unpriced panellists ([98b3696](https://github.com/irwin-r/consult-mcp-server/commit/98b369630652bf052544633feb5be695763f2e4a))


### Documentation

* document the otel extra, task mode, and the opt-in side-cars ([5d7cba0](https://github.com/irwin-r/consult-mcp-server/commit/5d7cba0e42c65838bee2379ab310058cc4005290))
* fix the redaction wording and fill in the repo-layout map ([cf9a6ad](https://github.com/irwin-r/consult-mcp-server/commit/cf9a6ad1ecd24bd1f4e73d247c67919cd19201cc))
* make the never-raised exception docstrings honest ([bdcf478](https://github.com/irwin-r/consult-mcp-server/commit/bdcf4780436ebd4c3d6865f6b7ccd2ad5471cdef))

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
- Opt-in aggregation side-cars (none run in the default flow): peer-ranking
  (`peer_rank`), medoid voting over `model:N` samples (`voting`), and refine
  elimination strategies (`strategies`).
- Long-running task mode (MCP SEP-1686): `consult` / `refine` / `sequence`
  accept `task: {ttl}` and return a task handle the client polls via
  `tasks/get`.
- OpenTelemetry spans via the `otel` extra, emitted per panellist call when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

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
