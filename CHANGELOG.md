# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the capsule schema and tool surface may change in minor releases.

## [0.5.0](https://github.com/irwin-r/consult-mcp-server/compare/v0.4.1...v0.5.0) (2026-06-14)


### ⚠ BREAKING CHANGES

* carry drop reasons and spend in peer rank forensics ([#65](https://github.com/irwin-r/consult-mcp-server/issues/65))

### Features

* report the cost per usable capsule in the run_summary rollup ([bb19427](https://github.com/irwin-r/consult-mcp-server/commit/bb19427d52207bb419b64dce6bebf75ecdbcac4f))


### Bug Fixes

* accumulate re-synthesis spend instead of clobbering run cost ([1a1e24f](https://github.com/irwin-r/consult-mcp-server/commit/1a1e24fd022bb272c2c4de5b8a33f7b672f53a96))
* carry drop reasons and spend in peer rank forensics ([#65](https://github.com/irwin-r/consult-mcp-server/issues/65)) ([be2b7e2](https://github.com/irwin-r/consult-mcp-server/commit/be2b7e27949d10206883bbc8a4f56b0ecf88ff4b))
* carry web citation URLs into bodies and research capsules ([#68](https://github.com/irwin-r/consult-mcp-server/issues/68)) ([522cbea](https://github.com/irwin-r/consult-mcp-server/commit/522cbea7b8597c3c958a65d6d8fc92158fa8137d))
* floor each panellist's output budget at its model budget ([8a68700](https://github.com/irwin-r/consult-mcp-server/commit/8a68700d4c522d33e5db0300765d4c5b28623986))
* give refine results the run_summary health rollup as well ([a71567b](https://github.com/irwin-r/consult-mcp-server/commit/a71567b57f311fc85cb3a06e6b2613af3ab940a0))
* harden the refine arbiter against verdict parse failures ([dac803b](https://github.com/irwin-r/consult-mcp-server/commit/dac803b2bd3a0dccefd8ce8139b49474e8fb875c))
* honour dry_run on refine and sequence ([fb23167](https://github.com/irwin-r/consult-mcp-server/commit/fb23167af108ea1e919d8ce3299915e6a709376a))
* kind-aware refine budgets and reasoning_content salvage ([6639845](https://github.com/irwin-r/consult-mcp-server/commit/66398454c5228ecf7d8ed20b296fae6afedc3628))
* make the capsule empty-extraction retry kind-aware ([474d008](https://github.com/irwin-r/consult-mcp-server/commit/474d008e1f8418a7c5bfa0fa2cd16b94b44dbfd4))
* pass capsule_kind through refine's round fanout ([46f5aff](https://github.com/irwin-r/consult-mcp-server/commit/46f5aff7168ac3a2620293af0f9dd9b296c738fc))
* salvage reasoning_content when content is empty ([2b2b8df](https://github.com/irwin-r/consult-mcp-server/commit/2b2b8df709d6b602dad0f53179d85210fb3c8784))
* stop paying for truncated panellists and harden the refine arbiter ([4f086a3](https://github.com/irwin-r/consult-mcp-server/commit/4f086a37634f82ce97fad0a033bdfffaca3d9a58))


### Documentation

* 2026-06 strategy dogfood review, plus two dogfood-found fixes ([c196b40](https://github.com/irwin-r/consult-mcp-server/commit/c196b4063f6aa17b3ff909fe43853bcd6ab299ea))
* add the 2026-06 strategy dogfood review ([edc0faf](https://github.com/irwin-r/consult-mcp-server/commit/edc0fafdf7bb2cacff28420a6c6f6f732f4bf0e0))
* log the 2026-06-11 strategy dogfood friction ([42861d6](https://github.com/irwin-r/consult-mcp-server/commit/42861d66c431e9dfdc52a21148c122a5256733a5))
* log the issue 55 fix-pass friction with the probe driver ([5502940](https://github.com/irwin-r/consult-mcp-server/commit/5502940a418cb8dcf74b5a5e67f172e01c71d819))

## [0.4.1](https://github.com/irwin-r/consult-mcp-server/compare/v0.4.0...v0.4.1) (2026-06-11)


### Bug Fixes

* bound caller-supplied arrays in the MCP schemas ([31899d1](https://github.com/irwin-r/consult-mcp-server/commit/31899d1274556e316ff9e94d8ce178a3c40614c4))
* clear the open-issue backlog ([f2ec732](https://github.com/irwin-r/consult-mcp-server/commit/f2ec7325be6f7465f62b7c71a76096b9d6f3d2c6))
* count substantive truncated entries in RunHandle.usable() ([f86eb15](https://github.com/irwin-r/consult-mcp-server/commit/f86eb15a18d5a7f2fdb8c3b4b6930e64b1d34e91))
* redact secrets on litellm loggers and exception objects ([38b0ee0](https://github.com/irwin-r/consult-mcp-server/commit/38b0ee08050e70f120748a5530f037797c78991e))

## [0.4.0](https://github.com/irwin-r/consult-mcp-server/compare/v0.3.0...v0.4.0) (2026-06-11)


### ⚠ BREAKING CHANGES

* a user config that relied on full replacement now inherits packaged entries it omitted; set unwanted entries to null to remove them.

### Features

* complete SEP-1686 task mode with tasks/result and tasks/cancel ([ac9b786](https://github.com/irwin-r/consult-mcp-server/commit/ac9b786a9d66aff5a42fbd0a91fef88da57cc379))
* deep-merge user registry overrides over the packaged config ([c0e2881](https://github.com/irwin-r/consult-mcp-server/commit/c0e288192665b0ff9ebe65b70c4c260dec06ef10))
* expose peer ranking and consult dry_run; retire dead surface area ([26c404d](https://github.com/irwin-r/consult-mcp-server/commit/26c404ddf276001dd6b7bd23f0fe7ccc0a6a4672))


### Bug Fixes

* **deps:** bump aiohttp, pyjwt, and pip past their advisories ([5beaf03](https://github.com/irwin-r/consult-mcp-server/commit/5beaf0301c8659ca574b9069697b5191de1f0bdc))
* enforce the cost cap across the whole consult pipeline ([18cb94c](https://github.com/irwin-r/consult-mcp-server/commit/18cb94cd7a61f6fc1d6108b334db53206e86f468))
* keep panellist identity stable across refine rounds ([06f1dd1](https://github.com/irwin-r/consult-mcp-server/commit/06f1dd12230bfa927c786e59b485c655adc4ca18))
* land the full-codebase review findings ([ebfbefe](https://github.com/irwin-r/consult-mcp-server/commit/ebfbefebed8e25b78b61c9d66ac125c6a31ceaf5))
* raise the documented typed exceptions from the engine ([bf793e3](https://github.com/irwin-r/consult-mcp-server/commit/bf793e381a42544f6a2d5bd5ca968d6a26ecbd3f))
* redact secrets at log, result, and disk boundaries ([#41](https://github.com/irwin-r/consult-mcp-server/issues/41)) ([4858c28](https://github.com/irwin-r/consult-mcp-server/commit/4858c2839f89c58346549fb8142bca94636c344b))
* tolerate malformed numeric env vars instead of crashing mid-run ([797e8e7](https://github.com/irwin-r/consult-mcp-server/commit/797e8e73a67d7a7cebd3c35df9f528da89a90851))


### Performance Improvements

* move heavy disk and render work off the event loop ([b7758df](https://github.com/irwin-r/consult-mcp-server/commit/b7758dfcc46b0d9695554993a8dcb7c0c597ad44))


### Documentation

* align security and registry docs with actual behaviour ([a13c603](https://github.com/irwin-r/consult-mcp-server/commit/a13c60345aa705daaccfe5826c0b1a15969c7206))
* refresh repo layout, test commands, and Docker volume ownership ([6f8ca37](https://github.com/irwin-r/consult-mcp-server/commit/6f8ca37d0c32f530bb67403995014fec7cb571bb))

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
