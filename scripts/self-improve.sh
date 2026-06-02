#!/usr/bin/env bash
# Warden for the consult-mcp-server self-improvement agent.
#
# The model is an untrusted worker. This script is the warden: it enforces, in
# deterministic shell, everything the panel said an LLM cannot enforce about
# itself — a single-run lock, a hard daily spend cap, repo preflight, and the
# merge gate. The agent only ever OPENS a PR and writes a verdict file; this
# script decides whether that PR actually merges, using ground truth (the real
# changed files and real CI), never the agent's self-report.
#
# Usage:
#   self-improve.sh            run one cycle
#   self-improve.sh --dry-run  preflight + lock only, never launch the agent
#   self-improve.sh --pr-only  force pr-only for this run regardless of $AUTONOMY
#   self-improve.sh --watch    stream a readable agent trace to the terminal
#
# Tunables come from the environment (launchd sets none, so defaults apply):
#   AUTONOMY        auto-merge | pr-only        (default auto-merge)
#   MAX_DAILY_USD   hard daily panel-spend cap  (default 50)
#   MAX_RUN_USD     per panel-call cap          (default 8)
#   MIN_INTERVAL_S  refuse to start if last run was newer than this (default 3600)
#   CYCLE_TIMEOUT_S watchdog on the agent run   (default 2700 = 45m)
#   CI_TIMEOUT_S    watchdog on the CI wait      (default 1800 = 30m)
set -euo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${SELF_IMPROVE_REPO:-$(cd "$SELF_DIR/.." && pwd)}"
STATE="$REPO/.self-improve"
LOG="$STATE/runs.log"
LOCKDIR="$STATE/cycle.lock"
VERDICT="$STATE/verdict.json"
JOURNAL="$REPO/docs/self-improve/journal.md"
mkdir -p "$STATE"
# machine-local overrides (gitignored): NODE_BIN, CLAUDE_BIN, MAX_DAILY_USD, ...
# shellcheck disable=SC1091
[ -f "$STATE/config.env" ] && . "$STATE/config.env"
NODE_BIN="${NODE_BIN:-$HOME/.nvm/versions/node/v20.20.1/bin}"
CLAUDE="${CLAUDE_BIN:-$NODE_BIN/claude}"

AUTONOMY="${AUTONOMY:-auto-merge}"
MAX_DAILY_USD="${MAX_DAILY_USD:-50}"
MAX_RUN_USD="${MAX_RUN_USD:-8}"
MIN_INTERVAL_S="${MIN_INTERVAL_S:-3600}"
CYCLE_TIMEOUT_S="${CYCLE_TIMEOUT_S:-2700}"
CI_TIMEOUT_S="${CI_TIMEOUT_S:-1800}"

DRY_RUN=0
WATCH=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --pr-only) AUTONOMY="pr-only" ;;
    --watch) WATCH=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

mkdir -p "$STATE" "$(dirname "$JOURNAL")"
export PATH="$NODE_BIN:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

now() { date +%s; }
stamp() { date "+%Y-%m-%dT%H:%M:%S%z"; }
log() { printf '%s  %s\n' "$(stamp)" "$1" >>"$LOG"; }
halt() { log "HALT: $1"; exit 0; }   # exit 0: a refused cycle is normal, not a launchd failure

# ---- single-run lock (mkdir is atomic; macOS has no flock) -------------------
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  # steal a stale lock (a crashed run that never cleaned up)
  age=$(( $(now) - $(stat -f %m "$LOCKDIR" 2>/dev/null || now) ))
  if [ "$age" -gt 7200 ]; then
    log "stealing stale lock (age ${age}s)"; rmdir "$LOCKDIR" 2>/dev/null || true
    mkdir "$LOCKDIR" 2>/dev/null || halt "could not take lock after steal"
  else
    halt "another cycle holds the lock (age ${age}s)"
  fi
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

# ---- minimum-interval guard --------------------------------------------------
LAST_FILE="$STATE/last-run"
if [ -f "$LAST_FILE" ]; then
  delta=$(( $(now) - $(cat "$LAST_FILE" 2>/dev/null || echo 0) ))
  [ "$delta" -lt "$MIN_INTERVAL_S" ] && halt "last run was ${delta}s ago (< ${MIN_INTERVAL_S}s)"
fi

# ---- hard daily spend cap (preflight) ----------------------------------------
LEDGER="$STATE/spend-$(date +%F).txt"
spent=$(cat "$LEDGER" 2>/dev/null || echo 0)
over=$(awk -v a="$spent" -v cap="$MAX_DAILY_USD" 'BEGIN{print (a>=cap)?1:0}')
[ "$over" = "1" ] && halt "daily spend \$$spent >= cap \$$MAX_DAILY_USD"
remaining=$(awk -v a="$spent" -v cap="$MAX_DAILY_USD" 'BEGIN{printf "%.2f", cap-a}')

log "=== cycle start (autonomy=$AUTONOMY dry=$DRY_RUN spent=\$$spent remaining=\$$remaining) ==="

# ---- load provider keys for the headless agent's MCP subprocess --------------
# Safe parse: export NAME=VALUE literally, never evaluate the value as shell.
if [ -f "$REPO/.env" ]; then
  while IFS= read -r line; do
    case "$line" in
      ''|\#*) continue ;;
      [A-Za-z_]*=*) export "$line" 2>/dev/null || true ;;
    esac
  done < "$REPO/.env"
fi

# ---- repo preflight ----------------------------------------------------------
cd "$REPO"
git fetch --quiet origin || halt "git fetch failed"
git checkout --quiet main || halt "cannot checkout main"
git pull --quiet --ff-only || halt "main is not fast-forwardable (diverged?)"

# main must be green: latest run of tests + audit must be success
ci_bad=$(gh run list --branch main --limit 15 \
  --json workflowName,conclusion \
  -q '[.[] | select(.workflowName=="tests" or .workflowName=="audit")]
      | group_by(.workflowName) | map(.[0])
      | map(select(.conclusion!="success")) | length' 2>/dev/null || echo 1)
[ "$ci_bad" != "0" ] && halt "main CI is not green (tests/audit)"

# Note: an open release-please PR is the normal steady state (it accumulates
# commits until a human cuts the release), so it does NOT halt a cycle. Release
# safety is covered elsewhere: the agent never touches that PR, the warden only
# merges the agent's own PR number, and version/changelog files are protected.

if [ "$DRY_RUN" = "1" ]; then
  log "dry-run: preflight passed, not launching agent"
  echo "$(now)" >"$LAST_FILE"
  log "=== cycle end (dry-run ok) ==="
  exit 0
fi

# ---- run the agent (it codes, branches, pushes, opens a PR, writes verdict) --
rm -f "$VERDICT"
export AUTONOMY MAX_RUN_USD
export DAILY_REMAINING="$remaining"
export SELF_IMPROVE_VERDICT="$VERDICT"
export SELF_IMPROVE_JOURNAL="$JOURNAL"

AGENT_LOG="$STATE/agent-$(date +%F).log"
PROMPT_TEXT="$(cat "$REPO/prompts/self-improver.md")"
log "launching agent (timeout ${CYCLE_TIMEOUT_S}s watch=$WATCH)"
set +e
if [ "$WATCH" = "1" ]; then
  # Manual, watchable run: stream a readable trace to the terminal and keep
  # the raw JSON in the log. No watchdog here, since a human is watching and
  # can Ctrl-C; the timeout matters for the unattended path below.
  "$CLAUDE" -p "$PROMPT_TEXT" --permission-mode bypassPermissions \
    --output-format stream-json --verbose < /dev/null 2>>"$AGENT_LOG" \
    | tee -a "$AGENT_LOG" | python3 "$REPO/scripts/stream-pretty.py"
  agent_rc=${PIPESTATUS[0]}
else
  "$CLAUDE" -p "$PROMPT_TEXT" --permission-mode bypassPermissions \
    < /dev/null >>"$AGENT_LOG" 2>&1 &
  CL=$!
  ( sleep "$CYCLE_TIMEOUT_S"; kill -TERM "$CL" 2>/dev/null ) & WD=$!
  wait "$CL"; agent_rc=$?
  kill "$WD" 2>/dev/null
fi
set -e
log "agent exited rc=$agent_rc"
echo "$(now)" >"$LAST_FILE"

# ---- read the agent's verdict (advisory gates only) --------------------------
if [ ! -f "$VERDICT" ]; then
  log "no verdict file; agent shipped nothing or crashed. nothing to merge."
  log "=== cycle end (no verdict) ==="
  exit 0
fi
IFS=$'\t' read -r DID PR WYM RISK SPEND < <(python3 - "$VERDICT" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("\t\t\t\t0"); sys.exit(0)
def b(x): return "1" if x is True else "0"
print("\t".join([b(d.get("did_work")), str(d.get("pr_number") or ""),
                 b(d.get("would_you_merge")), str(d.get("risk") or ""),
                 str(d.get("panel_spend_usd") or 0)]))
PY
)

# bank the spend regardless of merge outcome
newspent=$(awk -v a="$spent" -v b="${SPEND:-0}" 'BEGIN{printf "%.4f", a+b}')
echo "$newspent" >"$LEDGER"
log "panel spend this cycle \$${SPEND:-0}; day total \$$newspent"

[ "$DID" != "1" ] && { log "agent reports no actionable work this cycle"; log "=== cycle end (idle) ==="; exit 0; }
[ -z "$PR" ] && { log "did_work but no pr_number; leaving any branch for next cycle"; log "=== cycle end (no pr) ==="; exit 0; }

log "agent opened PR #$PR (risk=$RISK would_merge=$WYM)"

# ---- merge gate: ground-truth checks, not the agent's self-report ------------
if [ "$AUTONOMY" = "pr-only" ]; then
  gh pr comment "$PR" --body "Opened by the self-improvement agent in pr-only mode. Waiting for human review." >/dev/null 2>&1 || true
  log "pr-only: PR #$PR left for human review"; log "=== cycle end (pr-only) ==="; exit 0
fi

# protected paths never auto-merge (the agent must not weaken its own guardrails)
is_protected() {
  case "$1" in
    prompts/self-improver.md|scripts/*|\
    .github/*|consult/schemas/*|consult/config/models.json|pyproject.toml|uv.lock|.self-improve/*)
      return 0 ;;
    *) return 1 ;;
  esac
}
changed=$(gh pr diff "$PR" --name-only 2>/dev/null || git diff --name-only "main...HEAD")
blocked=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if is_protected "$f"; then blocked="$blocked $f"; fi
done <<EOF
$changed
EOF
if [ -n "$blocked" ]; then
  gh pr comment "$PR" --body "Touches protected path(s):${blocked}. Held for human review; not auto-merged." >/dev/null 2>&1 || true
  log "BLOCK auto-merge: protected paths:${blocked}"; log "=== cycle end (protected) ==="; exit 0
fi

# advisory gates from the verdict
[ "$RISK" = "high" ] && { gh pr comment "$PR" --body "Agent self-rated risk=high; held for human review." >/dev/null 2>&1 || true; log "BLOCK: risk=high"; log "=== cycle end (risk) ==="; exit 0; }
[ "$WYM" != "1" ] && { gh pr comment "$PR" --body "Agent's own review said do-not-merge; held for human." >/dev/null 2>&1 || true; log "BLOCK: would_you_merge=false"; log "=== cycle end (no-merge verdict) ==="; exit 0; }

# CI must pass (bounded wait)
log "waiting on CI for PR #$PR (timeout ${CI_TIMEOUT_S}s)"
set +e
( gh pr checks "$PR" --watch --interval 20 >/dev/null 2>&1 ) & CK=$!
( sleep "$CI_TIMEOUT_S"; kill -TERM "$CK" 2>/dev/null ) & CW=$!
wait "$CK"; ci_rc=$?
kill "$CW" 2>/dev/null
set -e
[ "$ci_rc" != "0" ] && { gh pr comment "$PR" --body "CI not green; held (did not auto-merge)." >/dev/null 2>&1 || true; log "BLOCK: CI not green (rc=$ci_rc)"; log "=== cycle end (ci) ==="; exit 0; }

# ---- merge + post-merge health check with auto-revert ------------------------
log "merging PR #$PR"
gh pr merge "$PR" --squash --delete-branch || { log "merge failed"; log "=== cycle end (merge-fail) ==="; exit 0; }

git checkout --quiet main && git pull --quiet --ff-only
if ! uv run --quiet pytest -q >>"$STATE/health-$(date +%F).log" 2>&1; then
  bad=$(git rev-parse HEAD)
  log "POST-MERGE HEALTH FAIL on $bad — reverting"
  if git revert --no-edit "$bad" && git push origin main; then
    gh issue create --title "Auto-reverted a self-improvement merge that broke main" \
      --body "PR #$PR merged green but the post-merge suite failed on \`main\`. Reverted $bad. See .self-improve/health logs." >/dev/null 2>&1 || true
    log "reverted $bad and pushed; filed incident issue"
  else
    log "REVERT FAILED for $bad — main may be broken, human needed"
  fi
  log "=== cycle end (reverted) ==="; exit 0
fi

log "PR #$PR merged and main is healthy"
log "=== cycle end (merged) ==="
