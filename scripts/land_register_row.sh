#!/usr/bin/env bash
# Land register rows / entries in P-2026-003's COORDINATION.md (REV 83 §4, R-717).
# Usage: land_register_row.sh <ids-regex> <commit-msg-file> [--dry]
#   <ids-regex>: the row/entry ids this landing adds, e.g. 'Q-DA-332' or '(Q-DE-111|Q-DE-112)' or 'R-717'.
#   The working-tree register must differ from HEAD ONLY by added lines whose id matches the regex.
# Closure = the post-condition on the commit's OWN diff, not the hold: added rows = exactly the caller's ids,
# zero foreign rows, no landed line changed or removed. Trailer: Landed-By: land_register_row.sh <sha256 of this file>.
set -u
IDS="${1:?ids-regex}"; MSGF="${2:?commit-msg-file}"; DRY="${3:-}"
ROOT=/home/yuqing/ctaNew; REG=orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md
cd "$ROOT" || exit 2
SELF_SHA=$(sha256sum "$0" | cut -c1-64)
diff_lines() { git diff --no-color -U0 -- "$REG" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)'; }
# 1. HOLD + SHAPE: only additions, and every added row/entry line carries one of the caller's ids
D=$(diff_lines)
if [ -z "$D" ]; then echo "NOTHING_TO_LAND: the register equals HEAD"; exit 3; fi
REMOVED=$(echo "$D" | grep -c '^-' || true)
[ "$REMOVED" -eq 0 ] || { echo "REFUSED REGISTER_EDITED: $REMOVED landed line(s) changed or removed (a landed row is never edited; supersede in band)"; echo "$D" | grep '^-' | head -3 | cut -c1-160; exit 4; }
FOREIGN=$(echo "$D" | grep -E '^\+(\| Q-|### R-)' | grep -vE "^\+(\| |### )?($IDS)\b" | grep -vE "^\+### ($IDS)\b" || true)
[ -z "$FOREIGN" ] && FOREIGN=$(echo "$D" | grep -E '^\+\| Q-' | grep -vE "^\+\| ($IDS) " || true)
[ -z "$FOREIGN" ] || { echo "REFUSED FOREIGN_ROW_IN_REGISTER: an added row is not among ($IDS):"; echo "$FOREIGN" | cut -c1-120 | head -3; exit 5; }
ADDED_IDS=$(echo "$D" | grep -oE "^\+(\| |### )($IDS)\b" | grep -oE "($IDS)" | sort -u | tr '\n' ' ')
[ -n "$ADDED_IDS" ] || { echo "REFUSED NO_ROW_WITH_THE_CALLER_IDS: the diff adds no line whose id matches ($IDS)"; exit 6; }
[ "$DRY" = "--dry" ] && { echo "DRY OK: would land [$ADDED_IDS] ($(echo "$D" | grep -c '^+') added lines)"; exit 0; }
# 2. COMMIT by file pathspec with the trailer
{ cat "$MSGF"; printf '\nLanded-By: land_register_row.sh %s\n' "$SELF_SHA"; } > "$MSGF.landed"
git add -- "$REG" && git commit -q -F "$MSGF.landed" -- "$REG" || { echo "COMMIT FAILED"; git restore -q --staged --worktree -- "$REG" 2>/dev/null; exit 7; }
# 3. POST-CONDITION on the commit's own diff
PATHS=$(git show --name-only --format= HEAD | grep -c .); PD=$(git show --no-color -U0 --format= HEAD -- "$REG" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)')
PREM=$(echo "$PD" | grep -c '^-' || true); PFOR=$(echo "$PD" | grep -E '^\+\| Q-' | grep -vcE "^\+\| ($IDS) " || true)
if [ "$PATHS" != "1" ] || [ "$PREM" != "0" ] || [ "$PFOR" != "0" ]; then echo "POST-CONDITION FAILED (paths=$PATHS removed=$PREM foreign=$PFOR) -- reverting"; git revert -q --no-edit HEAD; exit 8; fi
echo "POST-CONDITION OK: paths 1, removed 0, foreign 0, ids [$ADDED_IDS], trailer $SELF_SHA"
# 4. PUSH: capture, test, then trim
for i in 1 2 3; do
  out=$(git fetch -q origin mm-research 2>&1) || { echo "FETCH FAILED: $out"; exit 9; }
  if [ "$(git rev-list --count HEAD..origin/mm-research)" -gt 0 ]; then
    rb=$(git rebase -q origin/mm-research 2>&1) || { git rebase --abort 2>/dev/null; echo "REBASE FAILED: $rb"; exit 10; }
  fi
  out=$(git push -q origin mm-research 2>&1) && { echo "PUSHED $(git rev-parse --short HEAD)"; exit 0; }
  echo "push refused ($i): $(echo "$out" | tail -1)"; sleep 5
done; exit 11
