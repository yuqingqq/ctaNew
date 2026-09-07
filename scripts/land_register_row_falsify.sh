#!/usr/bin/env bash
# Falsifier for scripts/land_register_row.sh (R-751). In a SCRATCH repo only (never /home/yuqing/ctaNew):
#   1. two concurrent --row landings SERIALISE -- both land, each commit adds exactly its one row and removes nothing,
#      the register carries each row once, origin ends 2 commits ahead of base;
#   2. known-bads REFUSE by name: a dirty register HELD, an id mismatch REFUSED, a duplicate id REFUSED, a foreign row in
#      the row file REFUSED, and --dry undoes its own insertion.
# Exit 0 only if every property holds. SCRATCH overrides the temp parent.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); SCRIPT="$HERE/land_register_row.sh"
T=$(mktemp -d "${SCRATCH:-/tmp}/rowlock.XXXXXX"); REG=orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md
fail(){ echo "FALSIFIER FAIL: $*"; exit 1; }
git init -q --bare "$T/origin.git"; git init -q -b mm-research "$T/clone"; cd "$T/clone" || exit 2
git config user.email falsifier@scratch; git config user.name falsifier
mkdir -p "$(dirname "$REG")"; printf 'header\n\n| id | seat | text |\n|---|---|---|\n| Q-AA-1 | AA | first |\n\ntail text\n' > "$REG"
git add "$REG"; git commit -q -m base; git remote add origin "$T/origin.git"; git push -q -u origin mm-research 2>/dev/null
export LAND_ROOT="$T/clone" LAND_REMOTE=origin LAND_BRANCH=mm-research LOCK_WAIT_S=60
printf '| Q-BB-2 | BB | second |\n' > "$T/rowB"; printf '| Q-CC-3 | CC | third |\n' > "$T/rowC"; printf 'falsifier landing\n' > "$T/msg"
# 1. CONCURRENT
bash "$SCRIPT" --row "$T/rowB" 'Q-BB-2' "$T/msg" > "$T/outB" 2>&1 & p1=$!
bash "$SCRIPT" --row "$T/rowC" 'Q-CC-3' "$T/msg" > "$T/outC" 2>&1 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
sed 's/^/  B| /' "$T/outB"; sed 's/^/  C| /' "$T/outC"
[ $r1 -eq 0 ] && [ $r2 -eq 0 ] || fail "a concurrent landing did not exit 0 (B=$r1 C=$r2)"
[ "$(git rev-list --count origin/mm-research)" -eq 3 ] || fail "expected 3 commits on origin, got $(git rev-list --count origin/mm-research)"
[ "$(grep -c '^| Q-BB-2 ' "$REG")" -eq 1 ] && [ "$(grep -c '^| Q-CC-3 ' "$REG")" -eq 1 ] || fail "a row is missing or duplicated"
for c in HEAD HEAD~1; do a=$(git show -U0 --format= $c -- "$REG" | grep -c '^+| Q-'); d=$(git show -U0 --format= $c -- "$REG" | grep -c '^-[^-]'); [ "$a" -eq 1 ] && [ "$d" -eq 0 ] || fail "$c adds $a rows, removes $d lines"; done
git diff --quiet HEAD -- "$REG" || fail "register dirty after landings"
grep -q 'tail text' "$REG" && [ "$(grep -c '^| Q-' "$REG")" -eq 3 ] || fail "table shape damaged"
echo "CONCURRENT OK: both landed, serialised, each commit +1 row / -0 lines, origin at 3 commits"
# 2. KNOWN-BADS
printf '| Q-DD-4 | DD | fourth |\n' > "$T/rowD"
echo "| Q-ZZ-9 | ZZ | someone's uncommitted edit |" >> "$REG"
bash "$SCRIPT" --row "$T/rowD" 'Q-DD-4' "$T/msg" > "$T/o" 2>&1; grep -q 'HELD REGISTER_DIRTY' "$T/o" || fail "dirty register not held: $(cat "$T/o")"; git checkout -q -- "$REG"
bash "$SCRIPT" --row "$T/rowD" 'Q-EE-5' "$T/msg" > "$T/o" 2>&1; grep -q 'REFUSED ROW_ID_MISMATCH' "$T/o" || fail "id mismatch not refused: $(cat "$T/o")"; git diff --quiet HEAD -- "$REG" || fail "mismatch left an insertion"
bash "$SCRIPT" --row "$T/rowB" 'Q-BB-2' "$T/msg" > "$T/o" 2>&1; grep -q 'REFUSED DUPLICATE_ID' "$T/o" || fail "duplicate not refused: $(cat "$T/o")"; git diff --quiet HEAD -- "$REG" || fail "duplicate left an insertion"
printf '| Q-DD-4 | DD | fourth |\n| Q-FF-6 | FF | foreign |\n' > "$T/rowDF"
bash "$SCRIPT" --row "$T/rowDF" 'Q-DD-4' "$T/msg" > "$T/o" 2>&1; grep -q 'REFUSED ROW_ID_MISMATCH' "$T/o" || fail "foreign row in the row file not refused: $(cat "$T/o")"; git diff --quiet HEAD -- "$REG" || fail "foreign left an insertion"
bash "$SCRIPT" --row "$T/rowD" 'Q-DD-4' "$T/msg" --dry > "$T/o" 2>&1; grep -q 'DRY OK' "$T/o" || fail "dry run failed: $(cat "$T/o")"; git diff --quiet HEAD -- "$REG" || fail "dry run left its insertion"
echo "KNOWN-BADS OK: dirty HELD, id mismatch REFUSED, duplicate REFUSED, foreign row REFUSED, dry run undone"
echo "FALSIFIER PASS"; cd /; rm -rf "$T"; exit 0
