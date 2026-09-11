#!/usr/bin/env bash
# Land register rows in P-2026-003's COORDINATION.md (REV 83 §4, R-717; SERIALISED at R-751; id-shape guard + STRANDED exits at R-755).
# Usage:
#   land_register_row.sh <ids-regex> <commit-msg-file> [--dry]                  # legacy: the caller already inserted its row(s)
#   land_register_row.sh --row <rowfile> <ids-regex> <commit-msg-file> [--dry]  # serialised: the SCRIPT inserts the row(s) under the lock
#   <ids-regex>: the row ids this landing adds, e.g. 'Q-DA-332' or '(Q-DE-111|Q-DE-112)'.
#   <rowfile>: one complete table row per line, each beginning "| <id> |"; inserted after the LAST "| Q-" row of the table.
# THE LOCK (R-751): an exclusive flock on <root>/.git/p003_register.lock is held from before the fetch to after the push, so two
#   seats' landings SERIALISE instead of racing on one file (two whole-file read-modify-writes collided on 2026-09-07; the
#   FOREIGN_ROW guard caught both, the fix is serialisation). Waits LOCK_WAIT_S (default 600 s), then exits 12 LOCK_TIMEOUT.
# Closure = the post-condition on the commit's OWN diff: added rows = exactly the caller's ids, zero foreign rows, no landed line
#   changed or removed. Trailer: Landed-By: land_register_row.sh <sha256 of this file>.
# LAND_ROOT / LAND_REMOTE / LAND_BRANCH override the repo root, remote and branch -- for the falsifier
#   (scripts/land_register_row_falsify.sh), never for a real landing.
set -u
ROWF=""; if [ "${1:-}" = "--row" ]; then ROWF="${2:?rowfile}"; shift 2; fi
IDS="${1:?ids-regex}"; MSGF="${2:?commit-msg-file}"; DRY="${3:-}"
case "$IDS" in *'.*'*|*'.+'*) echo "REFUSED IDS_REGEX_TOO_LOOSE: an ids-regex containing .* or .+ disarms the foreign-row guard (REV 89 §5.1); name the ids"; exit 16;; esac
ROOT="${LAND_ROOT:-/home/yuqing/ctaNew}"; REMOTE="${LAND_REMOTE:-origin}"; BRANCH="${LAND_BRANCH:-mm-research}"
REG=orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md
cd "$ROOT" || exit 2
SELF_SHA=$(sha256sum "$0" | cut -c1-64)
LOCKF="$(git rev-parse --git-common-dir)/p003_register.lock"
exec 9>"$LOCKF" || { echo "LOCK_OPEN_FAILED $LOCKF"; exit 12; }
flock -w "${LOCK_WAIT_S:-600}" 9 || { echo "LOCK_TIMEOUT: another landing has held $LOCKF for ${LOCK_WAIT_S:-600} s"; exit 12; }
echo "LOCK HELD $(date -u +%H:%M:%SZ) pid $$"
diff_lines() { git diff --no-color -U0 -- "$REG" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)'; }
# 0. ROW MODE: fetch, fast-forward if the register is clean, insert the caller's rows from the file
if [ -n "$ROWF" ]; then
  out=$(git fetch -q "$REMOTE" "$BRANCH" 2>&1) || { echo "FETCH FAILED: $out"; exit 9; }
  if [ -n "$(git status --short -- "$REG")" ]; then echo "HELD REGISTER_DIRTY: another seat's uncommitted edit is in the register -- wait for its commit, do not withdraw it"; exit 3; fi
  if [ "$(git rev-list --count HEAD..$REMOTE/$BRANCH)" -gt 0 ]; then
    out=$(git merge -q --ff-only "$REMOTE/$BRANCH" 2>&1) || { echo "HELD BEHIND_AND_NOT_FF: $out"; exit 13; }
  fi
  python3 - "$REG" "$ROWF" "$IDS" <<'PY' || exit $?
import re,sys
reg,rowf,ids=sys.argv[1:4]
s=open(reg,encoding='utf-8').read().split('\n')
rows=[l.rstrip('\r') for l in open(rowf,encoding='utf-8').read().split('\n') if l.strip()]
if not rows: print("REFUSED EMPTY_ROWFILE"); sys.exit(6)
pat=re.compile(r'^\| ('+ids+r') \|')
bad=[l for l in rows if not pat.match(l)]
if bad: print("REFUSED ROW_ID_MISMATCH: a row does not begin '| <id> |' with an id in ("+ids+"): "+bad[0][:90]); sys.exit(6)
for l in rows:
    rid=l.split('|')[1].strip()
    if any(x.startswith('| '+rid+' |') for x in s): print("REFUSED DUPLICATE_ID: "+rid+" is already in the register (supersede in band with a new id)"); sys.exit(14)
q=[i for i,l in enumerate(s) if l.startswith('| Q-')]
if not q: print("REFUSED NO_TABLE: no '| Q-' row found"); sys.exit(15)
last=q[-1]; s[last+1:last+1]=rows
open(reg,'w',encoding='utf-8').write('\n'.join(s)); print("INSERTED %d row(s) after line %d" % (len(rows), last+1))
PY
fi
# 1. HOLD + SHAPE: only additions, and every added row/entry line carries one of the caller's ids
D=$(diff_lines)
if [ -z "$D" ]; then echo "NOTHING_TO_LAND: the register equals HEAD"; exit 3; fi
REMOVED=$(echo "$D" | grep -c '^-' || true)
[ "$REMOVED" -eq 0 ] || { echo "REFUSED REGISTER_EDITED: $REMOVED landed line(s) changed or removed (a landed row is never edited; supersede in band)"; echo "$D" | grep '^-' | head -3 | cut -c1-160; [ -n "$ROWF" ] && git checkout -q -- "$REG"; exit 4; }
FOREIGN=$(echo "$D" | grep -E '^\+(\| Q-|### R-)' | grep -vE "^\+(\| |### )?($IDS)\b" | grep -vE "^\+### ($IDS)\b" || true)
[ -z "$FOREIGN" ] && FOREIGN=$(echo "$D" | grep -E '^\+\| Q-' | grep -vE "^\+\| ($IDS) " || true)
[ -z "$FOREIGN" ] || { echo "REFUSED FOREIGN_ROW_IN_REGISTER: an added row is not among ($IDS):"; echo "$FOREIGN" | cut -c1-120 | head -3; exit 5; }
ADDED_IDS=$(echo "$D" | grep -oE "^\+(\| |### )($IDS)\b" | grep -oE "($IDS)" | sort -u | tr '\n' ' ')
[ -n "$ADDED_IDS" ] || { echo "REFUSED NO_ROW_WITH_THE_CALLER_IDS: the diff adds no line whose id matches ($IDS)"; exit 6; }
for _id in $ADDED_IDS; do case "$_id" in Q-[A-Z]*-[0-9]*|R-[0-9]*) [[ "$_id" =~ ^(Q-[A-Z]+-[0-9]+|R-[0-9]+)$ ]] || { echo "REFUSED ID_SHAPE: [$_id] is not Q-<SEAT>-<n> or R-<n>"; exit 17; };; *) echo "REFUSED ID_SHAPE: [$_id] is not Q-<SEAT>-<n> or R-<n>"; exit 17;; esac; done
[ "$DRY" = "--dry" ] && { echo "DRY OK: would land [$ADDED_IDS] ($(echo "$D" | grep -c '^+') added lines)"; [ -n "$ROWF" ] && git checkout -q -- "$REG" && echo "DRY: insertion undone"; exit 0; }
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
  out=$(git fetch -q "$REMOTE" "$BRANCH" 2>&1) || { echo "FETCH FAILED: $out"; exit 9; }
  if [ "$(git rev-list --count HEAD..$REMOTE/$BRANCH)" -gt 0 ]; then
    # The ONE sanctioned rebase in the shared tree (SEAT_PROTOCOL rule 21, R-755): only the commit this script just made,
    # only under the lock, only when NO OTHER PATH is dirty; abort on failure. Otherwise the commit is STRANDED and reported.
    OTHER_DIRTY=$(git status --short | grep -v '^??' | grep -v -- " $REG\$" || true)
    [ -z "$OTHER_DIRTY" ] || { echo "STRANDED (report it): origin moved during the landing and another path is dirty -- no rebase over another seat's files (rule 21); the coordinator rebases stranded commits at the first clean-tree moment (R-586)"; echo "$OTHER_DIRTY" | head -3; exit 10; }
    # DA 226: the comment above ASSERTS "only the commit this script just made" -- nothing CHECKED it.
    # A rebase replays EVERY unpushed commit on the branch. DA 223 began rewriting 13 belonging to BE, MEM
    # and DE and stopped on a conflict in another seat's file. REFUSE BY NAME instead of trusting the claim.
    FOREIGN=$(python3 - "$IDS" "$REMOTE/$BRANCH" <<'PYCHK'
import re, subprocess, sys
ids, upstream = sys.argv[1], sys.argv[2]
seats = sorted(set(re.findall(r'Q-([A-Z]+)-', ids)))
out = subprocess.run(['git','rev-list',upstream+'..HEAD'],capture_output=True,text=True).stdout.split()
foreign=[]
for c in out:
    subj = subprocess.run(['git','log','-1','--format=%s',c],capture_output=True,text=True).stdout.strip()
    own = any(subj.startswith(s+' ') or subj.startswith('Q-'+s+'-') or ('('+s+' ') in subj
              or (s=='REV' and subj.startswith('REVIEW ')) for s in seats)
    if not own: foreign.append(f"{c[:9]} {subj[:88]}")
# NAMED LIMIT: every commit here carries the same git author, so seat attribution is by SUBJECT.
# An unrecognised subject is treated as FOREIGN -- the loud direction, deliberately.
print("\n".join(foreign))
PYCHK
)
    if [ -n "$FOREIGN" ]; then
      echo "REFUSED FOREIGN_COMMIT_IN_REBASE_SET: rebasing onto $REMOTE/$BRANCH would replay $(echo "$FOREIGN" | wc -l) commit(s) this seat did not author:"
      echo "$FOREIGN" | sed 's/^/    /'
      echo "  The row is COMMITTED and UNPUSHED. Land it the rule-45 way: cherry-pick YOUR OWN commits onto a branch cut"
      echo "  from $REMOTE/$BRANCH and push that -- never rebase the shared branch to make your own push fast-forward."
      exit 18
    fi
    rb=$(git rebase -q "$REMOTE/$BRANCH" 2>&1) || { git rebase --abort 2>/dev/null; echo "STRANDED (report it): rebase onto $REMOTE/$BRANCH failed and was aborted: $rb"; exit 10; }
  fi
  out=$(git push -q "$REMOTE" "HEAD:$BRANCH" 2>&1) && { echo "PUSHED $(git rev-parse --short HEAD)"; exit 0; }
  echo "push refused ($i): $(echo "$out" | tail -1)"; sleep 5
done; exit 11
