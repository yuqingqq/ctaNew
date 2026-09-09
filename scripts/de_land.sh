#!/bin/bash
# THE LANDING GUARD, AS A SCRIPT (DE_PROCEDURE.md §9 open item 8, DE 164).
#
# Rule 21: a seat lands a worktree's exact bytes into the shared tree and
# commits BY PATHSPEC in the SAME act. The guard that makes that safe --
# "none of the paths I am about to overwrite has moved in the shared tree
# since my worktree read them" -- lived only in a seat's head and in
# scratch. Twice in one session a hand-written version PRINTED a warning
# and an `&&` chain sailed past it, once overwriting the USER's own commit
# 849bef2. So: this EXITS NON-ZERO, and it does the check BEFORE the copy.
#
#   scripts/de_land.sh <worktree> <msgfile> <path> [<path> ...]
#
# Exit codes:  2 usage · 3 the worktree is not clean · 4 a path moved in
# the shared tree since the worktree's base · 5 the copy or commit failed ·
# 6 the push was refused (the commit is STRANDED and is reported as such).
set -u
SHARED=/home/yuqing/ctaNew
WT="${1:-}"; MSG="${2:-}"; shift 2 || { echo "usage: de_land.sh <worktree> <msgfile> <paths...>"; exit 2; }
[ -n "$WT" ] && [ -n "$MSG" ] && [ "$#" -gt 0 ] || { echo "usage: de_land.sh <worktree> <msgfile> <paths...>"; exit 2; }
[ -f "$MSG" ] || { echo "REFUSED: no message file at $MSG"; exit 2; }

# (1) the worktree is clean except the ledger symlink (R-625)
DIRTY=$(git -C "$WT" status --short | grep -v '^?? data$' || true)
UNEXPECTED=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  f="${line:3}"
  keep=0
  for p in "$@"; do [ "$f" = "$p" ] && keep=1; done
  [ "$keep" = 0 ] && UNEXPECTED="$UNEXPECTED
$line"
done <<< "$DIRTY"
if [ -n "${UNEXPECTED// /}" ]; then
  echo "REFUSED: the worktree carries changes outside the pathspec:$UNEXPECTED"
  exit 3
fi

# (2) THE PRE-COPY GUARD. An afterwards-diff is how you learn you were lucky.
BASE=$(git -C "$WT" rev-parse HEAD) || exit 5
MOVED=$(git -C "$SHARED" diff --name-only "$BASE"..HEAD -- "$@" 2>/dev/null || true)
if [ -n "$MOVED" ]; then
  echo "REFUSED: these paths moved in the shared tree since $BASE:"
  printf '  %s\n' $MOVED
  echo "Refresh the worktree onto the tip and rebuild; never force past this."
  exit 4
fi

# (3) copy and commit by pathspec, in one act
for p in "$@"; do
  mkdir -p "$SHARED/$(dirname "$p")" || exit 5
  # A seat worktree's `data` is a SYMLINK to the shared tree's, so a path
  # under it is already the same file and `cp` would refuse. Skipping the
  # copy is correct there and ONLY there -- the test is inode identity,
  # not a path prefix, so it cannot be satisfied by a name that merely
  # looks like data/.
  if [ "$WT/$p" -ef "$SHARED/$p" ]; then
    echo "  same file (worktree symlink), no copy: $p"
  else
    cp -p "$WT/$p" "$SHARED/$p" || exit 5
  fi
done
git -C "$SHARED" add -f -- "$@" || exit 5
git -C "$SHARED" commit -F "$MSG" -- "$@" || exit 5
COMMIT=$(git -C "$SHARED" rev-parse HEAD)
if ! git -C "$SHARED" push origin mm-research; then
  echo "STRANDED: $COMMIT is committed and NOT pushed (another seat landed first)."
  echo "Report it; the coordinator rebases stranded commits. Do not rebase here."
  exit 6
fi
echo "LANDED $COMMIT"
for p in "$@"; do
  a=$(sha256sum "$WT/$p" | cut -d' ' -f1); b=$(sha256sum "$SHARED/$p" | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "  byte-identical $p" || { echo "  MISMATCH $p"; exit 5; }
done
