#!/bin/bash
# Refresh a seat worktree to origin's tip and keep data/ the ledger SYMLINK (R-553/R-554/R-625).
# Why a script: a bare `checkout --detach` materialises every NEW tracked artifact under data/
# (they arrive without the skip-worktree bit) and replaces the symlink with a directory
# (REV 59 §8). Sparse-checkout does not help: under it git ignores update-index --skip-worktree.
set -e
WT="${1:?usage: wt_refresh.sh <worktree> [ref]}"; REF="${2:-origin/mm-research}"; LEDGER=/home/yuqing/ctaNew/data
git -C "$WT" sparse-checkout disable >/dev/null 2>&1 || true
git -C "$WT" fetch -q origin
# R-554 sweep BEFORE the checkout: tracked data files that read as deleted behind a symlink block a checkout
git -C "$WT" ls-files data | xargs -r git -C "$WT" update-index --skip-worktree 2>/dev/null || true
# R-677 (BE 67's process note): a modified tracked file whose bytes EQUAL $REF's blob is a LANDED copy (rule 21's
# copy-into-the-shared-tree form leaves the worktree dirty at its old HEAD) -- restore it silently instead of refusing.
# Any modified file that differs from $REF is a real edit and still refuses below.
IDENT=0
while IFS= read -r line; do
  f="${line:3}"; [ -n "$f" ] || continue
  case "$line" in ' M '*|'M  '*|'MM '*|'?? '*) ;; *) continue ;; esac
  wt_blob=$(git -C "$WT" hash-object -- "$f" 2>/dev/null || echo none); ref_blob=$(git -C "$WT" rev-parse -q --verify "$REF:$f" 2>/dev/null || echo absent)
  if [ "$wt_blob" = "$ref_blob" ]; then
    case "$line" in '?? '*) rm -f -- "$WT/$f" && IDENT=$((IDENT+1)) ;;   # an untracked copy of a file $REF tracks: the checkout recreates it
                    *) git -C "$WT" checkout -q -- "$f" && IDENT=$((IDENT+1)) ;; esac
  fi
done < <(git -C "$WT" status --short | grep -v '^?? data$')
[ "$IDENT" -gt 0 ] && echo "restored $IDENT landed-identical file(s) (bytes equal to $REF's) before the checkout"
[ -L "$WT/data" ] && rm "$WT/data"                       # drop the symlink so the checkout cannot write through it
if ! git -C "$WT" checkout -q --detach "$REF" 2>/dev/null; then   # the seat's own uncommitted edits block a checkout
  echo "REFUSED: $WT has uncommitted edits that the checkout would overwrite -- preserve them (a WIP HELD commit) or land them first:"; git -C "$WT" status --short | grep -v '^?? data$' | head; [ -e "$WT/data" ] || ln -s "$LEDGER" "$WT/data"; exit 3
fi
git -C "$WT" ls-files data | xargs -r git -C "$WT" update-index --skip-worktree   # R-554, covering the new ones
rm -rf "$WT/data" && ln -s "$LEDGER" "$WT/data"          # the ledger symlink, R-553
[ "$(readlink -f "$WT/data")" = "$LEDGER" ] || { echo "REFUSED: $WT/data is not the ledger symlink"; exit 2; }
N=$(git -C "$WT" status --short | grep -vc '^?? data$' || true)
echo "$WT at $(git -C "$WT" rev-parse --short HEAD); data -> $(readlink "$WT/data"); skip-worktree $(git -C "$WT" ls-files -v data | grep -c '^S') / $(git -C "$WT" ls-files data | wc -l); other status lines: $N"
