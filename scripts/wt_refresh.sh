#!/bin/bash
# Refresh a seat worktree to origin's tip and keep data/ the ledger SYMLINK (R-553/R-554/R-625).
# Why a script: a bare `checkout --detach` materialises every NEW tracked artifact under data/
# (they arrive without the skip-worktree bit) and replaces the symlink with a directory
# (REV 59 §8). Sparse-checkout does not help: under it git ignores update-index --skip-worktree.
set -e
WT="${1:?usage: wt_refresh.sh <worktree> [ref]}"; REF="${2:-origin/mm-research}"; LEDGER=/home/yuqing/ctaNew/data
git -C "$WT" sparse-checkout disable >/dev/null 2>&1 || true
git -C "$WT" fetch -q origin
[ -L "$WT/data" ] && rm "$WT/data"                       # drop the symlink so the checkout cannot write through it
git -C "$WT" checkout -q --detach "$REF"                # materialises data/ as a directory of tracked files
git -C "$WT" ls-files data | xargs -r git -C "$WT" update-index --skip-worktree   # R-554, covering the new ones
rm -rf "$WT/data" && ln -s "$LEDGER" "$WT/data"          # the ledger symlink, R-553
[ "$(readlink -f "$WT/data")" = "$LEDGER" ] || { echo "REFUSED: $WT/data is not the ledger symlink"; exit 2; }
N=$(git -C "$WT" status --short | grep -vc '^?? data$' || true)
echo "$WT at $(git -C "$WT" rev-parse --short HEAD); data -> $(readlink "$WT/data"); skip-worktree $(git -C "$WT" ls-files -v data | grep -c '^S') / $(git -C "$WT" ls-files data | wc -l); other status lines: $N"
