#!/bin/bash
# Refresh a seat worktree to origin's tip WITHOUT materialising data/ (R-625).
# data/ is a symlink to the canonical ledger (R-553); sparse-checkout excludes
# the tracked artifacts under data/ so no checkout can replace the symlink (REV 59 §8).
set -e
WT="${1:?usage: wt_refresh.sh <worktree> [ref]}"; REF="${2:-origin/mm-research}"
git -C "$WT" fetch -q origin
git -C "$WT" sparse-checkout init --no-cone >/dev/null 2>&1 || true
git -C "$WT" sparse-checkout set '/*' '!/data/' >/dev/null
git -C "$WT" checkout -q --detach "$REF"
if [ ! -L "$WT/data" ]; then rm -rf "$WT/data"; ln -s /home/yuqing/ctaNew/data "$WT/data"; fi
[ "$(readlink -f "$WT/data")" = "/home/yuqing/ctaNew/data" ] || { echo "REFUSED: $WT/data is not the ledger symlink"; exit 2; }
echo "$WT at $(git -C "$WT" rev-parse --short HEAD); data -> $(readlink "$WT/data"); status lines: $(git -C "$WT" status --short | wc -l)"
