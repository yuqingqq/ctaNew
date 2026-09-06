#!/usr/bin/env bash
# A landed declaration version is immutable (R-711). For every <family>_v<N>.json under the
# given directory, the file must have exactly ONE commit in its git history; more means a
# landed version was EDITED in place (FORKED_BY_EDIT) -- the exit-map chain lost two seats'
# blocks that way on 2026-09-06 (a81484c -> ba635de -> 40c8903, all 'v2').
# Usage: declaration_immutability.sh <dir> [--falsify]
#   --falsify: positive control = the real producer_exit_maps_v2.json (3 commits) must FLAG;
#              known-good = producer_exit_maps_v1.json (1 commit) must PASS. Fails if either does not.
set -u
DIR="${1:?dir}"; MODE="${2:-}"; cd "$(git -C "$DIR" rev-parse --show-toplevel)" || exit 2
check() { local f="$1"; local n; n=$(git log --oneline --follow -- "$f" | wc -l); if [ "$n" -eq 1 ]; then echo "OK $f commits=1"; return 0; elif [ "$n" -eq 0 ]; then echo "UNTRACKED $f"; return 1; else echo "FORKED_BY_EDIT $f commits=$n"; return 1; fi; }
if [ "$MODE" = "--falsify" ]; then
  P=live/pm_research/declarations/producer_exit_maps_v2.json; G=live/pm_research/declarations/producer_exit_maps_v1.json
  r1=$(check "$P"); r2=$(check "$G"); echo "$r1"; echo "$r2"
  case "$r1" in FORKED_BY_EDIT*) ;; *) echo "FALSIFIER FAIL: positive control did not flag"; exit 1;; esac
  case "$r2" in OK*) ;; *) echo "FALSIFIER FAIL: known-good did not pass"; exit 1;; esac
  echo "FALSIFIER PASS"; exit 0
fi
RC=0; for f in "$DIR"/*_v[0-9]*.json; do [ -e "$f" ] || continue; check "$f" || RC=1; done; exit $RC
