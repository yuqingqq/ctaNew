#!/usr/bin/env bash
# A landed declaration version is immutable (R-711). For every <family>_v<N>.json under DIR,
# no commit AFTER the baseline may modify a file that already existed (its creation commit is
# the one allowed touch). The exit-map chain lost two seats' blocks to in-place edits on
# 2026-09-06 (a81484c -> ba635de -> 40c8903, all 'producer_exit_maps_v2.json').
# Usage: declaration_immutability.sh <dir> [--base <commit>] [--falsify]
#   default base: a3de2ef (the R-711 repair) -- edits before it are HISTORY, printed, not refusals.
#   --falsify: base 56d3894 (v1's landing): producer_exit_maps_v2.json must FLAG (3 edits after
#              creation), producer_exit_maps_v1.json must PASS. Exit 1 if either does not.
set -u
DIR="${1:?dir}"; shift; BASE=a3de2ef; MODE=""
while [ $# -gt 0 ]; do case "$1" in --base) BASE="$2"; shift 2;; --falsify) MODE=falsify; BASE=56d3894; shift;; *) echo "unknown arg $1"; exit 2;; esac; done
cd "$(git -C "$DIR" rev-parse --show-toplevel)" || exit 2
check() { # prints OK|FORKED_BY_EDIT|UNTRACKED ; returns 1 on FORKED/UNTRACKED
  local f="$1"; local created n
  created=$(git log --diff-filter=A --format=%H -- "$f" | tail -1)
  [ -n "$created" ] || { echo "UNTRACKED $f"; return 1; }
  n=$(git log --format=%H "$BASE..HEAD" -- "$f" | grep -vc "^$created\$")
  if [ "$n" -eq 0 ]; then echo "OK $f edits_after_base=0"; return 0; else echo "FORKED_BY_EDIT $f edits_after_base=$n"; return 1; fi
}
if [ "$MODE" = "falsify" ]; then
  r1=$(check live/pm_research/declarations/producer_exit_maps_v2.json); r2=$(check live/pm_research/declarations/producer_exit_maps_v1.json); echo "$r1"; echo "$r2"
  case "$r1" in FORKED_BY_EDIT*) ;; *) echo "FALSIFIER FAIL: positive control did not flag"; exit 1;; esac
  case "$r2" in OK*) ;; *) echo "FALSIFIER FAIL: known-good did not pass"; exit 1;; esac
  # REV 89 §5.2: the denominator line is checked on BOTH real directories -- one with pre-base history and one with none --
  # so the control fires on the empty-array defect whichever directory the caller named.
  for d in live/pm_research/declarations live/mm_research/declarations "$DIR"; do
    hist=$("$0" "$d" --base "$BASE" 2>&1 | grep -c "^HISTORY (not judged):"); [ "$hist" -eq 1 ] || { echo "FALSIFIER FAIL: no HISTORY denominator line for $d"; exit 1; }
  done
  echo "FALSIFIER PASS (base $BASE; denominator line present)"; exit 0
fi
RC=0; HF=0; declare -A HFAM; NF=0
for f in "$DIR"/*_v[0-9]*.json; do [ -e "$f" ] || continue; NF=$((NF+1)); check "$f" || RC=1
  created=$(git log --diff-filter=A --format=%H -- "$f" | tail -1)
  if [ -n "$created" ]; then h=$(git log --format=%H "$created..$BASE" -- "$f" 2>/dev/null | wc -l); if [ "$h" -gt 0 ]; then HF=$((HF+1)); fam=$(basename "$f" | sed -E 's/_v[0-9]+\.json$//'); HFAM[$fam]=1; fi; fi
done
# REV 81 §1.3: the denominator -- what this run did NOT judge, named as history, so exit 0 reads as
# "nothing edited since $BASE" and never as "the declarations are immutable".
NFAM=$(set +u; echo "${#HFAM[@]}")  # an EMPTY associative array is "unbound" under set -u (bash 5.1); counted with -u off so the denominator line always prints
echo "HISTORY (not judged): $HF of $NF version files in $NFAM families had in-place edits BEFORE base $BASE"
echo "base $BASE; exit $RC"; exit $RC
