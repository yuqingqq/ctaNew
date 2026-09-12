#!/bin/bash
# BE 188: the BUILD TREE rule, as BE 163 ruled it (option b), in ONE place.
#
# It replaces `[ "$HEAD" = "$PIN" ]` in launch_stage2.sh. That equality was a
# LITERAL THAT MUST TRACK A MOVING THING -- the same defect class as PARAMS_REL,
# RULED_DAYS and the V2 params pin. The build tree has legitimately advanced
# past the pin three times (v33, 6d22d78, bacb4e3, b34ed9f) while the five
# pinned digests never moved; under the equality every one of those advances
# silently disarmed the launcher, and it did: the preflight ADMITTED 20260909
# with 25/25 rows and the launcher refused it at rc=4.
#
# THE RULED PREDICATE, both halves required:
#   (1) the tree's HEAD DESCENDS FROM the pin        -- it is the same lineage
#   (2) each pinned module's WORKING-COPY BYTES equal the PIN'S BLOB, read from
#       the repository object store, never from the tree under test
# (2) is what actually protects the build; (1) is what stops an unrelated tree
# carrying identical files from passing for the pinned lineage.
#
# Usage:  be_tree_pin_guard.sh <worktree> <pin>
#   exit 0  admitted
#   exit 4  REFUSED PIN_NOT_ANCESTOR: <head>      (tree is not of the pin's lineage)
#   exit 4  REFUSED PINNED_DIGEST_MOVED: <file>   (a pinned module's bytes moved)
#   exit 2  usage / not a git worktree
set -u
PINNED_MODULES="be_daybook_build.py be_gate1_fragment.py be_gate1_state_tape.py de_phase4_diag_runner.py de_head_scoring.py"

guard() {   # guard <worktree> <pin> -> 0 admit / 4 refuse (reason on stdout)
  local WT="$1" PIN="$2" H
  H=$(git -C "$WT" rev-parse HEAD 2>/dev/null) || { echo "REFUSED NOT_A_GIT_TREE: $WT"; return 2; }
  git -C "$WT" rev-parse -q --verify "${PIN}^{commit}" >/dev/null 2>&1 || { echo "REFUSED PIN_NOT_IN_REPO: $PIN"; return 2; }
  git -C "$WT" merge-base --is-ancestor "$PIN" "$H" || { echo "REFUSED PIN_NOT_ANCESTOR: $H"; return 4; }
  local f pinblob wtblob
  for f in $PINNED_MODULES; do
    pinblob=$(git -C "$WT" rev-parse -q --verify "${PIN}:live/pm_research/$f" 2>/dev/null || echo absent)
    wtblob=$(git -C "$WT" hash-object -- "$WT/live/pm_research/$f" 2>/dev/null || echo missing)
    [ "$pinblob" = "$wtblob" ] || { echo "REFUSED PINNED_DIGEST_MOVED: $f (pin ${pinblob:0:12} tree ${wtblob:0:12})"; return 4; }
  done
  echo "tree admitted: $H descends from ${PIN:0:12}; all 5 pinned digests equal the pin's blobs"
  return 0
}

# ---- the guard's OWN falsifier. Every cell drives guard() itself, with REAL
# commits out of this repository -- no tree is moved, no worktree is created,
# nothing is written. Both directions, and each refusal by its own name.
if [ "${1:-}" = "--falsify" ]; then
  WT=/home/yuqing/ctaNew-wt-fwd
  PIN=7ed5a9015f75de64feeeeaad21d97e4eecc2b15c   # the build pin
  NOTANC=$(git -C "$WT" rev-parse origin/mm-research)  # a real commit, NOT of this lineage
  MOVED=5df2f468c77c                                    # a real ANCESTOR whose be_daybook_build.py blob differs
  RC=0
  _note() { if [ "$2" = "1" ]; then echo "  PASS  $1"; else echo "  FAIL  $1"; RC=1; fi; }
  o=$(guard "$WT" "$PIN"); r=$?
  _note "ADMITS the live build tree (descendant + 5 digests) with exit 0" "$([ "$r" = "0" ] && echo 1 || echo 0)"
  _note "and says WHY it admitted, computed, not asserted"               "$(printf '%s' "$o" | grep -q 'descends from' && echo 1 || echo 0)"
  o=$(guard "$WT" "$NOTANC"); r=$?
  _note "REFUSES a tree that is not of the pin's lineage with exit 4"    "$([ "$r" = "4" ] && echo 1 || echo 0)"
  _note "and names it PIN_NOT_ANCESTOR"                                  "$(printf '%s' "$o" | grep -q 'PIN_NOT_ANCESTOR' && echo 1 || echo 0)"
  o=$(guard "$WT" "$MOVED"); r=$?
  _note "REFUSES a DESCENDANT whose pinned module bytes moved, exit 4"   "$([ "$r" = "4" ] && echo 1 || echo 0)"
  _note "and names PINNED_DIGEST_MOVED and the file"                     "$(printf '%s' "$o" | grep -q 'PINNED_DIGEST_MOVED: be_daybook_build.py' && echo 1 || echo 0)"
  _note "  (the digest cell's commit really IS an ancestor -- so the cell tests (2), not (1))" "$(git -C "$WT" merge-base --is-ancestor "$MOVED" "$(git -C "$WT" rev-parse HEAD)" && echo 1 || echo 0)"
  o=$(guard /tmp "$PIN"); r=$?
  _note "REFUSES a path that is not a git tree with exit 2"              "$([ "$r" = "2" ] && echo 1 || echo 0)"
  echo "{\"falsifier\": \"be_tree_pin_guard\", \"n\": 8, \"failed\": $RC}"
  exit "$RC"
fi

WT="${1:?usage: be_tree_pin_guard.sh <worktree> <pin>}"; PIN="${2:?usage: be_tree_pin_guard.sh <worktree> <pin>}"
guard "$WT" "$PIN"; exit $?
