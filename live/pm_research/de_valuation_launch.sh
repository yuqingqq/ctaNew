#!/bin/bash
# LAUNCH A FORWARD VALUATION, OR REFUSE -- with the tree assertion the
# builds have and the valuations did not.
#
# WHY THIS EXISTS. DE 239: `deFV0907` was launched with its working
# directory in a tree at 3a7756a while the ruled pipeline commit is
# 7ed5a90. Had the heavy-run lock been free, day one's number would have
# come from the wrong instrument and NOTHING WOULD HAVE CAUGHT IT -- the
# BOOK's provenance verifies fine; it is the instrument reading it that
# would have been wrong. The lock conflict saved it, which is luck.
#
# THIS IS THE THIRD INSTANCE IN ONE SESSION of one shape: a unit whose
# ExecStart path looks right while the tree behind it is not. A path in a
# command line is a LABEL. `git rev-parse HEAD` in that path is the FACT.
#
# AND IT WAITS AS A UNIT, NOT AS A PLAN. An intention that lives in a
# seat's turn dies with the turn (it did, at 03:46). The waiting belongs
# to systemd.
set -u
PIN="7ed5a9015f75de64feeeeaad21d97e4eecc2b15c"
TREE="${VAL_TREE:-/home/yuqing/ctaNew-wt-deval}"
UNIT="${1:?usage: de_valuation_launch.sh <unit> <day> <book> [waitunit]}"
DAY="${2:?}"
BOOK="${3:?}"
WAITFOR="${4:-}"

# ---- THE ASSERTION, BEFORE ANYTHING ELSE -----------------------------
HEAD=$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo NONE)
if [ "$HEAD" != "$PIN" ]; then
  echo "REFUSED VALUATION_TREE_IS_NOT_THE_PIPELINE_COMMIT: $TREE is at" \
       "${HEAD:0:12} and the ruled pipeline commit is ${PIN:0:12}. A" \
       "valuation from the wrong tree produces a number from the wrong" \
       "instrument, and the book's provenance would still verify." >&2
  exit 2
fi
for m in de_forward_evaluator de_settlement_control_run \
         de_settlement_control_aggregate de_asymmetry_null_run \
         de_matched_cancel_control de_multiday_gate1_runner; do
  A=$(sha256sum "$TREE/live/pm_research/$m.py" | cut -d' ' -f1)
  B=$(git -C "$TREE" show "$PIN:live/pm_research/$m.py" | sha256sum | cut -d' ' -f1)
  if [ "$A" != "$B" ]; then
    echo "REFUSED VALUATION_MODULE_BYTES_DIFFER_FROM_THE_PIN: $m.py" >&2
    exit 2
  fi
done
echo "$(date -u +%H:%M:%SZ) tree $TREE at ${HEAD:0:12} == pin; 6/6 module digests match"

# ---- WAIT AS A UNIT, NOT AS A PLAN -----------------------------------
if [ -n "$WAITFOR" ]; then
  for _ in $(seq 1 480); do
    systemctl --user is-active --quiet "$WAITFOR" 2>/dev/null || break
    sleep 15
  done
  echo "$(date -u +%H:%M:%SZ) $WAITFOR is no longer active"
fi

# be_heavy_run.sh hardcodes WT=${BE_WORKTREE:-/home/yuqing/ctaNew-wt-be}
# and cds there, DISCARDING the tree selected above. That silently
# relocated the whole computation on 09-07 and left no error -- only the
# ABSENCE of a field in the record. Export both: BE_WORKTREE PREVENTS the
# relocation, DE_VALUATION_EXPECTED_TREE lets the pre-flight DETECT it if
# the prevention ever fails. One of these is not enough; a guard that can
# be bypassed without an error is worse than no guard.
export BE_WORKTREE="$TREE"
export DE_VALUATION_EXPECTED_TREE="$TREE"
cd "$TREE/live/pm_research" || exit 2
exec bash be_heavy_run.sh "$UNIT" de_forward_value_day.py \
  --day "$DAY" --book "$BOOK" \
  --out-dir /home/yuqing/ctaNew/data/pm_5min/derived/fwd \
  --n-draws 500 --seed "${DAY//-/}" --days-scored "$DAY" \
  --n-declared 7 --derived /home/yuqing/ctaNew/data/pm_5min/derived
