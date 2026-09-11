#!/bin/bash
# BE 148: launch one FWD stage, with be_build_preflight AS THE FIRST STAGE --
# it refuses BEFORE the lock is offered for, so a day is never half-built and
# never built twice. A NEW file: launch_stage.sh is being read right now by
# be147frag0909's offer loop and editing a running script is its own hazard.
set -u
STAGE="${1:?stage: frag|tape|book}"; DAY="${2:?day}"; UNIT="${3:?unit}"
WT=/home/yuqing/ctaNew-wt-fwd
PIN=7ed5a9015f75de64feeeeaad21d97e4eecc2b15c
D=/home/yuqing/ctaNew/data/pm_5min/derived
export BE_WORKTREE="$WT"          # BEFORE the preflight, which checks it
export BE_MEMORY_MAX=11869652313
export BE_MEMORY_MAX_BASIS="the envelope the coordinator used for p003fwd0907d, read from its own run record"
export BE_POLL_CEILING="${BE_POLL_CEILING:-250}"

case "$STAGE" in
  frag) MOD=be_gate1_fragment.py;   PFSTAGE=fragment
        OUT=$D/harmful_exposure_rows_v3_gate1_${DAY}_btc.json; ARGS="--day $DAY";;
  tape) MOD=be_gate1_state_tape.py; PFSTAGE=tape
        OUT=$D/phase2_state_tape_gate1_${DAY}_btc.json;        ARGS="--day $DAY";;
  book) MOD=be_daybook_build.py;    PFSTAGE=book
        OUT=$D/be_daybook_${DAY}_btc__L250ms__FWD1.pkl
        ARGS="--day $DAY --placement-latency-ms 250 --artifact-revision FWD1";;
  *) echo "REFUSED UNKNOWN_STAGE $STAGE"; exit 2;;
esac

# ---- STAGE 0: THE PREFLIGHT. Refuses before the lock, by name.
echo "--- preflight: $DAY $PFSTAGE ---"
if ! ( cd /home/yuqing/ctaNew && PM_DATA_ROOT=/home/yuqing/ctaNew \
         python3 live/pm_research/be_build_preflight.py --stage "$PFSTAGE" "$DAY" ); then
  echo "REFUSED BY PREFLIGHT -- the lock was never offered for. Nothing launched."
  exit 5
fi
# ---- the launcher's own guards, kept: they are cheap and they are the last
#      word between the preflight and the emit.
H=$(git -C "$WT" rev-parse HEAD)
[ "$H" = "$PIN" ] || { echo "REFUSED PIN_MISMATCH: $H"; exit 4; }
[ -e "$OUT" ] && { echo "REFUSED OUTPUT_EXISTS: $OUT"; exit 4; }
echo "preflight clean -- offering for the lock: $UNIT ($STAGE $DAY)"
bash /home/yuqing/ctaNew/live/pm_research/be_heavy_run.sh --poll "$UNIT" "$MOD" $ARGS
