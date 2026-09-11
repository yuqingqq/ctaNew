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
rc=$?

# ---- THE STAGE AFTER EVERY BOOK: the per-window generation census (BE 152).
# `--poll` returns when the LOCK IS TAKEN, not when the build finishes, so the
# census must wait for the UNIT -- a seat's own driver once `systemctl stop`ped
# a live build 15 s in by confusing the two. Only the book stage waits; frag
# and tape return as before.
if [ "$STAGE" = "book" ] && [ "$rc" -eq 0 ]; then
  _f() { systemctl --user show "$1" -p "$2" | sed "s/^$2=//"; }
  for _i in $(seq 1 400); do
    [ "$(_f "$UNIT" SubState)" != "running" ] && break
    sleep 20
  done
  if [ "$(_f "$UNIT" LoadState)" = "loaded" ] && [ "$(_f "$UNIT" ExecMainStatus)" = "0" ]; then
    echo "--- book landed; per-window generation census for $DAY ---"
    # NOT under the heavy lock: measured 3.89 GiB leaf against a 14.00 GiB
    # slice, so it coexists with a valuation under the 12 GB rule. It runs
    # INSIDE research.slice so it stays accounted and capped.
    systemctl --user reset-failed "census${DAY}" >/dev/null 2>&1
    systemd-run --user --unit="census${DAY}" --slice=research.slice       -p MemoryMax=6G -p CPUQuota=100% -p RemainAfterExit=yes       -p WorkingDirectory=/home/yuqing/ctaNew       --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew       -p StandardOutput=append:/home/yuqing/ctaNew/data/pm_5min/derived/be152census.log       -- /home/yuqing/pricer-sol/venv/bin/python3          live/pm_research/be_book_window_census.py "$DAY" >/dev/null 2>&1
  else
    echo "book unit did not exit 0 -- census NOT run (a census of a failed build certifies nothing)"
  fi
fi
exit $rc
