#!/usr/bin/env bash
# QUEUE PLACE TEST driver (2026-11-17, session aea5bfb9; USER "Ok go"): run_queue_place.sh all -> selftest, one job per day Oct23-27 (event
# study at new-best births: FIRST / LATE / EARLY; causal states: JOIN / REST1 / IMPROVE; controls inside), the report (readings in
# pf_queue_place_test.py). Each step through run_light on the claude slot.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/queue_place_20261117; mkdir -p $OUT/logs
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  all)
    job qp_selftest python3 predictfun/research/pf_queue_place_test.py selftest || exit 1
    for d in 20261023 20261024 20261025 20261026 20261027; do job qp_day_$d python3 predictfun/research/pf_queue_place_test.py day $d || exit 1; done
    job qp_report python3 predictfun/research/pf_queue_place_test.py report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) QUEUE-PLACE $PHASE DONE" >> $OUT/logs/driver.log
