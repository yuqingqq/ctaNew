#!/usr/bin/env bash
# PULL-DEPTH driver (2026-11-13, session ce68ebe0): selftest, per-day rows, leave-one-day-out fit, per-day eval, report; claude slot,
# waits for a free slot.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/pull_depth_20261113; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
job pd_selftest python3 predictfun/research/pf_pull_depth.py selftest
for d in $DAYS; do job pd_rows_$d python3 predictfun/research/pf_pull_depth.py rows $d; done
job pd_fit python3 predictfun/research/pf_pull_depth.py fit
for d in $DAYS; do job pd_eval_$d python3 predictfun/research/pf_pull_depth.py eval $d; done
job pd_report python3 predictfun/research/pf_pull_depth.py report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
