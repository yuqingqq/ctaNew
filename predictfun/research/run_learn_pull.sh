#!/usr/bin/env bash
# LEARN-PULL driver (2026-11-13, session ce68ebe0): selftest, per-day eval, report; claude slot, waits for a free slot.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/learn_pull_20261113; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
job lp_selftest python3 predictfun/research/pf_learn_pull.py selftest
for d in $DAYS; do job lp_day_$d python3 predictfun/research/pf_learn_pull.py day $d; done
job lp_report python3 predictfun/research/pf_learn_pull.py report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
