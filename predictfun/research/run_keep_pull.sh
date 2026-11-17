#!/usr/bin/env bash
# KEEP OR PULL driver (2026-11-17, session aea5bfb9; USER: "compare keeping versus pulling the same resting order at actual cancellation
# latency, measuring avoided losses against foregone gains"; the latest results review's next priority):
#   run_keep_pull.sh selftest   synthetic tape only (no market data is read): the selftest, then the mutation check of the selftest
#   run_keep_pull.sh smoke      selftest, then day 20261023 limited to 12 markets (reads the tape: only on the USER's go)
#   run_keep_pull.sh all        selftest, mutation check, one job per day Oct23-27, the report (reads the tape: only on the USER's go)
# Readings and controls are declared in pf_keep_pull.py. Each step through run_light on the claude slot.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?selftest|smoke|all}"
OUT=data/predict_fun/research/keep_pull_20261117; mkdir -p $OUT/logs
PY=predictfun/research/pf_keep_pull.py; MUT=predictfun/research/pf_keep_pull_mutants.py
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  selftest)
    job kp_selftest python3 $PY selftest || exit 1
    job kp_mutants python3 $MUT || exit 1;;
  smoke)
    job kp_selftest python3 $PY selftest || exit 1
    job kp_smoke python3 $PY day 20261023 12 || exit 1;;
  all)
    job kp_selftest python3 $PY selftest || exit 1
    job kp_mutants python3 $MUT || exit 1
    for d in 20261023 20261024 20261025 20261026 20261027; do job kp_day_$d python3 $PY day $d || exit 1; done
    job kp_report python3 $PY report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) KEEP-PULL $PHASE DONE" >> $OUT/logs/driver.log
