#!/usr/bin/env bash
# QUEUE PLACE LEDGER v2 driver (2026-11-17, session aea5bfb9; USER review: 2,054 counted states without a row or a status):
#   run_queue_place_ledger.sh selftest   synthetic tape only (no market data is read): the selftest, then the mutation check of the selftest
#   run_queue_place_ledger.sh smoke      selftest, then day 20261023 limited to 12 markets (reads the tape: only on the USER's go)
#   run_queue_place_ledger.sh all        selftest, mutation check, one job per day Oct23-27, the report (reads the tape: only on the USER's go)
# Readings and controls are declared in pf_queue_place_ledger.py. Each step through run_light on the claude slot.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?selftest|smoke|all}"
OUT=data/predict_fun/research/queue_place_20261117_v2; mkdir -p $OUT/logs
PY=predictfun/research/pf_queue_place_ledger.py; MUT=predictfun/research/pf_queue_place_ledger_mutants.py
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  selftest)
    job qpl_selftest python3 $PY selftest || exit 1
    job qpl_mutants python3 $MUT || exit 1;;
  smoke)
    job qpl_selftest python3 $PY selftest || exit 1
    job qpl_smoke python3 $PY day 20261023 12 || exit 1;;
  all)
    job qpl_selftest python3 $PY selftest || exit 1
    job qpl_mutants python3 $MUT || exit 1
    for d in 20261023 20261024 20261025 20261026 20261027; do job qpl_day_$d python3 $PY day $d || exit 1; done
    job qpl_report python3 $PY report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) QUEUE-PLACE-LEDGER $PHASE DONE" >> $OUT/logs/driver.log
