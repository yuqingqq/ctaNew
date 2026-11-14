#!/usr/bin/env bash
# STOP-MODEL driver (2026-11-14, session aea5bfb9): claude slot, waits for a free slot.
#   run_stop_model.sh data   -> selftest, per-day rows (with parity / falsifier / brute-force controls), scale choice + audit v2 (then stop: the USER asked
#                               for the data to be checked and its issues cleared before any fit)
#   run_stop_model.sh model  -> per-fold fits, per-test-day alignment lanes, report
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?data|model}"
OUT=data/predict_fun/research/stop_model_20261114; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
TESTS="20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
case "$PHASE" in
  data)
    job sm_selftest python3 predictfun/research/pf_stop_model.py selftest
    for d in $DAYS; do [ -f $OUT/rows_$d.parquet ] || job sm_rows_$d python3 predictfun/research/pf_stop_model.py rows $d; done
    job sm_scalecheck python3 predictfun/research/pf_stop_model.py scalecheck
    job sm_audit2 python3 predictfun/research/pf_stop_model.py audit2;;
  model)
    for t in $TESTS; do job sm_fit_$t python3 predictfun/research/pf_stop_model.py fit $t; done
    for t in $TESTS; do job sm_align_$t python3 predictfun/research/pf_stop_model.py align $t; done
    job sm_report python3 predictfun/research/pf_stop_model.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) PHASE $PHASE DONE" >> $OUT/logs/driver.log
