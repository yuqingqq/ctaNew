#!/usr/bin/env bash
# STOP-MODEL driver (2026-11-14, session aea5bfb9): claude slot, waits for a free slot.
#   run_stop_model.sh data   -> selftest, per-day rows (with parity / falsifier / brute-force controls), scale choice + audit v2 (then stop: the USER asked
#                               for the data to be checked and its issues cleared before any fit)
#   run_stop_model.sh model  -> per-fold fits, per-test-day alignment lanes, report
#   run_stop_model.sh v4     -> (2026-11-15, after the dev-day diagnostics) v3.1 report regression, v4 fits, v4 report
#   run_stop_model.sh play   -> (2026-11-15) STOP-PLAY: the v4 model's own decisions in the oracle's lane
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?data|model|v4|v4b|play}"
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
  v4)     # after pf_stop_model_diag.py run (its decisions): v3.1 report regression, v4 fits, v4 report
    job sm_report31 python3 predictfun/research/pf_stop_model.py report
    for t in $TESTS; do job sm_fit4_$t python3 predictfun/research/pf_stop_model.py fit4 $t; done
    job sm_report4 python3 predictfun/research/pf_stop_model.py report4;;
  v4b)    # (2026-11-15, after the input review / s5check / reg): within-side copies test, the loader-filter regression, v4 fits, v4 report
    job sm_dup python3 predictfun/research/pf_stop_model_review.py dup
    for d in 20261027 20261031 20261103; do PF_SM_ROWS_TAG=_d12 job sm_rows_d12_$d python3 predictfun/research/pf_stop_model.py rows $d; done
    job sm_d12cmp python3 predictfun/research/pf_stop_model_review.py d12cmp
    for t in $TESTS; do job sm_fit4_$t python3 predictfun/research/pf_stop_model.py fit4 $t; done
    job sm_report4 python3 predictfun/research/pf_stop_model.py report4;;
  play)   # STOP-PLAY on v4's predictions: calibration-day choice, test-day arms, report
    for t in $TESTS; do job sp_cal_$t python3 predictfun/research/pf_stop_play.py cal $t; done
    for t in $TESTS; do job sp_test_$t python3 predictfun/research/pf_stop_play.py test $t; done
    job sp_report python3 predictfun/research/pf_stop_play.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) PHASE $PHASE DONE" >> $OUT/logs/driver.log
