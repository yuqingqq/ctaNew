#!/usr/bin/env bash
# STOP-MODEL driver 2 (2026-11-15, session aea5bfb9): the phases after the input review (run_stop_model.sh is not edited while it runs).
#   run_stop_model2.sh x5    -> extra order-flow inputs rows_<day>_x5 for the 9 days, then the adoption test (x5test_20261115.json)
#   run_stop_model2.sh v5    -> v4 + the adopted x5 families on the 5 folds, report (fit5 refuses if no family was adopted)
#   PF_PLAY_V=<4|5> run_stop_model2.sh play -> STOP-PLAY on that version: calibration-day choice, test-day arms, report
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?x5|v5|play}"
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
  x5)
    for d in $DAYS; do job sm_x5_$d python3 predictfun/research/pf_stop_model_x5.py build $d; done
    job sm_x5test python3 predictfun/research/pf_stop_model_review.py x5test;;
  v5)
    for t in $TESTS; do job sm_fit5_$t python3 predictfun/research/pf_stop_model.py fit5 $t; done
    job sm_report5 python3 predictfun/research/pf_stop_model.py report5;;
  play)
    V="${PF_PLAY_V:?PF_PLAY_V=4|5}"
    for t in $TESTS; do job sp_cal_v${V}_$t python3 predictfun/research/pf_stop_play.py cal $t; done
    for t in $TESTS; do job sp_test_v${V}_$t python3 predictfun/research/pf_stop_play.py test $t; done
    job sp_report_v$V python3 predictfun/research/pf_stop_play.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) PHASE2 $PHASE DONE" >> $OUT/logs/driver.log
