#!/usr/bin/env bash
# STOP-MODEL driver 3 (2026-11-15, session aea5bfb9; USER: "We fixed all data issues then build the model and do model play"):
#   run_stop_model3.sh post  -> swap in the rebuilt rows (iff p30cmp passed), full step-5 recomputation on them, x5 inputs rebuilt, x6 test
#   run_stop_model3.sh v6    -> v6 fits on the 5 folds (T5 over 10 permutation draws), v6 report
#   PF_PLAY_V=6 [PF_PLAY_SIZE=<sh>] [PF_PLAY_QUNK=zero|inf] run_stop_model3.sh play -> STOP-PLAY
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?post|v6|play}"
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
  post)
    job sm_swaprows python3 predictfun/research/pf_stop_model_review.py swaprows
    job sm_s5check2b python3 predictfun/research/pf_stop_model_review.py s5check2
    for d in $DAYS; do job sm_x5b_$d python3 predictfun/research/pf_stop_model_x5.py build $d; done
    job sm_x6test python3 predictfun/research/pf_stop_model_review.py x6test;;
  v6)
    for t in $TESTS; do job sm_fit6_$t python3 predictfun/research/pf_stop_model.py fit6 $t; done
    job sm_report6 python3 predictfun/research/pf_stop_model.py report6;;
  play)
    V="${PF_PLAY_V:?PF_PLAY_V}"; Z="${PF_PLAY_SIZE:-30}"; Q="${PF_PLAY_QUNK:-zero}"; T=v${V}_s${Z}_q${Q}
    for t in $TESTS; do job sp_cal_${T}_$t python3 predictfun/research/pf_stop_play.py cal $t; done
    for t in $TESTS; do job sp_test_${T}_$t python3 predictfun/research/pf_stop_play.py test $t; done
    job sp_report_$T python3 predictfun/research/pf_stop_play.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) PHASE3 $PHASE DONE" >> $OUT/logs/driver.log
