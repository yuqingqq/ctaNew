#!/usr/bin/env bash
# STOP-EV-PLAY v2 driver (2026-11-15, session aea5bfb9; after the pipeline REVIEW, USER "Can we fix the critical issues and let the results be
# correct first"): run_stop_ev_play2.sh all -> selftests, per fold the calibration day (QR and MD1000 choices) then the test day (every arm,
# both clocks, RV, off-parity, sensitivity), the report (R980-R985 as declared in pf_stop_ev_play2.py). Each step through run_light on the
# claude slot; a busy slot is waited for.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/stop_ev_play2_20261115; mkdir -p $OUT/logs
TESTS="20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  all)
    job evp2_selftest python3 predictfun/research/pf_stop_ev_play2.py selftest || exit 1
    for t in $TESTS; do
      job evp2_cal_$t python3 predictfun/research/pf_stop_ev_play2.py cal $t || exit 1
      job evp2_test_$t python3 predictfun/research/pf_stop_ev_play2.py test $t || exit 1
    done
    job evp2_report python3 predictfun/research/pf_stop_ev_play2.py report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) STOP-EV-PLAY2 $PHASE DONE" >> $OUT/logs/driver.log
