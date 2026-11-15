#!/usr/bin/env bash
# STOP-EV-PLAY driver (2026-11-15, session aea5bfb9; USER: "We can test 1 first"): run_stop_ev_play.sh all -> selftests, the (theta, r) choice on
# each fold's calibration day, the five test days (KA / LDG / MD1000 / QR / QRS at M 50 and M 250, off-parity), the report. Each step through
# run_light on the claude slot; a busy slot is waited for.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/stop_ev_play_20261115; mkdir -p $OUT/logs
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
    job evp_selftest python3 predictfun/research/pf_stop_ev_play.py selftest || exit 1
    for t in $TESTS; do [ -f $OUT/cal_$t.json ] && continue; job evp_cal_$t python3 predictfun/research/pf_stop_ev_play.py cal $t || exit 1; done
    for t in $TESTS; do [ -f $OUT/test_$t.json ] && continue; job evp_test_$t python3 predictfun/research/pf_stop_ev_play.py test $t || exit 1; done
    job evp_report python3 predictfun/research/pf_stop_ev_play.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) STOP-EV-PLAY $PHASE DONE" >> $OUT/logs/driver.log
