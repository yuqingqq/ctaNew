#!/usr/bin/env bash
# STOP-EV vs TOP MAKERS driver (2026-11-16, session aea5bfb9; USER "check the result and how it compares against top makers"):
# run_stop_ev_vs_top.sh all -> selftest, one job per scorecard date (controls inside), the report. POST HOC, descriptive (no readings).
# Each step through run_light on the claude slot; a busy slot is waited for.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/stop_ev_vs_top_20261116; mkdir -p $OUT/logs
DAYS="20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  all)
    job evtop_selftest python3 predictfun/research/pf_stop_ev_vs_top.py selftest || exit 1
    for d in $DAYS; do job evtop_day_$d python3 predictfun/research/pf_stop_ev_vs_top.py day $d || exit 1; done
    job evtop_report python3 predictfun/research/pf_stop_ev_vs_top.py report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) STOP-EV-VS-TOP $PHASE DONE" >> $OUT/logs/driver.log
