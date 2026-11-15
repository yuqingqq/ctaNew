#!/usr/bin/env bash
# E2IZE driver (2026-11-15, session aea5bfb9; USER: "Go next"): run_stop_ev_e2ize.sh all -> selftests, the nine consumed days rebuilt with every
# WS-timed taker group re-timed by its own E2 book-time estimate, the report. Each step through run_light on the claude slot; a busy slot is waited for.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/stop_ev_e2ize_20261115; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  all)
    job e2z_selftest python3 predictfun/research/pf_stop_ev_e2ize.py selftest || exit 1
    for d in $DAYS; do [ -f $OUT/day_$d.json ] && continue; job e2z_day_$d python3 predictfun/research/pf_stop_ev_e2ize.py day $d || exit 1; done
    job e2z_report python3 predictfun/research/pf_stop_ev_e2ize.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) E2IZE $PHASE DONE" >> $OUT/logs/driver.log
