#!/usr/bin/env bash
# STOP-EV stage 1 driver (2026-11-15, session aea5bfb9; USER: "first, data processing, check its coverage, data quality, etc, then model,
# then the pipeline"): run_stop_ev.sh data -> synthetic selftests, the label data + checks E1-E7 of the nine consumed days, the report (E8,
# readings). Each step through run_light on the claude slot; a busy slot is waited for (as run_stop_model3.sh). PF_EV_SIZE -> _s<Z> outputs.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?data}"
OUT=data/predict_fun/research/stop_ev_20261115; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
Z="${PF_EV_SIZE:-30}"; T=$([ "$Z" = "30" ] && echo "" || echo "_s$Z")
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  data)
    job ev_selftest$T python3 predictfun/research/pf_stop_ev.py selftest || exit 1
    for d in $DAYS; do [ -f $OUT/day_${d}${T}.json ] && continue; job ev_day${T}_$d python3 predictfun/research/pf_stop_ev.py day $d; done
    job ev_report$T python3 predictfun/research/pf_stop_ev.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) STOP-EV $PHASE DONE" >> $OUT/logs/driver.log
