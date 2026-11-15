#!/usr/bin/env bash
# STOP-EV v3 driver (2026-11-15, session aea5bfb9; after the pipeline REVIEW, USER "Can we fix the critical issues and let the results be
# correct first"): run_stop_ev_model3.sh all -> selftests, the v3 model rows of the nine days (imputed-untimed labels, every eligible status),
# the five v3 fits (saved heads, reload gate before any prediction), the report (R900-R905 as declared in pf_stop_ev_model.py V3). Each step
# through run_light on the claude slot; a busy slot is waited for.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/stop_ev_model_20261115; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
TESTS="20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver3.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver3.log; return 1
  done; }
case "$PHASE" in
  all)
    job evm3_selftest python3 predictfun/research/pf_stop_ev_model.py selftest || exit 1
    for d in $DAYS; do job evm3_prep_$d python3 predictfun/research/pf_stop_ev_model.py prep3 $d || exit 1; done
    for t in $TESTS; do job evm3_fit_$t python3 predictfun/research/pf_stop_ev_model.py fit3 $t || exit 1; done
    job evm3_report python3 predictfun/research/pf_stop_ev_model.py report3 || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) STOP-EV v3 $PHASE DONE" >> $OUT/logs/driver3.log
