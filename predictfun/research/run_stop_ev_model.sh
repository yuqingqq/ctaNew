#!/usr/bin/env bash
# STOP-EV stage 2 driver (2026-11-15, session aea5bfb9; USER: "Yes model now"): run_stop_ev_model.sh all -> selftests, the model data of the
# nine consumed days (E10 join), the five chronological folds (hurdle + B2 + T5), the report. Each step through run_light on the claude
# slot; a busy slot is waited for (as run_stop_model3.sh).
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/stop_ev_model_20261115; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
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
    job evm_selftest python3 predictfun/research/pf_stop_ev_model.py selftest || exit 1
    for d in $DAYS; do job evm_prep_$d python3 predictfun/research/pf_stop_ev_model.py prep $d || exit 1; done
    for t in $TESTS; do job evm_fit_$t python3 predictfun/research/pf_stop_ev_model.py fit $t || exit 1; done
    job evm_report python3 predictfun/research/pf_stop_ev_model.py report;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) STOP-EV-MODEL $PHASE DONE" >> $OUT/logs/driver.log
