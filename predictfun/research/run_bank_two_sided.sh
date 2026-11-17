#!/usr/bin/env bash
# TWO-SIDED SCENARIO BANK driver (2026-11-17, session aea5bfb9; USER "Yes can you do the next and validate"): run_bank_two_sided.sh all ->
# selftest, build (OLD rebuilt = frozen bank, TWO, BID48), actions on the audited Oct26 / Oct27 neutral actions, the 12-market Oct27 pilot
# (OLD reproduction then TWO), report (A1 / A2 / R1 as declared in pf_bank_two_sided.py). Each step through run_light on the claude slot.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/bank_two_sided_20261117; mkdir -p $OUT/logs
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  all)
    job bank2_selftest python3 predictfun/research/pf_bank_two_sided.py selftest || exit 1
    job bank2_build python3 predictfun/research/pf_bank_two_sided.py build || exit 1
    for d in 20261026 20261027; do job bank2_actions_$d python3 predictfun/research/pf_bank_two_sided.py actions $d || exit 1; done
    job bank2_pilot python3 predictfun/research/pf_bank_two_sided.py pilot || exit 1
    job bank2_report python3 predictfun/research/pf_bank_two_sided.py report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) TWO-SIDED-BANK $PHASE DONE" >> $OUT/logs/driver.log
