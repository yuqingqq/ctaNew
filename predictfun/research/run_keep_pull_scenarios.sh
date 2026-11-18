#!/usr/bin/env bash
# EXECUTION-WINDOW LEDGER + QUEUE-AWARE SCENARIO REPLAY driver (2026-11-18, session aea5bfb9; USER: "Still have not cleared all issues,
# check the review, make sure you clear all issues"; the reviews of d1de7504 / c23e6f29 and the execution-windows addendum):
#   run_keep_pull_scenarios.sh selftest   synthetic only (no market data is read): both selftests and both mutation checks
#   run_keep_pull_scenarios.sh all        the selftests, the ledger (Oct23-27 through the replay's loader), the known-lifecycle check (the
#                                         11-09 audit's Nov 8 own legs), the scenario replay one job per day and its report
#                                         (reads consumed data: only on the USER's go)
# Readings and controls are declared in pf_keep_pull_exec_windows.py and pf_keep_pull_scenarios.py. Each step through run_light, claude slot.
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?selftest|all}"
OUT=data/predict_fun/research/keep_pull_scenarios_20261118; mkdir -p $OUT/logs
XW=predictfun/research/pf_keep_pull_exec_windows.py; SC=predictfun/research/pf_keep_pull_scenarios.py
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  selftest)
    job xw_selftest python3 $XW selftest || exit 1
    job xw_mutants python3 predictfun/research/pf_keep_pull_exec_windows_mutants.py || exit 1
    job sc_selftest python3 $SC selftest || exit 1
    job sc_mutants python3 predictfun/research/pf_keep_pull_scenarios_mutants.py || exit 1;;
  all)
    job sc_selftest python3 $SC selftest || exit 1
    job xw_run python3 $XW run || exit 1
    job xw_known python3 $XW known || exit 1
    for d in 20261023 20261024 20261025 20261026 20261027; do job sc_day_$d python3 $SC day $d || exit 1; done
    job sc_report python3 $SC report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) SCENARIOS $PHASE DONE" >> $OUT/logs/driver.log
