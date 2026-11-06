#!/usr/bin/env bash
# Latency-aware guard screen (STATUS 2026-11-04): runs after the matched-pull control.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
while ps -eo args | grep -q "[r]un_guard_matched.sh"; do sleep 30; done
while true; do
  $R badfill_info3 -- python3 -m predictfun.research.pf_badfill_info3 > $L/badfill_info3.log 2>&1 && { echo ALL_DONE; exit 0; }
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' $L/badfill_info3.log || { echo FAILED; tail -3 $L/badfill_info3.log; exit 1; }
  sleep 60
done
