#!/usr/bin/env bash
# Part C (STATUS 2026-11-04 02:06Z): big orders on other venues around bad fills; starts after run_badfill.sh ends.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
until grep -qE "ALL_DONE|FAILED" $L/run_badfill.out 2>/dev/null || ! ps -eo args | grep -q "[r]un_badfill.sh"; do sleep 60; done
while true; do
  $R big_prints -- python3 -m predictfun.research.pf_big_prints > $L/big_prints.log 2>&1 && { echo ALL_DONE; exit 0; }
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' $L/big_prints.log || { echo FAILED; tail -3 $L/big_prints.log; exit 1; }
  sleep 60
done
