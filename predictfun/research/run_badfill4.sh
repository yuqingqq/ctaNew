#!/usr/bin/env bash
# Longer look-back guard screen (STATUS 2026-11-04): runs after the 40 / 50% pull chain.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
while ps -eo args | grep -q "[r]un_guard_4050.sh"; do sleep 30; done
while true; do
  PF_RL_DDEC=2 $R badfill_info4 -- python3 -m predictfun.research.pf_badfill_info4 > $L/badfill_info4.log 2>&1 && { echo ALL_DONE; exit 0; }
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' $L/badfill_info4.log || { echo FAILED; tail -3 $L/badfill_info4.log; exit 1; }
  sleep 60
done
