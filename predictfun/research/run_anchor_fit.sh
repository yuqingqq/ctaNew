#!/usr/bin/env bash
# Fitted anchor (STATUS 2026-11-04 04:30Z): runs after the stage-3 chain.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
while ps -eo args | grep -q "[r]un_guard.sh"; do sleep 30; done
while true; do
  $R anchor_fit -- python3 -m predictfun.research.pf_anchor_fit > $L/anchor_fit.log 2>&1 && { echo ALL_DONE; exit 0; }
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' $L/anchor_fit.log || { echo FAILED; tail -3 $L/anchor_fit.log; exit 1; }
  sleep 60
done
