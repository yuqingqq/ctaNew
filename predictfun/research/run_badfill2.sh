#!/usr/bin/env bash
# Part B of the 01:57Z declaration after the sweep was cut at the 0.15 calibration check (STATUS 2026-11-04 02:4xZ).
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
while ps -eo args | grep -q "[p]ython3 -m predictfun.research.pf_step4_live"; do sleep 10; done
while true; do
  $R badfill_info -- python3 -m predictfun.research.pf_badfill_info > $L/badfill_info.log 2>&1 && { echo ALL_DONE; exit 0; }
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' $L/badfill_info.log || { echo FAILED; tail -3 $L/badfill_info.log; exit 1; }
  sleep 60
done
