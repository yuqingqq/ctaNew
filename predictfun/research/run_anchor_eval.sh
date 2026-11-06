#!/usr/bin/env bash
# Stage 1 anchor test (STATUS 2026-11-04): runs once pf_badfill_info2 has finished.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
while ps -eo args | grep -q "[p]ython3 -m predictfun.research.pf_badfill_info2"; do sleep 30; done
while true; do
  PF_RL_DDEC=2 $R anchor_eval -- python3 -m predictfun.research.pf_anchor_eval > $L/anchor_eval.log 2>&1 && { echo ALL_DONE; exit 0; }
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' $L/anchor_eval.log || { echo FAILED; tail -3 $L/anchor_eval.log; exit 1; }
  sleep 60
done
