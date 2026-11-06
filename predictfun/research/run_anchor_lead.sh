#!/usr/bin/env bash
# Lead-adjusted anchor (STATUS 2026-11-04 05:16Z): runs after the stage-3 control.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
while ps -eo args | grep -q "[r]un_guard_control.sh"; do sleep 30; done
while true; do
  PF_RL_DDEC=2 $R anchor_lead -- python3 -m predictfun.research.pf_anchor_lead > $L/anchor_lead.log 2>&1 && { echo ALL_DONE; exit 0; }
  grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' $L/anchor_lead.log || { echo FAILED; tail -3 $L/anchor_lead.log; exit 1; }
  sleep 60
done
