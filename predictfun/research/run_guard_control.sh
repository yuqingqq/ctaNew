#!/usr/bin/env bash
# Stage 3 control (STATUS 2026-11-04): the current guard at stricter thresholds; runs after the fitted anchor.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
while ps -eo args | grep -q "[r]un_anchor_fit.sh"; do sleep 30; done
run() { tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
B=G4+G2+G6+G5+G3+G1; A1=A1:5:25:0
A="MICRO|$B+M015+XNA+XMIN1+S60|mid|$A1;MICRO|$B+M010+XNA+XMIN1+S60|mid|$A1;MICRO|$B+M005+XNA+XMIN1+S60|mid|$A1"
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
  PF_PREDS=preds_g20 PF_ARMS="$A" PF_TAG=gdc run gdc_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
done
run guard_control_report python3 -m predictfun.research.pf_guard_control_report
echo ALL_DONE
