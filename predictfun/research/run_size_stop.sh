#!/usr/bin/env bash
# Size x stop grid for the trained guard (STATUS 2026-11-04 10:32Z): k = 60 first, then 50 / 70.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { local tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
B=G4+G2+G6+G5+G3+G1; A=""
for sz in A1:5:25:0 A1:10:50:0 A1:20:100:0; do
  for s in 60 120 180 240; do
    A="${A:+$A;}MICRO|$B+V0+XNA+XMIN1+S$s|mid|$sz"
  done
done
for k in 60 50 70; do
  for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    PF_PREDS=preds_g$k PF_ARMS="$A" PF_TAG=gz run gz${k}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  done
  ( run size_stop_report_$k python3 -m predictfun.research.pf_size_stop_report ) || echo "REPORT FAILED k=$k (grid continues)"
done
echo ALL_DONE
