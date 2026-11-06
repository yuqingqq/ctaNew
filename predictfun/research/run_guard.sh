#!/usr/bin/env bash
# Stage 3 (STATUS 2026-11-04): the trained per-side guard in the replay; 11-03 as the unseen day.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
run feat_1103 python3 -m predictfun.research.pf_step5_features_rl 20261103
run c3_1103 python3 -m predictfun.research.pf_step5_c3_preds apply 20261103
run guard_preds python3 -m predictfun.research.pf_step5_guard_preds 20261103
B=G4+G2+G6+G5+G3+G1; A1=A1:5:25:0
A="MICRO|$B+M025+XNA+XMIN1+S60|mid|$A1;MICRO|$B+V0+XNA+XMIN1+S60|mid|$A1;MICRO|$B+M025+V0+XNA+XMIN1+S60|mid|$A1"
for k in 20 10 30; do
  for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    PF_PREDS=preds_g$k PF_ARMS="$A" PF_TAG=gd run gd${k}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  done
done
run guard_report python3 -m predictfun.research.pf_guard_report 20261103
echo ALL_DONE
