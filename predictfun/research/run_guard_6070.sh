#!/usr/bin/env bash
# Pull 60 / 70% for the trained guard + matched-pull control (STATUS 2026-11-04).
set -u
cd /home/yuqing/ctaNew
while ps -eo args | grep -q "[r]un_badfill4.sh"; do sleep 30; done
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { local tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2 PF_GUARD_KS=60,70
run guard_preds_6070 python3 -m predictfun.research.pf_step5_guard_preds 20261103
run baseguard_preds_6070 python3 -m predictfun.research.pf_step5_baseguard_preds 20261103
B=G4+G2+G6+G5+G3+G1; A1=A1:5:25:0
A="MICRO|$B+V0+XNA+XMIN1+S60|mid|$A1;MICRO|$B+M025+V0+XNA+XMIN1+S60|mid|$A1"
for k in 60 70; do
  for pre in g b; do
    tag=gd; [ $pre = b ] && tag=gdb
    for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
      PF_PREDS=preds_$pre$k PF_ARMS="$A" PF_TAG=$tag run ${tag}${k}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
    done
  done
done
run guard_control_report python3 -m predictfun.research.pf_guard_control_report
echo ALL_DONE
