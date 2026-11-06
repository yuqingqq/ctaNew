#!/usr/bin/env bash
# Anchor quality sweep + bad-fill information test (declared in STATUS.yml 2026-11-04 01:57Z). Starts only after C2's 11-03 run.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs; C=data/predict_fun/research/c2
until grep -q '"day": "20261103"' $C/admission.jsonl 2>/dev/null && ! grep -q running $C/auto.heartbeat; do sleep 60; done
run() { tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
B=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1; A1=A1:5:25:0
A="MICRO|$B+S60|mid|$A1;MICRO|$B|mid|$A1"
run deg_preds python3 -m predictfun.research.pf_step5_degraded_preds
for lv in 15 30 50 70 90; do
  for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102; do
    PF_PREDS=preds_deg$lv PF_ARMS="$A" PF_TAG=anc run anc_d${lv}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  done
done
run anchor_sweep_report python3 -m predictfun.research.pf_anchor_sweep_report
run badfill_info python3 -m predictfun.research.pf_badfill_info
echo ALL_DONE
