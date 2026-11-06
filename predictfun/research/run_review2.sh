#!/usr/bin/env bash
# Second review stress runs + window / size grid (STATUS 2026-11-05 01:49Z). Starts after C2 has processed 11-04.
set -u
cd /home/yuqing/ctaNew
until grep -q "day 20261104 rc=" data/predict_fun/research/c2/auto.log 2>/dev/null || [ "$(date -u +%H%M)" -ge 0300 ]; do sleep 30; done
while ps -eo args | grep -qE "[p]f_step|[r]un_c2_day"; do sleep 15; done
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { local tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2 PF_TXTIME=ws PF_PREDS=preds_h60k
D9="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
B=G4+G2+G6+G5+G3+G1; A=""; W=""
for sz in A1:5:25:0 A1:20:100:0 AV:20:100:0.06 AV:5:25:0.06; do A="${A:+$A;}MICRO|$B+V0+XNA+XMIN1+XV+S60|mid|$sz"; done
for sz in AV:5:25:0.06 AV:20:100:0.06 AV:40:200:0.06; do for s in 60 120 180 240; do W="${W:+$W;}MICRO|$B+V0+XNA+XMIN1+XV+S$s|mid|$sz"; done; done
for d in $D9; do PF_TXSHIFT_E2=-300 PF_ARMS="$A" PF_TAG=rbtxE run rbtxE_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_TXSHIFT=-50 PF_ARMS="$A" PF_TAG=rbtxS50 run rbtxS50_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_ARMS="$W" PF_TAG=rbw run rbw_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_QACT=max PF_TXSHIFT=-25 PF_ARMS="$W" PF_TAG=rbwX run rbwX_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
run window_size_report python3 -m predictfun.research.pf_window_size_report
echo ALL_DONE
