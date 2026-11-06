#!/usr/bin/env bash
# Review follow-up (STATUS 2026-11-04 14:05Z): one time per taker transaction (PF_TXTIME), latency tails. After run_review.sh.
set -u
cd /home/yuqing/ctaNew
while ps -eo args | grep -q "[r]un_review.sh"; do sleep 15; done
while ps -eo args | grep -q "[p]f_step4_live 2026"; do sleep 10; done
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { local tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LPENDCANCEL=95 PF_RL_DDEC=2 PF_TIMING=e2
D9="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
B=G4+G2+G6+G5+G3+G1; A=""
for sz in A1:5:25:0 A1:20:100:0 AV:20:100:0.06; do A="${A:+$A;}MICRO|$B+V0+XNA+XMIN1+XV+S60|mid|$sz"; done
for d in $D9; do PF_RL_LCANCEL=22 PF_RL_LFILL=100 PF_TXTIME=ws PF_PREDS=preds_g60k PF_ARMS="$A" PF_TAG=rvtx run rvtx_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_RL_LCANCEL=34 PF_RL_LFILL=287 PF_TXTIME=ws PF_PREDS=preds_g60k PF_ARMS="$A" PF_TAG=rvtxL run rvtxL_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_RL_LCANCEL=22 PF_RL_LFILL=100 PF_TXTIME=min PF_PREDS=preds_g60k PF_ARMS="$A" PF_TAG=rvtxm run rvtxm_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
echo ALL_DONE
