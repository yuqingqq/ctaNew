#!/usr/bin/env bash
# Rebuild 3 after the second review (STATUS 2026-11-05 03:01Z): real taker-order key, within-frame match time, no sealed / forward input.
set -u
cd /home/yuqing/ctaNew
while ps -C python3 -o args= | grep -qE "pf_step|pf_txmap|run_c2_day"; do sleep 15; done
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs; S1=data/predict_fun/research/step1
run() { local tag=$1; shift
  while true; do
    [ "$(date -u +%H%M)" -ge 0205 ] && [ "$(date -u +%H%M)" -lt 0240 ] && { sleep 60; continue; }   # C2 window
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_DDEC=2
D9="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
for d in $D9; do
  [ -s $S1/$d/legs_tk.parquet ] && [ $S1/$d/txmap.json -nt predictfun/research/pf_txmap.py ] && continue
  run txmap_$d python3 -m predictfun.research.pf_txmap $d
done
PF_TXKEY=tk run grid_tk python3 -m predictfun.research.pf_badfill_grid_tx
PF_GRID=data/predict_fun/research/step4/badfill_info_tx_tk/grid.parquet PF_ROWS=rows_rl_d2_pk PF_PSTEM=t PF_GUARD_KS=60 run guard_preds_t60 python3 -m predictfun.research.pf_step5_guard_preds 20261103
run startknown_t60 python3 -m predictfun.research.pf_preds_startknown preds_t60 $D9
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k
B=G4+G2+G6+G5+G3+G1; A=""; W=""
for sz in A1:5:25:0 A1:20:100:0 AV:20:100:0.06 AV:5:25:0.06; do A="${A:+$A;}MICRO|$B+V0+XNA+XMIN1+XV+S60|mid|$sz"; done
W="MICRO|$B+V0+XNA+XMIN1+XV+S60|mid|A1:5:25:0;MICRO|$B+V0+XNA+XMIN1+XV+S60|mid|A1:20:100:0"
for sz in AV:5:25:0.06 AV:20:100:0.06 AV:40:200:0.06; do for s in 60 120 180 240; do W="$W;MICRO|$B+V0+XNA+XMIN1+XV+S$s|mid|$sz"; done; done
for d in $D9; do PF_TXJIT=frame PF_ARMS="$W" PF_TAG=kJ run kJ_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$W" PF_TAG=kX run kX_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_ARMS="$A" PF_TAG=kS run kS_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_TXJIT=frame PF_TXSHIFT_E2=-300 PF_ARMS="$A" PF_TAG=kE run kE_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
for d in $D9; do PF_TXTIME=min PF_TXJIT=frame PF_ARMS="$A" PF_TAG=kM run kM_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
run review3_report python3 -m predictfun.research.pf_review3_report
echo ALL_DONE
