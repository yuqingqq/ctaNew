#!/usr/bin/env bash
# Rehearsal before run_review3.sh (STATUS 2026-11-05 03:01Z): map 10-24 again (signer refinement), engine off-parity on 10-24,
# engaged tk + jitter on 10-24 (C3), input bounds engaged on 11-03 (C3, old key).
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
export PF_RL_DDEC=2
$R txmap_20261024 -- python3 -m predictfun.research.pf_txmap 20261024 > $L/txmap_20261024.log 2>&1 || { echo FAILED txmap; exit 1; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_PREDS=preds_h60k
B=G4+G2+G6+G5+G3+G1; A=""
for sz in A1:5:25:0 A1:20:100:0 AV:20:100:0.06 AV:5:25:0.06; do A="${A:+$A;}MICRO|$B+V0+XNA+XMIN1+XV+S60|mid|$sz"; done
C3="MICRO|$B+V0+XNA+XMIN1+XV+S60|mid|AV:20:100:0.06"
PF_ARMS="$A" PF_TAG=par3 $R par3_20261024 -- python3 -m predictfun.research.pf_step4_live 20261024 tokyoreal custom > $L/par3_20261024.log 2>&1 || { echo FAILED par3; exit 1; }
PF_TXKEY=tk PF_TXJIT=frame PF_ARMS="$C3" PF_TAG=smk $R smk_20261024 -- python3 -m predictfun.research.pf_step4_live 20261024 tokyoreal custom > $L/smk_20261024.log 2>&1 || { echo FAILED smk; exit 1; }
PF_ARMS="$C3" PF_TAG=smb $R smb_20261103 -- python3 -m predictfun.research.pf_step4_live 20261103 tokyoreal custom > $L/smb_20261103.log 2>&1 || { echo FAILED smb; exit 1; }
echo SMOKE_DONE
