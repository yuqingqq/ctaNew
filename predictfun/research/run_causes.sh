#!/usr/bin/env bash
# Order-change causes of the frozen candidate pm_ladder_v1 (pf_step4_ladder_causes.py), 8 read days + 11-03, timings J and X.
# One run_light job per day x timing (sequential). Outputs step4/causes_v1/<J|X>_<day>.json and ladder_..._ca<J|X>/<day>.
set -u
cd /home/yuqing/ctaNew
ARM='MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95'
O=data/predict_fun/research/step4/causes_v1; mkdir -p $O; LG=data/predict_fun/research/logs
COMMON="PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k"
for TM in J X; do
  if [ $TM = J ]; then EXTRA="PF_TXJIT=frame"; else EXTRA="PF_TXSHIFT=-50 PF_QACT=max"; fi
  for D in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    env $COMMON $EXTRA PF_ARMS="$ARM" PF_TAG=ca$TM PF_CAUSES_OUT=$O/${TM}_$D.json \
      predictfun/research/run_light.sh causes_${TM}_$D -- python3 -m predictfun.research.pf_step4_ladder_causes $D tokyoreal custom \
      > $LG/causes_${TM}_$D.log 2>&1 || { echo "FAILED $TM $D"; exit 1; }
  done
done
echo ALL DONE
