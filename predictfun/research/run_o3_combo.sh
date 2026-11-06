#!/usr/bin/env bash
# Round O3 step 2 (declared rule): the combined arm (base + adopted B2W100, Q025, BKA, S60), sequential + XV and concurrent twin.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_PREDS=preds_o3 PF_TIMING=e2
export PF_ARMS="MICRO|G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1+XS0+XV+B2W100+Q025+BKA+S60|mid|A1:5:25:0;MICRO|G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1+XV+B2W100+Q025+BKA+S60|mid|A1:5:25:0"
for d in $DAYS; do PF_TAG=o3c run o3c_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
run chk_o3c python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_preds_o3_o3c $DAYS
echo ALL_DONE
