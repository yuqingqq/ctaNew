#!/usr/bin/env bash
# Latency test L1 (declared in STATUS.yml 2026-11-03 04:21Z; user: real latency values). Read days only; one light process at a time.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
for d in $DAYS; do run l1f_$d python3 -m predictfun.research.pf_step5_features_rl $d; done
run l1lodo python3 -m predictfun.research.pf_step5_lodo_rl
ST=G4+G2+G6+G5+G3+G1; PL=$ST+M025+XNA+XMIN1; A1=A1:5:25:0
RL="MICRO|$PL|mid|$A1;MICRO|$PL+XT1|mid|$A1;MICRO|$PL+XT2|mid|$A1;MICRO|$PL+XV|mid|$A1"
RL="$RL;MICRO|$PL+XS0|mid|$A1;MICRO|$PL+XS0+XT1|mid|$A1;MICRO|$PL+XS0+XT2|mid|$A1;MICRO|$PL+XS0+XV|mid|$A1"
RL="$RL;MICRO|$ST+XNA+XMIN1|mid|5;MICRO|$ST+M025+XNA+XMIN1|mid|5"
T20="MICRO|$PL|mid|$A1;MICRO|$PL+XS0|mid|$A1;MICRO|$PL+XV|mid|$A1;MICRO|$PL+XS0+XV|mid|$A1;MICRO|$ST+XNA+XMIN1|mid|5;MICRO|$ST+M025+XNA+XMIN1|mid|5"
export PF_TIMING=e2
for d in $DAYS; do
  PF_ARMS="$RL" PF_PREDS=preds_rl_a PF_TAG=l1 run l1a_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  PF_ARMS="$RL" PF_PREDS=preds_rl_b PF_TAG=l1 run l1b_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  PF_ARMS="$T20" PF_PREDS=preds_lodo6 PF_TAG=l1 run l1t_$d python3 -m predictfun.research.pf_step4_live $d tokyo20 custom
done
for sub in live_tokyoreal_custom_e2_preds_rl_a_l1 live_tokyoreal_custom_e2_preds_rl_b_l1 live_tokyo20_custom_e2_preds_lodo6_l1; do
  run chk_$sub python3 -m predictfun.research.pf_check_fills $sub $DAYS; done
echo ALL_DONE
