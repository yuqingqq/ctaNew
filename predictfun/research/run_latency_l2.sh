#!/usr/bin/env bash
# Latency test L2c (STATUS.yml 2026-11-03; supersedes L2 / L2b): split order / cancel / pending-cancel / fill-known latency x decision.
# Read days only. The decision-2 ms features / model-A predictions were built by the L2 run and are reused (SKIP_FEATURES=1).
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
if [ "${SKIP_FEATURES:-0}" != 1 ]; then
  for d in $DAYS; do PF_RL_DDEC=2 run l2f_$d python3 -m predictfun.research.pf_step5_features_rl $d; done
  PF_RL_DDEC=2 PF_RL_ONLY_A=1 run l2lodo python3 -m predictfun.research.pf_step5_lodo_rl
fi
PL=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1; A1=A1:5:25:0
export PF_ARMS="MICRO|$PL|mid|$A1;MICRO|$PL+XV|mid|$A1;MICRO|$PL+XS0|mid|$A1;MICRO|$PL+XS0+XV|mid|$A1;MICRO|$PL+XS0+XT1|mid|$A1" PF_TIMING=e2
CFGS="40:22:95:100:10 22:22:95:100:10 40:22:95:30:10 40:22:95:100:2"
for d in $DAYS; do
  for cfg in $CFGS; do
    IFS=: read lp lc lq lf dd <<< "$cfg"; pre=preds_rl_a; [ "$dd" = 2 ] && pre=preds_rl_a_d2; tg=l2c_${lp}_${lc}_f${lf}_d$dd
    PF_RL_LPLACE=$lp PF_RL_LCANCEL=$lc PF_RL_LPENDCANCEL=$lq PF_RL_LFILL=$lf PF_RL_DDEC=$dd PF_PREDS=$pre PF_TAG=$tg run ${tg}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  done
done
for cfg in $CFGS; do
  IFS=: read lp lc lq lf dd <<< "$cfg"; pre=preds_rl_a; [ "$dd" = 2 ] && pre=preds_rl_a_d2; tg=l2c_${lp}_${lc}_f${lf}_d$dd
  run chk_$tg python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_${pre}_$tg $DAYS
done
echo ALL_DONE
