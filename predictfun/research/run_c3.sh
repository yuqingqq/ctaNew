#!/usr/bin/env bash
# Confirmation C3 (declared in STATUS.yml 2026-11-03 10:32Z): O3's combined strategy on fresh days, Tokyo clock. One light process at a time.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
DAYS="${C3_DAYS:-20261101 20261102}"
ST=G4+G2+G6+G5+G3+G1; B=$ST+M025+XNA+XMIN1; A1=A1:5:25:0; P3=$B+XS0+XV+B2W100+Q025+BKA+S60
ARMS="MICRO|$P3|mid|$A1;MICRO|$B+XV+B2W100+Q025+BKA+S60|mid|$A1;MICRO|$B+XS0+XV+BKA+S60|mid|$A1;MICRO|$B+XS0+XV+S60|mid|$A1"
ARMS="$ARMS;MICRO|$B+XS0+XV|mid|$A1;MICRO|$B+XS0+XV300+B2W100+Q025+BKA+S60|mid|$A1;MICRO|$P3+XCAP150|mid|$A1"
O=data/predict_fun/research/step4/live_tokyoreal_custom_e2_preds_c3_c3; PREV=""
for d in $DAYS; do
  PF_RL_DDEC=2 run c3f_$d python3 -m predictfun.research.pf_step5_features_rl $d
  run c3ap_$d python3 -m predictfun.research.pf_step5_c3_preds apply $d
  if [ -n "$PREV" ]; then export PF_LIVE_CAPSTATE_IN=$O/$PREV/capstate.json; else unset PF_LIVE_CAPSTATE_IN; fi
  PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_PREDS=preds_c3 PF_TIMING=e2 PF_TAG=c3 PF_ARMS="$ARMS" \
    run c3s4_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  PREV=$d
done
run chk_c3 python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_preds_c3_c3 $DAYS
echo ALL_DONE
