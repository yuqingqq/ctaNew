#!/usr/bin/env bash
# Live-deviation tests for the pmmmrust execution port (2026-11-02): C2-P and C2-S under no-amend, removal lock,
# 1 USD minimum order, and the 150 USD total-loss cap (carried day to day in date order). Read days only.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
SUB=live_tokyo20_custom_e2_preds_lodo6_livedev; O=data/predict_fun/research/step4/$SUB
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
P=G4+G2+G6+G5+G3+G1+M025; S=G5+G1+M025; A1=A1:5:25:0; A=""
for x in "" "+XNA" "+XNA+XL500A" "+XNA+XL1000A" "+XNA+XL500R" "+XNA+XL1000R" "+XMIN1" "+XNA+XMIN1" "+XCAP150" "+XNA+XMIN1+XCAP150"; do A="$A;MICRO|$P$x|mid|$A1"; done
for x in "" "+XNA" "+XNA+XL500A" "+XNA+XL1000A"; do A="$A;MICRO|$S$x|mid|$A1"; done
export PF_ARMS="${A#;}" PF_TIMING=e2 PF_PREDS=preds_lodo6 PF_TAG=livedev
PREV=""
for d in 20261023 20261024 20261025 20261026 20261027 20261031; do
  if [ -n "$PREV" ]; then export PF_LIVE_CAPSTATE_IN=$O/$PREV/capstate.json; else unset PF_LIVE_CAPSTATE_IN; fi
  run livedev_$d python3 -m predictfun.research.pf_step4_live $d tokyo20 custom
  PREV=$d
done
run livedev_chk python3 -m predictfun.research.pf_check_fills $SUB 20261023 20261024 20261025 20261026 20261027 20261031
echo ALL_DONE
