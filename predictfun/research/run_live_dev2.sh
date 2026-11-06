#!/usr/bin/env bash
# Live-deviation tests round 2 (exec-port request 2026-11-02): live-like C2-P (no amend + min 1 USD) under the SEQUENTIAL
# removal lock (XS0/200/500/1000), the concurrent lock at 200 ms (XL200A), and the exec-port throttle (XT1, XT2) at
# sequential L = 0 / 500 ms. Read days only; no loss cap (its cost was -31.67 in round 1).
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
SUB=live_tokyo20_custom_e2_preds_lodo6_livedev2
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
PL=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1; A1=A1:5:25:0; A=""
for x in "" "+XS0" "+XS200" "+XS500" "+XS1000" "+XL200A" "+XS0+XT1" "+XS500+XT1" "+XS0+XT2" "+XS500+XT2"; do A="$A;MICRO|$PL$x|mid|$A1"; done
export PF_ARMS="${A#;}" PF_TIMING=e2 PF_PREDS=preds_lodo6 PF_TAG=livedev2
for d in 20261023 20261024 20261025 20261026 20261027 20261031; do run livedev2_$d python3 -m predictfun.research.pf_step4_live $d tokyo20 custom; done
run livedev2_chk python3 -m predictfun.research.pf_check_fills $SUB 20261023 20261024 20261025 20261026 20261027 20261031
echo ALL_DONE
