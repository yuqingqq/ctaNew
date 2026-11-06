#!/usr/bin/env bash
# Live-deviation round 4 (exec-port request 2026-11-02): Predict's lock is per order, counted from placement
# (removalLockedUntil). Live-like C2-P, sequential, placement-relative lock XP<lambda> for lambda = 50/100/200/500/1000 ms,
# unthrottled and at 1 / 2 places/s; XP0 = parity control (must equal round 2's XS0 to the cent). Read days only.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
SUB=live_tokyo20_custom_e2_preds_lodo6_livedev4
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
PL=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1; A1=A1:5:25:0; A="MICRO|$PL+XP0|mid|$A1"
for th in "" "+XT1" "+XT2"; do for l in 50 100 200 500 1000; do A="$A;MICRO|$PL+XP$l$th|mid|$A1"; done; done
export PF_ARMS="$A" PF_TIMING=e2 PF_PREDS=preds_lodo6 PF_TAG=livedev4
for d in 20261023 20261024 20261025 20261026 20261027 20261031; do run livedev4_$d python3 -m predictfun.research.pf_step4_live $d tokyo20 custom; done
run livedev4_chk python3 -m predictfun.research.pf_check_fills $SUB 20261023 20261024 20261025 20261026 20261027 20261031
echo ALL_DONE
