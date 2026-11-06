#!/usr/bin/env bash
# Live-deviation round 5 (exec-port request 2026-11-02): the frozen Tokyo version replaces CONCURRENTLY. Live-like C2-P,
# placement-relative lock XQ<lambda> (old order fillable beside the new one until max(decision + 20, activation + lambda)),
# lambda = 0/50/100/200/500 ms x throttle none / 1 / 2 places/s. XQ0 unthrottled = parity control vs live-like (round 2).
# Two processes per day (memory: the 200/500 ms arms fill several times more shares). Read days only.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
PL=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1; A1=A1:5:25:0
arms() { A=""; for l in "$@"; do for th in "" "+XT1" "+XT2"; do A="$A;MICRO|$PL+XQ$l$th|mid|$A1"; done; done; echo "${A#;}"; }
export PF_TIMING=e2 PF_PREDS=preds_lodo6
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
for d in $DAYS; do PF_ARMS="$(arms 0 50 100)" PF_TAG=livedev5a run livedev5a_$d python3 -m predictfun.research.pf_step4_live $d tokyo20 custom; done
for d in $DAYS; do PF_ARMS="$(arms 200 500)" PF_TAG=livedev5b run livedev5b_$d python3 -m predictfun.research.pf_step4_live $d tokyo20 custom; done
run livedev5a_chk python3 -m predictfun.research.pf_check_fills live_tokyo20_custom_e2_preds_lodo6_livedev5a $DAYS
run livedev5b_chk python3 -m predictfun.research.pf_check_fills live_tokyo20_custom_e2_preds_lodo6_livedev5b $DAYS
echo ALL_DONE
