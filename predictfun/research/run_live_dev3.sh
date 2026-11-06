#!/usr/bin/env bash
# Live-deviation round 3 (user 2026-11-02: "Test between 0-200ms"): live-like C2-P, SEQUENTIAL lock L = 50/100/150 ms,
# each unthrottled and with the exec-port throttle at 1 and 2 places/s; plus L = 200 with the throttle. Read days only.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
SUB=live_tokyo20_custom_e2_preds_lodo6_livedev3
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
PL=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1; A1=A1:5:25:0; A=""
for x in "+XS50" "+XS100" "+XS150" "+XS50+XT1" "+XS100+XT1" "+XS150+XT1" "+XS200+XT1" "+XS50+XT2" "+XS100+XT2" "+XS150+XT2" "+XS200+XT2"; do A="$A;MICRO|$PL$x|mid|$A1"; done
export PF_ARMS="${A#;}" PF_TIMING=e2 PF_PREDS=preds_lodo6 PF_TAG=livedev3
for d in 20261023 20261024 20261025 20261026 20261027 20261031; do run livedev3_$d python3 -m predictfun.research.pf_step4_live $d tokyo20 custom; done
run livedev3_chk python3 -m predictfun.research.pf_check_fills $SUB 20261023 20261024 20261025 20261026 20261027 20261031
echo ALL_DONE
