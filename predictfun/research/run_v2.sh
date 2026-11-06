#!/usr/bin/env bash
# Rerun of the key arms with fill model v2 (code-review fixes, 2026-11-01). One light process at a time.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
ST=G4+G2+G6+G5+G3+G1
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
export PF_ARMS="MICRO|$ST|mid|5;MICRO|$ST+M05|mid|5;MICRO|$ST+M05|early|5;MICRO|$ST+M05|late|5"
for d in $DAYS; do PF_TIMING=e2 PF_TAG=v2c1 run v2c1_$d python3 -m predictfun.research.pf_step4 $d tokyo20 custom; done
B=""
for a in "$ST+M05|mid|5" "G5+G1+M025|mid|5" "G5+G1+M025|early|5" "G5+G1+M025|late|5" "$ST+M025|mid|5" "$ST+M025|mid|25" \
         "$ST+M025|mid|A1:5:25:0" "$ST+M025|early|A1:5:25:0" "$ST+M025|late|A1:5:25:0" "$ST+M025|mid|A1:5:25:0.25" \
         "G5+G1+M025|mid|25" "G5+G1+M025|mid|A6:5:25:120" "G5+G1+M025|mid|A1:5:25:0"; do B="$B;MICRO|$a"; done
export PF_ARMS="${B#;}"
for d in $DAYS; do PF_TIMING=e2 PF_PREDS=preds_lodo6 PF_TAG=v2o run v2o_$d python3 -m predictfun.research.pf_step4 $d tokyo20 custom; done
run v2chk1 python3 -m predictfun.research.pf_check_fills tokyo20_custom_e2_v2c1 $DAYS
run v2chk2 python3 -m predictfun.research.pf_check_fills tokyo20_custom_e2_preds_lodo6_v2o $DAYS
echo ALL_DONE
