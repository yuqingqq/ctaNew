#!/usr/bin/env bash
# Round O1 step 2: declared combination of adopted changes (theta 0.25, drop G2/G3/G4/G6) + post-hoc size checks.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
ST=G4+G2+G6+G5+G3+G1
export PF_ARMS="MICRO|G5+G1+M025|mid|5;MICRO|G5+G1+M025|early|5;MICRO|G5+G1+M025|late|5;MICRO|$ST+M025|early|5;MICRO|$ST+M025|late|5;MICRO|$ST+M05|mid|25;MICRO|$ST+M025|mid|25;MICRO|$ST+M05|mid|100;MICRO|$ST+M025|mid|100"
for d in 20261023 20261024 20261025 20261026 20261027 20261031; do
  PF_TIMING=e2 PF_PREDS=preds_lodo6 PF_TAG=o1combo run o1_combo_$d python3 -m predictfun.research.pf_step4 $d tokyo20 custom
done
echo ALL_DONE
