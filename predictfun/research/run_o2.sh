#!/usr/bin/env bash
# Round O2 (declared in STATUS.yml): adaptive size on two bases + restatement of C1 / O1 arms with the fixed loop.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
ST=G4+G2+G6+G5+G3+G1
A=""
for base in "G5+G1+M025" "$ST+M025"; do
  for sz in 5 25 A1:5:25:0 A1:5:25:0.25 A3:5:25:0.1 A3:5:25:0.25 A4:5:25:0 A6:5:25:120; do A="$A;MICRO|$base|mid|$sz"; done
done
export PF_ARMS="MICRO|$ST+M05|mid|5$A"
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
for d in $DAYS; do PF_TIMING=e2 PF_PREDS=preds_lodo6 PF_TAG=o2 run o2_$d python3 -m predictfun.research.pf_step4 $d tokyo20 custom; done
echo O2_DONE
export PF_ARMS="MICRO|$ST|mid|5;MICRO|$ST+M05|mid|5;MICRO|$ST+M05|early|5;MICRO|$ST+M05|late|5"
for d in $DAYS; do PF_TIMING=e2 PF_TAG=c1fix run c1fix_$d python3 -m predictfun.research.pf_step4 $d tokyo20 custom; done
echo ALL_DONE
