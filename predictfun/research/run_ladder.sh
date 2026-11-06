#!/usr/bin/env bash
# Ladder test (declared in STATUS.yml 2026-11-03): 14 arms x 8 days, Tokyo clock, decision 2 ms. One light process at a time.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
A=""
for S in 60 240; do for kz in 3:5 3:10 3:25 6:5 6:10; do A="$A;K=${kz%:*},z=${kz#*:},S=$S,TH=0.5"; done; done
for S in 60 240; do for th in 0.25 1.0; do A="$A;K=3,z=10,S=$S,TH=$th"; done; done
export PF_LADDER_ARMS="${A#;}" PF_RL_DDEC=2 PF_TAG=lad1
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102; do
  pre=preds_o3; case $d in 202611*) pre=preds_c3;; esac
  PF_PREDS=$pre run lad1_$d python3 -m predictfun.research.pf_ladder $d
done
echo ALL_DONE
