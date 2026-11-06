#!/usr/bin/env bash
# Queue-priority diagnostics (2026-11-03): market side, then one lad1 arm rerun with the order log (tag lad1q). One light process at a time.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
run qprio python3 -m predictfun.research.pf_queue_priority
export PF_LADDER_ARMS="K=3,z=5,S=60,TH=0.5" PF_RL_DDEC=2 PF_TAG=lad1q PF_ORDLOG=1
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102; do
  pre=preds_o3; case $d in 202611*) pre=preds_c3;; esac
  PF_PREDS=$pre run lad1q_$d python3 -m predictfun.research.pf_ladder $d
done
run lad1q_report python3 -m predictfun.research.pf_ladder_orders_report
echo ALL_DONE
