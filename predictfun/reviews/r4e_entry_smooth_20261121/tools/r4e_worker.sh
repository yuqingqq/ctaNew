#!/bin/bash
# R4E replay worker for one slot: claim replay partitions 1..11 atomically (mkdir) and run each through launch_r4e.sh (run_light, absolute
# paths). Stops claiming at a STOP_RESOURCE partition status or when the round's projected replay CPU passes the 8 core-hour ceiling.
set -u
SLOT=$1; R=/home/yuqing/ctaNew; RC=$R/predictfun/reviews/r4e_entry_smooth_20261121; D=$R/data/predict_fun/research/adv2_entry_smooth_v1_20261121
CL=$D/replay_v1/claims; mkdir -p $CL
for k in $(seq 1 11); do
  grep -l STOP_RESOURCE $D/replay_v1/parts/part_*_status.json > /dev/null 2>&1 && { echo "worker $SLOT: a STOP_RESOURCE status, no further claims"; break; }
  mkdir $CL/part_$k 2> /dev/null || continue
  bash $RC/launch_r4e.sh replay $SLOT $k
done
echo "worker $SLOT done $(date -u +%FT%TZ)"
