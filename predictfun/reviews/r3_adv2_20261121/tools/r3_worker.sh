#!/bin/bash
# R3 replay worker for one slot: optionally wait until the slot's label queue has finished (its last partition has an rc line), optionally
# run the label finalize + the offline evaluation first (once all 96 label partitions have rc lines), then claim replay partitions 1..11
# atomically (mkdir) and run each through launch_r3.sh (run_light, absolute paths). Usage: r3_worker.sh <slot> <wait-for-label-k|-> <post 0|1>
set -u
SLOT=$1; WAITK=$2; POST=$3; R=/home/yuqing/ctaNew; RC=$R/predictfun/reviews/r3_adv2_20261121; J=$RC/logs/jobs_rc.txt
D=$R/data/predict_fun/research/adv2_rich_v1_20261121; CL=$D/replay_v1/claims; mkdir -p $CL
if [ "$WAITK" != "-" ]; then
  until grep -q "^r3_eval_${WAITK} rc=" $J || grep -q "^r3_eval_${WAITK} skipped" $J; do sleep 30; done
  while pgrep -f "launch_r3.sh queue $SLOT " > /dev/null; do sleep 10; done
fi
if [ "$POST" = "1" ]; then
  until [ "$(grep -c '^r3_eval_[0-9]* \(rc=\|skipped\)' $J)" -ge 96 ]; do sleep 30; done
  while pgrep -f "launch_r3.sh queue " > /dev/null; do sleep 10; done
  if [ "$(grep '^r3_eval_[0-9]* rc=' $J | grep -vc ' rc=0')" = "0" ] && [ ! -f $RC/logs/STOP_LABEL_28_CORE_HOURS ]; then
    env PF_SLOT=$SLOT $R/predictfun/research/run_light.sh r3_eval_finalize -- python3 $R/predictfun/research/pf_adv2_sample.py finalize $D/eval_labels_v1 > $RC/logs/r3_eval_finalize.log 2>&1
    echo "r3_eval_finalize rc=$?" >> $J
    grep -q '^r3_eval_finalize rc=0' $J && bash $RC/launch_r3.sh eval $SLOT
  else
    echo "r3_eval_finalize skipped (a failed / stopped label partition)" >> $J
  fi
fi
for k in $(seq 1 11); do
  mkdir $CL/part_$k 2> /dev/null || continue
  bash $RC/launch_r3.sh replay $SLOT $k
done
echo "worker $SLOT done $(date -u +%FT%TZ)"
