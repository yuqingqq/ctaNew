#!/bin/bash
# pf_mutant_queue.sh -- one mutant per run_light job (one process, one slot core; review 4393ad19 B6). The control and the mutant indices are
# spread round robin over the given slots; each slot runs its jobs in order through run_light (absolute paths, 30-min timeout per job).
# Then: pf_mutant_summary.py <harness> <logdir> <prefix>.
# Usage: pf_mutant_queue.sh <harness file> <prefix> <log dir> <n mutants> <slot>...
set -u
H=$1; P=$2; L=$3; N=$4; shift 4; SL=("$@"); R=/home/yuqing/ctaNew/predictfun/research; mkdir -p $L
JOBS=(control $(seq 0 $((N - 1))))
for s in "${!SL[@]}"; do
  ( for j in "${!JOBS[@]}"; do
      [ $((j % ${#SL[@]})) -eq $s ] || continue
      env PF_SLOT=${SL[$s]} $R/run_light.sh ${P}_${JOBS[$j]} -- timeout 1800 python3 $R/pf_mutant_one.py $H ${JOBS[$j]} > $L/${P}_${JOBS[$j]}.log 2>&1
    done ) &
  sleep 2
done
wait
