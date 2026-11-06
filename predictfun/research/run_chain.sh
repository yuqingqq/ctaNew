#!/usr/bin/env bash
# run_chain.sh <module> <day>... : run one step module over several days, sequentially, each through run_light.sh.
# Stops at the first failure.
set -u
M="$1"; shift
for d in "$@"; do
  /home/yuqing/ctaNew/predictfun/research/run_light.sh "${M}_${d}" -- python3 -m "predictfun.research.${M}" "$d" \
    > "/home/yuqing/ctaNew/data/predict_fun/research/logs/${M}_${d}.log" 2>&1 || { echo "FAILED ${M} ${d}"; exit 1; }
  tail -1 "/home/yuqing/ctaNew/data/predict_fun/research/logs/${M}_${d}.log" | cut -c1-400
done
