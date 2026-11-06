#!/usr/bin/env bash
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; exit 1; }; }
for d in 20261023 20261024 20261025; do run booktime_$d python3 -m predictfun.research.pf_step1_booktime $d; done
for tm in e2 e4; do for d in 20261023 20261024 20261025; do
  PF_TIMING=$tm PF_POLICIES=MICRO run s4_t20_stack_${tm}_$d python3 -m predictfun.research.pf_step4 $d tokyo20 stack; done
  run rep_stack_$tm python3 -m predictfun.research.pf_step4_report tokyo20_stack_$tm 20261023 20261024 20261025; done
echo ALL_DONE
