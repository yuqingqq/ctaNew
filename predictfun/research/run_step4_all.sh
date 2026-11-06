#!/usr/bin/env bash
# Step-4 campaign (exploratory): sequential, one process at a time via run_light.sh.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; exit 1; }; tail -1 "$L/$tag.log" | cut -c1-200; }
for d in 20261024 20261025; do run s4_t20_singles_$d python3 -m predictfun.research.pf_step4 $d tokyo20 singles; done
for d in 20261023 20261024 20261025; do run s4_t20_stack_$d python3 -m predictfun.research.pf_step4 $d tokyo20 stack; done
for d in 20261023 20261024 20261025; do run s4_tape_ends_$d python3 -m predictfun.research.pf_step4 $d tape ends; done
for d in 20261023 20261024 20261025; do run s2_t20_$d python3 -m predictfun.research.pf_step2 $d tokyo20; done
for t in tokyo20_singles tokyo20_stack tape_ends; do run rep_$t python3 -m predictfun.research.pf_step4_report $t 20261023 20261024 20261025; done
echo ALL_DONE
