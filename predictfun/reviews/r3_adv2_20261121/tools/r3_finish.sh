#!/bin/bash
# After the evaluation: the declared checks + falsifier (launch_r3.sh checks), the receipt's output copies, the tables (r3_tables.py) and
# the generated report (fill_r3.py), each job through run_light on the given slot.
set -u
SLOT=${1:-claude}; R=/home/yuqing/ctaNew; RC=$R/predictfun/reviews/r3_adv2_20261121; L=$RC/logs; J=$L/jobs_rc.txt
S=/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad
grep -q '^r3_eval_readings rc=0' $J || { echo "REFUSED: the evaluation has not finished rc=0"; exit 1; }
env PF_SLOT=$SLOT $R/predictfun/research/run_light.sh r3_checks -- python3 $RC/r3_checks.py > $L/r3_checks.log 2>&1; echo "r3_checks rc=$?" >> $J
bash $S/r3_collect.sh > $L/r3_collect.log 2>&1; echo "r3_collect rc=$?" >> $J
env PF_SLOT=$SLOT $R/predictfun/research/run_light.sh r3_tables -- python3 $RC/r3_tables.py > $RC/tables.md 2> $L/r3_tables.log; echo "r3_tables rc=$?" >> $J
env PF_SLOT=$SLOT $R/predictfun/research/run_light.sh r3_report -- python3 $S/fill_r3.py > $L/r3_report.log 2>&1; echo "r3_report rc=$?" >> $J
grep '^r3_checks\|^r3_collect\|^r3_tables\|^r3_report' $J | tail -4; grep -A12 '"checks"' $L/r3_checks.log | head -14; tail -3 $L/r3_checks.log | cut -c1-300
