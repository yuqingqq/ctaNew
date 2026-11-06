#!/usr/bin/env bash
# Round O1 (declared in STATUS.yml): size test, 6-day LODO model, strategy variants. One light process at a time.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
for d in $DAYS; do PF_TIMING=e2 run o1_size_$d python3 -m predictfun.research.pf_step4 $d tokyo20 size; done
echo SIZE_DONE
run o1_lodo6 python3 -m predictfun.research.pf_step5_lodo6 $DAYS
echo LODO6_DONE
for d in $DAYS; do PF_TIMING=e2 PF_PREDS=preds_lodo6 run o1_opt_$d python3 -m predictfun.research.pf_step4 $d tokyo20 opt; done
echo ALL_DONE
