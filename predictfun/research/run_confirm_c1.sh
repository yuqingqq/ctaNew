#!/usr/bin/env bash
# Confirmation C1 (declared in STATUS.yml before reading these days). Sequential, one light process at a time.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
DAYS="20261026 20261027 20261031"
for d in $DAYS; do
  run s1_$d python3 -m predictfun.research.pf_step1 $d
  run t_$d  python3 -m predictfun.research.pf_step1_timing $d
  run bt_$d python3 -m predictfun.research.pf_step1_booktime $d
  run s5f_$d python3 -m predictfun.research.pf_step5_features $d
done
run s5_apply python3 -m predictfun.research.pf_step5_apply --train 20261023,20261024,20261025 --apply 20261026,20261027,20261031
for d in $DAYS; do PF_TIMING=e2 run s4_model_$d python3 -m predictfun.research.pf_step4 $d tokyo20 model; done
run rep_c1 python3 -m predictfun.research.pf_step4_report tokyo20_model_e2 20261026 20261027 20261031
echo ALL_DONE
