#!/bin/bash
# The pi0-50 fit / freeze / diagnostics / rollout chain (review 3d7eae6a's next priority; USER: "Yea 4 cores to make it faster").
# Gates, each computed, and a failed one stops the chain:
#   (0) the teacher's mutation suite passed on the exact sources now on disk, and the extraction is complete;
#   (1) four group fits in parallel, one per approved slot, each rc 0;
#   (2) the freeze;
#   (3) diag and rollout in parallel.
set -u
R=/home/yuqing/ctaNew/predictfun/research; H=/home/yuqing/ctaNew/predictfun/reviews/r2c_teacher_20261120; L=$H/logs
SC=/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad/teacher_logs
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd /home/yuqing/ctaNew
grep -q "r2 teacher mutation check passed" $SC/teacher_mutants.log || { echo "$(stamp) gate 0: the mutation suite did not pass"; exit 1; }
now=$(sha256sum $R/pf_r2_teacher.py $R/pf_r2_teacher_selftest.py $R/pf_r2_teacher_mutants.py | cut -c1-16 | tr '\n' ' ')
then=$(cat $SC/sources_at_mutants.txt | tr '\n' ' ')
[ "$now" = "$then" ] || { echo "$(stamp) gate 0: the sources changed since the mutation suite ($then -> $now)"; exit 1; }
grep -q " rc=0 " $SC/teacher_extract_v1.log && [ -f /home/yuqing/ctaNew/data/predict_fun/research/r2_teacher_20261120/train/manifest.json ] || { echo "$(stamp) gate 0: the extraction is not complete"; exit 1; }
[ "$(cat $SC/sources_at_extract.txt)" = "$(sha256sum $R/pf_r2_teacher.py | cut -c1-16)" ] || { echo "$(stamp) gate 0: the extraction ran another teacher source"; exit 1; }
sha256sum $R/pf_r2_teacher.py $R/pf_r2_teacher_selftest.py $R/pf_r2_teacher_mutants.py $R/pf_r2_shadow.py $R/pf_branch_engine.py $R/pf_pipeline_bench.py $R/run_light.sh > $H/sources_at_fit.sha256
echo "$(stamp) gate 0: mutation suite passed on these sources; extraction complete"
pids=()
for k in 0 1 2 3; do
  slot=$([ $k = 0 ] && echo claude || echo claude$((k + 1)))
  env PF_SLOT=$slot $R/run_light.sh teacher_fit_$k -- python3 $R/pf_r2_teacher.py fit $k > $L/teacher_fit_$k.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { echo "$(stamp) gate 1: a fit failed"; exit 1; }
echo "$(stamp) gate 1: four fits rc 0"
env PF_SLOT=claude $R/run_light.sh teacher_freeze -- python3 $R/pf_r2_teacher.py freeze > $L/teacher_freeze.log 2>&1 || { echo "$(stamp) gate 2: the freeze failed"; exit 1; }
echo "$(stamp) gate 2: frozen"
env PF_SLOT=claude $R/run_light.sh teacher_diag -- python3 $R/pf_r2_teacher.py diag > $L/teacher_diag.log 2>&1 & d=$!
env PF_SLOT=claude2 $R/run_light.sh teacher_rollout -- python3 $R/pf_r2_teacher.py rollout > $L/teacher_rollout.log 2>&1 & r=$!
wait $d; drc=$?; wait $r; rrc=$?
echo "$(stamp) gate 3: diag rc=$drc rollout rc=$rrc"
