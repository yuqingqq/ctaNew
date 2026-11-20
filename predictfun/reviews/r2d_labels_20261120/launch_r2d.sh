#!/bin/bash
# R2-D: the bounded Oct 24-25 integration build (review 8cc1dc49's next priority; USER: "Yea 4 cores to make it faster"; "accelerate while reliable").
# Subcommands, each with its computed gate (a failed gate stops):
#   gate      the runner v4 mutation suite passed on its recorded sources; the label selftest passed; the adapter's mutation suite passed
#             (SPECULATIVE=1: still running - checks.json's predicate F then accepts the builds only if it passed on exactly these sources);
#             the diag2 suites passed; writes sources_at_build.sha256 (refused if one exists with other content)
#   A         build A, uninterrupted: census -> pairs -> finalize (slot claude)
#   B         build B: census -> pairs (stopped after 40) -> pairs (resumed) -> finalize (slot claude2)
#   controls  the real-row controls on build A (slot claude)
#   checks    the declared checks on A and B (slot claude)
# Attempt 1 (manifest held every candidate in memory) is kept in logs/attempt1 and data/.../r2_labels_20261120_attempt1_manifest_in_memory.
set -u
R=/home/yuqing/ctaNew/predictfun/research; H=/home/yuqing/ctaNew/predictfun/reviews/r2d_labels_20261120; Lg=$H/logs
SC=/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad/teacher_logs; D=/home/yuqing/ctaNew/data/predict_fun/research/r2_labels_20261120
DG=/home/yuqing/ctaNew/predictfun/reviews/r2c_teacher_20261120/diag_v2/logs
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
h16() { sha256sum "$@" | cut -c1-16 | tr '\n' ' '; }
run() { env PF_SLOT=$1 $R/run_light.sh $2 -- python3 $R/pf_r2_labels.py "${@:3}" > $Lg/$2.log 2>&1; }
# the census CLI exits 1 after writing its outputs (its final print references the removed field 'candidates'); verify_census.py decides
census_ok() { run $1 r2d_$2_census census $D/$2; [ -f $D/$2/manifest.json ] && env PF_SLOT=$1 $R/run_light.sh r2d_$2_census_verify -- python3 $H/verify_census.py $D/$2 > $Lg/r2d_$2_census_verify.log 2>&1; }
cd /home/yuqing/ctaNew
case "${1:-}" in
gate)
  grep -q "r2 runner mutation check passed" $SC/runner_v4_mutants.log || { echo "$(stamp) gate: the runner v4 mutation suite did not pass"; exit 1; }
  [ "$(cat $SC/runner_v4_sources.txt | tr '\n' ' ')" = "$(h16 $R/pf_r2_runner.py $R/pf_r2_runner_mutants.py)" ] || { echo "$(stamp) gate: the runner changed since its suite"; exit 1; }
  grep -q "r2 labels selftests passed" $Lg/labels_selftest.log || { echo "$(stamp) gate: the label selftest did not pass"; exit 1; }
  if [ "${SPECULATIVE:-0}" != 1 ]; then
    for k in 0 1; do grep -q "r2 labels mutation check (shard $k/2) passed" $SC/labels_mutants_$k.log || { echo "$(stamp) gate: label mutation shard $k"; exit 1; }; done
  fi
  [ "$(cat $SC/labels_sources_at_mutants.txt | tr '\n' ' ')" = "$(h16 $R/pf_r2_labels.py $R/pf_r2_labels_selftest.py $R/pf_r2_labels_mutants.py)" ] || { echo "$(stamp) gate: the adapter differs from its suite's sources"; exit 1; }
  grep -q "selftest passed" $DG/diag2_selftest.log && grep -q "diag2 mutation check passed" $DG/diag2_mutants.log || { echo "$(stamp) gate: diag2 suites"; exit 1; }
  [ ! -e $D ] || { echo "$(stamp) gate: the build directory exists"; exit 1; }
  sha256sum $R/pf_r2_labels.py $R/pf_r2_labels_selftest.py $R/pf_r2_labels_mutants.py $R/pf_r2_teacher.py $R/pf_r2_runner.py $R/pf_r2_shadow.py $R/pf_branch_engine.py $R/run_light.sh > $H/sources_at_build.sha256
  echo "$(stamp) gate: the suites passed on these sources (speculative=${SPECULATIVE:-0})";;
A)
  [ "$(sha256sum $R/pf_r2_labels.py | cut -c1-64)" = "$(grep pf_r2_labels.py $H/sources_at_build.sha256 | head -1 | cut -c1-64)" ] || { echo "$(stamp) A: the adapter changed since the gate"; exit 1; }
  { [ -f $D/A/manifest.json ] || census_ok claude A; } && env PF_SLOT=claude $R/run_light.sh r2d_A_census_verify -- python3 $H/verify_census.py $D/A > $Lg/r2d_A_census_verify.log 2>&1 \
    && run claude r2d_A_pairs pairs $D/A && run claude r2d_A_finalize finalize $D/A; echo "$(stamp) A: rc=$?";;
B)
  [ "$(sha256sum $R/pf_r2_labels.py | cut -c1-64)" = "$(grep pf_r2_labels.py $H/sources_at_build.sha256 | head -1 | cut -c1-64)" ] || { echo "$(stamp) B: the adapter changed since the gate"; exit 1; }
  census_ok claude2 B && run claude2 r2d_B_pairs1 pairs $D/B 40 && run claude2 r2d_B_pairs2 pairs $D/B && run claude2 r2d_B_finalize finalize $D/B; echo "$(stamp) B: rc=$?";;
controls)
  env PF_SLOT=claude $R/run_light.sh r2d_controls -- python3 $H/r2d_controls.py $D/A > $Lg/r2d_controls.log 2>&1; echo "$(stamp) controls: rc=$?";;
checks)
  env PF_SLOT=claude $R/run_light.sh r2d_checks -- python3 $H/r2d_checks.py $D/A $D/B > $Lg/r2d_checks.log 2>&1; echo "$(stamp) checks: rc=$?";;
*) echo "usage: launch_r2d.sh gate|A|B|controls|checks"; exit 2;;
esac
