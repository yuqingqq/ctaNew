#!/bin/bash
# The full Oct 23 R2 build on 4 cores. USER 2026-11-20: "As long as the results are reliable, we can accelerate the process" /
# "Yea 4 cores to make it faster". A gated chain - every gate is a computed predicate and a failed one stops the chain:
#  1. the shadow mutation suite of c8d8fc8d passes;
#  2. the sharded runner (pf_r2_runner.py with shards) is swapped in, and its selftest and full mutation suite pass;
#  3. the frozen 256 as 4 parallel shards reproduces gate2A (shard_check.py, declared before the run);
#  4. the full Oct 23 plan (all 30,466 originals), 4 parallel shards (one per research slot), finalize.
# Review e6b0efa4 (2026-11-20) accepted c8d8fc8d's gate with no blocking finding; the sharded runner is reviewed with the full-day receipt.
# Its finding 2 workaround: before each finalize, preflight.py asserts that the finalizer's actual source map equals the plan's. No resource
# stop is resumed automatically (finding 1): a shard that is not PARTS_DONE stops the chain before finalize.
set -u
ROOT=/home/yuqing/ctaNew; RS=$ROOT/predictfun/research; G=$ROOT/predictfun/reviews/r2a_launch_gate_20261120/logs
F=$ROOT/predictfun/reviews/r2b_full_day_20261120; L=$F/logs; SC=/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad
R=$RS/pf_r2_runner.py; SL=(claude claude2 claude3 claude4); cd $ROOT
say() { echo "$(date -u +%FT%TZ) $*"; }
until grep -q "tag=shadow_mutants_final" $G/shadow_mutants_final.log 2>/dev/null; do sleep 30; done
grep -q "r2 shadow mutation check passed" $G/shadow_mutants_final.log || { say "GATE 1 FAILED: the shadow suite"; exit 1; }
say "gate 1: the shadow suite passed"
cp $SC/pf_r2_runner_sharded.py $RS/.pf_r2_runner.py.new && mv $RS/.pf_r2_runner.py.new $R
cp $SC/pf_r2_runner_mutants_sharded.py $RS/.pf_r2_runner_mutants.py.new && mv $RS/.pf_r2_runner_mutants.py.new $RS/pf_r2_runner_mutants.py
sha256sum $R $RS/pf_r2_runner_mutants.py $RS/pf_r2_shadow.py $RS/run_light.sh > $F/sources_at_launch.sha256
env PF_SLOT=claude $RS/run_light.sh shard_runner_selftest -- python3 $R selftest > $L/runner_selftest.log 2>&1 || { say "GATE 2 FAILED: the runner selftest"; exit 2; }
env PF_SLOT=claude $RS/run_light.sh shard_runner_mutants -- python3 $RS/pf_r2_runner_mutants.py > $L/runner_mutants.log 2>&1
grep -q "r2 runner mutation check passed" $L/runner_mutants.log || { say "GATE 2 FAILED: the runner mutation suite"; exit 2; }
say "gate 2: the sharded runner's selftest and mutation suite passed"
env PF_SLOT=claude $RS/run_light.sh shard4_plan -- python3 $R plan 20261023 _shard4 128 > $L/shard4_plan.log 2>&1 || { say "GATE 3 FAILED: the shard4 plan"; exit 3; }
P=(); for k in 0 1 2 3; do env PF_SLOT=${SL[$k]} $RS/run_light.sh shard4_run_$k -- python3 $R run 20261023 _shard4 $k/4 > $L/shard4_run_$k.log 2>&1 & P+=($!); sleep 2; done; wait ${P[@]}
for k in 0 1 2 3; do grep -qx PARTS_DONE $L/shard4_run_$k.log || { say "GATE 3 FAILED: shard4 run $k"; exit 3; }; done
env PF_SLOT=claude $RS/run_light.sh shard4_preflight -- python3 $F/preflight.py 20261023 _shard4 > $L/shard4_preflight.log 2>&1 || { say "GATE 3 FAILED: the shard4 finalizer's sources differ from its plan"; exit 3; }
env PF_SLOT=claude $RS/run_light.sh shard4_finalize -- python3 $R finalize 20261023 _shard4 > $L/shard4_finalize.log 2>&1 || { say "GATE 3 FAILED: the shard4 finalize"; exit 3; }
env PF_SLOT=claude $RS/run_light.sh shard4_check -- python3 $F/shard_check.py > $L/shard4_check.log 2>&1 || { say "GATE 3 FAILED: the shard check"; exit 3; }
say "gate 3: the 256 as 4 shards reproduces gate2A"
env PF_SLOT=claude $RS/run_light.sh r2full_plan -- python3 $R plan 20261023 _full all > $L/r2full_plan.log 2>&1 || { say "the full-day plan failed"; exit 4; }
say "full day: plan written; 4 shards start"
P=(); for k in 0 1 2 3; do env PF_SLOT=${SL[$k]} $RS/run_light.sh r2full_run_$k -- python3 $R run 20261023 _full $k/4 > $L/r2full_run_$k.log 2>&1 & P+=($!); sleep 2; done; wait ${P[@]}
for k in 0 1 2 3; do say "full day shard $k: $(grep -E '^(PARTS_DONE|STOPPED|STOP_RESOURCE)$' $L/r2full_run_$k.log | tail -1)"; done
for k in 0 1 2 3; do grep -qx PARTS_DONE $L/r2full_run_$k.log || { say "the full day: shard $k is not done - no finalize"; exit 5; }; done
env PF_SLOT=claude $RS/run_light.sh r2full_preflight -- python3 $F/preflight.py 20261023 _full > $L/r2full_preflight.log 2>&1 || { say "the full day: the finalizer's sources differ from its plan - no finalize"; exit 6; }
env PF_SLOT=claude $RS/run_light.sh r2full_finalize -- python3 $R finalize 20261023 _full > $L/r2full_finalize.log 2>&1; say "full day: finalize rc=$?"
