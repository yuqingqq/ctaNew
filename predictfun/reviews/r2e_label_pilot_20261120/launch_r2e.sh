#!/bin/bash
# R2-E launch gate (review 4393ad19): the r2_label_sample_v2 pilot on consumed Oct 24-25. Every job via run_light (absolute paths); the
# mutation suite holds one slot (2 processes), the pilot at most three (cap 5).
#   plans            -> plan A and plan B (independent, must be byte-identical) on claude2 / claude3
#   queue <slot> <job>...  -> jobs run in order on one slot; job = A<k> | B<k> | B<k>s<n> (B part k stopped after n worlds: the planted interruption)
#   finalize         -> finalize A and B
#   controls <slot> <part>...  -> the real-row controls on A's partitions (r2e_controls.py)
#   profile <slot>   -> cProfile of one world of part 0 in a profile-only copy of plan A (efficiency; its labels are never read)
set -u
R=/home/yuqing/ctaNew; M=$R/predictfun/research/pf_r2_label_sample.py; D=$R/data/predict_fun/research/r2_label_sample_v2_20261120
L=$R/predictfun/reviews/r2e_label_pilot_20261120/logs; mkdir -p $L; NS=${R2E_NS:-attempt4}   # attempts 1-3 kept (logs/attempt*)
lr() { env PF_SLOT=$1 $R/predictfun/research/run_light.sh $2 -- python3 $M "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $L/jobs_rc.txt; }
case "$1" in
  plans) lr claude2 r2e_plan_A plan $D/${NS}_pilot_A pilot & lr claude3 r2e_plan_B plan $D/${NS}_pilot_B pilot & wait ;;
  queue) S=$2; shift 2
         for j in "$@"; do
           case "$j" in
             A*) lr $S r2e_run_$j run $D/${NS}_pilot_A ${j#A} ;;
             B*s*) k=${j#B}; lr $S r2e_run_$j run $D/${NS}_pilot_B ${k%s*} ${k#*s} ;;
             B*) lr $S r2e_run_$j run $D/${NS}_pilot_B ${j#B} ;;
           esac
         done ;;
  finalize) lr claude2 r2e_finalize_A finalize $D/${NS}_pilot_A; lr claude2 r2e_finalize_B finalize $D/${NS}_pilot_B ;;
  profile) mkdir -p $D/${NS}_profile_only && cp $D/${NS}_pilot_A/plan.json $D/${NS}_profile_only/   # one world of part 0 under cProfile; labels never read
           env PF_SLOT=$2 $R/predictfun/research/run_light.sh r2e_pair_profile -- python3 -m cProfile -o $L/pair_profile.prof $M run $D/${NS}_profile_only 0 1 > $L/r2e_pair_profile.log 2>&1
           python3 -c "import pstats; p = pstats.Stats('$L/pair_profile.prof'); p.sort_stats('tottime').print_stats(25); p.sort_stats('cumulative').print_stats(40)" > $L/pair_profile.txt 2>&1 ;;
  controls) S=$2; shift 2                                    # r2e_controls.py on A's partitions, one process each
            for k in "$@"; do env PF_SLOT=$S $R/predictfun/research/run_light.sh r2e_controls_$k -- python3 $R/predictfun/reviews/r2e_label_pilot_20261120/r2e_controls.py $D/${NS}_pilot_A $k \
                > $L/r2e_controls_$k.log 2>&1; echo "r2e_controls_$k rc=$?" >> $L/jobs_rc.txt; done ;;
esac
