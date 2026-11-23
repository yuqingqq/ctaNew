#!/usr/bin/env bash
# MAKER-TRACE-RECONSTRUCTION-v1 launcher (P-2026-005; session aea5bfb9, 2026-11-22). Every Python run goes through run_light.sh with an
# absolute script path; each stage logs to logs/<tag>.log (write-once: an existing log refuses; pass a new suffix to retry) and appends
# "<tag> rc=<rc>" to logs/jobs_rc.txt. Order: selftest -> extract 20261024 (prefix) -> life-prefix -> project (write-once gate) -> the rest
# (extract 20261023/25/26/27/20261108 in parallel slots, life per day, truth, econ, report, checks), each rest stage refusing unless
# logs/projection.json says run_rest true.
# A retry suffix (_r<n>) is accepted by every stage; a run that run_light refused before starting (no summary line) does not consume its tag
# (its log is kept as <tag>.refused.<epoch>.log). FREEZE_CHECK=0 is honoured only by the selftest stage.
# Usage: launch_mt.sh selftest [sfx] | extract <day> <slot> [sfx] | life-prefix [sfx] | project [sfx] | life <day> <slot> [sfx] | truth [sfx] |
#        econ [sfx] | report [sfx] | checks [sfx]
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/maker_trace_20261122; L=$D/logs; RL=$P/run_light.sh
OUT=$R/data/predict_fun/research/maker_trace_v1_20261122; PREFIX=20261024
mkdir -p $L

SOURCES="predictfun/research/pf_mt_common.py predictfun/research/pf_mt_extract.py predictfun/research/pf_mt_extract_selftest.py
 predictfun/research/pf_mt_lifecycle.py predictfun/research/pf_mt_lifecycle_selftest.py predictfun/research/pf_mt_econ.py
 predictfun/research/pf_mt_econ_selftest.py predictfun/research/pf_mt_truth.py predictfun/research/pf_mt_truth_selftest.py
 predictfun/reviews/maker_trace_20261122/contract.md predictfun/reviews/maker_trace_20261122/declaration.md
 predictfun/reviews/maker_trace_20261122/manifest.json predictfun/reviews/maker_trace_20261122/launch_mt.sh
 predictfun/reviews/maker_trace_20261122/mt_checks.py predictfun/reviews/maker_trace_20261122/tools/rehearse_checks.py
 predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py predictfun/reviews/maker_trace_20261122/tools/project.py
 predictfun/reviews/maker_trace_20261122/tools/build_manifest.py predictfun/reviews/maker_trace_20261122/tools/rehearse_e2e.py
 predictfun/research/pf_mt_report.py predictfun/research/pf_mt_report_selftest.py"
frozen() {   # every source tracked and identical to HEAD (the freeze commit), else refuse
  git -C $R ls-files --error-unmatch $SOURCES > /dev/null 2>&1 && git -C $R diff --quiet HEAD -- $SOURCES \
    || { echo "REFUSED: a maker-trace source is untracked or differs from HEAD (freeze first)" >&2; exit 1; }
}
run() {   # run <slot> <tag> <cmd...>
  local slot=$1 tag=$2; shift 2
  [ -e "$L/$tag.log" ] && { echo "REFUSED: $L/$tag.log exists (write-once; retry with a new suffix)" >&2; return 1; }
  [ "$STAGE" = selftest ] && [ "${FREEZE_CHECK:-1}" = 0 ] || frozen
  echo "head=$(git -C $R rev-parse HEAD)" > "$L/$tag.log"
  env PF_SLOT=$slot $RL $tag -- "$@" >> "$L/$tag.log" 2>&1; local rc=$?
  if ! grep -q " tag=$tag rc=" "$L/$tag.log"; then    # run_light refused before starting: keep the evidence, free the tag
    mv "$L/$tag.log" "$L/$tag.refused.$(date +%s).log"; echo "$tag refused_before_start rc=$rc" >> $L/jobs_rc.txt
    echo "REFUSED before start: $tag (see logs/$tag.refused.*.log)" >&2; return 1
  fi
  echo "$tag rc=$rc" >> $L/jobs_rc.txt; tail -2 "$L/$tag.log" | cut -c1-300; return $rc
}
gate() {   # amendment 1: a write-once projection_v2.json (after a refused first projection and a USER top-up) supersedes projection.json
  local pj=$L/projection.json; [ -e $L/projection_v2.json ] && pj=$L/projection_v2.json
  python3 -c "import json,sys; j=json.load(open('$pj')); sys.exit(0 if j.get('run_rest') is True else 1)" 2>/dev/null \
    || { echo "REFUSED: $(basename $pj) is missing or run_rest is not true" >&2; exit 1; }
}
book_mids() {   # BOOK markets of a day (the frozen complete list), sorted
  python3 -c "import json; print(' '.join(str(m) for m in sorted(json.load(open('$R/predictfun/reviews/ladder_async_fast_20261122/manifest.json'))['complete'].get('$1', []))))"
}
depth_mids() {  # markets of a day whose depth the extractor wrote (BOOK on Oct 24-27, own-fill GT windows on Nov 8), sorted
  python3 -c "
import glob, os
ms = sorted({int(os.path.basename(p).split('_')[1]) for p in glob.glob('$OUT/depth/$1_*_frames.parquet')})
print(' '.join(map(str, ms)))"
}

sfx() {   # an optional retry suffix must look like _r<n>
  case "${1:-}" in ''|_r[0-9]|_r[0-9][0-9]) echo "${1:-}" ;; *) echo "REFUSED: retry suffix must be _r<n>, not '$1'" >&2; exit 2 ;; esac
}
STAGE=${1:-}
case "${1:-}" in
  selftest)   # synthetic only; allowed before the freeze (FREEZE_CHECK=0) and again after it
    rc=0
    S=$(sfx "${2:-}") || exit 2
    for t in extract lifecycle econ truth; do run claude mt_selftest_$t$S python3 $P/pf_mt_${t}_selftest.py || rc=1; done
    run claude mt_selftest_checks$S python3 $D/tools/rehearse_checks.py || rc=1
    exit $rc ;;
  extract)
    day=${2:?day}; slot=${3:?slot}
    S=$(sfx "${4:-}") || exit 2; [ "$day" = "$PREFIX" ] || gate
    run $slot mt_x_extract_$day$S python3 $P/pf_mt_extract.py day $day ;;
  life-prefix)
    S=$(sfx "${2:-}") || exit 2; m=$(book_mids $PREFIX | awk '{print $1}'); [ -n "$m" ] || { echo "REFUSED: no BOOK market on $PREFIX" >&2; exit 1; }
    run claude mt_life_${PREFIX}_$m$S python3 $P/pf_mt_lifecycle.py market $PREFIX $m ;;
  project)
    S=$(sfx "${2:-}") || exit 2; run claude mt_x_project$S python3 $D/tools/project.py ;;
  project-v2)   # amendment 1: only after a refused first projection and a USER top-up
    S=$(sfx "${2:-}") || exit 2; run claude mt_x_project_v2$S python3 $D/tools/project.py v2 ;;
  life)
    gate; day=${2:?day}; slot=${3:?slot}; S=$(sfx "${4:-}") || exit 2; m0=$(book_mids $PREFIX | awk '{print $1}'); rc=0
    ms=$(depth_mids $day); [ -n "$ms" ] || { echo "REFUSED: no depth market written for $day (extract it first)" >&2; exit 1; }
    for m in $ms; do
      [ "$day" = "$PREFIX" ] && [ "$m" = "$m0" ] && continue
      [ -e "$OUT/life/${day}_${m}.json" ] && continue     # already built (write-once outputs); a rerun only fills the gaps
      run $slot mt_life_${day}_$m$S python3 $P/pf_mt_lifecycle.py market $day $m || rc=1
    done; exit $rc ;;
  truth)
    gate; S=$(sfx "${2:-}") || exit 2
    run claude mt_x_truth_dry$S python3 $P/pf_mt_truth.py dry && run claude mt_x_truth_score$S python3 $P/pf_mt_truth.py score ;;
  econ)
    gate; S=$(sfx "${2:-}") || exit 2; run claude mt_x_econ$S python3 $P/pf_mt_econ.py run ;;
  report)   # the report module is frozen by its own later commit (before any reading); it must be tracked and equal HEAD
    gate; S=$(sfx "${2:-}") || exit 2; RS="predictfun/research/pf_mt_report.py predictfun/research/pf_mt_report_selftest.py"
    git -C $R ls-files --error-unmatch $RS > /dev/null 2>&1 && git -C $R diff --quiet HEAD -- $RS \
      || { echo "REFUSED: the report module is untracked or differs from HEAD" >&2; exit 1; }
    run claude mt_x_report_selftest$S python3 $P/pf_mt_report_selftest.py && run claude mt_x_report$S python3 $P/pf_mt_report.py ;;
  checks)
    gate; S=$(sfx "${2:-}") || exit 2; run claude mt_x_checks$S python3 $D/mt_checks.py run ;;
  *) echo "usage: launch_mt.sh selftest | extract <day> <slot> [sfx] | life-prefix | project | life <day> <slot> [sfx] | truth | econ | report | checks" >&2; exit 2 ;;
esac
