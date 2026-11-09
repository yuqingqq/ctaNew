#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/late_reduction_20261109
run() {
  local tag=$1; shift
  while ps -eo args= | rg -q '^(/[^ ]*/)?python(3([.][0-9]+)?)? .*predictfun[./]research[./]|^bash [^ ]*run_live_samewindow.sh'; do
    echo "QUEUED $tag: another research process/workflow is running"
    sleep 15
  done
  echo "START $tag"
  if ! "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1; then
    tail -30 "$OUT/logs/$tag.log"
    exit 1
  fi
  tail -2 "$OUT/logs/$tag.log"
}
run late_controls unittest predictfun.research.tests.test_late_reduction predictfun.research.tests.test_inflight_budget predictfun.research.tests.test_entry_budget predictfun.research.tests.test_passive_study predictfun.research.tests.test_sizing_study
run late_smoke predictfun.research.pf_late_run 20261023 J smoke
for timing in J X; do
  for day in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    run late_${timing}_${day} predictfun.research.pf_late_run "$day" "$timing" main
  done
done
run late_report predictfun.research.pf_late_report
echo 'COMPLETE: late-reduction experiment'
