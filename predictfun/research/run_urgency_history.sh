#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/urgency_history_20261109
run() {
  local tag=$1; shift
  echo "START $tag"
  if ! "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1; then
    tail -30 "$OUT/logs/$tag.log"
    exit 1
  fi
  tail -2 "$OUT/logs/$tag.log"
}
run urgency_history_controls unittest predictfun.research.tests.test_urgency_history predictfun.research.tests.test_passive_study predictfun.research.tests.test_sizing_study
for timing in J X; do
  for day in 20261023 20261024 20261025 20261026; do
    if [ -f "$OUT/research/step4/ladder_tokyoreal_custom_e2_preds_basis_study_urgency_history_${timing}/$day/passive_checks.json" ]; then
      run urgency_history_audit_${timing}_${day} predictfun.research.pf_urgency_history audit "$day" "$timing"
    else
      run urgency_history_${timing}_${day} predictfun.research.pf_urgency_history run "$day" "$timing"
    fi
  done
done
run urgency_history_report predictfun.research.pf_urgency_history report
echo 'COMPLETE: earlier urgency experiment'
