#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/entry_budget_20261109
run() {
  local tag=$1; shift
  echo "START $tag"
  if ! "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1; then
    tail -30 "$OUT/logs/$tag.log"
    exit 1
  fi
  tail -2 "$OUT/logs/$tag.log"
}
run entry_controls unittest predictfun.research.tests.test_entry_budget predictfun.research.tests.test_passive_study predictfun.research.tests.test_sizing_study
if [ ! -f "$OUT/research/step4/ladder_tokyoreal_custom_e2_preds_basis_study_entry_smoke_J/20261023/passive_checks.json" ]; then
  run entry_smoke predictfun.research.pf_entry_run 20261023 J smoke
fi
for timing in J X; do
  for day in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    if [ -f "$OUT/research/step4/ladder_tokyoreal_custom_e2_preds_basis_study_entry_main_${timing}/$day/passive_checks.json" ]; then
      run entry_audit_${timing}_${day} predictfun.research.pf_entry_report audit "$day" "$timing"
    else
      run entry_${timing}_${day} predictfun.research.pf_entry_run "$day" "$timing" main
    fi
  done
done
run entry_report predictfun.research.pf_entry_report
echo 'COMPLETE: entry-budget experiment'
