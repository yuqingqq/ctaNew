#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/passive_inventory_20261109
run() {
  local tag=$1; shift
  echo "START $tag"
  if ! "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; export PF_RL_DDEC=2; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1; then
    tail -30 "$OUT/logs/$tag.log"
    exit 1
  fi
  tail -2 "$OUT/logs/$tag.log"
}
if [ ! -f "$OUT/research/step4/ladder_tokyoreal_custom_e2_preds_basis_study_passive_smoke_X/20261027/passive_checks.json" ]; then
  run passive_smoke_X predictfun.research.pf_passive_run 20261027 X smoke
fi
for timing in J X; do
  for day in 20261027 20261031 20261101 20261102 20261103; do
    if [ -f "$OUT/research/step4/ladder_tokyoreal_custom_e2_preds_basis_study_passive_main_${timing}/$day/passive_checks.json" ]; then
      run passive_audit_${timing}_${day} predictfun.research.pf_passive_report audit "$day" "$timing" main
    else
      run passive_${timing}_${day} predictfun.research.pf_passive_run "$day" "$timing" main
    fi
  done
done
for timing in J X; do
  if [ -f "$OUT/research/step4/ladder_tokyoreal_custom_e2_preds_basis_study_passive_stress_${timing}/20261103/passive_checks.json" ]; then
    run passive_audit_stress_${timing} predictfun.research.pf_passive_report audit 20261103 "$timing" stress
  else
    run passive_stress_${timing} predictfun.research.pf_passive_run 20261103 "$timing" stress
  fi
done
run passive_final_controls unittest predictfun.research.tests.test_passive_study predictfun.research.tests.test_sizing_study
run passive_report predictfun.research.pf_passive_report
echo 'COMPLETE: passive inventory experiment'
