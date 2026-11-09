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
run passive_smoke_X predictfun.research.pf_passive_run 20261027 X smoke
for timing in J X; do
  for day in 20261027 20261031 20261101 20261102 20261103; do
    run passive_${timing}_${day} predictfun.research.pf_passive_run "$day" "$timing" main
  done
done
for timing in J X; do
  run passive_stress_${timing} predictfun.research.pf_passive_run 20261103 "$timing" stress
done
run passive_final_controls unittest predictfun.research.tests.test_passive_study predictfun.research.tests.test_sizing_study
run passive_report predictfun.research.pf_passive_report
echo 'COMPLETE: passive inventory experiment'
