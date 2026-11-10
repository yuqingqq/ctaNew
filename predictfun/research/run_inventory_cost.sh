#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/inventory_cost_20261110
run() {
  local tag=$1; shift
  "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1 || { tail -30 "$OUT/logs/$tag.log"; return 1; }
  tail -2 "$OUT/logs/$tag.log"
}
run controls pytest -q predictfun/research/tests/test_inventory_cost.py predictfun/research/tests/test_late_reduction.py predictfun/research/tests/test_inflight_budget.py predictfun/research/tests/test_entry_budget.py --tb=short
run smoke predictfun.research.pf_inventory_cost_run 20261023 J smoke
for timing in J X; do
  for day in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    run cost_${timing}_${day} predictfun.research.pf_inventory_cost_run "$day" "$timing" main
  done
done
run stress predictfun.research.pf_inventory_cost_run 20261103 X stress
run report predictfun.research.pf_inventory_cost_report
