#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/sizing_settlement_20261108
run() {
  local tag=$1; shift
  echo "START $tag"
  if ! "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; export PF_RL_DDEC=2; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1; then
    tail -30 "$OUT/logs/$tag.log"
    exit 1
  fi
  tail -2 "$OUT/logs/$tag.log"
}
run settlement_probability predictfun.research.pf_settlement_study
run settlement_inventory predictfun.research.pf_settlement_inventory
run sizing_engine_parity predictfun.research.pf_sizing_run smoke 20261023 J
for timing in J X; do
  for day in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    run sizing_${timing}_${day} predictfun.research.pf_sizing_run run "$day" "$timing"
  done
done
run sizing_report predictfun.research.pf_sizing_run report
echo 'COMPLETE: declared strategy and probability study'
