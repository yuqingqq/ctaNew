#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/sizing_settlement_20261108
test -f "$OUT/strategy_report.json"
run() {
  local tag=$1; shift
  echo "START $tag"
  if ! "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; export PF_RL_DDEC=2; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1; then
    tail -30 "$OUT/logs/$tag.log"
    exit 1
  fi
  tail -2 "$OUT/logs/$tag.log"
}
run sizing_cache_check predictfun.research.pf_sizing_checks
run settlement_nonlinear predictfun.research.pf_settlement_nonlinear
run settlement_nonlinear_inventory predictfun.research.pf_settlement_nonlinear shadow
run sizing_cancel_ack_measure predictfun.research.pf_sizing_latency measure
run sizing_cancel_ack_J predictfun.research.pf_sizing_latency stress J
run sizing_cancel_ack_X predictfun.research.pf_sizing_latency stress X
run sizing_source_tests unittest predictfun.research.tests.test_sizing_study
echo 'COMPLETE: declared followups and controls'
