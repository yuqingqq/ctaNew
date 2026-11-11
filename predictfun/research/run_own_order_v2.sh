#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yuqing/ctaNew
WORK=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/data/predict_fun/research/own_orders_calm_20261111
PY=/home/yuqing/pricer-sol/venv/bin/python3
run() { "$ROOT/predictfun/research/run_light.sh" "own_orders_v2_$1" -- bash -c 'cd "$1" && shift && exec "$@"' _ "$WORK" "$PY" "${@:2}"; }
case "${1:-}" in
  controls) run controls -m unittest predictfun.research.tests.test_order_policy predictfun.research.tests.test_order_fit_v2 -v ;;
  declare) run declare -m predictfun.research.pf_own_order_fit_v2 declare ;;
  all)
    for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
      if [ -f "$OUT/runs/build/$d/checks.json" ]; then echo "REUSE VERIFIED build $d"; else run "build_$d" -m predictfun.research.pf_order_policy_engine "$d" build; fi
    done
    run fit -m predictfun.research.pf_own_order_fit_v2
    for d in 20261026 20261027 20261031 20261101 20261102 20261103; do run "eval_$d" -m predictfun.research.pf_order_policy_engine "$d" eval; done
    for d in 20261026 20261027 20261031 20261101 20261102 20261103; do run "stress_$d" -m predictfun.research.pf_order_policy_engine "$d" stress; done
    run report -m predictfun.research.pf_order_policy_report ;;
  *) echo 'Usage: run_own_order_v2.sh controls|declare|all' >&2;exit 2 ;;
esac
