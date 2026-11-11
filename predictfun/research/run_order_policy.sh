#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yuqing/ctaNew
WORK=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/yuqing/pricer-sol/venv/bin/python3
run() { "$ROOT/predictfun/research/run_light.sh" "own_orders_$1" -- bash -c 'cd "$1" && shift && exec "$@"' _ "$WORK" "$PY" "${@:2}"; }
case "${1:-}" in
  controls) run controls -m unittest predictfun.research.tests.test_order_policy -v ;;
  declare) run declare -m predictfun.research.pf_order_policy_spec ;;
  smoke) run smoke -m predictfun.research.pf_order_policy_engine 20261023 smoke ;;
  all)
    run smoke -m predictfun.research.pf_order_policy_engine 20261023 smoke
    for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do run "build_$d" -m predictfun.research.pf_order_policy_engine "$d" build; done
    run fit -m predictfun.research.pf_order_policy_fit
    for d in 20261026 20261027 20261031 20261101 20261102 20261103; do run "eval_$d" -m predictfun.research.pf_order_policy_engine "$d" eval; done
    for d in 20261026 20261027 20261031 20261101 20261102 20261103; do run "stress_$d" -m predictfun.research.pf_order_policy_engine "$d" stress; done
    run report -m predictfun.research.pf_order_policy_report ;;
  *) echo 'Usage: run_order_policy.sh controls|declare|smoke|all' >&2;exit 2 ;;
esac
