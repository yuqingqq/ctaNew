#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yuqing/ctaNew
WORK=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/yuqing/pricer-sol/venv/bin/python3
run() { "$ROOT/predictfun/research/run_light.sh" "own_orders_v3_$1" -- bash -c 'cd "$1" && shift && exec "$@"' _ "$WORK" "$PY" "${@:2}"; }
case "${1:-}" in
  declare) run declare -m predictfun.research.pf_own_order_runner_v3 declare ;;
  all)
    for d in 20261026 20261027 20261031 20261101 20261102 20261103; do run "stress_$d" -m predictfun.research.pf_own_order_runner_v3 "$d" stress; done
    run report -m predictfun.research.pf_own_order_runner_v3 report ;;
  *) echo 'Usage: run_own_order_v3.sh declare|all' >&2;exit 2 ;;
esac
