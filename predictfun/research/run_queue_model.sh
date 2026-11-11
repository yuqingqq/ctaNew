#!/usr/bin/env bash
# One light process at a time. Commit sources, pass controls, declare and commit
# the declaration before `all`. Output refuses overwrites.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LAUNCH=/home/yuqing/ctaNew/predictfun/research/run_light.sh
PYTHON=/home/yuqing/pricer-sol/venv/bin/python3
run() {
  local tag=$1; shift
  "$LAUNCH" "$tag" -- bash -c 'cd "$1" && shift && exec "$@"' _ "$TASK_ROOT" "$PYTHON" "$@"
}
case "${1:-}" in
  controls) run queue_controls -m unittest predictfun.research.tests.test_queue_model -v ;;
  declare) run queue_declare -m predictfun.research.pf_queue_model_spec ;;
  build) run "queue_build_${2:?day}" -m predictfun.research.pf_queue_model_data "$2" ;;
  fit) run queue_fit -m predictfun.research.pf_queue_model_fit ;;
  verify) run queue_verify -m predictfun.research.pf_queue_model_verify ;;
  all)
    git -C "$TASK_ROOT" ls-files --error-unmatch predictfun/reviews/queue_model_20261111/declaration.json >/dev/null
    for day in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
      run "queue_build_$day" -m predictfun.research.pf_queue_model_data "$day"
    done
    run queue_fit -m predictfun.research.pf_queue_model_fit
    run queue_verify -m predictfun.research.pf_queue_model_verify
    ;;
  *) echo 'Usage: run_queue_model.sh controls|declare|build DAY|fit|verify|all' >&2; exit 2 ;;
esac
