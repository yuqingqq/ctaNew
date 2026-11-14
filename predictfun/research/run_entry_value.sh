#!/usr/bin/env bash
# Separate research candidate. Does not modify/resume ORDER-HARM outputs.
# Each stage refuses existing artifacts; failed attempts remain as provenance.
set -euo pipefail
ENTRY_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENTRY_OUT="${PF_ENTRY_VALUE_OUT:-/home/yuqing/ctaNew/data/predict_fun/research/entry_value_20261114_v1}"
ENTRY_PYTHON="${PF_ENTRY_VALUE_PYTHON:-python3}"
entry_job() {
    local tag=$1
    shift
    env PF_SLOT="${PF_SLOT:-codex}" "$ENTRY_CODE_DIR/run_light.sh" "entry_value_$tag" -- \
        "$ENTRY_PYTHON" "$ENTRY_CODE_DIR/pf_entry_value.py" "$@" --out "$ENTRY_OUT"
}
entry_job selftest selftest
for day in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    entry_job "rows_$day" rows "$day"
done
for day in 20261027 20261031 20261101 20261102 20261103; do
    entry_job "fit_$day" fit "$day"
    entry_job "eval_$day" eval "$day"
done
entry_job report report
