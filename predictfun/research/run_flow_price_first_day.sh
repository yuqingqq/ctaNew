#!/usr/bin/env bash
# User requested the first chronological readout before the remaining dates.
set -euo pipefail
FLOW_SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLOW_PY=/home/yuqing/pricer-sol/venv/bin/python3
FLOW_WRAPPER=/home/yuqing/ctaNew/predictfun/research/run_light.sh
FLOW_OUT=/home/yuqing/ctaNew/data/predict_fun/research/flow_price_20261116_v1
FLOW_LOGS="$FLOW_OUT/logs"
# The old scheduler is stopped; let its Oct24 child finish without interruption.
while [[ -e /proc/1420788 ]]; do sleep 5; done
if [[ ! -s "$FLOW_OUT/20261024/receipt.json" ]]; then
  printf 'REFUSED: Oct24 build did not finish successfully\n' >&2
  exit 1
fi
"$FLOW_PY" - <<'PY'
from pathlib import Path
import os,signal
p=Path('/proc/1351263/cmdline')
if p.exists():
    cmd=p.read_bytes().replace(b'\0',b' ').decode().strip()
    assert cmd=='bash /tmp/cta-flow-price-model-20261116/predictfun/research/run_flow_price_predictions.sh',cmd
    os.kill(1351263,signal.SIGTERM)
    os.kill(1351263,signal.SIGCONT)
PY
for day in 20261025 20261026 20261027; do
  printf 'Building %s for first test fold\n' "$day"
  PF_SLOT=codex "$FLOW_WRAPPER" "flow_price_data_$day" -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_data.py" "$day" >"$FLOW_LOGS/data_$day.log" 2>&1
done
printf 'Fitting first chronological test day 20261027\n'
PF_SLOT=codex "$FLOW_WRAPPER" flow_price_fit_20261027 -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_model.py" 20261027 >"$FLOW_LOGS/fit_20261027.log" 2>&1
printf 'Auditing and reporting first-day predictions\n'
PF_SLOT=codex "$FLOW_WRAPPER" flow_price_first_day -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_first_day.py" >"$FLOW_LOGS/first_day.log" 2>&1
printf 'First-day readout complete; continuing remaining frozen dates\n'
for day in 20261031 20261101 20261102 20261103; do
  PF_SLOT=codex "$FLOW_WRAPPER" "flow_price_data_$day" -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_data.py" "$day" >"$FLOW_LOGS/data_$day.log" 2>&1
  PF_SLOT=codex "$FLOW_WRAPPER" "flow_price_fit_$day" -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_model.py" "$day" >"$FLOW_LOGS/fit_$day.log" 2>&1
done
PF_SLOT=codex "$FLOW_WRAPPER" flow_price_report -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_report.py" >"$FLOW_LOGS/report.log" 2>&1
PF_SLOT=codex "$FLOW_WRAPPER" flow_price_audit -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_check.py" audit >"$FLOW_LOGS/audit.log" 2>&1
printf 'Full prediction experiment complete\n'
