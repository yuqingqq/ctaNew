#!/usr/bin/env bash
# Regenerate all engine-dependent linear-model results with the CAUSAL-FIXED
# engines (s64.run / s65.runL bugfix 2026-05-16). Per-symbol/Step-62 predictions
# are engine-independent and reused. Continue-on-error so one failure does not
# block the rest; each step's own OUT dir + *_run.log are overwritten with the
# corrected output. Original stale logs were already superseded in memory.
set -u
cd /home/yuqing/ctaNew
LOGD=linear_model/results
ts() { date -u +%H:%M:%S; }
run() {
  local n="$1" script="$2"
  echo "==== [$(ts)] STEP $n START: $script ====" | tee -a "$LOGD/regen_corrected.log"
  if timeout 5400 python3 "linear_model/scripts/$script" \
        > "$LOGD/step${n}_run.log" 2>&1; then
    echo "==== [$(ts)] STEP $n DONE ====" | tee -a "$LOGD/regen_corrected.log"
  else
    echo "==== [$(ts)] STEP $n FAILED (rc=$?) — see step${n}_run.log ====" \
        | tee -a "$LOGD/regen_corrected.log"
  fi
}
echo "==== [$(ts)] REGEN START (causal-fixed engines) ====" > "$LOGD/regen_corrected.log"
run 64 64_meanrev_v2_backtest.py
run 65 65_tail_attrib_deconc.py
run 67 67_persymbol_meanrev.py
run 68 68_persymbol_selfstd.py
run 69 69_verify_selfstd.py
run 70 70_pit_fix_window.py
run 66 66_coststress_iterdrop.py
run 71 71_battery_alleligible.py
echo "==== [$(ts)] REGEN COMPLETE ====" | tee -a "$LOGD/regen_corrected.log"
