#!/bin/bash
# Final HONEST replay (audit remediation): regen base+long books from the gap-clean + stale-label-refreshed
# panel (V4_PANEL), then replay recent + OOS through KEEPSET4 with the gate .shift(1) + regime wall-clock fixes
# active. Fail-loud (set -e + explicit exits). Portable scratch (HONEST_SCR).
set -e
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"
SCR="${HONEST_SCR:-$(mktemp -d)}"; echo "scratch: $SCR"
export V4_PANEL=$ROOT/outputs/vBTC_features/panel_expanded_v0_clean.parquet V4_BOOK_SUFFIX=_honest

echo "=== regen RECENT books from clean panel ==="
python3 live/gen_residual_target.py > "$SCR/regen_recent.log" 2>&1 || { echo FAIL; tail -6 "$SCR/regen_recent.log"; exit 1; }
echo "=== regen OOS books from clean panel ==="
python3 live/gen_oos_v4.py 2023-01-01 2025-10-01 > "$SCR/regen_oos.log" 2>&1 || { echo FAIL; tail -6 "$SCR/regen_oos.log"; exit 1; }

# KEEPSET4 config (verbatim from replay_clean_confirm.sh)
export COST_BPS_LEG=9 FEE_BPS_FILL=4.5 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1 CHARGE_FUNDING=1
export DEPTH_COST_CSV=$ROOT/live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
export STRAT_K=2 STRAT_K_LONG=1 BEAR_K=0 SIZING_MODE=inv_sqrt_vol
export BEAR_DEPTH_RAMP=0 CONC_CAP=0.99 CONC_CAP_SINGLE_EXEMPT=1 SHORT_MIN_RET3D=-999 LONG_MAX_RET3D=999 BEAR_MODE=equal
export REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full
export STOP_SKIP_REGIMES=bear STOP_K_SIGMA=2.0 BULL_MODE=default BULL_GROSS_MULT=0 BULL_DEEP_THR=0.15 BULL_DEEP_MODE=mom1d_long
export GLOBAL_GROSS_MULT=1.0 CONVEXITY_DVOL_CACHE_PKL=$ROOT/live/state/v3loop/ddloop/_dvol_cache.pkl
rm -f "$CONVEXITY_DVOL_CACHE_PKL"   # audit #6 self-contained: force a FRESH .shift(1) rebuild; don't silently inherit a stale (leaked) pkl

run() {  # name base_dir long_dir from [end]
  export CONVEXITY_PREDS_PATH=$ROOT/live/state/convexity/$2/v0full_hl60.parquet
  export CONVEXITY_PREDS_LONG=$ROOT/live/state/convexity/$3/v0full_hl60.parquet
  export CONVEXITY_STATE=$SCR/$1; rm -rf "$CONVEXITY_STATE"; mkdir -p "$CONVEXITY_STATE"
  local endarg=""; [ -n "$5" ] && endarg="--replay-end $5"
  echo "== replay $1 =="
  python3 -m live.convexity_paper_bot --replay-from "$4" $endarg > "$SCR/replay_$1.log" 2>&1 || { echo FAIL; tail -6 "$SCR/replay_$1.log"; exit 1; }
  echo "  done -> $CONVEXITY_STATE/cycles.csv"
}
run recent_honest hl_tgt_res_base_honest hl_tgt_res_long_honest 2025-10-04
run oos_honest    hl_v4base_oos_honest   hl_v4long_oos_honest    2023-01-01 2025-10-01
echo "HONEST_FINAL_DONE: $SCR"
