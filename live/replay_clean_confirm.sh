#!/bin/bash
# Confirmatory full-stack replay: LEAKED vs label-CLEAN books through IDENTICAL KEEPSET4 config.
# Self-validating: if the LEAKED replay reproduces the documented canonical Sharpe, the config is
# faithful and the CLEAN delta is trustworthy. Parity: GLOBAL_GROSS_MULT=1.0 (pre-cap, as the
# validated cell), universe-meta defaults to PANEL (historical replay). Isolates the LABEL fix
# (both books are not-universe-cleaned, so the only difference is the 317 gap-label NaNs).
set -e
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"
SCR="${REPLAY_SCRATCH:-$(mktemp -d)}"   # portable (audit #6): no hardcoded session path; override with REPLAY_SCRATCH
echo "scratch dir: $SCR"
# --- KEEPSET4 env (verbatim from run_convexity_v4_live.sh, GLOBAL_GROSS_MULT->1.0 parity) ---
export COST_BPS_LEG=9 FEE_BPS_FILL=4.5 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1 CHARGE_FUNDING=1
export DEPTH_COST_CSV=$ROOT/live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
export STRAT_K=2 STRAT_K_LONG=1 BEAR_K=0 SIZING_MODE=inv_sqrt_vol
export BEAR_DEPTH_RAMP=0 CONC_CAP=0.99 CONC_CAP_SINGLE_EXEMPT=1 SHORT_MIN_RET3D=-999 LONG_MAX_RET3D=999
export BEAR_MODE=equal
export REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full
export STOP_SKIP_REGIMES=bear STOP_K_SIGMA=2.0
export BULL_MODE=default BULL_GROSS_MULT=0
export BULL_DEEP_THR=0.15 BULL_DEEP_MODE=mom1d_long
export GLOBAL_GROSS_MULT=1.0
export CONVEXITY_DVOL_CACHE_PKL=$ROOT/live/state/v3loop/ddloop/_dvol_cache.pkl
# NB: do NOT set CONVEXITY_UNIVERSE_META -> defaults to PANEL (historical replay sees full universe)

run_one() {  # name base_dir long_dir from [end]
  local name=$1 base=$2 long=$3 fromd=$4 endd=$5
  export CONVEXITY_PREDS_PATH=$ROOT/live/state/convexity/$base/v0full_hl60.parquet
  export CONVEXITY_PREDS_LONG=$ROOT/live/state/convexity/$long/v0full_hl60.parquet
  export CONVEXITY_STATE=$SCR/replay_$name; rm -rf "$CONVEXITY_STATE"; mkdir -p "$CONVEXITY_STATE"
  local endarg=""; [ -n "$endd" ] && endarg="--replay-end $endd"
  echo "== replay $name ($base | from $fromd $endd) =="
  python3 -m live.convexity_paper_bot --replay-from "$fromd" $endarg > "$SCR/replay_$name.log" 2>&1 || { echo "  FAILED (see log)"; tail -5 "$SCR/replay_$name.log"; return 1; }   # audit #6: FAIL LOUD (was return 0 -> swallowed crashes)
  echo "  done -> $CONVEXITY_STATE/cycles.csv"
}

run_one recent_leaked hl_tgt_res_base          hl_tgt_res_long          2025-10-04
run_one recent_clean  hl_tgt_res_base_cleanfix hl_tgt_res_long_cleanfix 2025-10-04
run_one oos_leaked    hl_v4base_oos            hl_v4long_oos            2023-01-01 2025-10-01
run_one oos_clean     hl_v4base_oos_cleanfix   hl_v4long_oos_cleanfix   2023-01-01 2025-10-01
echo "REPLAYDONE"
