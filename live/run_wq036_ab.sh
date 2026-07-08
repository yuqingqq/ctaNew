#!/usr/bin/env bash
# Controlled wq036 A/B: identical v3_native strategy + matched execution (from controlled_v1v3.sh IS cell),
# vary ONLY the pred set (row-matched regenerated baseline vs +wq036 variants).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
D=live/state/convexity
EXEC=(COST_BPS_LEG=9 CHARGE_FUNDING=1 CONVEXITY_PIT_DVOL=1 XS_LEAN=1
      CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet
      DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
      CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl)
V3=(STRAT_K=2 STRAT_K_LONG=1 BEAR_K=2 SIDE_MODE=default SIZING_MODE=inv_sqrt_vol SIDE_BETA_NEUT=0
    BEAR_MODE=equal STOP_SKIP_REGIMES=bear LONG_MAX_RET3D=999 SHORT_MIN_RET3D=-0.20
    BEAR_DEPTH_RAMP=1 BEAR_DEPTH_D0=0.10 BEAR_DEPTH_D1=0.30 CONC_CAP=0.40 CONC_CAP_SINGLE_EXEMPT=1
    REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full
    BULL_MODE=sidealpha BULL_GROSS_MULT=1 BULL_LONG_MULT=0.25 BULL_LONG_INSTRUMENT=btc BTC_HEDGE_COST_BPS=2 BULL_K=2 STRAT_HOLD_BULL=1 BULL_SHORT_RANK=return_1d BULL_DEEP_THR=0.15)
run(){ local tag=$1 pb=$2 pl=$3; local sd=live/state/longtail/wq036_ab/$tag; rm -rf "$sd"; mkdir -p "$sd"
  env "${EXEC[@]}" "${V3[@]}" CONVEXITY_PREDS_PATH="$D/$pb/v0full_hl60.parquet" CONVEXITY_PREDS_LONG="$D/$pl/v0full_hl60.parquet" \
      CONVEXITY_STATE=$sd PYTHONPATH=. $PY -m live.convexity_paper_bot --replay-all > "$sd/run.log" 2>&1
  echo "$tag rc=$?"; }
run lean hl_wq036base_lean hl_wq036long_lean &
run bn   hl_wq036base_bn   hl_wq036long_bn &
run raw  hl_wq036base_raw  hl_wq036long_raw &
wait; echo ABALLDONE
