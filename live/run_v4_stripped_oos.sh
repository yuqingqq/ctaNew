#!/usr/bin/env bash
# Structural-only (gates STRIPPED) v4 through the SAME cost engine on OOS — isolate fitted-gate overfit vs -1.68 full-stack.
# Keep: K_long=1/K_short=2, inv_vol sizing. OFF: REGIME_GATE, bear ramp, bull hedge, DD-stop, conc_cap.
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
D=live/state/convexity
EXEC=(COST_BPS_LEG=9 FEE_BPS_FILL=4.5 CHARGE_FUNDING=1 CONVEXITY_PIT_DVOL=1 XS_LEAN=1
      CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet
      DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
      CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl)
STRIP=(STRAT_K=2 STRAT_K_LONG=1 BEAR_K=0 SIDE_MODE=default SIZING_MODE=inv_sqrt_vol
       REGIME_GATE=0 BEAR_DEPTH_RAMP=0 BULL_MODE=default CONC_CAP=0.99 CONC_CAP_SINGLE_EXEMPT=1
       STOP_SKIP_REGIMES=side,bear,bull SHORT_MIN_RET3D=-999 LONG_MAX_RET3D=999)
sd=live/state/longtail/v4_stripped_oos; rm -rf "$sd"; mkdir -p "$sd"
env "${EXEC[@]}" "${STRIP[@]}" CONVEXITY_PREDS_PATH="$D/hl_v4base_oos/v0full_hl60.parquet" CONVEXITY_PREDS_LONG="$D/hl_v4long_oos/v0full_hl60.parquet" \
    CONVEXITY_STATE=$sd PYTHONPATH=. $PY -m live.convexity_paper_bot --replay-all > "$sd/run.log" 2>&1
echo "stripped_oos rc=$?"; echo V4STRIPDONE
