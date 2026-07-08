#!/usr/bin/env bash
# Controlled v1-vs-v3: MATCHED execution (cost 9 + funding + depth-cost + same model preds + same universe meta),
# vary only STRATEGY. Cells: v1_native (K3,equal,beta-neut,BEAR=flat,BULL=mom,truncation,no gates),
# v1_notrunc (isolate truncation on v1's config), v3_native (K1/2,inv_sqrt_vol,gates,no truncation).
# Usage: bash live/controlled_v1v3.sh <is|oos>
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
PER=${1:-is}
if [ "$PER" = "oos" ]; then
  PB=live/state/convexity/hl_lean175_oos/v0full_hl60.parquet; PL=live/state/convexity/hl_residrev_oos/v0full_hl60.parquet
  DA=live/state/longtail/va_oos.parquet
else
  PB=live/state/convexity/hl_lean175/v0full_hl60.parquet; PL=live/state/convexity/hl_residrev_lean/v0full_hl60.parquet
  DA=live/state/longtail/va_is.parquet
fi
# MATCHED execution (identical on all cells)
EXEC=(COST_BPS_LEG=9 CHARGE_FUNDING=1 CONVEXITY_PIT_DVOL=1 XS_LEAN=1
      CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet
      DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
      CONVEXITY_PREDS_PATH="$PB" CONVEXITY_PREDS_LONG="$PL"
      CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl)
run(){ local tag=$1; shift; local sd=live/state/longtail/ctl_${PER}_${tag}; rm -rf "$sd"; mkdir -p "$sd"
  env "${EXEC[@]}" "$@" CONVEXITY_STATE=$sd PYTHONPATH=. $PY -m live.convexity_paper_bot --replay-all > "$sd/run.log" 2>&1
  echo "[$PER] $tag done"; }

# v1_native: v1 strategy (its real defaults) + truncation
run v1_native   STRAT_K=3 SIDE_MODE=default SIZING_MODE=equal SIDE_BETA_NEUT=1 BEAR_MODE=flat BULL_MODE=mom \
                REGIME_GATE=0 BULL_DEEP_THR=99 BEAR_DEPTH_RAMP=0 SHORT_MIN_RET3D=-999 CONC_CAP=0 CONC_CAP_SINGLE_EXEMPT=0 \
                CONVEXITY_DYNAMIC_ALLOWLIST_PATH="$DA" &
# v1_notrunc: identical to v1_native but NO truncation (isolate truncation on v1's config)
run v1_notrunc  STRAT_K=3 SIDE_MODE=default SIZING_MODE=equal SIDE_BETA_NEUT=1 BEAR_MODE=flat BULL_MODE=mom \
                REGIME_GATE=0 BULL_DEEP_THR=99 BEAR_DEPTH_RAMP=0 SHORT_MIN_RET3D=-999 CONC_CAP=0 CONC_CAP_SINGLE_EXEMPT=0 &
# v3_native: full v3 stack (matched execution cost 9 = its native)
run v3_native   STRAT_K=2 STRAT_K_LONG=1 BEAR_K=2 SIDE_MODE=default SIZING_MODE=inv_sqrt_vol SIDE_BETA_NEUT=0 \
                BEAR_MODE=equal STOP_SKIP_REGIMES=bear LONG_MAX_RET3D=999 SHORT_MIN_RET3D=-0.20 \
                BEAR_DEPTH_RAMP=1 BEAR_DEPTH_D0=0.10 BEAR_DEPTH_D1=0.30 CONC_CAP=0.40 CONC_CAP_SINGLE_EXEMPT=1 \
                REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full \
                BULL_MODE=sidealpha BULL_GROSS_MULT=1 BULL_LONG_MULT=0.25 BULL_LONG_INSTRUMENT=btc BTC_HEDGE_COST_BPS=2 BULL_K=2 STRAT_HOLD_BULL=1 BULL_SHORT_RANK=return_1d BULL_DEEP_THR=0.15 &
wait; echo "[$PER] all cells done"
