#!/usr/bin/env bash
# #17 end-to-end NET (fee 4.5 + funding + depth) — full v3 production stack + v4 OOS preds (honest 2023-25),
# baseline K_short=2 vs the picked K_short=3. K_long=1, bear keeps production ramp (BEAR_K=2). Isolates the
# side-regime K_short bump inside the real cost engine.
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
D=live/state/convexity
EXEC=(COST_BPS_LEG=9 FEE_BPS_FILL=4.5 CHARGE_FUNDING=1 CONVEXITY_PIT_DVOL=1 XS_LEAN=1
      CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet
      DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
      CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl)
# full v3_native stack; only STRAT_K (=K_short) varies between the two runs
V3=(STRAT_K_LONG=1 BEAR_K=2 SIDE_MODE=default SIZING_MODE=inv_sqrt_vol SIDE_BETA_NEUT=0
    BEAR_MODE=equal STOP_SKIP_REGIMES=bear LONG_MAX_RET3D=999 SHORT_MIN_RET3D=-0.20
    BEAR_DEPTH_RAMP=1 BEAR_DEPTH_D0=0.10 BEAR_DEPTH_D1=0.30 CONC_CAP=0.40 CONC_CAP_SINGLE_EXEMPT=1
    REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full
    BULL_MODE=sidealpha BULL_GROSS_MULT=1 BULL_LONG_MULT=0.25 BULL_LONG_INSTRUMENT=btc BTC_HEDGE_COST_BPS=2 BULL_K=2 STRAT_HOLD_BULL=1 BULL_SHORT_RANK=return_1d BULL_DEEP_THR=0.15)
run(){ local tag=$1 ks=$2; local sd=live/state/longtail/v4_net_k/$tag; rm -rf "$sd"; mkdir -p "$sd"
  env "${EXEC[@]}" "${V3[@]}" STRAT_K=$ks \
      CONVEXITY_PREDS_PATH="$D/hl_v4base_oos/v0full_hl60.parquet" CONVEXITY_PREDS_LONG="$D/hl_v4long_oos/v0full_hl60.parquet" \
      CONVEXITY_STATE=$sd PYTHONPATH=. $PY -m live.convexity_paper_bot --replay-all > "$sd/run.log" 2>&1
  echo "$tag(K_short=$ks) rc=$?"; }
run kshort2 2 &
run kshort3 3 &
wait; echo V4NETKDONE
