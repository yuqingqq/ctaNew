#!/usr/bin/env bash
# FROZEN driver for the convexity v3 regime-gated stack (validated 2026-06-30).
# This is the exact, reproducible env recipe behind the reported numbers — do NOT rely on bare REGIME_GATE=1
# (code defaults differ from the validated config in subtle ways; everything operative is pinned here).
#
# WHAT IT IS: convexity short-alt mean-reversion + a market-wide PERFORMANCE regime gate that de-grosses the whole
# book when the strategy's own trailing-30d realized cross-sectional L/S edge has gone negative (momentum regime).
# The gate — NOT the static btc_ret_30d bull/side/bear label — is the operative classifier.
#
# Usage: bash live/run_convexity_v3_regime_gate.sh <STATE_DIR> [extra ENV=val ...]
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd "$ROOT"
OUT=${1:?usage: run_convexity_v3_regime_gate.sh <STATE_DIR>}; shift || true
mkdir -p "$OUT"

env \
  COST_BPS_LEG=9 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1 CHARGE_FUNDING=1 \
  CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet \
  DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k \
  STRAT_K=2 BEAR_K=2 CONC_CAP=0.40 LONG_MAX_RET3D=999 SIZING_MODE=inv_sqrt_vol \
  BEAR_MODE=equal STOP_SKIP_REGIMES=bear SIDE_BETA_NEUT=0 \
  `# --- VALIDATED CORE (2026-07-01): +20% return at equal DD, broad (6/6 blocks, ex-Nov +3440), universe-robust 6/6 ---` \
  `# K_LONG=1: side long alpha lives ENTIRELY in the top-conviction pick; the 2nd long is neg-EV (-7.7bp). side-only. ` \
  `# SHORT_MIN=-0.20: veto shorting recent crashers (ret_3d<-20%); they squeeze/bounce (-57bp cohort). side+bear. ` \
  `# BEAR_DEPTH_RAMP: bear gross scales continuously w/ drawdown depth (0 at -10%, full at -30%) — short works only ` \
  `# in deep capitulation (t+3.2), is anti-alpha in the shallow grind (no reversion). Smooth, cliff-free risk control.` \
  STRAT_K_LONG=1 SHORT_MIN_RET3D=-0.20 \
  BEAR_DEPTH_RAMP=1 BEAR_DEPTH_D0=0.10 BEAR_DEPTH_D1=0.30 \
  `# --- REGIME GATE (performance-based, binary, full-universe thermometer) ---` \
  REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 \
  REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full \
  `# --- BULL: short-only(ish) + 1-sleeve(4h) hold (front-loaded edge); long=0.25 caps net-short risk ---` \
  BULL_MODE=sidealpha BULL_GROSS_MULT=1 BULL_LONG_MULT=0.25 BULL_LONG_INSTRUMENT=btc BTC_HEDGE_COST_BPS=2 BULL_K=2 STRAT_HOLD_BULL=1 \
  `# bull long ballast = 25% BTC. funding now CHARGED from data/ml/cache/funding_BTCUSDT.parquet (tiny: 0.21bps/8h, ` \
  `# +7bps total vs alt). BTC cost=2bps (not in HL capacity file; conservative for BTC liquidity). vs alt: tied/slightly ` \
  `# better Sharpe (+2.234), ~4% shallower maxDD, lower cost, less net-short (conc_cap exempts the single-name hedge).` \
  `# --- regime-robust bull short ranker (modest, env-gated; comment out to revert to pred) ---` \
  BULL_SHORT_RANK=return_1d \
  CONVEXITY_STATE="$OUT/state" \
  CONVEXITY_PREDS_PATH=live/state/convexity/hl_lean175/v0full_hl60.parquet \
  CONVEXITY_PREDS_LONG=live/state/convexity/hl_residrev_lean/v0full_hl60.parquet \
  CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl \
  PYTHONPATH=. "$@" python3 -m live.convexity_paper_bot --replay-all > "$OUT/run.log" 2>&1
echo "rc=$? -> $OUT/run.log"
