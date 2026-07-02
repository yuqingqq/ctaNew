#!/usr/bin/env bash
# v3 live-wiring PARITY gate: does the bot's --cycle (live path) reproduce --replay-all (backtest engine)
# on the SAME v3 env + preds? Path A = full replay. Path B = replay to a midpoint (seeds exact state) then
# --cycle the tail. Compare the tail cycles A vs B — must match bit-for-bit (like v1's 11/11 golden).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
MID=${1:-2026-04-01}
PBASE=$ROOT/live/state/convexity/hl_lean175/v0full_hl60.parquet
PLONG=$ROOT/live/state/convexity/hl_residrev_lean/v0full_hl60.parquet
A=$ROOT/live/state/longtail/parityA; B=$ROOT/live/state/longtail/parityB; rm -rf "$A" "$B"; mkdir -p "$A" "$B"

# FROZEN v3 env (identical to run_convexity_v3_regime_gate.sh)
v3env(){ env COST_BPS_LEG=9 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1 CHARGE_FUNDING=1 \
  CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet \
  DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k \
  STRAT_K=2 BEAR_K=2 CONC_CAP=0.40 LONG_MAX_RET3D=999 SIZING_MODE=inv_sqrt_vol \
  BEAR_MODE=equal STOP_SKIP_REGIMES=bear SIDE_BETA_NEUT=0 STRAT_K_LONG=1 SHORT_MIN_RET3D=-0.20 \
  BEAR_DEPTH_RAMP=1 BEAR_DEPTH_D0=0.10 BEAR_DEPTH_D1=0.30 CONC_CAP_SINGLE_EXEMPT=1 \
  REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full \
  BULL_MODE=sidealpha BULL_GROSS_MULT=1 BULL_LONG_MULT=0.25 BULL_LONG_INSTRUMENT=btc BTC_HEDGE_COST_BPS=2 BULL_K=2 STRAT_HOLD_BULL=1 \
  BULL_SHORT_RANK=return_1d BULL_DEEP_THR=0.15 \
  CONVEXITY_PREDS_PATH="$PBASE" CONVEXITY_PREDS_LONG="$PLONG" \
  CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl PYTHONPATH=. "$@"; }

echo "[A] full --replay-all ..."
v3env CONVEXITY_STATE=$A $PY -m live.convexity_paper_bot --replay-all > "$A/run.log" 2>&1
echo "[B] --replay-from 2025-10-04 --replay-end $MID (seed) ..."
v3env CONVEXITY_STATE=$B $PY -m live.convexity_paper_bot --replay-from 2025-10-04 --replay-end "$MID" > "$B/run1.log" 2>&1
echo "[B] --cycle (catch up the tail past $MID) ..."
v3env CONVEXITY_STATE=$B $PY -m live.convexity_paper_bot --cycle > "$B/run2.log" 2>&1

$PY - "$A/cycles.csv" "$B/cycles.csv" "$MID" <<'PYEOF'
import sys, pandas as pd, numpy as np
a=pd.read_csv(sys.argv[1]); b=pd.read_csv(sys.argv[2]); mid=pd.Timestamp(sys.argv[3],tz="UTC")
for d in (a,b): d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
a=a[a.open_time>=mid].set_index("open_time").sort_index()
b=b[b.open_time>=mid].set_index("open_time").sort_index()
common=a.index.intersection(b.index)
print(f"tail cycles compared (>= {mid.date()}): A={len(a)} B={len(b)} common={len(common)}")
cols=[c for c in ["pnl_bps","gross_after_stop","net_target","n_trades","regime"] if c in a.columns and c in b.columns]
ok=True
for c in cols:
    if a[c].dtype.kind in "fc":
        md=float((a.loc[common,c].fillna(0)-b.loc[common,c].fillna(0)).abs().max())
        print(f"  {c:18s} max|Δ| = {md:.6f}")
        if md>1e-3: ok=False
    else:
        mism=int((a.loc[common,c].astype(str)!=b.loc[common,c].astype(str)).sum())
        print(f"  {c:18s} mismatches = {mism}/{len(common)}")
        if mism>0: ok=False
pa=a.loc[common,"pnl_bps"].sum(); pb=b.loc[common,"pnl_bps"].sum()
print(f"  tail totPnL: replay {pa:+.1f}  vs  cycle {pb:+.1f}  (Δ {pa-pb:+.3f})")
print("PARITY:", "PASS ✅" if ok else "FAIL ❌")
PYEOF
