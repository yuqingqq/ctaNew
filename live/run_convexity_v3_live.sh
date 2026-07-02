#!/usr/bin/env bash
# Convexity v3 (regime-gate stack) LIVE forward-test — PARALLEL to v1 (separate state dir), every bar OOS.
# Mirrors run_convexity_v1_live.sh's loop (refresh → predict → bot --cycle) but with the FROZEN v3 env
# (regime gate + validated core + BULL_DEEP_THR=0.15) and the v3 deployable artifact via predict_v3_incremental.
#
# Model: convexity_v3_{base,residrev}_model.pkl (train_v3_artifact.py, fit_cut = latest panel - 1d).
# Deploy: on the training box, retrain monthly (train_v3_artifact) + push; live box git pulls + runs this.
# Launch: tmux new -d -s cvx3 'bash /home/yuqing/ctaNew/live/run_convexity_v3_live.sh'
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
OUT=$ROOT/live/state/convexity/v3_live; mkdir -p "$OUT/state"; LOG=$OUT/run.log
PANEL=$ROOT/outputs/vBTC_features/panel_expanded_v0.parquet
export V3_PREDS_DIR=$OUT
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

# ---- FROZEN v3 env (must match run_convexity_v3_regime_gate.sh, the validated backtest driver) ----
export COST_BPS_LEG=9 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1 CHARGE_FUNDING=1
export DEPTH_COST_CSV=$ROOT/live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
export STRAT_K=2 BEAR_K=2 CONC_CAP=0.40 LONG_MAX_RET3D=999 SIZING_MODE=inv_sqrt_vol
export BEAR_MODE=equal STOP_SKIP_REGIMES=bear SIDE_BETA_NEUT=0
export STRAT_K_LONG=1 SHORT_MIN_RET3D=-0.20 BEAR_DEPTH_RAMP=1 BEAR_DEPTH_D0=0.10 BEAR_DEPTH_D1=0.30
export CONC_CAP_SINGLE_EXEMPT=1
export REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full
export BULL_MODE=sidealpha BULL_GROSS_MULT=1 BULL_LONG_MULT=0.25 BULL_LONG_INSTRUMENT=btc BTC_HEDGE_COST_BPS=2 BULL_K=2 STRAT_HOLD_BULL=1
export BULL_SHORT_RANK=return_1d
export BULL_DEEP_THR=0.15                                     # this-session adopted, OOS-validated hot-bull cut
# LIVE universe meta: onboardDate maturity (as v1 live) — the panel-coverage default collapses to ~24 large-caps
export CONVEXITY_UNIVERSE_META=$ROOT/live/state/convexity/maturity_meta.parquet
export CONVEXITY_STATE=$OUT/state
export CONVEXITY_PREDS_PATH=$OUT/base.parquet CONVEXITY_PREDS_LONG=$OUT/long.parquet
export CONVEXITY_DVOL_CACHE_PKL=$ROOT/live/state/v3loop/ddloop/_dvol_cache.pkl

nap(){ S=$($PY -c "
import datetime as dt, pandas as pd
now=dt.datetime.now(dt.timezone.utc)
def fl(t): return t.replace(hour=(t.hour//4)*4, minute=0, second=0, microsecond=0)
try: edge=pd.to_datetime(pd.read_parquet('$PANEL',columns=['open_time'])['open_time'],utc=True).max().to_pydatetime()
except Exception: edge=now-dt.timedelta(hours=8)
expected=fl(now-dt.timedelta(hours=4,minutes=35))
if edge<expected: print(600)                                  # bar overdue → retry in 10m
else:
    nxt=fl(now)+dt.timedelta(hours=4,minutes=35); print(int(max(60,(nxt-now).total_seconds())))
" 2>/dev/null); sleep "${S:-600}"; }

log "== convexity v3 LIVE (regime-gate stack | parallel forward-test | BULL_DEEP_THR=0.15) =="
[ -f "$OUT/base.parquet" ] || log " NOTE: seed $OUT/{base,long}.parquet from the validated backtest preds first (see deploy notes)"
while true; do
  $PY live/ingest_funding_fapi.py            >> "$LOG" 2>&1 && log " funding OK" || log " funding WARN"
  if ! $PY live/incremental_xs_feats.py --workers 6 >> "$LOG" 2>&1; then log " xs_feats FAIL — skip"; nap; continue; fi
  if ! $PY live/incremental_panel.py    --workers 6 >> "$LOG" 2>&1; then log " panel FAIL — skip"; nap; continue; fi
  if ! $PY live/predict_v3_incremental.py            >> "$LOG" 2>&1; then log " predict FAIL — skip"; nap; continue; fi
  if $PY -m live.convexity_paper_bot --cycle          >> "$LOG" 2>&1; then log " cycle OK"; else log " cycle FAIL"; fi
  nap
done
