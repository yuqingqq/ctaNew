#!/usr/bin/env bash
# Convexity v4 (residual target + KEEPSET4 + deep-bull momentum) LIVE forward-test — PARALLEL to v3
# (separate state dir), every bar OOS. Mirrors run_convexity_v3_live.sh's loop (refresh → predict →
# bot --cycle) but with:
#   - v4 preds: residual-target artifacts (train_v4_artifact.py) via predict_v4_incremental.py
#   - KEEPSET4 risk layer (V4_PERFORMANCE §1): BEAR_MODE=equal (plain) + REGIME_GATE(180/K2/binary/full)
#     + DD-stop (2σ, skip bear) + BULL_GROSS_MULT=0. Four levers, no fitted knobs.
#   - DEEP-BULL MOMENTUM overlay ENABLED (V4_PERFORMANCE §6.1, forward-test candidate 2026-07-07):
#     BULL_DEEP_MODE=mom1d_long — in deep bull (btc30>=0.15) long top-2 by return_1d at half gross,
#   - GLOBAL_GROSS_MULT=0.5 live gross cap on both books (2022-holdout FAIL consequence,
#     V4_PERFORMANCE §7 BLOCKING; lifted only on forward bear-farm confirmation).
#     pred-independent book, shared DD-stop/regime-gate. Only candidate positive BOTH windows
#     (recent Δ+0.04 Sh bull +379; OOS Δ+0.09 Sh bull +1,531); CIs cross 0 → FORWARD TEST, not adopted.
#
# Model: convexity_v4_{base,residrev}_model.pkl (train_v4_artifact.py, fit_cut = latest panel - 1d).
# Setup (done 2026-07-07; repeat only on reset — ORDER MATTERS):
#   1. Seed the pred books from the validated backtest (PIT-clean; warm regime gate). Since the
#      2026-07-08 audit fixes, the predictor refuses to re-predict bars at-or-before the artifact's
#      fit_cut, AND fails closed if the seed is missing (unseeded start requires
#      PREDICT_ALLOW_UNSEEDED=1 and is fit_cut-floored). Seed:
#        cp live/state/convexity/hl_tgt_res_base_clean/v0full_hl60.parquet live/state/convexity/v4_live/base.parquet
#        cp live/state/convexity/hl_tgt_res_long_clean/v0full_hl60.parquet live/state/convexity/v4_live/long.parquet
#   2. Bootstrap state with a replay over the seed — MUST use the PANEL as universe meta (the live
#      maturity_meta grid is floored at now-400d, so a historical replay against it sees an empty
#      universe): run this file's env with CONVEXITY_UNIVERSE_META=$PANEL, GLOBAL_GROSS_MULT=1.0
#      (the 0.5x live cap postdates the validated cell — override it for parity replays), and
#      --replay-from 2025-10-04. Verified 2026-07-07 (pre-cap): reproduces the validated
#      k4_deepmom cell exactly (Sh +2.26 / +19,628).
# Launch: tmux new -d -s cvx4 'bash /home/yuqing/ctaNew/live/run_convexity_v4_live.sh'
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
OUT=$ROOT/live/state/convexity/v4_live; mkdir -p "$OUT/state"; LOG=$OUT/run.log
PANEL=$ROOT/outputs/vBTC_features/panel_expanded_v0.parquet
export V4_PREDS_DIR=$OUT
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

# ---- cost/exec (matches the validated KEEPSET4 harness cells) ----
export COST_BPS_LEG=9 FEE_BPS_FILL=4.5 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1 CHARGE_FUNDING=1
export DEPTH_COST_CSV=$ROOT/live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
# ---- base strategy (locked harness VANILLA base: K=1L/2S, inv_sqrt_vol, plain bear-K, no fitted filters) ----
# NB: SIDE_BETA_NEUT deliberately NOT set (bot default 1 = beta-neut side, the canonical KEEPSET4 harness
# behavior). The fitted v3 live script's SIDE_BETA_NEUT=0 is a fitted-stack knob — untested atop KEEPSET4.
export STRAT_K=2 STRAT_K_LONG=1 BEAR_K=0 SIZING_MODE=inv_sqrt_vol
export BEAR_DEPTH_RAMP=0 CONC_CAP=0.99 CONC_CAP_SINGLE_EXEMPT=1 SHORT_MIN_RET3D=-999 LONG_MAX_RET3D=999
# ---- KEEPSET4 risk layer (tune/_cfg/v2_KEEPSET4_m.json) ----
export BEAR_MODE=equal
export REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2 REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full
export STOP_SKIP_REGIMES=bear STOP_K_SIGMA=2.0
export BULL_MODE=default BULL_GROSS_MULT=0
# ---- deep-bull momentum overlay (ENABLED; forward-test candidate) ----
export BULL_DEEP_THR=0.15 BULL_DEEP_MODE=mom1d_long
# ---- 2022-holdout FAIL consequence (pre-registered, BINDING — RESEARCH_LOOP_20260707 Iter 4 F10) ----
# Live gross capped 0.5x until the forward ledger confirms the bear farm: BEAR_MODE=equal bear-regime
# NET PnL (bot's own labels) over >=2 calendar months of forward data containing >=1 bear episode,
# day-block 95% CI excluding 0. Months without bear cycles do not advance the clock.
export GLOBAL_GROSS_MULT=0.5
# ---- live universe + state ----
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

log "== convexity v4 LIVE (KEEPSET4 + deep-bull mom1d_long | parallel forward-test) =="
[ -f "$OUT/base.parquet" ] || log " NOTE: seed $OUT/{base,long}.parquet from hl_tgt_res_{base,long}_clean first (see header)"
while true; do
  $PY live/ingest_funding_fapi.py            >> "$LOG" 2>&1 && log " funding OK" || log " funding WARN"
  if ! $PY live/incremental_xs_feats.py --workers 6 >> "$LOG" 2>&1; then log " xs_feats FAIL — skip"; nap; continue; fi
  if ! $PY live/incremental_panel.py    --workers 6 --rebuild-days 10 >> "$LOG" 2>&1; then log " panel FAIL — skip"; nap; continue; fi  # rebuild>=RECOMPUTE_DAYS: backfills late klines + propagates xs-feats tail repairs (audit 2026-07-08)
  if ! $PY live/predict_v4_incremental.py            >> "$LOG" 2>&1; then log " predict FAIL — skip"; nap; continue; fi
  if $PY -m live.convexity_paper_bot --cycle          >> "$LOG" 2>&1; then log " cycle OK"; else log " cycle FAIL"; fi
  nap
done
