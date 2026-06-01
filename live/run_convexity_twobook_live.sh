#!/usr/bin/env bash
# Convexity TWO-BOOK champion LIVE forward test — 4h cadence (Phase-IX volatility-split).
#   BookA = flow model (V0+flow) on the FROZEN rvol top-80 (live/models/twobook_split.json, as-of retrain)
#   BookB = price model (V0) on the rest; combined 50/50. Each book trades $50k (→ $100k total).
# Data edge fed real-time by the WS collector (klines + aggTrades).
#
# ROBUSTNESS:
#  - FAIL-CLOSED on critical steps (xs_feats/panel/predict/split): on failure SKIP the cycle rather than
#    trade on stale data.
#  - NO-OP guard: only trade/combine/report when the panel actually ADVANCED a bar this run (a 4h bar only
#    settles ~30min after its boundary). Avoids re-combining/re-notifying/re-logging the same stale cycle.
#  - ADAPTIVE cadence: retry every 10min while a bar is overdue; else sleep to next boundary+35min.
#  - maturity uses onboardDate (live/build_maturity_meta.py → maturity_meta.parquet via CONVEXITY_UNIV_META,
#    NOT the shared x132 preds).
#
# Launch:  tmux new -d -s cvx2 'bash /home/yuqing/ctaNew/live/run_convexity_twobook_live.sh'
# Watch:   tail -f /home/yuqing/ctaNew/live/state/convexity_twobook/run.log
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd $ROOT
PY=$ROOT/.venv/bin/python
STA=$ROOT/live/state/convexity_bookA; STB=$ROOT/live/state/convexity_bookB
PRA=$ROOT/live/state/convexity/split2/bookA_flow.parquet
PRB=$ROOT/live/state/convexity/split2/bookB_price.parquet
OUT=$ROOT/live/state/convexity_twobook; mkdir -p "$OUT"
LOG=$OUT/run.log
export STRAT_K=3 SIDE_MODE=default CONVEXITY_EQUITY=50000
export CONVEXITY_UNIV_META=$ROOT/live/state/convexity/maturity_meta.parquet   # dedicated maturity meta
export CONVEXITY_PIT_DVOL=1                                                    # honest PIT liquidity gate
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
panel_edge(){ $PY -c "import pandas as pd; print(pd.to_datetime(pd.read_parquet('outputs/vBTC_features/panel_expanded_v0.parquet',columns=['open_time'])['open_time'],utc=True).max())" 2>/dev/null; }
nap(){  # adaptive: retry-10m if a bar is overdue, else sleep to next boundary+35min
  S=$($PY -c "
import datetime as dt, pandas as pd
now=dt.datetime.now(dt.timezone.utc)
def fl(t): return t.replace(hour=(t.hour//4)*4, minute=0, second=0, microsecond=0)
edge=pd.to_datetime(pd.read_parquet('outputs/vBTC_features/panel_expanded_v0.parquet',columns=['open_time'])['open_time'],utc=True).max().to_pydatetime()
expected=fl(now - dt.timedelta(hours=4, minutes=35))
print(600 if edge < expected else max(120, int((edge + dt.timedelta(hours=8, minutes=35) - now).total_seconds())))" 2>/dev/null)
  [ -z "$S" ] && S=600
  log "-- sleep ${S}s (adaptive) --"; sleep "$S"; }

log "== convexity TWO-BOOK LIVE supervisor (K=3, FROZEN split, PIT-dvol, fail-closed) =="
while true; do
  log "-- cycle start --"
  EDGE0=$(panel_edge)
  $PY live/ingest_flow_daily.py --workers 4 >> "$LOG" 2>&1 && log " 1 flow-ingest OK" || log " 1 flow-ingest WARN (non-critical)"
  $PY live/reconcile_aggtrades.py          >> "$LOG" 2>&1 && log " 1b reconcile OK"  || log " 1b reconcile WARN"
  if ! $PY live/incremental_xs_feats.py --workers 6 >> "$LOG" 2>&1; then log " 2 xs_feats FAIL — skip cycle (fail-closed)"; nap; continue; fi
  log " 2 xs_feats OK"
  if ! $PY live/incremental_panel.py --workers 6 >> "$LOG" 2>&1; then log " 3 panel FAIL — skip cycle (fail-closed)"; nap; continue; fi
  log " 3 panel OK"
  EDGE1=$(panel_edge)
  if [ "$EDGE1" = "$EDGE0" ]; then log " -- no new bar settled (edge $EDGE1) — skip trade/report --"; nap; continue; fi
  log " -- new bar: $EDGE0 -> $EDGE1 --"
  $PY live/build_maturity_meta.py          >> "$LOG" 2>&1 && log " 4 maturity OK"    || log " 4 maturity WARN (uses last meta)"
  if ! $PY live/predict_twobook_incremental.py >> "$LOG" 2>&1; then log " 5 predict FAIL — skip cycle"; nap; continue; fi
  log " 5 predict OK"
  if ! $PY live/rvol_split.py >> "$LOG" 2>&1; then log " 6 split FAIL — skip cycle"; nap; continue; fi
  log " 6 split OK"
  if CONVEXITY_STATE=$STA CONVEXITY_PREDS_PATH=$PRA $PY -m live.convexity_paper_bot --cycle >> "$LOG" 2>&1; then A_OK=1; log " 7 BookA OK"; else A_OK=0; log " 7 BookA FAIL"; fi
  if CONVEXITY_STATE=$STB CONVEXITY_PREDS_PATH=$PRB $PY -m live.convexity_paper_bot --cycle >> "$LOG" 2>&1; then B_OK=1; log " 8 BookB OK"; else B_OK=0; log " 8 BookB FAIL"; fi
  if [ "$A_OK" = 1 ] && [ "$B_OK" = 1 ]; then
    $PY live/convexity_twobook_combine.py --book-a $STA/cycles.csv --book-b $STB/cycles.csv --out "$OUT" >> "$LOG" 2>&1 && log " 9 combine OK" || log " 9 combine FAIL"
    $PY live/convexity_slippage.py --state $STA --book A --out "$OUT/slippage.csv" >> "$LOG" 2>&1 || true
    $PY live/convexity_slippage.py --state $STB --book B --out "$OUT/slippage.csv" >> "$LOG" 2>&1 || true
    $PY live/convexity_notify_twobook.py >> "$LOG" 2>&1 && log " 10-11 slippage+telegram OK" || log " 10-11 SKIP"
  else
    log " 9-11 SKIPPED (a book failed — no partial combine/report)"
  fi
  nap
done
