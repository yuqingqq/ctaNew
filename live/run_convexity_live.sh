#!/usr/bin/env bash
# Convexity PRICE-book LIVE forward test — 4h cadence (the strategy decides every 4h).
# Data edge is fed in real time by the WS collector (live/run_ws_collector.sh): closed 5m klines
# stream into data/ml/test/parquet/klines/, so this supervisor only runs the incremental compute
# + the cached frozen-model prediction + the cycle. ~1 min of work per 4h window.
#
# Each cycle (just after every 00/04/08/12/16/20 UTC boundary):
#   1. incremental_xs_feats   recompute trailing-window xs_feats, append new bars
#   2. incremental_panel      append the newly-settled 4h bar(s) to the panel
#   3. build_maturity_meta    refresh per-sym earliest dates (maturity≥180d eligibility)
#   4. predict_twobook_incremental  cached frozen model → FORWARD (live OOS) price preds
#   5. convexity_paper_bot --cycle  advance the book, mark PnL, persist positions.json
#
# Launch:  tmux new -d -s cvxlive 'bash /home/yuqing/ctaNew/live/run_convexity_live.sh'
# Watch:   tail -f /home/yuqing/ctaNew/live/state/convexity_bookB/live.log
# Stop:    tmux kill-session -t cvxlive
# Inspect: CONVEXITY_STATE=.../convexity_bookB .venv/bin/python -m live.convexity_paper_bot --check-state
#
# PREREQS: ws collector running (tmux wscol); panel + v0full seed + maturity meta + xs_feats present;
# positions.json bootstrapped. Frozen model is monthly-retrained elsewhere (this box pulls it).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd $ROOT
PY=$ROOT/.venv/bin/python
ST=$ROOT/live/state/convexity_bookB
PR=$ROOT/live/state/convexity/hl/v0full_hl60.parquet
LOG=$ST/live.log
export STRAT_K=3 SIDE_MODE=default
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

log "== convexity PRICE-book LIVE supervisor started (4h, K=3) =="
while true; do
  log "-- cycle start --"
  $PY live/incremental_xs_feats.py --workers 6        >> "$LOG" 2>&1 && log "  1 xs_feats OK"  || log "  1 xs_feats FAIL"
  $PY live/incremental_panel.py --workers 6           >> "$LOG" 2>&1 && log "  2 panel OK"     || log "  2 panel FAIL"
  $PY live/build_maturity_meta.py                     >> "$LOG" 2>&1 && log "  3 maturity OK"  || log "  3 maturity FAIL"
  $PY live/predict_twobook_incremental.py             >> "$LOG" 2>&1 && log "  4 preds OK"     || log "  4 preds FAIL"
  CONVEXITY_STATE=$ST CONVEXITY_PREDS_PATH=$PR \
    $PY -m live.convexity_paper_bot --cycle           >> "$LOG" 2>&1 && log "  5 cycle OK"     || log "  5 cycle FAIL"
  # 6. realized execution cost off the LIVE Hyperliquid L2 orderbook (fee + book-walk + fill-completeness)
  $PY live/convexity_slippage.py --state "$ST" --book B --out "$ST/slippage.csv" >> "$LOG" 2>&1 \
    && log "  6 slippage OK" || log "  6 slippage SKIP"
  # 7. 4h snapshot to Telegram (equity, regime, positions, realized slip, kill-switch)
  $PY live/convexity_notify_price.py                  >> "$LOG" 2>&1 && log "  7 telegram OK"  || log "  7 telegram SKIP"

  NEXT_4H=$($PY -c "
import datetime as dt
now=dt.datetime.now(dt.timezone.utc); hr=(now.hour//4 + 1)*4
nxt=now.replace(hour=hr%24, minute=0, second=0, microsecond=0)
if hr>=24: nxt+=dt.timedelta(days=1)
print(int(nxt.timestamp()))")
  SLEEP=$(( NEXT_4H - $(date -u +%s) + 90 )); [ $SLEEP -lt 90 ] && SLEEP=90
  log "-- cycle done; sleep ${SLEEP}s to next 4h boundary --"
  sleep $SLEEP
done
