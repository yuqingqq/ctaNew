#!/usr/bin/env bash
# convexity TWO-BOOK champion forward-test supervisor (Phase-VII champion, +3.71 / DD -1417).
#
# Each 4h boundary:
#   1. refresh panel (klines + X132)
#   2. BookA: flow model on flow-syms   -> CONVEXITY_STATE=convexity_bookA, K=3
#   3. BookB: price model on non-flow   -> CONVEXITY_STATE=convexity_bookB, K=3
#   4. combine 50/50 -> live/state/convexity_twobook/{twobook_equity.csv,twobook_summary.json}
#
# Launch:  tmux new -d -s cvx2 'bash /home/yuqing/ctaNew/live/run_convexity_twobook.sh'
# Watch:   tail -f /home/yuqing/ctaNew/live/state/convexity_twobook/run.log
#
# PREREQUISITES (see CONVEXITY_BOT_LAUNCH.md "Two-book forward test"):
#   - BOOKA_PREDS / BOOKB_PREDS env must point at preds files kept current through panel-end.
#     For a true live run these must be REGENERATED each cycle from fresh features (flow model
#     for A, price model for B) — wire that into the refresh step. Until then this replays a
#     fixed preds file forward as the panel extends (paper-test of a frozen model).
#   - Daily flow ingestion (live/ingest_flow_daily.sh via cron) keeps BookA's flow features fresh.
set -uo pipefail
ROOT=/home/yuqing/ctaNew
export PYTHONPATH=$ROOT
cd $ROOT
OUT=$ROOT/live/state/convexity_twobook; mkdir -p "$OUT"
LOG=$OUT/run.log
PY=python3
: "${STRAT_K:=3}"; export STRAT_K
: "${SIDE_MODE:=default}"; export SIDE_MODE
# Volatility-split champion (Phase-IX): BookA = flow model on high-rvol top-80, BookB = price model on rest.
: "${BOOKA_PREDS:=$ROOT/live/state/convexity/split2/bookA_hv80.parquet}"
: "${BOOKB_PREDS:=$ROOT/live/state/convexity/split2/bookB_hv80.parquet}"

log(){ echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "== convexity TWO-BOOK supervisor started (K=$STRAT_K, SIDE_MODE=$SIDE_MODE) =="
while true; do
  log "-- cycle start --"
  $PY -m live.refresh_convexity_panel --days-back 14 >> "$LOG" 2>&1 \
    && log "refresh OK" || log "refresh FAILED (continuing on stale panel)"

  log "BookA (flow) --cycle"
  CONVEXITY_STATE=$ROOT/live/state/convexity_bookA CONVEXITY_PREDS_PATH="$BOOKA_PREDS" \
    $PY -m live.convexity_paper_bot --cycle >> "$LOG" 2>&1 && log "BookA OK" || log "BookA FAILED"

  log "BookB (price) --cycle"
  CONVEXITY_STATE=$ROOT/live/state/convexity_bookB CONVEXITY_PREDS_PATH="$BOOKB_PREDS" \
    $PY -m live.convexity_paper_bot --cycle >> "$LOG" 2>&1 && log "BookB OK" || log "BookB FAILED"

  log "combine 50/50"
  $PY live/convexity_twobook_combine.py \
      --book-a $ROOT/live/state/convexity_bookA/cycles.csv \
      --book-b $ROOT/live/state/convexity_bookB/cycles.csv \
      --out "$OUT" >> "$LOG" 2>&1 && log "combine OK" || log "combine FAILED"

  log "measure realized slippage (HL L2)"
  $PY live/convexity_slippage.py --state $ROOT/live/state/convexity_bookA --book A --out "$OUT/slippage.csv" >> "$LOG" 2>&1 || log "slippage A skip"
  $PY live/convexity_slippage.py --state $ROOT/live/state/convexity_bookB --book B --out "$OUT/slippage.csv" >> "$LOG" 2>&1 || log "slippage B skip"

  NOW=$(date -u '+%s')
  NEXT_4H=$(python3 -c "
import datetime as dt
now=dt.datetime.now(dt.timezone.utc); hr=(now.hour//4 + 1)*4
nxt=now.replace(hour=hr%24, minute=0, second=0, microsecond=0)
if hr>=24: nxt+=dt.timedelta(days=1)
print(int(nxt.timestamp()))")
  SLEEP=$(( NEXT_4H - NOW + 60 )); [ $SLEEP -lt 60 ] && SLEEP=60
  log "sleeping ${SLEEP}s until next 4h boundary"
  sleep $SLEEP
done
