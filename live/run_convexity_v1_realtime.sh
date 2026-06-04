#!/usr/bin/env bash
# Convexity v1 REAL-TIME forward test — one boundary loop that fires as soon as each 4h close bar flushes
# (event-driven via wait_bar_ready, not a fixed grace):
#   1. refresh realtime data (collector klines → xs_feats → panel; funding)
#   2. SETTLE the just-labeled bar = the MODELED reference track (bot --cycle, close-to-close − flat cost)
#   3. DECIDE the current bar at the boundary (decide_v1 → bot --decide → decision.json: legs + turnover)
#   4. probe the live Hyperliquid L2 book NOW for the turnover legs (real execution price at the boundary)
#   5. book the REAL-FILL round-trip PnL (convexity_realfill: FIFO lots, HL entry→exit, mark at HL mid)
#   6. snapshot to Telegram (real-fill PnL + exec-cost decomposition, with the modeled track as reference)
#
# Replaces the settle-only run_convexity_v1_live.sh (cvx1). Real-fill track is forward-only (no historical
# HL book): first fills appear next side regime, first completed 24h round-trip ~24h after a side entry.
# Launch:  tmux new -d -s cvx1rt 'bash /home/yuqing/ctaNew/live/run_convexity_v1_realtime.sh'
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=$ROOT/.venv/bin/python
OUT=$ROOT/live/state/convexity_v1; mkdir -p "$OUT/realfill"; LOG=$OUT/realtime.log
export COST_BPS_LEG=4.5 STRAT_K=3 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1
export CONVEXITY_UNIVERSE_META=$ROOT/live/state/convexity/maturity_meta.parquet
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
bot_edge(){ $PY -c "import json;print(json.load(open('$OUT/state/positions.json')).get('last_open_time') or '')" 2>/dev/null; }

apply_universe(){ $PY - << 'PY'
import json, pandas as pd
excl=set(json.load(open("live/models/convexity_v1_universe.json"))["exclude_high_vol"])
for src,dst in [("hl/v0full_hl60.parquet","base.parquet"),("hl_residrev/v0full_hl60.parquet","long.parquet")]:
    d=pd.read_parquet("live/state/convexity/"+src); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d[~d["symbol"].isin(excl)].to_parquet("live/state/convexity_v1/"+dst)
PY
}

# EVENT-DRIVEN: block until the next boundary's CLOSING 5m kline has actually flushed to the collector files,
# then return immediately (≈ the few-second flush latency) — not a guessed fixed grace.
wait_ready(){ log "-- waiting for next boundary's close bar --"; \
  $PY live/wait_bar_ready.py >> "$LOG" 2>&1 || { log " wait_bar_ready err — fallback sleep 900"; sleep 900; }; }

log "== convexity v1 REAL-TIME (settle ref + decide@boundary → HL execute → real-fill round-trip PnL) =="
while true; do
  log "-- cycle start --"
  # 1) refresh realtime data
  $PY live/ingest_funding_fapi.py >> "$LOG" 2>&1 && log " funding OK" || log " funding WARN"
  if ! $PY live/incremental_xs_feats.py --workers 6 >> "$LOG" 2>&1; then log " xs_feats FAIL"; wait_ready; continue; fi
  if ! $PY live/incremental_panel.py    --workers 6 >> "$LOG" 2>&1; then log " panel FAIL"; wait_ready; continue; fi
  $PY live/build_maturity_meta.py >> "$LOG" 2>&1 || true

  # 2) SETTLE the modeled reference track (advance any newly-labeled bar)
  BOT0=$(bot_edge)
  if $PY live/predict_twobook_incremental.py >> "$LOG" 2>&1 && apply_universe >> "$LOG" 2>&1; then
    CONVEXITY_STATE=$OUT/state CONVEXITY_PREDS_PATH=$OUT/base.parquet CONVEXITY_PREDS_LONG=$OUT/long.parquet \
      $PY -m live.convexity_paper_bot --cycle >> "$LOG" 2>&1 || log " settle WARN"
    BOT1=$(bot_edge); [ "$BOT1" != "$BOT0" ] && log " settled modeled $BOT0 -> $BOT1" || true
  else log " settle-predict WARN"; fi

  # 3) DECIDE the current bar at the boundary
  if ! $PY live/decide_v1.py >> "$LOG" 2>&1; then log " decide_v1 FAIL"; wait_ready; continue; fi
  if CONVEXITY_STATE=$OUT/state CONVEXITY_PREDS_PATH=$OUT/decide/base_decide.parquet CONVEXITY_PREDS_LONG=$OUT/decide/long_decide.parquet \
       $PY -m live.convexity_paper_bot --decide >> "$LOG" 2>&1; then
    log " decided: $($PY -c "import json;d=json.load(open('$OUT/state/decision.json'));print(d['open_time'],d['regime'],'-',len(d.get('turnover',{})),'legs to execute')" 2>/dev/null)"
    # 4) probe live HL L2 for the turnover legs (real execution price at the boundary)
    $PY live/convexity_slippage.py --decide --state $OUT/state --book v1 --out $OUT/realfill/decide_slip.csv >> "$LOG" 2>&1 || log " HL probe WARN"
    # 5) book the real-fill round-trip PnL
    $PY live/convexity_realfill.py --state $OUT >> "$LOG" 2>&1 && log " ledger updated" || log " ledger WARN"
    # 6) snapshot (real-fill + modeled reference)
    $PY live/convexity_notify_v1.py >> "$LOG" 2>&1 && log " tg snapshot sent" || log " tg snapshot WARN"
  else log " decide FAIL"; fi
  wait_ready
done
