#!/usr/bin/env bash
# Convexity v1 LIVE forward test — single low-vol book + resid_rev, 4h cadence.
# Frozen models (convexity_v1_{short,long}_model.pkl, fit_cut 2026-05-29); every forward bar is OOS.
#
# Wires the two parity fixes found 2026-06-04 (verified 11/11 exact vs golden_cycles_v1.json):
#   (a) ingest_funding EVERY cycle — v1 restored funding to the model; a stale feed silently NaNs
#       funding_rate_z_7d and corrupts the preds (the 03-31 stall = wrong legs). Vision monthly lags a
#       few days into the current month → tolerable (z degraded, not NaN); FAPI top-up is a TODO.
#   (b) onboardDate maturity (CONVEXITY_UNIVERSE_META=maturity_meta) — the panel-coverage default
#       collapses this box's shallow universe to 24 large-caps; onboardDate keeps the full low-vol book.
#
# ROBUSTNESS: fail-closed on xs_feats/panel/predict; NO-OP guard (only advance when a new bar settled);
# adaptive nap (retry 10m if a bar is overdue, else sleep to next boundary+35m).
# Launch:  tmux new -d -s cvx1 'bash /home/yuqing/ctaNew/live/run_convexity_v1_live.sh'
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=$ROOT/.venv/bin/python
OUT=$ROOT/live/state/convexity_v1; mkdir -p "$OUT"; LOG=$OUT/run.log
PANEL=$ROOT/outputs/vBTC_features/panel_expanded_v0.parquet
export COST_BPS_LEG=4.5 STRAT_K=3 SIDE_MODE=default
export XS_LEAN=1                                                                  # lean xs_feats: 6 cols only, 23× faster (validated 11/11 golden)
export CONVEXITY_UNIVERSE_META=$ROOT/live/state/convexity/maturity_meta.parquet   # onboardDate maturity (fix b)
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
panel_edge(){ $PY -c "import pandas as pd; print(pd.to_datetime(pd.read_parquet('$PANEL',columns=['open_time'])['open_time'],utc=True).max())" 2>/dev/null; }
nap(){
  S=$($PY -c "
import datetime as dt, pandas as pd
now=dt.datetime.now(dt.timezone.utc)
def fl(t): return t.replace(hour=(t.hour//4)*4, minute=0, second=0, microsecond=0)
edge=pd.to_datetime(pd.read_parquet('$PANEL',columns=['open_time'])['open_time'],utc=True).max().to_pydatetime()
expected=fl(now - dt.timedelta(hours=4, minutes=35))
print(600 if edge < expected else max(120, int((edge + dt.timedelta(hours=8, minutes=35) - now).total_seconds())))" 2>/dev/null)
  [ -z "$S" ] && S=600; log "-- sleep ${S}s (adaptive) --"; sleep "$S"; }

apply_universe(){ $PY - << 'PY'
import json, pandas as pd
excl=set(json.load(open("live/models/convexity_v1_universe.json"))["exclude_high_vol"])
for src,dst in [("hl/v0full_hl60.parquet","base.parquet"),("hl_residrev/v0full_hl60.parquet","long.parquet")]:
    d=pd.read_parquet("live/state/convexity/"+src); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d[~d["symbol"].isin(excl)].to_parquet("live/state/convexity_v1/"+dst)
PY
}

log "== convexity v1 LIVE (single low-vol book + resid_rev | funding-refresh + onboardDate maturity) =="
while true; do
  log "-- cycle start --"
  EDGE0=$(panel_edge)
  # SIGNALS COME FROM THE REALTIME FEED ONLY: WS collector klines (5m, written by wscol) + FAPI funding.
  # No Vision in the live loop (Vision lags 1-2d → would generate stale signals). reconcile_aggtrades
  # (Vision flow) is removed — v1 uses no flow features anyway.
  $PY live/ingest_funding_fapi.py        >> "$LOG" 2>&1 && log " funding FAPI(current) OK" || log " funding FAPI WARN"
  if ! $PY live/incremental_xs_feats.py --workers 6 >> "$LOG" 2>&1; then log " xs_feats FAIL — skip"; nap; continue; fi
  if ! $PY live/incremental_panel.py    --workers 6 >> "$LOG" 2>&1; then log " panel FAIL — skip"; nap; continue; fi
  EDGE1=$(panel_edge)
  BOT0=$($PY -c "import json;print(json.load(open('$OUT/state/positions.json')).get('last_open_time') or '')" 2>/dev/null)
  # Advance whenever the panel (labeled bars only) is AHEAD OF THE BOT — not merely when the panel changed
  # this cycle. The panel can jump ahead out-of-band (e.g. a manual rebuild) leaving preds/bot behind, which
  # the old EDGE0==EDGE1 guard couldn't detect. Equal panel/bot edge ⇒ caught up ⇒ skip. (EDGE0 logs movement.)
  if [ "$EDGE1" = "$BOT0" ]; then log " -- bot caught up to panel ($EDGE1) — skip --"; nap; continue; fi
  log " -- panel $EDGE0->$EDGE1 | bot $BOT0 -> advancing --"
  $PY live/build_maturity_meta.py        >> "$LOG" 2>&1 && log " maturity OK" || log " maturity WARN"
  if ! $PY live/predict_twobook_incremental.py >> "$LOG" 2>&1; then log " predict FAIL — skip"; nap; continue; fi
  if ! apply_universe                    >> "$LOG" 2>&1; then log " universe FAIL — skip"; nap; continue; fi
  if CONVEXITY_STATE=$OUT/state CONVEXITY_PREDS_PATH=$OUT/base.parquet CONVEXITY_PREDS_LONG=$OUT/long.parquet \
       $PY -m live.convexity_paper_bot --cycle >> "$LOG" 2>&1; then
    BOT1=$($PY -c "import json;print(json.load(open('$OUT/state/positions.json')).get('last_open_time') or '')" 2>/dev/null)
    if [ "$BOT1" != "$BOT0" ]; then                          # bot actually settled ≥1 new labeled cycle
      log " advance OK ($BOT0 -> $BOT1)"
      $PY live/convexity_slippage.py --state $OUT/state --book v1 --out $OUT/slippage.csv >> "$LOG" 2>&1 || true
      # per-cycle portfolio + PnL snapshot → Telegram (only on a real advance, so it's never stale/duplicate)
      $PY live/convexity_notify_v1.py >> "$LOG" 2>&1 && log " tg snapshot sent" || log " tg snapshot WARN"
    else
      log " no new labeled cycle (bot at $BOT1) — no snapshot"
    fi
  else
    log " advance FAIL"
  fi
  nap
done
