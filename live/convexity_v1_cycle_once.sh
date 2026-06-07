#!/usr/bin/env bash
# ONE convexity-v1 decision cycle. Triggered DIRECTLY by the collector the instant a 4h bar closes (push),
# and also by the fallback watchdog as a safety net. Safe to call repeatedly:
#   - flock (-n) → only ever one cycle at a time (collector + fallback can't collide).
#   - idempotent → if this 4h boundary is already booked in the ledger, exit immediately (no dup HL probe /
#     no dup Telegram snapshot).
# Pipeline: refresh data → settle modeled reference → decide current bar → probe HL → book real-fill → snapshot.
# Arg $1 = caller tag (collector|fallback|manual) for the log.
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=$ROOT/.venv/bin/python
export CONVEXITY_BOOK=convexity_v2                 # v2 lives in its OWN state dir (v1 = untouched reference)
OUT=$ROOT/live/state/$CONVEXITY_BOOK; mkdir -p "$OUT/realfill"; LOG=$OUT/realtime.log
export COST_BPS_LEG=4.5 STRAT_K=3 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1
export CONVEXITY_UNIVERSE_META=$ROOT/live/state/convexity/maturity_meta.parquet
# === v2 candidate (2026-06-05 mechanism audit) — trade the bear edge instead of sitting flat ===
#   BEAR_MODE=equal      : trade bear via equal-weight K=3 L/S (was flat) — the bear edge the DD-stop hid
#   STOP_SKIP_REGIMES=bear: DD-stop OFF in bear (it's pro-cyclical vs mean-rev; engaged 78% of bear)
#   SIDE_BETA_NEUT=0     : equal-weight sizing (drop the noisy beta-neutral a/b reweighting)
# Scorecard: gross +3.82 / net-of-funding +3.33 Sharpe (vs v1 +2.92); CAVEAT ~2x maxDD. Forward test = arbiter.
export BEAR_MODE=equal STOP_SKIP_REGIMES=bear SIDE_BETA_NEUT=0 BEAR_K=2 SIZING_MODE=inv_vol
SRC="${1:-manual}"
log(){ echo "[$(date -u '+%F %T')] [cycle/$SRC] $*" | tee -a "$LOG"; }
bot_edge(){ $PY -c "import json;print(json.load(open('$OUT/state/positions.json')).get('last_open_time') or '')" 2>/dev/null; }
apply_universe(){ $PY - << 'PY'
import json, os, pandas as pd
book=os.environ.get("CONVEXITY_BOOK","convexity_v1")   # write to the ACTIVE book dir, not a hardcoded v1
excl=set(json.load(open("live/models/convexity_v1_universe.json"))["exclude_high_vol"])
for src,dst in [("hl/v0full_hl60.parquet","base.parquet"),("hl_residrev/v0full_hl60.parquet","long.parquet")]:
    d=pd.read_parquet("live/state/convexity/"+src); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d[~d["symbol"].isin(excl)].to_parquet(f"live/state/{book}/"+dst)
PY
}

# ---- no overlap: only one cycle at a time ----
exec 9>"$OUT/.cycle.lock"
flock -n 9 || { log "another cycle holds the lock — skip"; exit 0; }
# PAUSE switch: if PAUSE file present, skip (e.g. preds under investigation) — collector trigger + fallback both no-op
[ -f "$OUT/PAUSE" ] && { log "PAUSED ($OUT/PAUSE present) — skip cycle"; exit 0; }

# ---- idempotent: skip if this 4h boundary is already booked ----
B=$($PY -c "import pandas as pd; print(pd.Timestamp.utcnow().floor('4h'))" 2>/dev/null)
LED=$($PY -c "import json,os;p='$OUT/realfill/ledger.json';print(json.load(open(p)).get('last_open_time','') if os.path.exists(p) else '')" 2>/dev/null)
[ -n "$B" ] && [ "$B" = "$LED" ] && { log "boundary $B already booked — skip"; exit 0; }

log "=== boundary $B: cycle start ==="
# Capture HL DECISION mids NOW (~bar close) for the candidate universe → realfill measures the TRUE HL→HL
# latency drift (exec_mid − decision_mid), basis-free. Async/non-blocking; finishes long before the ledger reads it.
[ -f "$OUT/long.parquet" ] && $PY live/convexity_slippage.py --snapshot-mids --preds "$OUT/long.parquet" \
    --out "$OUT/decide/decision_mids.json" >> "$LOG" 2>&1 &
# 1) refresh realtime data (collector already flushed the boundary bar)
# Funding comes from the @markPrice subscription (collector writes the caches on each settlement). Skip the
# ~39s FAPI pull when the caches are fresh; FAPI only as a fallback if the subscription ever went stale
# (funding settles every 4h → fresh if the newest cached settlement is < 5h old).
FAGE=$($PY -c "
import pandas as pd, glob
fs=glob.glob('data/ml/cache/funding_*.parquet')
mx=max((pd.to_datetime(pd.read_parquet(f,columns=['calc_time'])['calc_time'],utc=True).max() for f in fs[:30]), default=None)
print(round((pd.Timestamp.utcnow()-mx).total_seconds()/3600,1) if mx is not None else 999)" 2>/dev/null)
if $PY -c "import sys;sys.exit(0 if float('${FAGE:-999}')<5 else 1)" 2>/dev/null; then
  log "funding fresh via subscription (${FAGE}h old)"
else
  log "funding STALE (${FAGE}h) — FAPI fallback"; $PY live/ingest_funding_fapi.py >> "$LOG" 2>&1 || log "funding WARN"
fi
# Wait (bounded) for the just-closed boundary 5m bar to propagate to ~the full cohort BEFORE features. The fast
# cycle otherwise decides at boundary+5m+~3s, before the slow (thin/illiquid) symbols' bar arrives via WS/FAPI,
# collapsing the xs-rank cohort and tripping the decide guard (the old 89s backfill hid this by waiting).
$PY live/wait_boundary_bar.py "$B" 155 25 >> "$LOG" 2>&1 || true
# Repair 5m bars the WS feed dropped (FAPI backfill) BEFORE features — gaps make row-offset features
# (return_1d=pct_change(288), bars_since_high, obv) reach back the wrong distance / use wrong values. After the
# wait FAPI has the boundary bar, so this fills any genuine remaining gaps. Still required: a silent gap
# corrupts the row-offset features with no error.
$PY live/backfill_klines_gaps.py >> "$LOG" 2>&1 || log "backfill WARN"
# xs_feats recomputes a trailing window so corrected/backfilled klines propagate. build_bar (decide) reads ONLY
# the xs_feats caches + the panel SYMBOL LIST (not the panel data), so the panel rebuild + modeled settle are
# DEFERRED below — off the decision→fill critical path — to minimise execution latency.
$PY live/incremental_xs_feats.py --workers 6 >> "$LOG" 2>&1 || { log "xs_feats FAIL — abort"; exit 1; }

# 2) DECIDE + EXECUTE FIRST, so the fill lands ASAP after the signal (cuts decision→fill latency ~40s→~20s)
$PY live/decide_v1.py >> "$LOG" 2>&1 || { log "decide_v1 FAIL — abort"; exit 1; }
if CONVEXITY_STATE=$OUT/state CONVEXITY_PREDS_PATH=$OUT/decide/base_decide.parquet CONVEXITY_PREDS_LONG=$OUT/decide/long_decide.parquet \
     $PY -m live.convexity_paper_bot --decide >> "$LOG" 2>&1; then
  log "decided: $($PY -c "import json;d=json.load(open('$OUT/state/decision.json'));print(d['open_time'],d['regime'],'-',len(d.get('turnover',{})),'legs')" 2>/dev/null)"
  # 3) probe live HL L2 for the turnover legs (real execution price at the boundary)
  $PY live/convexity_slippage.py --decide --state $OUT/state --book v1 --out $OUT/realfill/decide_slip.csv >> "$LOG" 2>&1 || log "HL probe WARN"
  # 4) book the real-fill round-trip PnL (THE FILL — latency window ends here)
  $PY live/convexity_realfill.py --state $OUT >> "$LOG" 2>&1 && log "ledger updated" || log "ledger WARN"
else log "decide FAIL"; fi

# 5) DEFERRED (post-fill): panel rebuild + maturity meta + settle the modeled REFERENCE track. None feed the
# decision (build_bar uses only the panel symbol list + xs_feats; decide uses the prior-cycle panel/meta, both
# slow-changing), so they run AFTER the fill. Failures here are WARN, not abort — the trade is already booked.
$PY live/incremental_panel.py --workers 6 --rebuild-days 10 >> "$LOG" 2>&1 || log "panel WARN"
$PY live/build_maturity_meta.py >> "$LOG" 2>&1 || true
BOT0=$(bot_edge)
if $PY live/predict_twobook_incremental.py >> "$LOG" 2>&1 && apply_universe >> "$LOG" 2>&1; then
  CONVEXITY_STATE=$OUT/state CONVEXITY_PREDS_PATH=$OUT/base.parquet CONVEXITY_PREDS_LONG=$OUT/long.parquet \
    $PY -m live.convexity_paper_bot --cycle >> "$LOG" 2>&1 || log "settle WARN"
  BOT1=$(bot_edge); [ "$BOT1" != "$BOT0" ] && log "settled modeled $BOT0 -> $BOT1" || true
else log "settle-predict WARN"; fi

# 6) snapshot (real-fill primary)
$PY live/convexity_notify_v1.py >> "$LOG" 2>&1 && log "tg snapshot sent" || log "tg WARN"
log "=== boundary $B: done ==="
