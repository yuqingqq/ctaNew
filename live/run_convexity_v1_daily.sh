#!/usr/bin/env bash
# Convexity v1 — SINGLE low-vol book, daily forward pipeline. Models trained MONTHLY (train_twobook_models.py);
# this daily job is fully incremental (append new bars + predict, no refit). NO flow/book-A, NO combine.
#
#   1. klines/funding   refreshed by the DATA-FEED server (subscription). This script only rebuilds features.
#   2. xs_feats         incremental_xs_feats.py     (45d-window recompute + append)
#   3. panel            incremental_panel.py        (append new 4h bars)
#   4. predict          predict_twobook_incremental.py -> base (short ranker) + resid_rev (long ranker) preds
#   5. universe         exclude frozen top-N high-vol (convexity_v1_universe.json, static-at-retrain) -> low-vol book
#   6. advance          ONE book, K=3 L/S beta-neutral, 24h/6-sleeve, dual-pred (long=resid_rev, short=base)
#   7. slippage         realized HL-L2 slippage on the single book
# MONTHLY: monthly_retrain.sh refits + re-ranks the universe (static within the month).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
OUT=$ROOT/live/state/convexity_v1; mkdir -p "$OUT"; LOG=$OUT/daily.log
: "${COST_BPS_LEG:=4.5}"; export COST_BPS_LEG
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

log "== convexity v1 daily (single low-vol book) =="
# 1. (klines + funding delivered by the feed server). Optional self-fetch fallback (idempotent, skip-rebuild):
[ "${SELF_FETCH:-0}" = "1" ] && { log "1. self-fetch klines (fallback)"; $PY -m live.refresh_convexity_panel --days-back 7 --skip-rebuild >>"$LOG" 2>&1 && log "  klines OK" || log "  klines FAIL"; }
log "2. xs_feats (incremental)"; $PY live/incremental_xs_feats.py --workers 6 >>"$LOG" 2>&1 && log "  xs_feats OK" || log "  xs_feats FAIL"
log "3. panel (incremental, append new 4h bars)"; $PY live/incremental_panel.py --workers 6 >>"$LOG" 2>&1 && log "  panel OK" || log "  panel FAIL"
log "4. predict (cached frozen models -> base + resid_rev)"; $PY live/predict_twobook_incremental.py >>"$LOG" 2>&1 && log "  preds OK" || log "  preds FAIL"
log "5. apply universe (exclude frozen top-N high-vol)"; $PY - >>"$LOG" 2>&1 << 'PY' || log "  universe FAIL"
import json, pandas as pd
excl=set(json.load(open("live/models/convexity_v1_universe.json"))["exclude_high_vol"])
for src,dst in [("hl/v0full_hl60.parquet","base.parquet"),("hl_residrev/v0full_hl60.parquet","long.parquet")]:
    d=pd.read_parquet("live/state/convexity/"+src); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d[~d["symbol"].isin(excl)].to_parquet("live/state/convexity_v1/"+dst)
print(f"   low-vol universe: excluded {len(excl)} high-vol names")
PY
log "6. advance single book (K=3, dual-pred: long=resid_rev, short=base)"
CONVEXITY_STATE=$OUT/state CONVEXITY_PREDS_PATH=$OUT/base.parquet CONVEXITY_PREDS_LONG=$OUT/long.parquet \
  STRAT_K=3 SIDE_MODE=default $PY -m live.convexity_paper_bot --cycle >>"$LOG" 2>&1 && log "  advance OK" || log "  advance FAIL"
log "7. slippage"; $PY live/convexity_slippage.py --state $OUT/state --book v1 --out $OUT/slippage.csv >>"$LOG" 2>&1 || log "  slippage skip"
log "== convexity v1 daily done =="
