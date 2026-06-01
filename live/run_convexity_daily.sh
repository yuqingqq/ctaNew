#!/usr/bin/env bash
# Convexity two-book DAILY forward-test pipeline (champion = volatility-split, Phase-IX).
#
# FULLY INCREMENTAL (2026-06-01): ~6-7 min/cycle (was 2hr+). Every stage appends only new bars; models
# are cached (trained MONTHLY by train_twobook_models.py, NOT here). All steps validated machine-precision.
#   1. flow ingest      ingest_flow_daily.py        (Kyle/VPIN on new aggTrade files only)   ~1-1.5min
#   2a. klines fetch    refresh --skip-rebuild       (new days, no invalidate)               ~2min
#   2b. xs_feats        incremental_xs_feats.py      (45d-window recompute + append)          ~1min
#   2c. panel           incremental_panel.py         (append new 4h bars, mem-safe)           ~17s
#   3. preds            predict_twobook_incremental.py (CACHED models, predict new bars)      ~7s
#   4. rvol-split top-80 -> flow book
#   5. --cycle both books (advance; mom30/beta windowed 45d → ~10s, was ~3.5min)
#   6. combine 50/50 + realized HL-L2 slippage
# MEMORY: workers capped (≤6 incremental / ≤4 full) on the 30GB box. MONTHLY: train_twobook_models.py refits.
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd $ROOT
SP=$ROOT/live/state/convexity/split2; OUT=$ROOT/live/state/convexity_twobook; mkdir -p $OUT
LOG=$OUT/daily.log; PY=python3
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a $LOG; }

log "== daily pipeline start =="
log "1. flow ingest"; $PY live/ingest_flow_daily.py >> $LOG 2>&1 && log "  flow OK" || log "  flow FAIL"
log "2a. klines fetch (no invalidate)"; $PY -m live.refresh_convexity_panel --days-back 7 --skip-rebuild >> $LOG 2>&1 && log "  klines OK" || log "  klines FAIL"
log "2b. INCREMENTAL xs_feats (append new bars; 6w, mem-safe)"; $PY live/incremental_xs_feats.py --workers 6 >> $LOG 2>&1 && log "  xs_feats OK" || log "  xs_feats FAIL"
log "2c. INCREMENTAL panel (append new 4h bars only; 6w, append-only → no full-history frames, no OOM)"; $PY live/incremental_panel.py --workers 6 >> $LOG 2>&1 && log "  panel OK" || log "  panel FAIL"
# (build_panel_fast.py remains for a full from-scratch rebuild, e.g. monthly retrain or universe change.)
log "3. INCREMENTAL preds (cached models → predict new bars only, ~7s vs ~4min refit)"; $PY live/predict_twobook_incremental.py >> $LOG 2>&1 && log "  preds OK" || log "  preds FAIL"
log "4. rvol-split top-80"; $PY - >> $LOG 2>&1 << 'PY' || log "  split FAIL"
import pandas as pd, numpy as np
asof=pd.Timestamp.utcnow().normalize()+pd.Timedelta(days=1)
lo=asof-pd.Timedelta(days=30)
p=pd.read_parquet("outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","rvol_7d"])
p["open_time"]=pd.to_datetime(p["open_time"],utc=True); p=p[(p.open_time>=lo)&(p.open_time<asof)]
rv=p.groupby("symbol")["rvol_7d"].mean().to_dict()
ff=pd.read_parquet("live/state/convexity/hl/fullflow_hl60.parquet"); v0=pd.read_parquet("live/state/convexity/hl/v0full_hl60.parquet")
for d in (ff,v0): d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
oos=sorted(set(ff[ff.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].symbol.unique()))
ranked=sorted([s for s in oos if np.isfinite(rv.get(s,np.nan))],key=lambda s:-rv[s])
A=set(ranked[:80])
ff[ff.symbol.isin(A)].to_parquet("live/state/convexity/split2/bookA_hv80.parquet")
v0[~v0.symbol.isin(A)].to_parquet("live/state/convexity/split2/bookB_hv80.parquet")
print(f"   split: flow {len(A)}, price {len([s for s in ranked if s not in A])}")
PY
log "5. advance both books"
for bk in A B; do
  CONVEXITY_STATE=$ROOT/live/state/convexity_book$bk CONVEXITY_PREDS_PATH=$SP/book${bk}_hv80.parquet STRAT_K=3 SIDE_MODE=default \
    $PY -m live.convexity_paper_bot --cycle >> $LOG 2>&1 && log "  book$bk OK" || log "  book$bk FAIL"
done
log "6. combine + slippage"
$PY live/convexity_twobook_combine.py --book-a $ROOT/live/state/convexity_bookA/cycles.csv --book-b $ROOT/live/state/convexity_bookB/cycles.csv --out $OUT >> $LOG 2>&1 && log "  combine OK" || log "  combine FAIL"
for bk in A B; do $PY live/convexity_slippage.py --state $ROOT/live/state/convexity_book$bk --book $bk --out $OUT/slippage.csv >> $LOG 2>&1; done
log "== daily pipeline done =="
