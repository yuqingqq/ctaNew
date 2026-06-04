#!/usr/bin/env bash
# convexity v1 — SINGLE low-vol book. No two books, no flow, no combine.
#   universe : eligible symbols MINUS the top-N high-vol (convexity_v1_universe.json) -> trade the rest (~94 low-vol)
#   construct: K=3 long/short, beta-neutral, 24h hold / 6 sleeves, regime gate
#   dual-pred: LONG ranked by resid_rev model, SHORT ranked by base model
#   models   : live/models/convexity_v1_{short,long}_model.pkl ; universe convexity_v1_universe.json
#
# Usage:
#   bash live/run_convexity_v1.sh            # replay-forward on current preds
#   REGEN=1 bash live/run_convexity_v1.sh    # refresh preds from frozen models first (predict new bars)
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=python3
: "${REGEN:=0}"; : "${COST_BPS_LEG:=4.5}"; export COST_BPS_LEG
OUT=$ROOT/live/state/convexity_v1; mkdir -p "$OUT"; LOG=$OUT/run.log
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
log "== convexity v1 (single low-vol book) | cost=${COST_BPS_LEG}bps =="

[ "$REGEN" = "1" ] && { log "refresh preds from frozen models"; $PY live/predict_twobook_incremental.py >>"$LOG" 2>&1 || log "WARN predict failed"; }

# Filter base + resid_rev preds to the low-vol universe (exclude the frozen top-N high-vol set).
$PY - <<PY >>"$LOG" 2>&1 || { log "filter FAILED"; exit 1; }
import json, pandas as pd
from pathlib import Path
R=Path("/home/yuqing/ctaNew")
excl=set(json.load(open(R/"live/models/convexity_v1_universe.json"))["exclude_high_vol"])
o=R/"live/state/convexity_v1"; o.mkdir(parents=True,exist_ok=True)
for src,dst in [("hl/v0full_hl60.parquet","base.parquet"),("hl_residrev/v0full_hl60.parquet","long.parquet")]:
    d=pd.read_parquet(R/"live/state/convexity"/src)
    d[~d["symbol"].isin(excl)].to_parquet(o/dst)
print("filtered to low-vol universe")
PY

# Run the bot on the single low-vol book (dual-pred: short=base, long=resid_rev).
export CONVEXITY_PREDS_PATH=$OUT/base.parquet
export CONVEXITY_PREDS_LONG=$OUT/long.parquet
export CONVEXITY_STATE=$OUT/state STRAT_K=3 SIDE_MODE=default
$PY -m live.convexity_paper_bot --replay-all >>"$LOG" 2>&1 || { log "bot FAILED"; exit 1; }
$PY - <<PY | tee -a "$LOG"
import json
s=json.load(open("$OUT/state/summary.json")) if __import__("os").path.exists("$OUT/state/summary.json") else None
import pandas as pd,numpy as np
c=pd.read_csv("$OUT/state/cycles.csv"); p=c["pnl_bps"]/1e4
print(f"\n=== convexity v1 (single low-vol book + resid_rev) ===")
print(f"  Sharpe {p.mean()/p.std()*np.sqrt(6*365):+.3f}  totPnL {c['pnl_bps'].sum():+.0f}bps  cycles {len(c)}")
PY
log "done -> $OUT/state/cycles.csv"
