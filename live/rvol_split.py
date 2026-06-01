"""Refresh the two-book per-book preds against the FROZEN volatility-split.

The split MEMBERSHIP is the single source of truth in live/models/twobook_split.json (the 'flow_book'
list), set as-of the model fit_cut / retrain date by train_twobook_models.py (PIT, NOT re-ranked every
cycle — research: per-cycle re-rank +2.11 vs static +2.74). This step just LOADS that frozen set and
refreshes the per-book preds with the latest bars:
  top of frozen flow_book (with fresh flow caches) → BookA (flow model, V0+flow)
  everything else in the universe                  → BookB (price model, V0)

NOTE: previously this re-ranked rvol as-of the panel edge into flow_set.json — that put the split's
as-of date at "now" and applied it backward over the bootstrap (a look-ahead). Using the committed
twobook_split.json (as-of the retrain) removes that. Re-ranking now happens only at retrain.

Usage: PYTHONPATH=. .venv/bin/python live/rvol_split.py
"""
import glob, json
from pathlib import Path
import pandas as pd
REPO = Path("/home/yuqing/ctaNew")
SP = REPO/"live/state/convexity/split2"; HL = REPO/"live/state/convexity/hl"
SPLIT = REPO/"live/models/twobook_split.json"


def main():
    SP.mkdir(parents=True, exist_ok=True)
    A = set(json.loads(SPLIT.read_text())["flow_book"])          # FROZEN flow set (as-of retrain, PIT)
    # FRESHNESS GUARD: a flow sym with a stale/missing cache (ingest_flow failed) would feed dead flow
    # features into BookA — drop it from BookA (it falls through to BookB, which needs no flow data).
    pe = pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet", columns=["open_time"])
    asof = pd.to_datetime(pe["open_time"], utc=True).max()
    fresh_cut = asof - pd.Timedelta(hours=12)
    fresh = set()
    for f in glob.glob(str(REPO/"data/ml/cache/flow_*.parquet")):
        s = Path(f).stem.replace("flow_", "")
        if s not in A:
            continue
        try:
            if pd.to_datetime(pd.read_parquet(f).index.max(), utc=True) >= fresh_cut:
                fresh.add(s)
        except Exception:
            continue
    A = A & fresh
    ff = pd.read_parquet(HL/"fullflow_hl60.parquet"); v0 = pd.read_parquet(HL/"v0full_hl60.parquet")
    for d in (ff, v0): d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    ff[ff.symbol.isin(A)].to_parquet(SP/"bookA_flow.parquet")
    v0[~v0.symbol.isin(A)].to_parquet(SP/"bookB_price.parquet")
    meta = json.loads(SPLIT.read_text())
    print(f"[rvol_split] FROZEN split (asof {meta.get('asof')}): BookA(flow) {len(A)}/{len(meta['flow_book'])} fresh "
          f"| BookB(price) {v0[~v0.symbol.isin(A)].symbol.nunique()} (preds refreshed)")


if __name__ == "__main__":
    main()
