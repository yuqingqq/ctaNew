"""Binance USDM bookDepth (coarse-L2) loader -> 4h PIT features, for integration into the convexity panel.

SOURCE: data.binance.vision/data/futures/um/daily/bookDepth/<SYM>/<SYM>-bookDepth-<DATE>.zip  (free, no auth).
  Format: rows of (timestamp, percentage, depth, notional); 30s snapshots; percentage in
  {-5,-4,-3,-2,-1,-0.2,+0.2,+1,+2,+3,+4,+5} = cumulative depth/notional within that % of mid (neg=bid, pos=ask).
  Coverage ~2023-01 -> present (NOT 2021-22), which matches the strategy's OOS+recent eval window.

FEATURES (per 30s snapshot -> aggregated over each 4h bar the snapshot falls in; the TEST shifts by one bar so a
panel row at open_time=T uses the book observed during [T-4h, T) = strictly PIT):
  l2_imb1   (bidN1-askN1)/(bidN1+askN1)   book imbalance within 1% of mid   [alpha candidate: directional pressure]
  l2_imb02  same at 0.2% (near touch)
  l2_liq1   log(bidN1+askN1)              total near liquidity within 1%    [capacity; alpha is illiquidity-bound]
  l2_touch  (bidN02+askN02)/(bidN1+askN1) near-touch concentration          [low = thin at touch = fragile]
  l2_slope  (bidN5+askN5)/(bidN1+askN1)   deep-vs-near book shape
  l2_asym1  log(bidN1/askN1)              bid/ask depth asymmetry           [squeeze/dump fragility, limitation #4]
  l2_imbstd  std of imb1 within the bar   quoting instability
Aggregates cached per symbol at data/ml/cache/l2_<SYM>.parquet (merge-on-write). Raw zips are discarded after agg.
"""
import io, sys, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")

def _fetch_day(sym, day):
    url = f"{BASE}/{sym}/{sym}-bookDepth-{day:%Y-%m-%d}.zip"
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError as e:
        return None if e.code == 404 else None
    except Exception:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(raw)); d = pd.read_csv(z.open(z.namelist()[0]))
    except Exception:
        return None
    if not {"timestamp", "percentage", "notional"}.issubset(d.columns) or not len(d):
        return None
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    # pivot notional by percentage level per snapshot
    p = d.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
    def col(x):
        return p[x] if x in p.columns else pd.Series(np.nan, index=p.index)
    bN1, aN1 = col(-1.0), col(1.0); bN02, aN02 = col(-0.2), col(0.2); bN5, aN5 = col(-5.0), col(5.0)
    bN2, aN2 = col(-2.0), col(2.0); bN3, aN3 = col(-3.0), col(3.0)   # for imb2/imb3/imb5 (wider-book imbalance)
    liq1 = (bN1 + aN1).replace(0, np.nan)
    snap = pd.DataFrame({
        "imb1": (bN1 - aN1) / liq1,
        "imb02": (bN02 - aN02) / (bN02 + aN02).replace(0, np.nan),
        "imb2": (bN2 - aN2) / (bN2 + aN2).replace(0, np.nan),
        "imb3": (bN3 - aN3) / (bN3 + aN3).replace(0, np.nan),
        "imb5": (bN5 - aN5) / (bN5 + aN5).replace(0, np.nan),
        "liq1": np.log(liq1),
        "touch": (bN02 + aN02) / liq1,
        "slope": (bN5 + aN5) / liq1,
        "asym1": np.log((bN1 / aN1.replace(0, np.nan)).clip(1e-6, 1e6)),
    }, index=p.index)
    return snap

def _agg_4h(snap):
    """aggregate 30s snapshots into the 4h bar they fall in (observation bar; TEST shifts +1 bar for PIT)."""
    g = snap.groupby(snap.index.floor("4h"))
    out = g.mean()
    out["l2_imbstd"] = g["imb1"].std()
    out.columns = [c if c.startswith("l2_") else f"l2_{c}" for c in out.columns]
    out.index.name = "obs_bar"
    return out

def load_symbol(sym, days, workers=16):
    # MEMORY-SAFE: aggregate each day's 30s snapshots to 4h AS IT ARRIVES (4h bars never span a day), so we never
    # hold a full symbol's raw snapshots at once. Result is identical to aggregating the concatenated snapshots.
    aggs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_day, sym, d): d for d in days}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r):
                a = _agg_4h(r[~r.index.duplicated()])
                aggs.append(a)
    if not aggs: return None
    out = pd.concat(aggs).sort_index()
    return out[~out.index.duplicated(keep="last")]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", required=True, help="comma-separated")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    syms = a.syms.split(","); days = pd.date_range(a.start, a.end, freq="D")
    CACHE.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(syms, 1):
        out = load_symbol(sym, days, a.workers)
        if out is None:
            print(f"  [{i}/{len(syms)}] {sym}: no data", flush=True); continue
        p = CACHE / f"l2_{sym}.parquet"
        if p.exists():
            old = pd.read_parquet(p); out = pd.concat([old, out])
            out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_parquet(p)
        print(f"  [{i}/{len(syms)}] {sym}: {len(out)} 4h-bars {str(out.index.min())[:10]}..{str(out.index.max())[:10]}", flush=True)
    print("BDLOADDONE")

if __name__ == "__main__":
    main()
