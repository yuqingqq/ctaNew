"""Build a DYN_ALLOW parquet implementing v1's vol-truncation for the v3 A/B: each MONTH (frozen at month-start,
v1-style — 'rolling re-rank hurts'), exclude the top-N symbols by trailing-30d mean rvol_7d. Allowlist =
universe minus that top-N, applied to all cycles in the month. Output: (symbol, open_time) rows for ALLOWED names.
Usage: python3 live/build_volexclude_allow.py <N> <out.parquet> [preds_path]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/home/yuqing/ctaNew")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "live/state/longtail/volexcl_allow.parquet"
PREDS = sys.argv[3] if len(sys.argv) > 3 else str(R / "live/state/convexity/hl_lean175/v0full_hl60.parquet")

pan = pd.read_parquet(R / "outputs/vBTC_features/panel_expanded_v0.parquet", columns=["symbol", "open_time", "rvol_7d"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
pr = pd.read_parquet(PREDS, columns=["symbol", "open_time"]); pr["open_time"] = pd.to_datetime(pr["open_time"], utc=True)
pr["mon"] = pr["open_time"].dt.to_period("M")
pan = pan.sort_values(["symbol", "open_time"])

rows = []
for mon, g in pr.groupby("mon"):
    cyc = sorted(g["open_time"].unique())
    ms = pd.Timestamp(min(cyc))                                 # first cycle of the month (tz-aware)
    lo = ms - pd.Timedelta(days=30)
    win = pan[(pan.open_time >= lo) & (pan.open_time < ms)]
    rv = win.groupby("symbol")["rvol_7d"].mean().dropna()
    univ = set(g["symbol"].unique())
    cand = [s for s in univ if s in rv.index]
    excl = set(sorted(cand, key=lambda s: -rv[s])[:N])          # top-N high-vol EXCLUDED (frozen this month)
    allowed = [s for s in univ if s not in excl]
    for ot in cyc:
        for s in allowed:
            rows.append((s, ot))
out = pd.DataFrame(rows, columns=["symbol", "open_time"])
out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
OUT.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(OUT)
print(f"DYN_ALLOW (exclude top-{N} high-vol, monthly-frozen): {out['symbol'].nunique()} distinct allowed syms, "
      f"{out['open_time'].nunique()} cycles, {len(out)} rows | avg allowed/cyc {len(out)/out['open_time'].nunique():.0f} -> {OUT}")
