"""Crowding-feature panel for SQ1 (RESEARCH_LOOP_20260707 addendum 14).

Per-symbol 4h-cadence crowding features, ALL free, DISJOINT from V0_LEAN. Availability-lagged for
Vision daily archive (T1 36h worst-case convention on metrics; funding known at settlement so a
lighter lag). Outputs live/state/convexity/crowding_panel.parquet with columns
symbol, open_time, + crowding features. NO forward windows, NO pred/V0 features, NO label.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
CACHE = REPO / "data/ml/cache"
LAG_METRICS = 432   # 36h in 5m bars (T1 Vision-availability convention)

def load_panel_syms():
    p = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "funding_rate_z_7d",
                                            "funding_rate_1d_change"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    return p[(p.open_time.dt.hour % 4 == 0) & (p.open_time.dt.minute == 0)]

def metrics_feats(sym):
    """OI-change + positioning ratios from the metrics cache, 4h grid, availability-lagged."""
    f = CACHE / f"metrics_{sym}.parquet"
    if not f.exists(): return None
    m = pd.read_parquet(f, columns=["sum_open_interest", "sum_toptrader_long_short_ratio",
                                    "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    m = m[~m.index.duplicated(keep="last")].sort_index()
    m.index = pd.to_datetime(m.index, utc=True, format="mixed")
    grid = pd.date_range(m.index.min().ceil("5min"), m.index.max(), freq="5min", tz="UTC")
    m = m.reindex(grid).ffill(limit=288)   # carry last known up to 1 day
    oi = m["sum_open_interest"].astype(float)
    out = pd.DataFrame(index=grid)
    # OI change: 1d log-change, z-scored over trailing 7d (crowding build-up), then lagged
    oi_chg = np.log(oi / oi.shift(288)).replace([np.inf, -np.inf], np.nan)
    out["oi_change_z"] = ((oi_chg - oi_chg.rolling(2016).mean()) /
                          oi_chg.rolling(2016).std().replace(0, np.nan))
    out["toptrader_ls"] = m["sum_toptrader_long_short_ratio"].astype(float)
    out["ls_ratio"] = m["count_long_short_ratio"].astype(float)
    out["taker_ls"] = m["sum_taker_long_short_vol_ratio"].astype(float)
    out = out.shift(LAG_METRICS)   # availability lag (PIT)
    out = out[(out.index.hour % 4 == 0) & (out.index.minute == 0)]
    out["symbol"] = sym
    return out.reset_index().rename(columns={"index": "open_time"})

def main():
    P = load_panel_syms()   # funding features (lighter lag: settlement-known, shift 1 cycle)
    P = P.sort_values(["symbol", "open_time"])
    for c in ("funding_rate_z_7d", "funding_rate_1d_change"):
        P[c] = P.groupby("symbol")[c].shift(1)
    syms = sorted(P["symbol"].unique())
    parts = []
    for i, sym in enumerate(syms):
        m = metrics_feats(sym)
        if m is not None: parts.append(m)
        if (i + 1) % 40 == 0: print(f"  {i+1}/{len(syms)}", flush=True)
    M = pd.concat(parts, ignore_index=True)
    M["open_time"] = pd.to_datetime(M["open_time"], utc=True)
    out = P.merge(M, on=["symbol", "open_time"], how="left")
    # cycle-level crowding regime state: XS dispersion of funding
    g = out.groupby("open_time")
    out["funding_dispersion"] = g["funding_rate_z_7d"].transform("std")
    feats = ["funding_rate_z_7d", "funding_rate_1d_change", "oi_change_z", "toptrader_ls",
             "ls_ratio", "taker_ls", "funding_dispersion"]
    cov = out[feats].notna().mean().round(3)
    print("feature coverage:\n" + cov.to_string())
    D = REPO / "live/state/convexity"; D.mkdir(parents=True, exist_ok=True)
    out[["symbol", "open_time"] + feats].to_parquet(D / "crowding_panel.parquet")
    print(f"wrote {D/'crowding_panel.parquet'} ({len(out)} rows)")
    print("CROWDDONE")

if __name__ == "__main__":
    main()
