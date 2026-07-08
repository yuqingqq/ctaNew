"""Build PIT BTC positioning axes from the Binance futures metrics cache and merge them onto the
regime-discovery gate dataset. Committed 2026-07-07 per design-review F1 (the original battery ran
from an uncommitted inline script; this is the reproducible version).

Axes (all trailing/PIT, from 5-min metrics of BTCUSDT):
  btc_oi_chg_24h  log change of sum_open_interest (CONTRACTS, price-neutral — NOT the USD value
                  column, which is price-contaminated and fires mechanically after drawdowns)
  btc_oi_chg_3d   same, 3d
  btc_oi_z_30d    (OI - 30d rolling mean) / 30d rolling std; min_periods = 20d of 5-min bars;
                  window coverage < 80% -> NaN (feed gaps must not silently shorten the window)
  btc_taker_ls_24h / btc_top_ls_24h / btc_glob_ls_24h  trailing-24h means of the three L/S ratios

Merge rule: asof backward (last snapshot <= cycle open), staleness cap 2h -> else NaN.
NaN axis value => downstream gates must be INERT (never fire on missing data).

Usage: python3 live/build_positioning_axes.py [--dataset live/V4_GATE_MODEL_DATASET.parquet]
       [--metrics data/ml/cache/metrics_BTCUSDT.parquet] [--out live/V4_GATE_DATASET_POSITIONING.parquet]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent

def build_axes(metrics_path: Path) -> pd.DataFrame:
    m = pd.read_parquet(metrics_path).sort_index()
    m = m[~m.index.duplicated(keep="last")]
    oi = m["sum_open_interest"].astype(float).replace(0, np.nan)   # contracts, price-neutral
    tls = m["sum_taker_long_short_vol_ratio"].astype(float)
    pls = m["sum_toptrader_long_short_ratio"].astype(float)
    gls = m["count_long_short_ratio"].astype(float)
    W30 = 288 * 30
    cov = oi.notna().rolling(W30, min_periods=1).sum() / W30       # window coverage guard (>=80%)
    z = (oi - oi.rolling(W30, min_periods=288 * 20).mean()) / oi.rolling(W30, min_periods=288 * 20).std()
    feat = pd.DataFrame({
        "btc_oi_chg_24h": np.log(oi / oi.shift(288)),
        "btc_oi_chg_3d":  np.log(oi / oi.shift(288 * 3)),
        "btc_oi_z_30d":   z.where(cov >= 0.8),
        "btc_taker_ls_24h": tls.rolling(288, min_periods=144).mean(),
        "btc_top_ls_24h":   pls.rolling(288, min_periods=144).mean(),
        "btc_glob_ls_24h":  gls.rolling(288, min_periods=144).mean(),
    }).dropna(how="all")
    fr = feat.reset_index()
    fr.columns = ["open_time"] + list(feat.columns)
    fr["open_time"] = pd.to_datetime(fr["open_time"], utc=True).astype("datetime64[ns, UTC]")
    return fr.sort_values("open_time")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "live/V4_GATE_MODEL_DATASET.parquet"))
    ap.add_argument("--metrics", default=str(REPO / "data/ml/cache/metrics_BTCUSDT.parquet"))
    ap.add_argument("--out", default=str(REPO / "live/V4_GATE_DATASET_POSITIONING.parquet"))
    a = ap.parse_args()
    fr = build_axes(Path(a.metrics))
    d = pd.read_parquet(a.dataset)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True).astype("datetime64[ns, UTC]")
    d = d.sort_values("open_time")
    d = pd.merge_asof(d, fr, on="open_time", direction="backward", tolerance=pd.Timedelta("2h"))
    newcols = [c for c in fr.columns if c != "open_time"]
    print("coverage:"); print(d[newcols].notna().mean().round(3))
    d.to_parquet(a.out, index=False)
    print(f"wrote {a.out} ({len(d)} rows)")

if __name__ == "__main__":
    main()
