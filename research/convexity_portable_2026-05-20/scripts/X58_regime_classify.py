"""X58 — Classify each cycle in current sample by market regime.

Regimes per cycle (4h bar):
  - btc_trend_30d: BTC 30d return sign (bull/sideways/bear)
  - btc_vol_regime: BTC realized vol 90d percentile (low/mid/high)
  - btc_drawdown: distance from 90d high (recovered/correcting/drawdown)
  - month: cyclical (Jan, Feb, ...) for seasonal check

Saves: results/X58_regime_labels.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"
KLINES = REPO / "data/ml/test/parquet/klines"


def main():
    print("=== X58 regime classification ===\n")
    # Load BTC 5m closes
    btc_files = sorted((KLINES / "BTCUSDT" / "5m").glob("*.parquet"))
    btc_dfs = [pd.read_parquet(f, columns=["open_time","close"]) for f in btc_files]
    btc = pd.concat(btc_dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float32)
    print(f"BTC closes: {len(btc):,} bars, {btc.index.min()} → {btc.index.max()}")

    # Subset to 4h cycle boundaries
    btc_4h = btc[btc.index.hour % 4 == 0][btc[btc.index.hour % 4 == 0].index.minute == 0]
    btc_4h = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    print(f"4h cycles: {len(btc_4h):,}")

    # Compute regime metrics
    # 30d return = ~7,776 5m bars (30 × 24 × 12) at 5m, or 180 4h bars
    btc_log = np.log(btc_4h)
    ret_30d = btc_log.diff(180)  # 30d in 4h bars
    ret_7d = btc_log.diff(42)    # 7d in 4h bars
    ret_90d = btc_log.diff(540)  # 90d
    # vol 30d (realized vol)
    rv_30d = btc_log.diff().rolling(180, min_periods=30).std() * np.sqrt(180)
    rv_90d_pct = rv_30d.rolling(540, min_periods=90).rank(pct=True)
    # drawdown from 90d high
    high_90d = btc_4h.rolling(540, min_periods=90).max()
    dd_90d = (btc_4h - high_90d) / high_90d

    # Classify
    regimes = pd.DataFrame(index=btc_4h.index)
    regimes["btc_close"] = btc_4h
    regimes["ret_7d"] = ret_7d
    regimes["ret_30d"] = ret_30d
    regimes["ret_90d"] = ret_90d
    regimes["rv_30d"] = rv_30d
    regimes["rv_90d_pct"] = rv_90d_pct
    regimes["dd_90d"] = dd_90d

    # Categorical regimes
    regimes["trend_30d"] = pd.cut(ret_30d,
                                    bins=[-np.inf, -0.10, 0.10, np.inf],
                                    labels=["bear", "sideways", "bull"])
    regimes["vol_regime"] = pd.cut(rv_90d_pct,
                                     bins=[-0.01, 0.33, 0.67, 1.01],
                                     labels=["low", "mid", "high"])
    regimes["drawdown"] = pd.cut(dd_90d,
                                   bins=[-1.0, -0.15, -0.05, 0.001],
                                   labels=["dd_deep", "dd_mid", "near_high"])
    regimes["month"] = regimes.index.month

    print(f"\nRegime distribution (trend_30d × vol_regime):")
    print(regimes.dropna(subset=["trend_30d", "vol_regime"]).groupby(["trend_30d", "vol_regime"]).size().unstack(fill_value=0))
    print(f"\nDrawdown distribution:")
    print(regimes["drawdown"].value_counts(dropna=False))

    out = regimes.reset_index().rename(columns={"open_time": "open_time"})
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out_path = OUT / "X58_regime_labels.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"\nDate range: {out['open_time'].min()} → {out['open_time'].max()}")


if __name__ == "__main__":
    main()
