"""Add funding features to panel_variants.parquet.

For each symbol, computes:
  funding_rate            : last published rate at bar t
  funding_rate_z_7d       : 7d rolling z-score
  funding_rate_1d_change  : rate now minus rate 1d ago
  funding_streak_pos      : consecutive bars with rate > 0 (capped 21)
  funding_streak_neg      : consecutive bars with rate < 0 (capped 21)

Saves updated panel as panel_variants_with_funding.parquet.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_collectors.funding_rate_loader import load_funding_rate, align_funding_to_bars

PANEL_IN = REPO / "outputs/vBTC_features/panel_variants.parquet"
PANEL_OUT = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"


def compute_funding_features(bar_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    funding = load_funding_rate(symbol, start_month="2025-01", end_month="2026-04")
    if funding.empty:
        return pd.DataFrame(index=bar_index, columns=[
            "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
            "funding_streak_pos", "funding_streak_neg"
        ])

    # Aligned current rate (PIT)
    rate = align_funding_to_bars(funding, bar_index)
    df = pd.DataFrame({"funding_rate": rate.to_numpy()}, index=bar_index)

    # 7d rolling z-score: window of 2016 bars (= 7d × 288 bars/day at 5min)
    win_7d = 2016
    rolling_mean = df["funding_rate"].rolling(win_7d, min_periods=288).mean()
    rolling_std = df["funding_rate"].rolling(win_7d, min_periods=288).std()
    df["funding_rate_z_7d"] = ((df["funding_rate"] - rolling_mean)
                                  / rolling_std.replace(0, np.nan))

    # 1d change (3 funding publications back ≈ 1 day)
    df["funding_rate_1d_change"] = df["funding_rate"] - df["funding_rate"].shift(288)

    # Streaks
    rate_arr = df["funding_rate"].to_numpy()
    streak_pos = np.zeros(len(rate_arr), dtype=int)
    streak_neg = np.zeros(len(rate_arr), dtype=int)
    p = n = 0
    for i, r in enumerate(rate_arr):
        if pd.isna(r):
            p = n = 0
        elif r > 0:
            p = min(p + 1, 21); n = 0
        elif r < 0:
            n = min(n + 1, 21); p = 0
        else:
            p = n = 0
        streak_pos[i] = p
        streak_neg[i] = n
    df["funding_streak_pos"] = streak_pos
    df["funding_streak_neg"] = streak_neg
    return df


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_IN)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    funding_cols = ["funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
                     "funding_streak_pos", "funding_streak_neg"]
    for c in funding_cols:
        panel[c] = np.nan

    syms = sorted(panel["symbol"].unique())
    print(f"\nComputing funding features per symbol...", flush=True)
    for s in syms:
        t0 = time.time()
        sub_idx = panel.index[panel["symbol"] == s]
        sub = panel.loc[sub_idx].copy()
        # Convert open_time to DatetimeIndex
        if sub["open_time"].dtype.kind == "i":
            bar_index = pd.to_datetime(sub["open_time"], unit="ms", utc=True)
        else:
            bar_index = pd.to_datetime(sub["open_time"], utc=True)
        bar_index = pd.DatetimeIndex(bar_index.values, tz="UTC")
        funding_df = compute_funding_features(bar_index, s)
        # Assign back
        for c in funding_cols:
            panel.loc[sub_idx, c] = funding_df[c].to_numpy()
        n_nonnan = (~funding_df["funding_rate"].isna()).sum()
        print(f"  {s:<14} bars={len(sub):,} funding_nonnan={n_nonnan:,}  ({time.time()-t0:.0f}s)",
              flush=True)

    # Stats
    print(f"\nFunding feature stats (across all symbols, 51-name panel):", flush=True)
    for c in funding_cols:
        v = panel[c].dropna()
        print(f"  {c:<28}  n={len(v):>10,}  mean={v.mean():+.6f}  std={v.std():.6f}", flush=True)

    print(f"\nSaving...", flush=True)
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"  saved → {PANEL_OUT}", flush=True)


if __name__ == "__main__":
    main()
