"""Raw-data exploration to find potential insights about why performance varies.

Looks at:
  1. Per-fold realized alpha distribution (mean, std, skew, kurt)
  2. Per-fold cross-sectional dispersion at each timestamp
  3. Cross-fold autocorrelation of returns
  4. Per-fold volume profile per symbol
  5. Realized alpha vs forward return scatter (any obvious patterns?)
  6. Alpha autocorrelation per symbol (is alpha persistent or mean-reverting?)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
OUT_DIR = REPO / "outputs/vBTC_raw_insights"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC
PROD_FOLDS = [5, 6, 7, 8, 9]


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms\n", flush=True)

    folds_all = _multi_oos_splits(panel)

    # === 1. Per-fold realized alpha (alpha_A) statistics ===
    print(f"=== 1. Per-fold realized alpha (alpha_A = my_fwd - basket_A_fwd) statistics ===",
          flush=True)
    print(f"  fold |  split  |    mean    |   std   |  skew   |  kurt  |  n_obs", flush=True)
    print(f"  {'-' * 70}", flush=True)
    for fid in range(len(folds_all)):
        train, cal, test = _slice(panel, folds_all[fid])
        for label, df in [("train", train), ("cal", cal), ("test", test)]:
            a = df["alpha_A"].dropna()
            if len(a) < 100: continue
            print(f"  {fid:>4} | {label:<6}  | {a.mean():+10.6f} | {a.std():.5f} | "
                  f"{a.skew():+.3f} | {a.kurt():+.3f} | {len(a):>8,}", flush=True)
        print()

    # === 2. Cross-sectional dispersion per timestamp per fold ===
    print(f"=== 2. Cross-sectional realized alpha dispersion per fold ===", flush=True)
    print(f"  At each timestamp, std of alpha_A across all symbols. ", flush=True)
    print(f"  fold | mean_disp | std_disp | min_disp | max_disp ", flush=True)
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        _, _, test = _slice(panel, folds_all[fid])
        disp_per_t = test.groupby("open_time")["alpha_A"].std()
        print(f"  {fid:>4} | {disp_per_t.mean():.5f} | {disp_per_t.std():.5f} | "
              f"{disp_per_t.min():.5f} | {disp_per_t.max():.5f}", flush=True)

    # === 3. Per-symbol alpha autocorrelation (is alpha persistent?) ===
    print(f"\n=== 3. Per-symbol alpha autocorrelation (lag=1, lag=12, lag=288) ===", flush=True)
    print(f"  AC1: 5-min autocorr; AC12: 1h autocorr; AC288: 1d autocorr", flush=True)
    print(f"  Strong AC suggests momentum/persistence. Weak AC = white noise alpha.", flush=True)
    syms = sorted(panel["symbol"].unique())
    rows = []
    for s in syms:
        sub = panel[panel["symbol"] == s].sort_values("open_time")
        a = sub["alpha_A"].dropna()
        if len(a) < 1000: continue
        ac1 = a.autocorr(lag=1)
        ac12 = a.autocorr(lag=12)
        ac288 = a.autocorr(lag=288)
        rows.append({"symbol": s, "ac1": ac1, "ac12": ac12, "ac288": ac288, "n": len(a)})
    df_ac = pd.DataFrame(rows).sort_values("ac1", ascending=False)
    print(f"  {'symbol':<14} {'AC1':>7} {'AC12':>7} {'AC288':>7}", flush=True)
    print(f"  Top 10 by AC1:", flush=True)
    for _, r in df_ac.head(10).iterrows():
        print(f"  {r['symbol']:<14} {r['ac1']:>+7.4f} {r['ac12']:>+7.4f} {r['ac288']:>+7.4f}",
              flush=True)
    print(f"  Bottom 10 by AC1:", flush=True)
    for _, r in df_ac.tail(10).iterrows():
        print(f"  {r['symbol']:<14} {r['ac1']:>+7.4f} {r['ac12']:>+7.4f} {r['ac288']:>+7.4f}",
              flush=True)

    print(f"\n  Median across symbols:  AC1={df_ac['ac1'].median():+.4f}  "
          f"AC12={df_ac['ac12'].median():+.4f}  AC288={df_ac['ac288'].median():+.4f}", flush=True)

    # === 4. Volume profile change per fold ===
    print(f"\n=== 4. Volume per fold (median daily volume_ma_50) per symbol ===", flush=True)
    syms_v = ["BTCUSDT", "ETHUSDT", "FILUSDT", "DOTUSDT", "ICPUSDT", "TAOUSDT", "HYPEUSDT", "ASTERUSDT"]
    print(f"  symbol         | f5      | f6      | f7      | f8      | f9", flush=True)
    for s in syms_v:
        cells = [s.ljust(14)]
        for fid in PROD_FOLDS:
            if fid >= len(folds_all): cells.append("n/a")
            else:
                _, _, test = _slice(panel, folds_all[fid])
                sub = test[test["symbol"] == s]
                if sub.empty:
                    cells.append("n/a")
                else:
                    med = sub["volume_ma_50"].dropna().median()
                    cells.append(f"{med:>7.0f}" if not pd.isna(med) else "n/a")
        print(f"  {' | '.join(cells)}", flush=True)

    # === 5. Conv_gate behavior per fold: how often does it trigger? ===
    print(f"\n=== 5. Cross-sectional dispersion of alpha_A by fold (low → conv_gate skips) ===",
          flush=True)
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        _, _, test = _slice(panel, folds_all[fid])
        # autocorr_pctile_7d filters cycles. how often is it below threshold?
        cycles_total = len(test["open_time"].unique())
        # check how many timestamps have full universe present
        ts_disp = test.groupby("open_time")["alpha_A"].agg(["std", "count"])
        ts_disp = ts_disp[ts_disp["count"] >= 30]   # need most names present
        disp_pctiles = ts_disp["std"].quantile([0.1, 0.3, 0.5, 0.7, 0.9])
        print(f"  fold {fid}: dispersion percentiles  10%={disp_pctiles.iloc[0]:.5f}  "
              f"30%={disp_pctiles.iloc[1]:.5f}  50%={disp_pctiles.iloc[2]:.5f}  "
              f"70%={disp_pctiles.iloc[3]:.5f}  90%={disp_pctiles.iloc[4]:.5f}",
              flush=True)

    df_ac.to_csv(OUT_DIR / "per_symbol_autocorr.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
