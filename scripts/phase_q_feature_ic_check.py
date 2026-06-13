"""Phase Q-0 preliminary: IC sanity check.

Before committing to a full LGBM retrain, compute the IC of the 2 new candidate
features (ethbtc_change_24h, xs_ret_disp_1d) against target_A on the existing
panel. If IC is decent (|IC| > 0.005), proceed with full WINNER_23 retrain. If
near zero, the cohort-PnL spread was driven by something other than predictive
power on this target.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def load_eth_btc_ratio():
    bdir = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"
    edir = REPO / "data/ml/test/parquet/klines/ETHUSDT/5m"
    bf = sorted(bdir.glob("*.parquet"))
    ef = sorted(edir.glob("*.parquet"))
    bdfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in bf]
    edfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in ef]
    btc = pd.concat(bdfs, ignore_index=True).rename(columns={"close": "btc_close"})
    eth = pd.concat(edfs, ignore_index=True).rename(columns={"close": "eth_close"})
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    eth["open_time"] = pd.to_datetime(eth["open_time"], utc=True)
    btc = btc.drop_duplicates("open_time")
    eth = eth.drop_duplicates("open_time")
    df = btc.merge(eth, on="open_time", how="inner").sort_values("open_time")
    df["ethbtc"] = df["eth_close"] / df["btc_close"]
    df["ethbtc_change_24h"] = df["ethbtc"].pct_change(288)
    return df[["open_time", "ethbtc_change_24h"]]


def main():
    print("=== Phase Q-0: WINNER_23 feature IC sanity check ===\n", flush=True)
    print("  loading panel...", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                              columns=["open_time", "symbol", "target_A", "return_1d"])
    print(f"  panel {len(panel):,} rows ({time.time()-t0:.0f}s)\n", flush=True)

    # Compute xs_ret_disp_1d (cross-symbol std of return_1d at each timestamp)
    print(f"  computing xs_ret_disp_1d...", flush=True)
    t0 = time.time()
    xs_disp = panel.groupby("open_time")["return_1d"].std().reset_index()
    xs_disp.columns = ["open_time", "xs_ret_disp_1d"]
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Load ETH/BTC ratio
    print(f"  loading ETH/BTC ratio...", flush=True)
    t0 = time.time()
    ethbtc = load_eth_btc_ratio()
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Attach to panel
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel.merge(xs_disp, on="open_time", how="left")
    panel = panel.merge(ethbtc, on="open_time", how="left")

    # IC computation: rank corr per timestamp, then average
    print(f"\n  computing IC vs target_A per cross-section, then averaging...", flush=True)
    valid = panel.dropna(subset=["target_A"])
    print(f"  {len(valid):,} valid rows", flush=True)

    # Filter to one sample per timestamp for cross-asset / xs-disp features
    # (since they're broadcast same-value across symbols at each timestamp)
    per_t = valid.drop_duplicates("open_time")[
        ["open_time", "ethbtc_change_24h", "xs_ret_disp_1d"]]
    print(f"  unique timestamps: {len(per_t):,}", flush=True)

    # Time-series IC (linear correlation with cross-section-averaged target_A)
    ts_target = valid.groupby("open_time")["target_A"].mean().reset_index()
    ts_target.columns = ["open_time", "target_A_mean"]
    ts_target = ts_target.merge(per_t, on="open_time", how="inner").dropna()
    print(f"\n  time-series IC (cross-section-mean target_A vs scalar feature):", flush=True)
    for col in ["ethbtc_change_24h", "xs_ret_disp_1d"]:
        sub = ts_target.dropna(subset=[col])
        if len(sub) < 100:
            print(f"    {col}: too few ({len(sub)})")
            continue
        spearman = sub["target_A_mean"].rank().corr(sub[col].rank())
        pearson = sub["target_A_mean"].corr(sub[col])
        print(f"    {col:>22}:  Spearman {spearman:+.4f}   Pearson {pearson:+.4f}  (n={len(sub)})",
              flush=True)

    # Per-symbol IC (more relevant for the LGBM model since it learns per-symbol)
    print(f"\n  per-symbol time-series IC (mean across 51 symbols):", flush=True)
    for col in ["ethbtc_change_24h", "xs_ret_disp_1d"]:
        ic_per_sym = []
        for sym, g in valid.groupby("symbol"):
            sub = g.dropna(subset=[col, "target_A"])
            if len(sub) < 1000: continue
            ic_per_sym.append(sub[col].rank().corr(sub["target_A"].rank()))
        ic_arr = np.array(ic_per_sym)
        print(f"    {col:>22}:  mean IC {ic_arr.mean():+.4f}   "
              f"std {ic_arr.std():.4f}   "
              f"frac_positive {(ic_arr > 0).mean()*100:.0f}%   "
              f"(n_syms={len(ic_arr)})", flush=True)

    # Compare against a known WINNER_21 feature for benchmark
    print(f"\n  benchmark: existing WINNER_21 features (per-symbol IC mean):", flush=True)
    panel_full = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                    columns=["symbol", "target_A", "funding_rate",
                                              "corr_to_btc_1d", "atr_pct", "return_1d"])
    for col in ["funding_rate", "corr_to_btc_1d", "atr_pct", "return_1d"]:
        if col not in panel_full.columns: continue
        ic_per_sym = []
        for sym, g in panel_full.dropna(subset=[col, "target_A"]).groupby("symbol"):
            if len(g) < 1000: continue
            ic_per_sym.append(g[col].rank().corr(g["target_A"].rank()))
        ic_arr = np.array(ic_per_sym)
        print(f"    {col:>22}:  mean IC {ic_arr.mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
