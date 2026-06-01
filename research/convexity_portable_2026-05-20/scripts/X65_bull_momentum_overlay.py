"""X65 — Bull-momentum overlay: switch to long-only momentum in bull regime.

When BTC 30d return > threshold (BULL):
  - Set V5_minus_v3_7cx predictions to a momentum signal:
    pred_bull = z-score of sym's 30d return (so K=3 picks become top-3 momentum, bottom-3 mean-reversion)
  - OR simple: pred_bull = sym's 30d return (rank ordering)

In non-bull regime:
  - Keep V5_minus_v3_7cx predictions as is

Sweep: threshold {0.10, 0.15, 0.20} × overlay strategy {momentum_30d, momentum_7d}
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
KLINES = REPO / "data/ml/test/parquet/klines"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def load_30d_returns_per_sym(syms):
    """Load 30d returns per sym (lookback)."""
    out = []
    for sym in syms:
        sd = KLINES / sym / "5m"
        if not sd.exists(): continue
        dfs = [pd.read_parquet(f, columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
        df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time").sort_index()
        ret_30d = (df["close"] / df["close"].shift(8640) - 1).astype(np.float32).shift(1)  # PIT
        ret_7d = (df["close"] / df["close"].shift(2016) - 1).astype(np.float32).shift(1)
        sym_df = pd.DataFrame({
            "symbol": sym,
            "open_time": df.index,
            "ret_30d": ret_30d.values,
            "ret_7d": ret_7d.values,
        })
        out.append(sym_df)
    return pd.concat(out, ignore_index=True)


def main():
    print("=== X65 bull-momentum overlay ===\n")
    # Load BTC 30d return for regime detection
    btc_files = sorted((KLINES / "BTCUSDT" / "5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in btc_files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float32)
    btc_30d_ret = (btc / btc.shift(8640) - 1).astype(np.float32)

    # Load V5_minus_v3_7cx predictions on HL-50 (champion)
    preds = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    syms = sorted(preds["symbol"].unique())
    print(f"HL-50 syms: {len(syms)}")

    # Compute per-sym 30d/7d returns
    print("Loading per-sym 30d/7d returns...")
    sym_rets = load_30d_returns_per_sym(syms)
    sym_rets["open_time"] = pd.to_datetime(sym_rets["open_time"], utc=True)
    print(f"  sym_rets: {len(sym_rets):,} rows")

    # Merge
    btc_df = btc_30d_ret.to_frame("btc_30d_ret").reset_index()
    btc_df["open_time"] = pd.to_datetime(btc_df["open_time"], utc=True)
    m = preds.merge(btc_df, on="open_time", how="left")
    m = m.merge(sym_rets, on=["symbol","open_time"], how="left")
    print(f"  merged: {len(m):,} rows")

    # Baseline
    m_base = x6.run_sleeve_on_preds(CACHE / "x54_V5_minus_v3_7cx_preds.parquet",
                                      "x65_baseline")
    base_sh = m_base.get("sharpe", 0) or 0
    print(f"\nBaseline V5_minus_v3_7cx HL-50: Sharpe={base_sh:+.2f}")
    print(f"\n{'threshold':<12} {'overlay':<14} {'Sharpe':>8} {'folds+':>8} {'conc':>6}")

    results = []
    for thr in [0.10, 0.15, 0.20]:
        for overlay_name, overlay_col in [("mom_30d", "ret_30d"), ("mom_7d", "ret_7d")]:
            m_overlay = m.copy()
            in_bull = m_overlay["btc_30d_ret"] > thr
            # In bull: replace pred with cross-sectional z-score of overlay_col
            # Group by open_time, compute z-score, set as new pred
            grp = m_overlay.groupby("open_time")[overlay_col]
            z = grp.transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
            m_overlay.loc[in_bull, "pred"] = z[in_bull].astype(np.float32).fillna(0)
            apd = m_overlay[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]].copy()
            tmp = CACHE / f"x65_thr{thr}_{overlay_name}_preds.parquet"
            apd.to_parquet(tmp, index=False)
            mm = x6.run_sleeve_on_preds(tmp, f"x65_thr{thr}_{overlay_name}")
            sh = mm.get("sharpe", 0) or 0
            fp = mm.get("folds_pos", "?")
            conc = mm.get("concentration", "?")
            print(f"  {thr:<12.2f} {overlay_name:<14} {sh:>+8.2f} {str(fp):>8} {str(conc):>6}", flush=True)
            results.append({"threshold": thr, "overlay": overlay_name,
                             "sharpe": sh, "delta_vs_base": sh - base_sh,
                             "folds_pos": fp, "concentration": conc})

    import csv
    out_csv = OUT / "X65_bull_momentum_overlay.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["threshold","overlay","sharpe","delta_vs_base","folds_pos","concentration"])
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved → {out_csv}")
    print(f"\nReference: V5_minus_v3_7cx HL-50 = +1.74, X64 best bull-skip = ?")


if __name__ == "__main__":
    main()
