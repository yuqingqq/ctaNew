"""Phase 1 EDA: empirical characterization of alpha-vs-BTC structure.

Computes β-adjusted residuals for FULL39 universe at multiple horizons and
analyzes:
  1. Per-name alpha quality   — Sharpe-if-perfect, autocorr, distribution
  2. Time-series structure    — autocorr at lags 1, 6, 12, 48, 288 (5min → 1d)
  3. Cross-sectional factor structure — PCA on residual matrix
  4. Beta stability           — β rolling time series, regime variability
  5. Regime dependence        — alpha conditional on BTC vol bucket, hour-of-day
  6. Tail behavior            — skew, kurtosis, fat-tail diagnostics

Output: Phase 1 EDA report (markdown + CSVs).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import list_universe, build_kline_features

OUT_DIR = REPO / "outputs/vBTC_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BTC_SYMBOL = "BTCUSDT"
BETA_WINDOW = 288    # 1d at 5-min cadence
HORIZONS = [12, 48, 288]   # 1h, 4h, 1d


def compute_per_name_alpha(panel_dict: dict, btc_close: pd.Series) -> pd.DataFrame:
    """For each name, compute β-adjusted alpha-vs-BTC at multiple horizons,
    plus per-bar β. Returns long-format DataFrame.
    """
    btc_ret = btc_close.pct_change()
    rows = []
    for s, f in panel_dict.items():
        if s == BTC_SYMBOL: continue
        my_close = f["close"].copy()
        my_close.index = pd.to_datetime(my_close.index, utc=True)
        my_ret = my_close.pct_change()
        joined = pd.DataFrame({"my_close": my_close, "my_ret": my_ret,
                                 "btc_close": btc_close.reindex(my_close.index, method="ffill"),
                                 "btc_ret": btc_ret.reindex(my_close.index, method="ffill")}).dropna()
        if len(joined) < BETA_WINDOW + 100: continue
        # Rolling β
        cov = (joined["my_ret"] * joined["btc_ret"]).rolling(BETA_WINDOW).mean() - \
              joined["my_ret"].rolling(BETA_WINDOW).mean() * joined["btc_ret"].rolling(BETA_WINDOW).mean()
        var = joined["btc_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        beta = (cov / var).clip(-5, 5).shift(1)

        # β-adjusted residuals at multiple horizons
        for h in HORIZONS:
            my_fwd = joined["my_close"].pct_change(h).shift(-h)
            btc_fwd = joined["btc_close"].pct_change(h).shift(-h)
            alpha_h = my_fwd - beta * btc_fwd
            df_h = pd.DataFrame({
                "open_time": joined.index, "symbol": s, "horizon": h,
                "beta": beta.values, "alpha": alpha_h.values,
                "my_fwd": my_fwd.values, "btc_fwd": btc_fwd.values,
            })
            rows.append(df_h)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    print("Loading panel data for FULL39 universe...")
    universe = sorted(list_universe(min_days=200))
    print(f"  {len(universe)} symbols available")

    feats = {s: build_kline_features(s) for s in universe}
    feats = {s: f for s, f in feats.items() if not f.empty}
    btc_close = feats[BTC_SYMBOL]["close"].copy()
    btc_close.index = pd.to_datetime(btc_close.index, utc=True)
    print(f"  BTC has {len(btc_close):,} 5-min bars over "
          f"{btc_close.index.min().date()} → {btc_close.index.max().date()}")

    print("\nComputing β-adjusted alpha-vs-BTC at horizons", HORIZONS, "...")
    alpha_df = compute_per_name_alpha(feats, btc_close)
    print(f"  Alpha rows: {len(alpha_df):,}")

    # ====== 1. PER-NAME QUALITY (using h=48 horizon, the strategy cadence) ======
    print("\n" + "=" * 100)
    print("1. PER-NAME ALPHA-VS-BTC QUALITY (horizon=48 bars = 4 hours)")
    print("=" * 100)
    h = 48
    df_h = alpha_df[alpha_df["horizon"] == h].dropna(subset=["alpha", "beta"])

    per_name = []
    for s, g in df_h.groupby("symbol"):
        g = g.dropna(subset=["alpha"])
        if len(g) < 1000: continue
        # Sample every h bars for non-overlapping cycles
        sub = g.iloc[::h]
        a = sub["alpha"].dropna().values
        if len(a) < 30: continue
        ann_factor = np.sqrt(365 * 24 / 4)   # 4-hour cycles annualized
        # Sharpe-if-perfect (oracle): the realized stdev of |alpha| as upper bound for Sharpe
        # Actually: per-name autocorrelation tells us if alpha is predictable by simple persistence
        ac1 = float(pd.Series(a).autocorr(lag=1)) if len(a) > 2 else 0.0
        ac6 = float(pd.Series(a).autocorr(lag=6)) if len(a) > 7 else 0.0
        per_name.append({
            "symbol": s, "n_cycles": len(a),
            "alpha_mean_bps": a.mean() * 1e4,
            "alpha_std_bps": a.std() * 1e4,
            "alpha_skew": float(pd.Series(a).skew()),
            "alpha_kurt": float(pd.Series(a).kurtosis()),
            "abs_mean_bps": np.abs(a).mean() * 1e4,
            "ac1": ac1, "ac6": ac6,
            "beta_mean": g["beta"].mean(),
            "beta_std": g["beta"].std(),
        })
    pn_df = pd.DataFrame(per_name).sort_values("abs_mean_bps", ascending=False)
    print(f"\n  {'symbol':<14} {'cyc':>4} {'mean_bps':>9} {'std_bps':>8} {'|μ|_bps':>9} "
          f"{'skew':>5} {'kurt':>5} {'AC1':>7} {'AC6':>7} {'β_mean':>7} {'β_std':>7}")
    for _, r in pn_df.iterrows():
        print(f"  {r['symbol']:<14} {r['n_cycles']:>4d} "
              f"{r['alpha_mean_bps']:>+9.2f} {r['alpha_std_bps']:>+8.2f} {r['abs_mean_bps']:>+9.2f} "
              f"{r['alpha_skew']:>+5.2f} {r['alpha_kurt']:>+5.1f} "
              f"{r['ac1']:>+7.3f} {r['ac6']:>+7.3f} "
              f"{r['beta_mean']:>+7.2f} {r['beta_std']:>+7.2f}")
    print(f"\n  Universe-wide stats:")
    print(f"    Mean per-name |α| (bps/cycle):  {pn_df['abs_mean_bps'].mean():+.2f}")
    print(f"    Std of α across names:          {pn_df['alpha_std_bps'].mean():+.2f}")
    print(f"    Mean β to BTC:                  {pn_df['beta_mean'].mean():.2f}  (range [{pn_df['beta_mean'].min():.2f}, {pn_df['beta_mean'].max():.2f}])")
    print(f"    β std (regime variability):     {pn_df['beta_std'].mean():.3f}")

    # ====== 2. AUTOCORRELATION STRUCTURE ======
    print("\n" + "=" * 100)
    print("2. AUTOCORRELATION STRUCTURE  (mean across names)")
    print("=" * 100)
    print(f"  {'horizon':<10} {'AC1':>8} {'AC2':>8} {'AC6':>8} {'AC12':>8} {'AC48':>8}")
    for h in HORIZONS:
        df_h = alpha_df[alpha_df["horizon"] == h].dropna(subset=["alpha"])
        ac_list = {1: [], 2: [], 6: [], 12: [], 48: []}
        for s, g in df_h.groupby("symbol"):
            sub = g.iloc[::h]
            a = sub["alpha"].dropna().values
            if len(a) < 50: continue
            ser = pd.Series(a)
            for lag in ac_list:
                if len(ser) > lag + 1:
                    ac_list[lag].append(ser.autocorr(lag=lag))
        means = {lag: np.mean([x for x in xs if not np.isnan(x)]) for lag, xs in ac_list.items()}
        print(f"  h={h:<8} {means[1]:>+8.3f} {means[2]:>+8.3f} {means[6]:>+8.3f} "
              f"{means[12]:>+8.3f} {means[48]:>+8.3f}")
    print(f"\n  Interpretation:")
    print(f"    AC1 ≈ 0:           no momentum/reversal at next-cycle scale")
    print(f"    AC1 > +0.05:       momentum (alpha persists)")
    print(f"    AC1 < -0.05:       reversal (alpha mean-reverts)")
    print(f"    AC6, AC12, AC48:   how fast alpha mean-reverts")

    # ====== 3. CROSS-SECTIONAL FACTOR STRUCTURE (PCA) ======
    print("\n" + "=" * 100)
    print("3. CROSS-SECTIONAL FACTOR STRUCTURE (PCA on h=48 alpha matrix)")
    print("=" * 100)
    h = 48
    df_h = alpha_df[alpha_df["horizon"] == h].dropna(subset=["alpha"])
    # Pivot: rows = time, cols = symbol
    pivot = df_h.pivot_table(index="open_time", columns="symbol", values="alpha", aggfunc="first")
    pivot = pivot.iloc[::h].dropna(thresh=int(0.7 * len(pivot.columns)))   # cycles with >70% sym coverage
    pivot = pivot.fillna(0)
    print(f"  Matrix shape: {pivot.shape}  (cycles × symbols)")
    if pivot.shape[0] > 50 and pivot.shape[1] > 5:
        # Standardize per column
        std = pivot.std()
        std[std == 0] = 1
        normed = (pivot - pivot.mean()) / std
        # Eigendecomposition of correlation matrix
        corr = normed.corr().fillna(0)
        eigvals = np.sort(np.linalg.eigvalsh(corr.values))[::-1]
        eigvals = eigvals[eigvals > 0]
        evr = eigvals / eigvals.sum() * 100
        cum = np.cumsum(evr)
        print(f"\n  Top eigenvalues (% variance explained):")
        for i, (v, c) in enumerate(zip(evr[:10], cum[:10]), 1):
            print(f"    PC{i}: {v:>5.1f}%  (cumulative: {c:>5.1f}%)")
        n_for_80 = int(np.argmax(cum >= 80) + 1) if (cum >= 80).any() else len(cum)
        print(f"\n  → {n_for_80} factors explain ≥80% of cross-sectional variance")
        print(f"  → idiosyncratic share (after factors): {100 - cum[n_for_80-1]:.1f}%")

    # ====== 4. REGIME DEPENDENCE: BTC volatility ======
    print("\n" + "=" * 100)
    print("4. REGIME DEPENDENCE — α conditional on BTC realized volatility")
    print("=" * 100)
    btc_returns_4h = btc_close.resample("4h").last().pct_change().dropna()
    btc_vol_30d = btc_returns_4h.rolling(180).std() * np.sqrt(365 * 24 / 4)   # annualized
    btc_vol_df = pd.DataFrame({"open_time_4h": btc_vol_30d.index, "btc_vol": btc_vol_30d.values}).dropna()
    df_h = alpha_df[alpha_df["horizon"] == 48].dropna(subset=["alpha"]).copy()
    df_h["open_time_4h"] = pd.to_datetime(df_h["open_time"], utc=True).dt.floor("4h")
    df_h = df_h.merge(btc_vol_df, on="open_time_4h", how="left")
    df_h = df_h.dropna(subset=["btc_vol"])
    df_h["vol_quartile"] = pd.qcut(df_h["btc_vol"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    print(f"  {'BTC vol quartile':<18} {'mean |α|_bps':>12} {'σ(α)_bps':>10} "
          f"{'mean β':>8} {'σ(β)':>7}")
    for q, g in df_h.groupby("vol_quartile"):
        print(f"  {str(q):<18} "
              f"{g['alpha'].abs().mean()*1e4:>+12.2f} "
              f"{g['alpha'].std()*1e4:>+10.2f} "
              f"{g['beta'].mean():>+8.2f} "
              f"{g['beta'].std():>+7.2f}")

    # ====== 5. HOUR-OF-DAY EFFECTS ======
    print("\n" + "=" * 100)
    print("5. HOUR-OF-DAY pattern (h=48 alpha, sampled every 4h)")
    print("=" * 100)
    df_h = alpha_df[alpha_df["horizon"] == 48].dropna(subset=["alpha"])
    df_h["hour"] = pd.to_datetime(df_h["open_time"], utc=True).dt.hour
    print(f"  {'hour':<5} {'mean |α|_bps':>12} {'σ(α)_bps':>10} {'cycles':>8}")
    for hr, g in df_h.groupby("hour"):
        if len(g) < 100: continue
        print(f"  {hr:<5d} {g['alpha'].abs().mean()*1e4:>+12.2f} "
              f"{g['alpha'].std()*1e4:>+10.2f} {len(g):>8}")

    # Save per-name table
    pn_df.to_csv(OUT_DIR / "per_name_quality.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
