"""Characterize the recent regime — what's actually different?

For each calendar month from May 2025 to April 2026, compute:
  - Mean pairwise correlation across 25 symbols (high → less differentiation)
  - Basket realized vol (low → markets quiet, less alpha)
  - Cross-sectional dispersion (std of returns across symbols)
  - BTC absolute return (large moves → BTC-driven regime)
  - Mean |funding rate| (high → positioning extremes)
  - 99th-pctile cross-sectional spread (top-K vs bot-K realized return gap)

Compare bad months (high dispersion-fail) to good months.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import (
    build_kline_features, list_universe, build_basket,
)
from data_collectors.metrics_loader import fetch_metrics  # for stuff if needed

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
CACHE_DIR = REPO / "data/ml/cache"


def load_funding(s):
    p = CACHE_DIR / f"funding_{s}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p).set_index("calc_time")["funding_rate"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[~df.index.duplicated(keep="last")].sort_index()


def main():
    universe = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe if s not in NEW_SYMBOLS])
    print(f"Loading {len(orig25)} ORIG25 symbols...", flush=True)

    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()

    # Resample to 4-hour bars (h=48 cadence)
    closes_4h = closes.resample("4h").last().ffill()
    rets_4h = closes_4h.pct_change()
    log_rets_4h = np.log(closes_4h / closes_4h.shift(1))

    # Daily returns for stability
    closes_d = closes.resample("1D").last().ffill()
    rets_d = closes_d.pct_change()

    # Build per-month regime characterization
    months = pd.date_range("2025-05-01", "2026-05-01", freq="MS", tz="UTC")
    print(f"Months to characterize: {len(months) - 1}", flush=True)

    regime_data = []
    for i in range(len(months) - 1):
        m_start, m_end = months[i], months[i + 1]
        # Daily returns within month
        d_mask = (rets_d.index >= m_start) & (rets_d.index < m_end)
        d_window = rets_d[d_mask].dropna(how="all")
        # 4h returns within month
        h_mask = (rets_4h.index >= m_start) & (rets_4h.index < m_end)
        h_window = rets_4h[h_mask].dropna(how="all")

        if d_window.empty or h_window.empty:
            continue

        # 1. Mean pairwise correlation (daily returns)
        corr_matrix = d_window.corr().values
        upper_triu = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        mean_corr = np.nanmean(upper_triu)

        # 2. Basket realized vol (4h returns, annualized to 365)
        basket_h = h_window.mean(axis=1)
        basket_vol = basket_h.std() * np.sqrt(2190)  # annualized at h=48 cadence

        # 3. Cross-sectional dispersion: std of monthly returns across symbols
        monthly_ret_per_sym = (1 + d_window).prod() - 1
        cs_dispersion = monthly_ret_per_sym.std()

        # 4. BTC monthly return magnitude
        btc_ret = monthly_ret_per_sym.get("BTCUSDT", np.nan)

        # 5. Cross-sectional realized spread per 4h cycle (top-7 vs bot-7)
        # For each 4h bar, rank symbols by RETURN (not pred), compute top7 - bot7
        per_bar_spread = []
        for t, row in h_window.iterrows():
            row = row.dropna()
            if len(row) < 14:
                continue
            sorted_r = row.sort_values()
            spread = sorted_r.iloc[-7:].mean() - sorted_r.iloc[:7].mean()
            per_bar_spread.append(spread)
        if per_bar_spread:
            mean_realized_spread_bps = np.mean(per_bar_spread) * 1e4
            std_realized_spread_bps = np.std(per_bar_spread) * 1e4
        else:
            mean_realized_spread_bps = std_realized_spread_bps = np.nan

        # 6. Mean |funding rate| across universe
        all_fund = []
        for s in orig25:
            f = load_funding(s)
            if f is None: continue
            f_in_month = f[(f.index >= m_start) & (f.index < m_end)]
            if not f_in_month.empty:
                all_fund.append(f_in_month.abs().mean())
        mean_abs_funding = np.mean(all_fund) if all_fund else np.nan

        regime_data.append({
            "month": m_start.date().isoformat(),
            "n_days": len(d_window),
            "mean_pair_corr": mean_corr,
            "basket_vol_annualized": basket_vol,
            "cs_dispersion_monthly": cs_dispersion,
            "btc_return_monthly": btc_ret,
            "mean_realized_spread_4h_bps": mean_realized_spread_bps,
            "std_realized_spread_4h_bps": std_realized_spread_bps,
            "mean_abs_funding_monthly": mean_abs_funding,
        })

    df = pd.DataFrame(regime_data)
    print("\n" + "=" * 130, flush=True)
    print("REGIME CHARACTERIZATION", flush=True)
    print("=" * 130, flush=True)
    print(f"  {'month':>12} {'corr':>6} {'bk_vol':>7} {'cs_disp':>8} "
          f"{'btc_ret':>8} {'realized_spread':>16} {'spread_std':>11} {'abs_fund':>10}", flush=True)
    for _, r in df.iterrows():
        print(f"  {r['month']:>12} {r['mean_pair_corr']:>+5.3f} "
              f"{r['basket_vol_annualized']:>6.3f} {r['cs_dispersion_monthly']:>7.3f} "
              f"{r['btc_return_monthly']:>+7.3f} {r['mean_realized_spread_4h_bps']:>+15.1f} "
              f"{r['std_realized_spread_4h_bps']:>10.1f} {r['mean_abs_funding_monthly']*100:>+9.4f}%", flush=True)

    # Compare bad months (multi-OOS bad: fold 3=Sep, 6=Dec/Jan, 9=Apr) to good (fold 2=Aug-Sep, 4=Oct-Nov, 7=Jan-Feb)
    bad_months = ["2025-09-01", "2025-12-01", "2026-04-01"]
    good_months = ["2025-08-01", "2025-10-01", "2026-01-01"]

    bad_df = df[df["month"].isin(bad_months)]
    good_df = df[df["month"].isin(good_months)]

    print(f"\n  --- BAD vs GOOD month comparison ---", flush=True)
    print(f"  Bad months (multi-OOS Sharpe negative): {bad_months}", flush=True)
    print(f"  Good months (multi-OOS Sharpe positive): {good_months}", flush=True)
    if not bad_df.empty and not good_df.empty:
        print(f"\n  metric                          BAD mean       GOOD mean      Δ")
        for col in ["mean_pair_corr", "basket_vol_annualized", "cs_dispersion_monthly",
                     "mean_realized_spread_4h_bps", "std_realized_spread_4h_bps",
                     "mean_abs_funding_monthly"]:
            b = bad_df[col].mean()
            g = good_df[col].mean()
            print(f"  {col:<32} {b:>+10.4f}    {g:>+10.4f}    {b-g:>+8.4f}", flush=True)

    # Compare RECENT (last 3 months) to overall median
    recent_df = df.tail(3)
    print(f"\n  --- RECENT 3 months vs FULL period median ---", flush=True)
    print(f"  metric                          RECENT mean    full median    Δ")
    for col in ["mean_pair_corr", "basket_vol_annualized", "cs_dispersion_monthly",
                 "mean_realized_spread_4h_bps", "std_realized_spread_4h_bps",
                 "mean_abs_funding_monthly"]:
        r = recent_df[col].mean()
        m = df[col].median()
        print(f"  {col:<32} {r:>+10.4f}    {m:>+10.4f}    {r-m:>+8.4f}", flush=True)

    out = REPO / "outputs/h48_regime_characterize"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "regime_per_month.csv", index=False)
    print(f"\n  saved → {out}", flush=True)


if __name__ == "__main__":
    main()
