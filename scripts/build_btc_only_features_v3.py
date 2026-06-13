"""Build v3 BTC-only feature panel — 39 candidate features for the BTC-residual
hybrid model. Per docs/vBTC_V3_FEATURE_PLAN.md.

Sources:
- outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet — base panel
  (alpha_beta, return_pct, funding_rate, microstructure, process fingerprint).
- data/ml/test/parquet/klines/{SYM}/5m/*.parquet — for close + volume series.

PIT discipline:
- All multi-day stats computed at 1d granularity, then shift(1) at the daily
  level before forward-filling to 5m bars. So at time t (any 5m bar within
  day D), features use stats computed through day D-1 only.
- 4h-window microstructure features pass through from the source panel
  (already PIT-safe per the original cross_sectional pipeline).

Output: outputs/vBTC_features_btc_v3/panel_v3.parquet
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SRC_PANEL = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_features_btc_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PANEL = OUT_DIR / "panel_v3.parquet"


# Feature blocks per plan
# (A) Liquidity / tradability — 6
A_LIQ = ["log_dollar_volume_7d", "log_dollar_volume_30d", "volume_stability_30d",
         "amihud_illiq_30d", "roll_spread_proxy_30d", "turnover_volatility_30d"]
# (B) BTC relationship — 7
B_BTC = ["beta_btc_30d", "beta_btc_90d", "beta_btc_180d", "beta_btc_instability",
         "corr_btc_30d", "corr_btc_90d", "corr_breakdown"]
# (C) Residual behavior — 9
C_RES = ["resid_vol_7d", "resid_vol_30d", "resid_vol_90d",
         "resid_skew_30d", "resid_kurt_30d", "resid_jump_count_30d",
         "resid_autocorr_1d", "resid_reversal_score_7d", "resid_trend_score_30d"]
# (D) Trend / anchoring — 5
D_TRD = ["dist_from_30d_high", "dist_from_90d_high", "dist_from_365d_high",
         "multi_horizon_trend_score", "volume_confirmed_trend_score"]
# (E) Funding crowding — 6
E_FND = ["funding_mean_7d", "funding_mean_30d", "funding_z_30d",
         "funding_persistence_7d", "funding_abs_30d", "funding_sign_streak"]
# (F) Microstructure — 4 (passthrough + derived)
F_MIC = ["aggr_ratio_4h", "signed_volume_4h_z", "tfi_4h", "avg_trade_size_4h_z"]
# (G) Process fingerprint — 3 (passthrough)
G_PRC = ["idio_skew_1d", "idio_kurt_1d", "idio_max_abs_12b"]

V3_ALL = A_LIQ + B_BTC + C_RES + D_TRD + E_FND + F_MIC + G_PRC
assert len(V3_ALL) == 40, f"expected 40, got {len(V3_ALL)}"  # 6+7+9+5+6+4+3 = 40
# Plan says 39 — funding_sign_streak is partial in panel; we count it as one feature


def load_daily_klines(sym):
    """Aggregate 5m klines to 1d (UTC) close + dollar volume."""
    d = KLINES_DIR / sym / "5m"
    if not d.exists():
        return None
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close", "quote_volume"])
                    for f in files], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["date"] = df["open_time"].dt.floor("1D")
    daily = df.groupby("date").agg(
        close=("close", "last"),
        dollar_volume=("quote_volume", "sum"),
    ).reset_index()
    daily["return_1d"] = daily["close"].pct_change()
    daily["symbol"] = sym
    return daily


def rolling_ols_beta(y, x, win, min_obs=None):
    """Rolling β = Cov(y, x) / Var(x) over `win` daily obs."""
    if min_obs is None:
        min_obs = max(10, win // 3)
    cov_yx = y.rolling(win, min_periods=min_obs).cov(x)
    var_x = x.rolling(win, min_periods=min_obs).var()
    return cov_yx / var_x.replace(0, np.nan)


def rolling_corr(y, x, win, min_obs=None):
    if min_obs is None:
        min_obs = max(10, win // 3)
    return y.rolling(win, min_periods=min_obs).corr(x)


def compute_daily_features(daily, btc_daily):
    """Compute v3 daily features per symbol. Input: daily frame with
    [date, close, dollar_volume, return_1d, symbol]. Returns same frame
    augmented with feature columns (PIT-shifted)."""
    d = daily.merge(btc_daily[["date", "btc_return_1d", "btc_close"]], on="date", how="left")
    d = d.sort_values("date").reset_index(drop=True)
    y = d["return_1d"]
    x = d["btc_return_1d"]

    # (B) BTC relationship — β / corr / instability / breakdown
    for win in (30, 90, 180):
        d[f"beta_btc_{win}d"] = rolling_ols_beta(y, x, win)
        if win < 180:
            d[f"corr_btc_{win}d"] = rolling_corr(y, x, win)
    d["beta_btc_instability"] = d["beta_btc_30d"] - d["beta_btc_180d"]
    d["corr_breakdown"] = d["corr_btc_30d"] - d["corr_btc_90d"]

    # idio_ret = residual after applying the 90d β (centered window for
    # daily residual stats; use 90d β since it's the "reference" stable β)
    d["idio_ret_1d"] = d["return_1d"] - d["beta_btc_90d"] * d["btc_return_1d"]

    # (C) Residual behavior — vol / skew / kurt / jump / autocorr / reversal / trend
    for win in (7, 30, 90):
        d[f"resid_vol_{win}d"] = d["idio_ret_1d"].rolling(win, min_periods=max(5, win//3)).std()
    d["resid_skew_30d"] = d["idio_ret_1d"].rolling(30, min_periods=15).skew()
    d["resid_kurt_30d"] = d["idio_ret_1d"].rolling(30, min_periods=15).kurt()
    # jump_count_30d: count of |idio| > 3 * trailing_30d_std
    sigma = d["idio_ret_1d"].rolling(30, min_periods=15).std()
    threshold = 3 * sigma
    d["resid_jump_count_30d"] = (
        (d["idio_ret_1d"].abs() > threshold).rolling(30, min_periods=15).sum()
    )
    # AR(1) of idio over 30d window: ad-hoc via rolling .apply (slow but small data)
    d["resid_autocorr_1d"] = d["idio_ret_1d"].rolling(30, min_periods=15).apply(
        lambda s: s.autocorr(lag=1) if s.notna().sum() >= 15 else np.nan, raw=False
    )
    # Reversal score: -idio_ret_7d / idio_vol_7d  (Blitz canonical)
    idio_7d = d["idio_ret_1d"].rolling(7, min_periods=5).sum()
    d["resid_reversal_score_7d"] = -idio_7d / d["resid_vol_7d"].replace(0, np.nan)
    # Trend score: idio_ret_30d / idio_vol_30d
    idio_30d = d["idio_ret_1d"].rolling(30, min_periods=15).sum()
    d["resid_trend_score_30d"] = idio_30d / d["resid_vol_30d"].replace(0, np.nan)
    # Multi-horizon trend score (used by D)
    idio_90d = d["idio_ret_1d"].rolling(90, min_periods=45).sum()
    trend_7 = (d["idio_ret_1d"].rolling(7, min_periods=5).sum()
               / d["resid_vol_7d"].replace(0, np.nan))
    trend_30 = d["resid_trend_score_30d"]
    trend_90 = idio_90d / d["resid_vol_90d"].replace(0, np.nan)
    d["multi_horizon_trend_score"] = (trend_7 + trend_30 + trend_90) / 3.0

    # (A) Liquidity
    d["log_dollar_volume_7d"] = np.log1p(d["dollar_volume"].rolling(7, min_periods=4).mean())
    d["log_dollar_volume_30d"] = np.log1p(d["dollar_volume"].rolling(30, min_periods=15).mean())
    d["volume_stability_30d"] = (
        d["dollar_volume"].rolling(30, min_periods=15).std()
        / d["dollar_volume"].rolling(30, min_periods=15).mean().replace(0, np.nan)
    )
    # Amihud: mean(|ret_1d| / dollar_volume) over 30d  (units: 1/$ — scaled for sanity)
    d["amihud_illiq_30d"] = (
        (d["return_1d"].abs() / d["dollar_volume"].replace(0, np.nan))
        .rolling(30, min_periods=15).mean() * 1e9  # rescale so feature is order-1
    )
    # Roll's spread proxy: 2 * sqrt(-cov(Δp_t, Δp_{t-1})) for negative cov
    dp = d["close"].diff()
    dp_lag = dp.shift(1)
    cov_dp = dp.rolling(30, min_periods=15).cov(dp_lag)
    d["roll_spread_proxy_30d"] = np.where(
        cov_dp < 0, 2 * np.sqrt(np.maximum(-cov_dp, 0)) / d["close"], 0.0
    )
    # Turnover (= dollar_volume / close * close = dollar_volume in $; use ratio of vol to its long-mean as a "turnover" proxy)
    turnover = d["dollar_volume"] / d["dollar_volume"].rolling(90, min_periods=30).mean()
    d["turnover_volatility_30d"] = turnover.rolling(30, min_periods=15).std()

    # (D) Trend / anchoring
    for win in (30, 90, 365):
        max_close = d["close"].rolling(win, min_periods=min(win, 30)).max()
        d[f"dist_from_{win}d_high"] = d["close"] / max_close - 1.0
    # Volume-confirmed trend: trend_30 * z(volume_30d)
    vol_mean = d["dollar_volume"].rolling(30, min_periods=15).mean()
    vol_std = d["dollar_volume"].rolling(30, min_periods=15).std()
    vol_z = (d["dollar_volume"] - vol_mean) / vol_std.replace(0, np.nan)
    d["volume_confirmed_trend_score"] = d["resid_trend_score_30d"] * vol_z

    # PIT discipline: shift all daily features by 1 day so they only use data
    # through prior day. Pass-through return_1d is what *generated* the features
    # but feature values themselves are next-day-known once shifted.
    new_feats_daily = A_LIQ + B_BTC + C_RES + D_TRD
    for c in new_feats_daily:
        d[c] = d[c].shift(1)

    return d[["symbol", "date"] + new_feats_daily]


def compute_funding_features(panel_sym):
    """Funding extensions per symbol on the source 5-min panel.

    funding_rate in Binance USDM perps is updated every 8h (288 bars / 3 = 96 bars).
    We treat funding_rate as a discrete signal: roll on the *unique values* per
    funding period, then forward-fill.
    """
    g = panel_sym.sort_values("open_time").copy()
    g["date"] = g["open_time"].dt.floor("1D")

    # Per-day funding stats: aggregate to day-end (last value within day)
    daily_fund = g.groupby("date")["funding_rate"].last().reset_index()
    daily_fund["abs_fund"] = daily_fund["funding_rate"].abs()
    daily_fund["sign"] = np.sign(daily_fund["funding_rate"])

    daily_fund["funding_mean_7d"] = daily_fund["funding_rate"].rolling(7, min_periods=4).mean()
    daily_fund["funding_mean_30d"] = daily_fund["funding_rate"].rolling(30, min_periods=15).mean()
    mu = daily_fund["funding_rate"].rolling(30, min_periods=15).mean()
    sd = daily_fund["funding_rate"].rolling(30, min_periods=15).std()
    daily_fund["funding_z_30d"] = (daily_fund["funding_rate"] - mu) / sd.replace(0, np.nan)
    daily_fund["funding_persistence_7d"] = (
        (daily_fund["funding_rate"] > 0).rolling(7, min_periods=4).mean()
    )
    daily_fund["funding_abs_30d"] = daily_fund["abs_fund"].rolling(30, min_periods=15).mean()
    # Sign-streak: current run-length of same-sign funding
    sign = daily_fund["sign"].fillna(0).astype(int)
    grp = (sign != sign.shift()).cumsum()
    daily_fund["funding_sign_streak"] = sign.groupby(grp).cumcount() + 1
    daily_fund["funding_sign_streak"] = daily_fund["funding_sign_streak"] * sign  # signed

    # PIT shift
    for c in E_FND:
        daily_fund[c] = daily_fund[c].shift(1)

    return daily_fund[["date"] + E_FND]


def compute_microstructure_zscores(panel_sym):
    """Z-scores for microstructure features that need normalization."""
    g = panel_sym.sort_values("open_time").copy()
    # signed_volume_4h_z: 7d trailing z on per-bar signed_volume_4h
    if "signed_volume_4h" in g.columns:
        # 7d × 288 bars/day = 2016 bars
        mu = g["signed_volume_4h"].rolling(2016, min_periods=288).mean()
        sd = g["signed_volume_4h"].rolling(2016, min_periods=288).std()
        g["signed_volume_4h_z"] = ((g["signed_volume_4h"] - mu) / sd.replace(0, np.nan)).shift(1)
    else:
        g["signed_volume_4h_z"] = np.nan
    if "avg_trade_size_4h" in g.columns:
        mu = g["avg_trade_size_4h"].rolling(2016, min_periods=288).mean()
        sd = g["avg_trade_size_4h"].rolling(2016, min_periods=288).std()
        g["avg_trade_size_4h_z"] = ((g["avg_trade_size_4h"] - mu) / sd.replace(0, np.nan)).shift(1)
    else:
        g["avg_trade_size_4h_z"] = np.nan
    return g[["open_time", "signed_volume_4h_z", "avg_trade_size_4h_z"]]


def main():
    print("=== Build v3 BTC-only feature panel (39 candidate features) ===\n", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(SRC_PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"Source panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols, "
          f"{panel.shape[1]} cols", flush=True)

    # Build BTC daily series first
    print("Loading BTC daily klines...", flush=True)
    btc_daily = load_daily_klines("BTCUSDT")
    btc_daily = btc_daily.rename(columns={"close": "btc_close", "return_1d": "btc_return_1d"})
    btc_daily = btc_daily.drop(columns=["dollar_volume", "symbol"])
    print(f"  BTC daily: {len(btc_daily)} days", flush=True)

    # Per-symbol daily features
    print("Computing daily features per symbol...", flush=True)
    all_daily = []
    syms = sorted(panel["symbol"].unique())
    for i, sym in enumerate(syms):
        ts = time.time()
        daily = load_daily_klines(sym)
        if daily is None or len(daily) < 30:
            print(f"  [{i+1:2d}/{len(syms)}] {sym}: SKIP (no klines)", flush=True)
            continue
        feat = compute_daily_features(daily, btc_daily)
        all_daily.append(feat)
        if (i + 1) % 10 == 0 or i == len(syms) - 1:
            print(f"  [{i+1:2d}/{len(syms)}] {sym}: {len(feat)} days ({time.time()-ts:.1f}s)",
                  flush=True)
    daily_panel = pd.concat(all_daily, ignore_index=True)
    print(f"Daily panel: {len(daily_panel):,} rows × {daily_panel.shape[1]} cols, "
          f"{time.time()-t0:.0f}s elapsed", flush=True)

    # Funding features per symbol (from source panel)
    print("Computing funding features...", flush=True)
    all_fund = []
    for sym, g in panel.groupby("symbol"):
        f = compute_funding_features(g)
        f["symbol"] = sym
        all_fund.append(f)
    fund_panel = pd.concat(all_fund, ignore_index=True)
    print(f"Funding panel: {len(fund_panel):,} rows, {time.time()-t0:.0f}s elapsed", flush=True)

    # Merge daily features (with PIT shift) onto 5m panel
    print("Merging daily features onto 5m panel via forward-fill within day...", flush=True)
    panel["date"] = panel["open_time"].dt.floor("1D")
    panel = panel.merge(daily_panel, on=["symbol", "date"], how="left")
    panel = panel.merge(fund_panel, on=["symbol", "date"], how="left")

    # Microstructure z-scores
    print("Computing microstructure z-scores...", flush=True)
    all_micro = []
    for sym, g in panel.groupby("symbol"):
        m = compute_microstructure_zscores(g)
        m["symbol"] = sym
        all_micro.append(m)
    micro_panel = pd.concat(all_micro, ignore_index=True)
    panel = panel.merge(micro_panel, on=["symbol", "open_time"], how="left")
    panel = panel.drop(columns=["date"])

    # Sanity checks
    print(f"\nFinal panel: {len(panel):,} rows × {panel.shape[1]} cols, "
          f"total {time.time()-t0:.0f}s", flush=True)
    print("\nFeature availability check (target: <30% NaN):", flush=True)
    new_features = A_LIQ + B_BTC + C_RES + D_TRD + E_FND + ["signed_volume_4h_z", "avg_trade_size_4h_z"]
    failures = 0
    for c in new_features:
        if c not in panel.columns:
            print(f"  ✗ {c:<35} MISSING", flush=True)
            failures += 1
            continue
        n_nan = panel[c].isna().sum()
        pct_nan = n_nan / len(panel) * 100
        non_finite = (~np.isfinite(panel[c].fillna(0))).sum()
        mark = "✓" if pct_nan < 30 else ("⚠" if pct_nan < 50 else "✗")
        if pct_nan >= 30: failures += 1
        print(f"  {mark} {c:<35} NaN: {pct_nan:5.1f}%  inf: {non_finite}", flush=True)

    # Per-symbol variance check on a few new features
    print("\nPer-symbol variance check (top-3 new features, n_zero_var_syms):", flush=True)
    for c in ["beta_btc_30d", "resid_vol_30d", "amihud_illiq_30d", "funding_z_30d"]:
        if c not in panel.columns: continue
        zv = panel.groupby("symbol")[c].nunique()
        n_zero_var = (zv <= 1).sum()
        print(f"  {c:<25}: {n_zero_var}/51 symbols with ≤1 unique value", flush=True)

    if failures > 0:
        print(f"\nWARNING: {failures} features failed availability check", flush=True)

    # Pass-through existing microstructure + process fingerprint
    for c in ["aggr_ratio_4h", "tfi_4h", "idio_skew_1d", "idio_kurt_1d", "idio_max_abs_12b"]:
        if c not in panel.columns:
            print(f"  WARN: passthrough feature missing: {c}", flush=True)

    panel.to_parquet(OUT_PANEL, index=False)
    print(f"\nSaved: {OUT_PANEL} ({len(panel):,} rows × {panel.shape[1]} cols)", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
