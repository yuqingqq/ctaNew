"""v3 5m-resolution feature builder. Replaces 1d-granularity v3 with
5m-bar-window rolling features that match the target's 5m granularity.

Features (19 after dedupe vs WINNER_17):
  A_liquidity (3): log_dollar_volume_7d, volume_stability_30d, amihud_illiq_30d
  B_btc       (4): beta_btc_30d, beta_btc_90d, corr_btc_30d, corr_breakdown
  C_resid     (6): resid_vol_30d, resid_vol_90d,
                   resid_skew_30d, resid_kurt_30d,
                   resid_jump_count_30d, resid_trend_score_30d
  D_trend     (3): dist_from_30d_high, dist_from_365d_high,
                   multi_horizon_trend_score
  E_funding   (1): funding_mean_30d
  G_process   (2): idio_skew_1d, idio_max_abs_12b  (passthrough from base panel)

Source: data/ml/test/parquet/klines/{SYM}/5m/*.parquet (raw OHLCV)
Output: outputs/vBTC_features_btc_v3/panel_v3_5m.parquet (keys + new features only)

Computational notes:
  - β/corr/resid_vol/dist_from_high/liquidity: 5m bar windows via pandas .rolling()
    (streaming O(n) per symbol — fast)
  - skew/kurt/jump_count/trend_score: 1h-resampled idio returns (pandas .rolling().skew()
    is O(n·window), prohibitive at 5m granularity with 8640-bar window). Compute
    at 1h cadence then asof-merge back to 5m bars and shift(1) for PIT.
  - PIT discipline: all rolling stats .shift(1) at native resolution (5m or 1h)
    BEFORE the cross-resolution merge.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
BASE_PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_features_btc_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PANEL = OUT_DIR / "panel_v3_5m.parquet"

BAR_PER_DAY = 288  # 5m bars
WIN_7D   = 7 * BAR_PER_DAY      # 2016
WIN_30D  = 30 * BAR_PER_DAY     # 8640
WIN_90D  = 90 * BAR_PER_DAY     # 25920
WIN_365D = 365 * BAR_PER_DAY    # 105120
BAR_1H   = 12  # 5m bars per hour

# 1h-cadence windows (in 1h bars)
H_30D  = 30 * 24
H_365D = 365 * 24


def load_5m(sym):
    d = KLINES_DIR / sym / "5m"
    files = sorted(d.glob("*.parquet"))
    if not files: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close", "quote_volume"])
                    for f in files], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df


def compute_5m_features(sym_df, btc_df):
    """Compute 5m features for one symbol against BTC."""
    g = sym_df.merge(
        btc_df[["open_time", "close", "quote_volume"]].rename(
            columns={"close": "btc_close", "quote_volume": "btc_qv"}),
        on="open_time", how="left",
    )
    g = g.sort_values("open_time").reset_index(drop=True)

    sym_ret = g["close"].pct_change()
    btc_ret = g["btc_close"].pct_change()

    # ------- B block: β, corr -------
    out = {}
    for win_d, win_b in [(30, WIN_30D), (90, WIN_90D)]:
        min_obs = max(1000, win_b // 4)
        cov = sym_ret.rolling(win_b, min_periods=min_obs).cov(btc_ret)
        var = btc_ret.rolling(win_b, min_periods=min_obs).var()
        beta = cov / var.replace(0, np.nan)
        out[f"beta_btc_{win_d}d"] = beta.shift(1)  # PIT
        corr = sym_ret.rolling(win_b, min_periods=min_obs).corr(btc_ret)
        out[f"corr_btc_{win_d}d"] = corr.shift(1)
    out["corr_breakdown"] = out["corr_btc_30d"] - out["corr_btc_90d"]
    # drop corr_btc_90d after using it for breakdown (Phase 2 trim + dedupe)
    del out["corr_btc_90d"]

    # ------- idio_ret_5m using β_90d -------
    beta_for_idio = beta.shift(1)  # PIT β at 90d
    idio_ret = sym_ret - beta_for_idio * btc_ret

    # ------- C block: residual vol (5m bar windows) -------
    for win_d, win_b in [(30, WIN_30D), (90, WIN_90D)]:
        min_obs = max(1000, win_b // 4)
        rv = idio_ret.rolling(win_b, min_periods=min_obs).std()
        out[f"resid_vol_{win_d}d"] = rv.shift(1)

    # ------- A block: liquidity (5m bar windows) -------
    # log_dollar_volume_7d
    out["log_dollar_volume_7d"] = (
        np.log1p(g["quote_volume"].rolling(WIN_7D, min_periods=WIN_7D//4).mean()).shift(1)
    )
    # volume_stability_30d = std / mean
    vol_mean = g["quote_volume"].rolling(WIN_30D, min_periods=WIN_30D//4).mean()
    vol_std  = g["quote_volume"].rolling(WIN_30D, min_periods=WIN_30D//4).std()
    out["volume_stability_30d"] = (vol_std / vol_mean.replace(0, np.nan)).shift(1)
    # amihud_illiq_30d = rolling mean of (|return| / quote_volume), rescaled
    amihud_raw = sym_ret.abs() / g["quote_volume"].replace(0, np.nan)
    out["amihud_illiq_30d"] = (
        amihud_raw.rolling(WIN_30D, min_periods=WIN_30D//4).mean().shift(1) * 1e9
    )

    # ------- D block: anchoring (5m bar windows) -------
    max30  = g["close"].rolling(WIN_30D,  min_periods=WIN_30D//4).max()
    max365 = g["close"].rolling(WIN_365D, min_periods=WIN_30D).max()  # min_periods allows partial year
    out["dist_from_30d_high"]  = (g["close"] / max30  - 1.0).shift(1)
    out["dist_from_365d_high"] = (g["close"] / max365 - 1.0).shift(1)

    # ------- 1h-cadence distributional features (skew, kurt, jump, trend) -------
    # Resample idio_ret_5m to 1h (sum over 12 5m bars)
    df_1h = pd.DataFrame({"open_time": g["open_time"], "idio_5m": idio_ret})
    df_1h["hour"] = df_1h["open_time"].dt.floor("1h")
    h = df_1h.groupby("hour")["idio_5m"].sum().reset_index()
    h.columns = ["hour", "idio_1h"]

    # Rolling on 1h cadence
    h["resid_skew_30d"] = h["idio_1h"].rolling(H_30D, min_periods=H_30D//4).skew().shift(1)
    h["resid_kurt_30d"] = h["idio_1h"].rolling(H_30D, min_periods=H_30D//4).kurt().shift(1)
    sigma_30d = h["idio_1h"].rolling(H_30D, min_periods=H_30D//4).std()
    h["resid_jump_count_30d"] = (
        (h["idio_1h"].abs() > 3 * sigma_30d).rolling(H_30D, min_periods=H_30D//4).sum().shift(1)
    )
    # trend_score = sum_30d / std_30d
    sum_30d = h["idio_1h"].rolling(H_30D, min_periods=H_30D//4).sum()
    h["resid_trend_score_30d"] = (sum_30d / sigma_30d.replace(0, np.nan)).shift(1)
    # Multi-horizon trend: 7d / 30d / 90d at 1h cadence
    H_7D = 7 * 24; H_90D = 90 * 24
    sum_7  = h["idio_1h"].rolling(H_7D,  min_periods=H_7D//4 ).sum()
    sum_90 = h["idio_1h"].rolling(H_90D, min_periods=H_90D//4).sum()
    sig_7  = h["idio_1h"].rolling(H_7D,  min_periods=H_7D//4 ).std()
    sig_90 = h["idio_1h"].rolling(H_90D, min_periods=H_90D//4).std()
    trend_7  = sum_7  / sig_7.replace(0, np.nan)
    trend_30 = sum_30d / sigma_30d.replace(0, np.nan)
    trend_90 = sum_90 / sig_90.replace(0, np.nan)
    h["multi_horizon_trend_score"] = ((trend_7 + trend_30 + trend_90) / 3.0).shift(1)

    # Merge 1h features back to 5m via hour key
    g["hour"] = g["open_time"].dt.floor("1h")
    g = g.merge(h[["hour", "resid_skew_30d", "resid_kurt_30d",
                    "resid_jump_count_30d", "resid_trend_score_30d",
                    "multi_horizon_trend_score"]],
                on="hour", how="left").drop(columns=["hour"])

    # Pack features
    feat_df = pd.DataFrame({"open_time": g["open_time"], **out})
    feat_df["resid_skew_30d"] = g["resid_skew_30d"].values
    feat_df["resid_kurt_30d"] = g["resid_kurt_30d"].values
    feat_df["resid_jump_count_30d"] = g["resid_jump_count_30d"].values
    feat_df["resid_trend_score_30d"] = g["resid_trend_score_30d"].values
    feat_df["multi_horizon_trend_score"] = g["multi_horizon_trend_score"].values
    return feat_df


def compute_funding_mean_30d(panel_sym):
    """Funding's update cadence is 8h; compute 30d rolling mean from 5m
    forward-filled funding_rate, sampled at funding cadence."""
    g = panel_sym.sort_values("open_time").copy()
    if "funding_rate" not in g.columns:
        g["funding_mean_30d"] = np.nan
        return g[["open_time", "funding_mean_30d"]]
    g["funding_mean_30d"] = (
        g["funding_rate"].rolling(WIN_30D, min_periods=WIN_30D//4).mean().shift(1)
    )
    return g[["open_time", "funding_mean_30d"]]


def main():
    print("=== Build v3 5m-resolution feature panel (19 features) ===\n", flush=True)
    t0 = time.time()

    # Load base panel (just keys + funding_rate for the funding feature)
    print("Loading base panel keys + funding...", flush=True)
    base = pd.read_parquet(BASE_PANEL, columns=["symbol", "open_time", "funding_rate"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    syms = sorted(base["symbol"].unique())
    print(f"  base: {len(base):,} rows × {len(syms)} symbols", flush=True)

    # Load BTC klines
    print("Loading BTC 5m klines...", flush=True)
    btc_5m = load_5m("BTCUSDT")
    print(f"  BTC: {len(btc_5m):,} bars", flush=True)

    # Per-symbol feature computation
    print("Computing 5m features per symbol...", flush=True)
    all_feats = []
    for i, sym in enumerate(syms):
        ts = time.time()
        sym_5m = load_5m(sym)
        if sym_5m is None:
            print(f"  [{i+1:2d}/{len(syms)}] {sym}: SKIP (no klines)", flush=True)
            continue
        # 5m features
        feats = compute_5m_features(sym_5m, btc_5m)
        feats["symbol"] = sym
        # funding feature from base panel
        base_sym = base[base["symbol"] == sym].copy()
        fund = compute_funding_mean_30d(base_sym)
        feats = feats.merge(fund, on="open_time", how="left")
        all_feats.append(feats)
        if (i + 1) % 10 == 0 or i == len(syms) - 1:
            print(f"  [{i+1:2d}/{len(syms)}] {sym}: {len(feats):,} bars "
                  f"({time.time()-ts:.1f}s)", flush=True)

    panel = pd.concat(all_feats, ignore_index=True)
    print(f"\nMerged 5m feature panel: {len(panel):,} rows × {panel.shape[1]} cols, "
          f"{time.time()-t0:.0f}s elapsed", flush=True)

    # NaN report
    new_features = [
        "log_dollar_volume_7d", "volume_stability_30d", "amihud_illiq_30d",
        "beta_btc_30d", "beta_btc_90d", "corr_btc_30d", "corr_breakdown",
        "resid_vol_30d", "resid_vol_90d",
        "resid_skew_30d", "resid_kurt_30d", "resid_jump_count_30d", "resid_trend_score_30d",
        "dist_from_30d_high", "dist_from_365d_high", "multi_horizon_trend_score",
        "funding_mean_30d",
    ]
    print("\nNaN rate per feature:", flush=True)
    for c in new_features:
        nan_pct = panel[c].isna().mean() * 100
        non_finite = (~np.isfinite(panel[c].fillna(0))).sum()
        mark = "✓" if nan_pct < 30 else ("⚠" if nan_pct < 50 else "✗")
        print(f"  {mark} {c:<35} NaN: {nan_pct:5.1f}%  inf: {non_finite}", flush=True)

    # Save (keys + new features only — to be merged on top of base panel at train time)
    keep_cols = ["symbol", "open_time"] + new_features
    panel = panel[keep_cols]
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"\nSaved: {OUT_PANEL}", flush=True)
    print(f"  shape: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)
    print(f"  features ({len(new_features)}): {new_features}", flush=True)
    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
