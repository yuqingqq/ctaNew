"""Audit horizons of v3_5m features against the 4h prediction target.

For each v3_5m feature, report:
- Window size in bars + days
- Whether window matches target horizon, exceeds it, or is microstructure-noisy
- Cross-sectional IC vs alpha_beta at entry cadence (4h)
- NaN rate
- Per-symbol variance
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
PANEL_V3_5M = REPO / "outputs/vBTC_features_btc_v3/panel_v3_5m.parquet"
BASE_PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"

ENTRY_STRIDE = 48  # 4h
BAR_PER_DAY = 288

V3_5M_FEATURES = {
    # name: (input_resolution, rolling_window_in_input_units, computed_purpose, target_match)
    "beta_btc_30d":              ("5m", 30*BAR_PER_DAY,   "rolling β cov/var on 5m raw returns over 30d", "30d"),
    "beta_btc_90d":              ("5m", 90*BAR_PER_DAY,   "rolling β over 90d",                            "90d"),
    "corr_btc_30d":              ("5m", 30*BAR_PER_DAY,   "rolling corr on 5m returns over 30d",           "30d"),
    "corr_breakdown":            ("derived", 0,           "corr_30d - corr_90d",                           "derived"),
    "resid_vol_30d":             ("5m", 30*BAR_PER_DAY,   "rolling std of idio_5m over 30d",               "30d"),
    "resid_vol_90d":             ("5m", 90*BAR_PER_DAY,   "rolling std over 90d",                          "90d"),
    "resid_skew_30d":            ("1h", 30*24,            "rolling skew of idio_1h over 30d",              "30d"),
    "resid_kurt_30d":            ("1h", 30*24,            "rolling kurt of idio_1h over 30d",              "30d"),
    "resid_jump_count_30d":      ("1h", 30*24,            "count |idio_1h|>3σ over 30d",                    "30d"),
    "resid_trend_score_30d":     ("1h", 30*24,            "sum_30d / std_30d on idio_1h",                  "30d"),
    "multi_horizon_trend_score": ("1h", 0,                "avg of trend_7d/30d/90d (1h)",                   "composite"),
    "log_dollar_volume_7d":      ("5m", 7*BAR_PER_DAY,    "log(rolling mean of quote_volume over 7d)",     "7d"),
    "volume_stability_30d":      ("5m", 30*BAR_PER_DAY,   "rolling std/mean of quote_volume over 30d",     "30d"),
    "amihud_illiq_30d":          ("5m", 30*BAR_PER_DAY,   "rolling mean |ret_5m|/qv_5m over 30d (×1e9)",   "30d"),
    "dist_from_30d_high":        ("5m", 30*BAR_PER_DAY,   "5m close / rolling max(close, 30d) - 1",        "30d"),
    "dist_from_365d_high":       ("5m", 365*BAR_PER_DAY,  "5m close / rolling max(close, 365d) - 1",       "365d"),
    "funding_mean_30d":          ("5m", 30*BAR_PER_DAY,   "rolling mean of funding_rate over 30d (5m)",    "30d"),
}


def per_cycle_ic(df, feat, target="alpha_beta"):
    samp = df.dropna(subset=[feat, target])
    if len(samp) == 0: return np.nan, 0
    ics = []
    for t, g in samp.groupby("open_time"):
        if len(g) < 10: continue
        ic = g[feat].rank().corr(g[target].rank())
        if not pd.isna(ic): ics.append(ic)
    if not ics: return np.nan, 0
    return float(np.mean(ics)), len(ics)


def main():
    print("=== v3_5m horizon audit ===\n", flush=True)
    t0 = time.time()
    panel_v3 = pd.read_parquet(PANEL_V3_5M)
    panel_v3["open_time"] = pd.to_datetime(panel_v3["open_time"], utc=True)
    print(f"v3_5m panel: {len(panel_v3):,} rows × {panel_v3.shape[1]} cols\n", flush=True)

    # Load alpha_beta from base panel (compute target if needed)
    base = pd.read_parquet(BASE_PANEL, columns=["symbol","open_time","return_pct"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    btc = base[base.symbol=="BTCUSDT"][["open_time","return_pct"]].rename(
        columns={"return_pct":"btc_ret"}).drop_duplicates("open_time")
    # Quick PIT β at 90d×288 = 25920 bar window
    print("Computing alpha_beta from base panel (90d × 288 bar β)...", flush=True)
    out_list = []
    for sym, g in base.groupby("symbol"):
        gg = g.merge(btc, on="open_time", how="left").sort_values("open_time").reset_index(drop=True)
        if sym == "BTCUSDT":
            gg["beta_pit"] = 1.0
        else:
            y = gg["return_pct"]; x = gg["btc_ret"]
            cov = y.rolling(90*BAR_PER_DAY, min_periods=1000).cov(x)
            var = x.rolling(90*BAR_PER_DAY, min_periods=1000).var()
            gg["beta_pit"] = (cov / var.replace(0, np.nan)).shift(1)
        gg["alpha_beta"] = gg["return_pct"] - gg["beta_pit"] * gg["btc_ret"]
        gg["symbol"] = sym
        out_list.append(gg[["symbol","open_time","alpha_beta"]])
    base_ab = pd.concat(out_list, ignore_index=True)
    print(f"  done {time.time()-t0:.0f}s\n", flush=True)

    # Merge alpha_beta to v3_5m panel
    panel_v3 = panel_v3.merge(base_ab, on=["symbol","open_time"], how="left")

    # Subsample to 4h entry cadence
    times = sorted(panel_v3["open_time"].unique())
    keep_t = set(times[::ENTRY_STRIDE])
    samp = panel_v3[panel_v3["open_time"].isin(keep_t)].copy()
    print(f"4h-cadence sample: {len(samp):,} rows, {len(keep_t)} cycles\n", flush=True)

    # ====== HORIZON AUDIT ======
    print(f"{'='*120}", flush=True)
    print(f"  HORIZON AUDIT vs 4h target", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"  {'feature':<32} {'res':>4} {'window':>10} {'days':>7} {'horizon_match':<14} "
          f"{'nan%':>6} {'cs_IC':>8} {'sym_var':>9}", flush=True)

    rows = []
    for feat, (res, win_bars, _, target_label) in V3_5M_FEATURES.items():
        if feat not in panel_v3.columns:
            print(f"  ⚠ {feat:<32} MISSING", flush=True)
            continue
        if res == "5m":
            days = win_bars / BAR_PER_DAY
        elif res == "1h":
            days = win_bars / 24
        else:
            days = 0
        # Horizon analysis vs 4h target (= 48 5m bars = 4 1h bars)
        if target_label == "derived":
            mh = "derived"
        elif res == "5m":
            if win_bars < 48:
                mh = "TOO SHORT"  # below target
            elif win_bars < 288:
                mh = "near-target"
            elif win_bars < 30*288:
                mh = "regime-mid"
            else:
                mh = "regime-long"
        elif res == "1h":
            if win_bars < 4:
                mh = "TOO SHORT"
            elif win_bars < 24:
                mh = "near-target"
            elif win_bars < 30*24:
                mh = "regime-mid"
            else:
                mh = "regime-long"
        else:
            mh = "n/a"

        nan_pct = samp[feat].isna().mean() * 100
        cs_ic, n_cyc = per_cycle_ic(samp, feat)
        psy_var = samp.groupby("symbol")[feat].std().median()
        if pd.isna(psy_var): psy_var = 0
        rows.append({
            "feature": feat, "res": res,
            "window_bars": win_bars, "days": days,
            "horizon_match": mh,
            "nan_pct": nan_pct,
            "cs_IC": cs_ic if not pd.isna(cs_ic) else 0,
            "sym_var_median": psy_var,
        })
        ic_str = f"{cs_ic:+.4f}" if not pd.isna(cs_ic) else "  n/a "
        print(f"  {feat:<32} {res:>4} {win_bars:>10} {days:>7.1f} {mh:<14} "
              f"{nan_pct:>5.1f}% {ic_str:>8} {psy_var:>9.4f}", flush=True)

    df_audit = pd.DataFrame(rows)
    out_path = REPO / "outputs/vBTC_features_btc_v3/v3_5m_horizon_audit.csv"
    df_audit.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}", flush=True)

    # ====== COMPARE TO WINNER_17 HORIZONS ======
    print(f"\n{'='*120}", flush=True)
    print(f"  WINNER_17 horizon coverage (for reference)", flush=True)
    print(f"{'='*120}", flush=True)
    w17_horizons = [
        ("return_1d",                "5m",   288, "1d backward return"),
        ("atr_pct",                  "5m",   "?", "ATR pct (likely 14-bar EMA)"),
        ("dom_level_vs_bk",          "5m",   "?", "dominance level vs basket (1-bar)"),
        ("dom_change_288b_vs_bk",    "5m",   288, "dom change vs bk over 288 bars (1d)"),
        ("bk_ema_slope_4h",          "5m",   48,  "basket EMA slope over 48 bars (4h) **MATCHES TARGET**"),
        ("corr_change_3d_vs_bk",     "5m",   864, "Δ in corr_vs_bk over 3 days"),
        ("obv_z_1d",                 "5m",   288, "OBV z-score 1d window"),
        ("vwap_slope_96",            "5m",   96,  "VWAP slope over 96 bars (8h)"),
        ("bars_since_high_xs_rank",  "5m",   "?", "time since recent high (xs rank)"),
        ("idio_vol_1d_vs_bk_xs_rank","5m",   288, "1d idio vol vs basket"),
        ("sym_id",                   "id",   0,   "categorical encoding"),
        ("funding_rate",             "8h",   1,   "current funding (8h tick)"),
        ("funding_rate_z_7d",        "8h",   21,  "z-score over 7d (= 21 funding ticks)"),
        ("corr_to_btc_1d",           "5m",   288, "1d (288 bar) corr to BTC"),
        ("idio_vol_to_btc_1h",       "5m",   12,  "1h (12 bar) idio vol vs BTC **NEAR-TARGET**"),
        ("beta_to_btc_change_5d",    "5m",   1440,"Δ β over 5d"),
        ("funding_rate_1d_change",   "8h",   3,   "Δ funding over 1d (= 3 ticks)"),
    ]
    print(f"  {'feature':<32} {'res':>4} {'window':>10} {'description':<42}", flush=True)
    for name, res, win, desc in w17_horizons:
        print(f"  {name:<32} {res:>4} {str(win):>10} {desc:<42}", flush=True)

    # ====== MISSING HORIZON BUCKETS ======
    print(f"\n{'='*120}", flush=True)
    print(f"  WHICH HORIZONS ARE MISSING (focus on near-target = 4h)", flush=True)
    print(f"{'='*120}", flush=True)
    bucket_features = {
        "< 1h (microstructure)":       [],
        "1h to 4h (near-target)":      ["idio_vol_to_btc_1h"],
        "4h to 24h (short-context)":   ["bk_ema_slope_4h", "vwap_slope_96"],
        "1d-3d (mid-context)":         ["return_1d", "dom_change_288b_vs_bk", "obv_z_1d",
                                         "corr_to_btc_1d", "idio_vol_1d_vs_bk_xs_rank"],
        "3d-7d":                       ["corr_change_3d_vs_bk", "funding_rate_z_7d"],
        "7d-30d":                      ["log_dollar_volume_7d", "beta_to_btc_change_5d"],
        "30d (regime-mid)":            ["beta_btc_30d", "corr_btc_30d", "resid_vol_30d",
                                         "resid_skew_30d", "resid_kurt_30d", "resid_jump_count_30d",
                                         "resid_trend_score_30d", "volume_stability_30d",
                                         "amihud_illiq_30d", "dist_from_30d_high",
                                         "funding_mean_30d", "multi_horizon_trend_score"],
        "90d-365d (regime-long)":      ["beta_btc_90d", "resid_vol_90d", "dist_from_365d_high"],
        "categorical/identity":        ["sym_id"],
    }
    print(f"  {'bucket':<32} {'count':>5}  features", flush=True)
    for b, feats in bucket_features.items():
        print(f"  {b:<32} {len(feats):>5}  {feats}", flush=True)

    print(f"\n  Note: target horizon = 4h. Buckets at-target (4h-24h) and 1d-3d should carry most signal.", flush=True)
    print(f"  CURRENT BLIND SPOT: there are NO new v3_5m features in the 4h-24h or 1d-3d bucket.", flush=True)
    print(f"  All v3_5m additions are at 30d+ (regime-mid) horizons.", flush=True)


if __name__ == "__main__":
    main()
