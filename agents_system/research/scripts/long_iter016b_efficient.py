"""iter-016b — efficient version of long-horizon features + regime detector"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS_HL14 = REPO/"live/state/convexity/x132_p2_hl14_full_fullOOS_preds.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K = 5
CYCLES_PER_DAY = 6

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","bars_since_high_xs_rank","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def main():
    t0 = time.time()
    print("=== iter-016b: efficient long-horizon features + regime detector ===\n", flush=True)

    print("loading panel...", flush=True)
    panel = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"]+V0_FEATURES)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"  {len(panel):,} rows", flush=True)

    # Efficient long-horizon features via groupby + rolling sum (much faster than apply)
    print("computing long-horizon features (vectorized)...", flush=True)
    # Use log returns approximation: log(1+r) ≈ r for small r → ret_Nd ≈ exp(sum(log(1+r_1d))) - 1
    # But return_1d is already 1d return (4h × 6 = 1d). For ret_Nd we need N-day cumulative.
    # Simpler approximation: rolling sum of return_1d (works for small returns)
    g = panel.groupby("symbol", group_keys=False)
    for days in [7, 14, 30, 60, 90]:
        bars = days * CYCLES_PER_DAY
        # Approximate Nd return as sum of 1d returns (good for small magnitudes)
        panel[f"ret_{days}d"] = g["return_1d"].transform(lambda x: x.rolling(bars, min_periods=bars//2).sum().shift(1))
        print(f"  ret_{days}d done [{time.time()-t0:.0f}s]", flush=True)
    # vol ratio recent vs 30d
    panel["idio_vol_30d_avg"] = g["idio_vol_to_btc_1d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["vol_ratio_recent_vs_30d"] = panel["idio_vol_to_btc_1d"] / panel["idio_vol_30d_avg"]
    # cumulative funding 30d
    panel["funding_30d_cum"] = g["funding_rate"].transform(lambda x: x.rolling(180, min_periods=90).sum().shift(1))
    print(f"new features done [{time.time()-t0:.0f}s]", flush=True)

    NEW_FEATURES = ["ret_7d","ret_14d","ret_30d","ret_60d","ret_90d",
                    "vol_ratio_recent_vs_30d","funding_30d_cum"]

    # Load model pred
    print("merging pred...", flush=True)
    preds = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred"])
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    panel = panel.merge(preds, on=["symbol","open_time"], how="left")
    feat_set = V0_FEATURES + NEW_FEATURES + ["pred"]

    # ============ (A) PER-FEATURE LONG EDGE H1 vs H2 ============
    print("\n=== (A) Per-feature top-K long edge ===\n")
    print(f"{'feature':<30} {'H1 bps':>9} {'H1 t':>6} {'H2 bps':>9} {'H2 t':>6} {'sig H1/H2':>10}")
    print("-"*78)
    for feat in feat_set:
        if feat not in panel.columns: continue
        rows = []
        for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
            sub = panel[(panel["open_time"]>=s) & (panel["open_time"]<e)].dropna(subset=[feat,"return_pct"])
            if len(sub) < 1000: continue
            # For each cycle, take top-K by feat
            top_means = sub.groupby("open_time").apply(
                lambda g: g.nlargest(K, feat)["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
            if len(top_means) < 50: continue
            arr = top_means.values * 1e4
            mean = arr.mean(); se = arr.std()/np.sqrt(len(arr))
            t = mean/se if se>0 else float("nan")
            rows.append((period_label, mean, t, len(arr)))
        if len(rows)==2:
            h1_mean, h1_t = rows[0][1], rows[0][2]
            h2_mean, h2_t = rows[1][1], rows[1][2]
            sig1 = "★" if abs(h1_t)>1.96 else " "
            sig2 = "★" if abs(h2_t)>1.96 else " "
            marker = " (NEW)" if feat in NEW_FEATURES else " (pred)" if feat=="pred" else ""
            print(f"  {feat:<30} {h1_mean:>+7.2f}  {h1_t:>+5.2f}  {h2_mean:>+7.2f}  {h2_t:>+5.2f}    {sig1}/{sig2}{marker}")

    # ============ (B) REGIME DETECTOR ============
    print("\n\n=== (B) Regime Detector — monthly frac of features with positive rolling long edge ===\n")
    # For each cycle, compute per-feature long edge, then rolling-30d mean, then frac positive
    cycle_results = {}
    for feat in feat_set:
        if feat not in panel.columns: continue
        per_cycle = panel.dropna(subset=[feat,"return_pct"]).groupby("open_time").apply(
            lambda g: g.nlargest(K, feat)["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        cycle_results[feat] = per_cycle
        print(f"  {feat} computed [{time.time()-t0:.0f}s]", flush=True)

    edges_df = pd.DataFrame(cycle_results)
    bars_30 = 30*CYCLES_PER_DAY
    rolling = edges_df.rolling(bars_30, min_periods=bars_30//2).mean()
    # exclude btc_rvol_7d which is universe-broadcast not symbol-level
    drop_cols = [c for c in ["btc_rvol_7d"] if c in rolling.columns]
    feat_rolling = rolling.drop(columns=drop_cols)
    rolling["frac_positive"] = (feat_rolling > 0).sum(axis=1) / max(1, feat_rolling.shape[1])

    print("\nMonthly average of (% features with positive 30d trailing long edge):")
    rolling.index = pd.to_datetime(rolling.index, utc=True)
    monthly = rolling.groupby(rolling.index.to_period("M"))["frac_positive"].mean()
    for month, frac in monthly.items():
        bar = "█"*int(frac*40)
        print(f"  {month}:  {frac*100:>5.1f}%  {bar}")

    # Save data
    rolling[["frac_positive"]].to_parquet(REPO/"agents_system/research/outputs/long_iter016_frac.parquet")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
