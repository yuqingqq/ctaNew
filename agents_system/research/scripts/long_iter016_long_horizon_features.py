"""LONG-PRED iter-016 — Long-horizon features + regime detector via feature health

Two purposes:
  (A) Test long-horizon features (ret_7d, 14d, 30d, 60d, 90d, funding_30d_mean)
      that V0 doesn't include. Maybe one captures narrative cycles → positive
      long edge in H2.
  (B) Show how per-feature rolling 30d long edge can serve as a regime detector:
      when MANY features have positive trailing long edge → V0-favored regime
      when NO features have positive long edge → V6-favored regime
      This becomes an action signal for V0 ↔ V6 switching.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS_HL14 = REPO/"live/state/convexity/x132_p2_hl14_full_fullOOS_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K = 5
CYCLES_PER_DAY = 6

# Existing V0 features
V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","bars_since_high_xs_rank","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def compute_long_horizon_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ret_7d, ret_14d, ret_30d, ret_60d, ret_90d to the panel.
    Approximate via cumulative product of (1 + return_1d) over rolling windows.
    Also compute vol_change_30d (idio_vol_30d / idio_vol_7d - 1)."""
    print("computing long-horizon features...", flush=True)
    out = panel.sort_values(["symbol","open_time"]).copy()
    # ret_Nd computed per sym by rolling product of (1 + return_1d) over N*6 cycles
    for days in [7, 14, 30, 60, 90]:
        bars = days * CYCLES_PER_DAY
        col = f"ret_{days}d"
        out[col] = out.groupby("symbol",group_keys=False)["return_1d"].apply(
            lambda x: ((1+x).rolling(bars, min_periods=bars//2).apply(np.prod, raw=True) - 1).shift(1))
    # vol expansion ratio: idio_vol_30d / idio_vol_7d (where 30d is rolling 30d avg of idio_vol_1d)
    out["idio_vol_30d_avg"] = out.groupby("symbol",group_keys=False)["idio_vol_to_btc_1d"].apply(
        lambda x: x.rolling(30*CYCLES_PER_DAY, min_periods=30*CYCLES_PER_DAY//2).mean().shift(1))
    out["vol_ratio_recent_vs_30d"] = out["idio_vol_to_btc_1d"] / out["idio_vol_30d_avg"]
    # funding 30d sum (positive = persistently overcrowded longs paying)
    out["funding_30d_cum"] = out.groupby("symbol",group_keys=False)["funding_rate"].apply(
        lambda x: x.rolling(30*CYCLES_PER_DAY, min_periods=30*CYCLES_PER_DAY//2).sum().shift(1))
    return out

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-016: Long-horizon features + regime detector ===\n", flush=True)

    # Load panel
    print("loading panel...", flush=True)
    panel = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"]+V0_FEATURES)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]

    # Compute long-horizon features
    panel = compute_long_horizon_features(panel)

    NEW_FEATURES = ["ret_7d","ret_14d","ret_30d","ret_60d","ret_90d",
                    "vol_ratio_recent_vs_30d","funding_30d_cum"]

    # ============ (A) PER-FEATURE LONG EDGE — NEW FEATURES ============
    print("\n=== (A) NEW LONG-HORIZON FEATURE EDGE in H1 vs H2 ===\n")
    print(f"{'feature':<30} {'H1 long bps':>12} {'H1 t':>7} {'H2 long bps':>12} {'H2 t':>7} {'sig H1/H2':>10}")
    print("-"*82)
    results = []
    for feat in V0_FEATURES + NEW_FEATURES + ["pred"]:
        if feat == "pred":
            # add pred from preds file
            preds = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred"])
            preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
            test_df = panel.merge(preds, on=["symbol","open_time"], how="inner")
        else:
            if feat not in panel.columns: continue
            test_df = panel

        for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
            sub = test_df[(test_df["open_time"]>=s) & (test_df["open_time"]<e)].dropna(subset=[feat,"return_pct"])
            top_rets = []
            for ot, g in sub.groupby("open_time"):
                if len(g)<2*K: continue
                top_rets.append(g.nlargest(K, feat)["return_pct"].mean())
            top_rets = np.array(top_rets)
            if len(top_rets) < 50: continue
            mean_bps = top_rets.mean()*1e4
            se = (top_rets.std()/np.sqrt(len(top_rets)))*1e4
            t = mean_bps/se if se>0 else float("nan")
            results.append(dict(feature=feat, period=period_label, mean=mean_bps, se=se, t=t, n=len(top_rets)))

    rdf = pd.DataFrame(results)
    for feat in V0_FEATURES + NEW_FEATURES + ["pred"]:
        try:
            h1 = rdf[(rdf["feature"]==feat)&(rdf["period"]=="H1")].iloc[0]
            h2 = rdf[(rdf["feature"]==feat)&(rdf["period"]=="H2")].iloc[0]
        except IndexError: continue
        sig1 = "★" if abs(h1["t"])>1.96 else " "
        sig2 = "★" if abs(h2["t"])>1.96 else " "
        marker = " (NEW)" if feat in NEW_FEATURES else " (pred)" if feat=="pred" else ""
        print(f"  {feat:<30} {h1['mean']:>+9.2f}  {h1['t']:>+5.2f}  {h2['mean']:>+9.2f}  {h2['t']:>+5.2f}    {sig1}/{sig2} {marker}")

    # ============ (B) REGIME DETECTOR via FEATURE HEALTH ============
    print("\n\n=== (B) REGIME DETECTOR — fraction of features with positive 30d trailing long edge ===\n")
    print("For each calendar day, compute trailing-30d long edge for each feature.")
    print("Count how many features have positive trailing long edge.")
    print("Fraction high → V0 regime (long signal available); fraction low → V6 regime\n")

    # Per-cycle per-feature long edge
    panel_for_detector = panel.copy()
    preds_local = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred"])
    preds_local["open_time"] = pd.to_datetime(preds_local["open_time"], utc=True)
    panel_for_detector = panel_for_detector.merge(preds_local, on=["symbol","open_time"], how="left")
    feat_set = V0_FEATURES + NEW_FEATURES + ["pred"]
    # per cycle, per feature: long edge = mean(top-K returns) over cycle
    cycle_edges_rows = []
    for ot, g in panel_for_detector.groupby("open_time"):
        if len(g)<2*K: continue
        row = dict(open_time=ot)
        for feat in feat_set:
            if feat not in g.columns: continue
            sub = g.dropna(subset=[feat,"return_pct"])
            if len(sub)<2*K: continue
            row[feat] = sub.nlargest(K, feat)["return_pct"].mean()
        cycle_edges_rows.append(row)
    edges_df = pd.DataFrame(cycle_edges_rows).set_index("open_time").sort_index()
    # rolling 30-day mean
    bars_30 = 30*CYCLES_PER_DAY
    rolling = edges_df.rolling(bars_30, min_periods=bars_30//2).mean()
    # fraction of features with positive rolling long edge
    rolling["frac_positive"] = (rolling[feat_set].drop(columns=["btc_rvol_7d"], errors="ignore") > 0).sum(axis=1) / max(1, len(feat_set)-1)
    # monthly summary
    rolling.index = pd.to_datetime(rolling.index, utc=True)
    rolling["month"] = rolling.index.to_period("M").astype(str)
    monthly = rolling.groupby("month")["frac_positive"].mean()
    print(f"  Monthly fraction of features with positive 30d trailing long edge:")
    for month, frac in monthly.items():
        bar = "█"*int(frac*40)
        print(f"    {month}:  {frac*100:>5.1f}%  {bar}")

    print(f"\n  Interpretation:")
    print(f"    > 60% features positive → V0/full regime (long signal available across feature set)")
    print(f"    40-60% → mixed regime")
    print(f"    < 30% → V6 regime (long signal collapsed)")

    # Save for plotting
    rolling[["frac_positive"]].to_parquet(REPO/"agents_system/research/outputs/long_iter016_frac_positive.parquet")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
