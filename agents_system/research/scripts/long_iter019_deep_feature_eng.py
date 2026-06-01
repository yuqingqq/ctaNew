"""LONG-PRED iter-019 — Deep feature engineering — REALLY think hard, no superficial give-up.

Test 40+ engineered features grouped by mechanism. For each, measure H1 and H2 long edge,
then build composite signals from the best.

Categories:
  A. Cross-sectional ranks of every V0 feature (per cycle, percentile rank in universe)
  B. Momentum divergences (recent vs longer-term returns)
  C. Vol/funding ratios (regime within sym)
  D. Interactions (2-way products of top features)
  E. Anomaly z-scores (each feature normalized by trailing distribution)
  F. Multi-signal compounds (pump-setup heuristics)

If ANY engineered feature has positive H2 long edge with significance (|t|>2), we have
a real fix candidate. Build feature set + test model with it.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K = 5
CYCLES_PER_DAY = 6

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def main():
    t0 = time.time()
    print("=== iter-019: Deep feature engineering ===\n", flush=True)

    print("loading panel...", flush=True)
    panel = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"]+V0_FEATURES)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    # Add longer-horizon returns (vectorized)
    print("computing long-horizon returns + funding_30d...", flush=True)
    g = panel.groupby("symbol", group_keys=False)
    for days in [7, 14, 30, 60, 90]:
        bars = days * CYCLES_PER_DAY
        panel[f"ret_{days}d"] = g["return_1d"].transform(lambda x: x.rolling(bars, min_periods=bars//2).sum().shift(1))
    panel["idio_vol_30d"] = g["idio_vol_to_btc_1d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["funding_30d"] = g["funding_rate"].transform(lambda x: x.rolling(180, min_periods=90).sum().shift(1))
    panel["rvol_30d"] = g["rvol_7d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))

    # ============ ENGINEER NEW FEATURES ============
    print("engineering new features...", flush=True)

    # A. Cross-sectional ranks per cycle
    print("  A. Cross-sectional ranks...", flush=True)
    for feat in V0_FEATURES:
        panel[f"xsrank_{feat}"] = panel.groupby("open_time")[feat].rank(pct=True)
    # B. Momentum divergences
    print("  B. Momentum divergences...", flush=True)
    panel["mom_div_1d_7d"] = panel["return_1d"] - panel["ret_7d"]/7
    panel["mom_div_3d_30d"] = panel["ret_3d"]/3 - panel["ret_30d"]/30
    panel["mom_div_7d_30d"] = panel["ret_7d"]/7 - panel["ret_30d"]/30
    panel["mom_div_7d_60d"] = panel["ret_7d"]/7 - panel["ret_60d"]/60
    panel["mom_accel"] = panel["return_1d"] - 0.5*(panel["ret_7d"]/7 + panel["ret_30d"]/30)
    # C. Vol/funding ratios
    print("  C. Vol/funding ratios...", flush=True)
    panel["vol_ratio_1h_1d"] = panel["idio_vol_to_btc_1h"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)
    panel["vol_ratio_7d_30d"] = panel["rvol_7d"] / panel["rvol_30d"].replace(0, np.nan)
    panel["vol_ratio_recent_30d"] = panel["idio_vol_to_btc_1d"] / panel["idio_vol_30d"].replace(0, np.nan)
    panel["funding_change_rate"] = panel["funding_rate_1d_change"] / panel["funding_rate"].replace(0, np.nan)
    panel["funding_per_vol"] = panel["funding_rate"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)
    # D. Interactions
    print("  D. Interactions...", flush=True)
    panel["vol_x_mom"] = panel["rvol_7d"] * panel["return_1d"]
    panel["vol_x_neg_mom"] = panel["rvol_7d"] * (-panel["return_1d"])  # high vol + low return
    panel["funding_x_beta"] = panel["funding_rate"] * panel["beta_to_btc_change_5d"]
    panel["funding_x_neg_mom"] = panel["funding_rate"] * (-panel["return_1d"])
    panel["bars_x_vol"] = panel["bars_since_high"] * panel["rvol_7d"]
    panel["vol_x_idio"] = panel["rvol_7d"] * panel["idio_vol_to_btc_1d"]
    # E. Anomaly z-scores (feature vs its trailing distribution per sym)
    print("  E. Anomaly z-scores...", flush=True)
    g = panel.groupby("symbol", group_keys=False)
    for feat in ["return_1d","rvol_7d","funding_rate","idio_vol_to_btc_1d"]:
        rolling_mean = g[feat].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
        rolling_std = g[feat].transform(lambda x: x.rolling(180, min_periods=90).std().shift(1))
        panel[f"z_{feat}"] = (panel[feat] - rolling_mean) / rolling_std.replace(0, np.nan)
    # F. Multi-signal compound features (hand-crafted pump-setup heuristics)
    print("  F. Multi-signal compounds...", flush=True)
    # Pump setup 1: recent dip + low/neg funding + rising vol
    panel["pump_setup_1"] = (-panel["return_1d"].clip(lower=-0.05, upper=0.05)) * \
                            (-panel["funding_rate"]) * \
                            panel["vol_ratio_recent_30d"]
    # Breakout: low bars_since_high (or high xs_rank) + rising vol
    panel["breakout"] = panel["xsrank_bars_since_high"] * panel["vol_ratio_recent_30d"]
    # Cooled-down: high bars_since_high + low recent vol (consolidation)
    panel["consolidating"] = (1 - panel["xsrank_bars_since_high"]) * (1/panel["vol_ratio_recent_30d"].clip(0.1, 10))
    # Anti-overcrowded: low absolute funding (not heavily bid up)
    panel["not_overcrowded"] = -panel["funding_rate"].abs()
    # Momentum acceleration positive AND not extreme funding
    panel["fresh_breakout"] = panel["mom_accel"].clip(lower=0, upper=0.05) * (1 / (1 + panel["funding_rate"].abs()))

    NEW_FEATURES = (
        [f"xsrank_{f}" for f in V0_FEATURES]
        + ["mom_div_1d_7d","mom_div_3d_30d","mom_div_7d_30d","mom_div_7d_60d","mom_accel"]
        + ["vol_ratio_1h_1d","vol_ratio_7d_30d","vol_ratio_recent_30d","funding_change_rate","funding_per_vol"]
        + ["vol_x_mom","vol_x_neg_mom","funding_x_beta","funding_x_neg_mom","bars_x_vol","vol_x_idio"]
        + ["z_return_1d","z_rvol_7d","z_funding_rate","z_idio_vol_to_btc_1d"]
        + ["pump_setup_1","breakout","consolidating","not_overcrowded","fresh_breakout"]
    )
    print(f"  Total NEW features: {len(NEW_FEATURES)}", flush=True)

    # ============ TEST EACH NEW FEATURE FOR LONG EDGE ============
    print("\n=== Per-feature long-K=5 edge H1 vs H2 (sorted by H2 edge descending) ===\n")
    rows = []
    for feat in NEW_FEATURES:
        if feat not in panel.columns: continue
        for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
            sub = panel[(panel["open_time"]>=s) & (panel["open_time"]<e)].dropna(subset=[feat,"return_pct"])
            if len(sub) < 1000: continue
            top_means = sub.groupby("open_time").apply(
                lambda g: g.nlargest(K, feat)["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
            if len(top_means) < 50: continue
            arr = top_means.values * 1e4
            mean = arr.mean(); se = arr.std()/np.sqrt(len(arr))
            t = mean/se if se>0 else float("nan")
            rows.append(dict(feature=feat, period=period_label, mean=mean, se=se, t=t, n=len(arr)))

    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index="feature", columns="period", values=["mean","t"]).fillna(0)
    pivot.columns = [f"{c[1]}_{c[0]}" for c in pivot.columns]
    pivot = pivot.sort_values("H2_mean", ascending=False)
    print(f"{'feature':<28} {'H1 bps':>9} {'H1 t':>7} {'H2 bps':>9} {'H2 t':>7} {'sig':>4}")
    print("-"*70)
    for feat, row in pivot.iterrows():
        sig = "★" if abs(row.get("H2_t",0))>1.96 else " "
        sig_h1 = "★" if abs(row.get("H1_t",0))>1.96 else " "
        print(f"{feat:<28} {row.get('H1_mean',0):>+7.2f} {row.get('H1_t',0):>+5.2f}  {row.get('H2_mean',0):>+7.2f} {row.get('H2_t',0):>+5.2f}  {sig_h1}/{sig}")

    # ============ FIND BEST CANDIDATES ============
    print(f"\n=== TOP 10 features by H2 long edge ===\n")
    h2 = rdf[rdf["period"]=="H2"].sort_values("mean", ascending=False).head(10)
    for _, r in h2.iterrows():
        sig = "★" if abs(r["t"])>1.96 else " "
        print(f"  {r['feature']:<28} H2 = {r['mean']:+.2f} bps (t={r['t']:+.2f}) {sig}")

    print(f"\n=== Features with positive H2 long edge AND H1 not destroyed ===\n")
    # H1 mean > -5 (not destructively negative), H2 mean > 0
    h1 = rdf[rdf["period"]=="H1"].set_index("feature")
    h2 = rdf[rdf["period"]=="H2"].set_index("feature")
    common = h1.index.intersection(h2.index)
    survivors = []
    for f in common:
        if h2.loc[f,"mean"] > 0 and h1.loc[f,"mean"] > -5:
            survivors.append((f, h1.loc[f,"mean"], h1.loc[f,"t"], h2.loc[f,"mean"], h2.loc[f,"t"]))
    survivors.sort(key=lambda x: x[3], reverse=True)
    if survivors:
        print(f"  {'feature':<28} {'H1 bps':>9} {'H1 t':>7} {'H2 bps':>9} {'H2 t':>7}")
        for f, h1m, h1t, h2m, h2t in survivors:
            print(f"  {f:<28} {h1m:>+7.2f} {h1t:>+5.2f}  {h2m:>+7.2f} {h2t:>+5.2f}")
    else:
        print("  NONE — every feature either destroys H1 or fails H2")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
