"""LONG-PRED iter-015 — Why doesn't the long leg work?

Three hypotheses to test, each with a clean diagnostic:

  (A) FEATURE-LEVEL: V0 features individually have no long-side info
      Test: per-feature univariate top-K=5 long edge. If NO single feature has
      meaningful positive long edge, features can't predict pumps.

  (B) MODEL-LEVEL: Features DO have info but per-sym Ridge mis-combines them
      Test: compare model pred top-K long edge vs best-single-feature top-K
      long edge. If single feature beats Ridge, the model is over-engineering.

  (C) TARGET-LEVEL: Target (per-sym z-residual) is wrong for long selection
      Test: would a simpler target (e.g., raw return rank, momentum) produce
      meaningful long edge with the same features?
"""
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

# V0 features
FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
            "bars_since_high","bars_since_high_xs_rank","autocorr_pctile_7d",
            "corr_to_btc_1d","beta_to_btc_change_5d",
            "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
            "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
            "rvol_7d","ret_3d","btc_rvol_7d"]

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-015: Why doesn't long work? ===\n", flush=True)

    # Load panel + preds
    print("loading panel + preds...", flush=True)
    panel = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"]+FEATURES)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    preds = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred"])
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds = preds[(preds["open_time"].dt.hour%4==0) & (preds["open_time"].dt.minute==0)]
    df = panel.merge(preds, on=["symbol","open_time"], how="inner")
    print(f"  {len(df):,} rows × {df['symbol'].nunique()} syms")

    # ============ (A) FEATURE-LEVEL: per-feature long edge ============
    print("\n=== (A) PER-FEATURE LONG EDGE — does ANY single feature predict pumps? ===\n")
    print("For each feature, sort syms within each cycle by feature value.")
    print("Compute mean of top-K=5 absolute realized return (the 'long P&L' if you used that feature alone).\n")
    print(f"{'feature':<28} {'H1 long bps':>12} {'H1 std/n':>10} {'H2 long bps':>12} {'H2 std/n':>10} {'sig H1/H2':>10}")
    print("-"*90)
    rows = []
    for feat in FEATURES + ["pred"]:
        for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
            sub = df[(df["open_time"]>=s) & (df["open_time"]<e)].dropna(subset=[feat,"return_pct"])
            top_rets = []
            for ot, g in sub.groupby("open_time"):
                if len(g)<2*K: continue
                g_top = g.nlargest(K, feat)
                top_rets.append(g_top["return_pct"].mean())
            top_rets = np.array(top_rets)
            if len(top_rets) < 50: continue
            mean_bps = top_rets.mean()*1e4
            se = (top_rets.std()/np.sqrt(len(top_rets)))*1e4
            t = mean_bps/se if se>0 else float("nan")
            rows.append(dict(feature=feat, period=period_label, mean_bps=mean_bps, se=se, t=t, n=len(top_rets)))

    rdf = pd.DataFrame(rows)
    # Pivot to compare H1 vs H2 per feature
    for feat in FEATURES + ["pred"]:
        try:
            h1 = rdf[(rdf["feature"]==feat)&(rdf["period"]=="H1")].iloc[0]
            h2 = rdf[(rdf["feature"]==feat)&(rdf["period"]=="H2")].iloc[0]
        except IndexError: continue
        sig1 = "★" if abs(h1["t"])>1.96 else " "
        sig2 = "★" if abs(h2["t"])>1.96 else " "
        print(f"  {feat:<28} {h1['mean_bps']:>+9.2f}    {h1['se']:>5.2f}/{int(h1['n']):<4} {h2['mean_bps']:>+9.2f}    {h2['se']:>5.2f}/{int(h2['n']):<4}  {sig1}/{sig2}")

    # ============ (B) MODEL vs BEST-SINGLE-FEATURE ============
    print("\n\n=== (B) MODEL vs BEST-SINGLE-FEATURE — is per-sym Ridge over-engineering? ===\n")
    h2_only = rdf[rdf["period"]=="H2"].sort_values("mean_bps", ascending=False)
    print(f"  TOP-3 single features for H2 long edge:")
    for i, r in h2_only.head(3).iterrows():
        print(f"    {r['feature']:<28} H2 long={r['mean_bps']:+.2f} bps (t={r['t']:+.2f}, n={int(r['n'])})")
    print(f"\n  Model's pred:")
    pred_h2 = rdf[(rdf["feature"]=="pred")&(rdf["period"]=="H2")].iloc[0]
    print(f"    {'pred (model)':<28} H2 long={pred_h2['mean_bps']:+.2f} bps (t={pred_h2['t']:+.2f}, n={int(pred_h2['n'])})")
    best_feat = h2_only.iloc[0]
    if best_feat["mean_bps"] > pred_h2["mean_bps"] + 1.0:
        print(f"\n  → BEST SINGLE FEATURE BEATS MODEL by {best_feat['mean_bps']-pred_h2['mean_bps']:+.2f} bps. Model is over-engineering.")
    else:
        print(f"\n  → Model's pred is at/above the best single feature. Model uses available info.")

    # ============ (C) TARGET-LEVEL: simpler targets ============
    print("\n\n=== (C) ALTERNATIVE TARGET — would different ranking signal work? ===\n")
    print("Test: rank by ANTI-momentum (low recent return = expected to bounce) vs PRO-momentum (high recent = expected to continue)\n")
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)].dropna(subset=["return_1d","return_pct"])
        anti_mom = []; pro_mom = []
        for ot, g in sub.groupby("open_time"):
            if len(g)<2*K: continue
            # ANTI-mom = recent losers as longs (mean-rev hypothesis)
            anti_mom.append(g.nsmallest(K, "return_1d")["return_pct"].mean())
            # PRO-mom = recent winners as longs (momentum hypothesis)
            pro_mom.append(g.nlargest(K, "return_1d")["return_pct"].mean())
        anti_mom = np.array(anti_mom); pro_mom = np.array(pro_mom)
        am_se = anti_mom.std()/np.sqrt(len(anti_mom))*1e4
        pm_se = pro_mom.std()/np.sqrt(len(pro_mom))*1e4
        am_mean = anti_mom.mean()*1e4
        pm_mean = pro_mom.mean()*1e4
        print(f"  {period_label}:")
        print(f"    Anti-momentum long (buy recent losers):  {am_mean:+.2f} bps  t={am_mean/am_se:+.2f}  n={len(anti_mom)}")
        print(f"    Pro-momentum long (buy recent winners):  {pm_mean:+.2f} bps  t={pm_mean/pm_se:+.2f}  n={len(pro_mom)}")
        print(f"    Δ pro − anti:                            {pm_mean-am_mean:+.2f} bps  (positive = momentum wins, negative = mean-rev wins)")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
