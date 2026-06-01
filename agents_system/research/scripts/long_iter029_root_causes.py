"""LONG-PRED iter-029 — Root cause investigation for two clear issues.

ISSUE A: Why does long selection alpha degrade so much from H1 (+40) to H2 (+7)?
ISSUE B: Why does short selection alpha always fail (negative in both periods)?

Uses LGBM V0 pooled predictions from iter-025 (the best model).

Phase A: Long degradation
  A1: Per-month long selection alpha — is it gradual decay or step change?
  A2: Cross-sectional dispersion per month
  A3: Per-decile analysis — does top-decile work, or only top-K?
  A4: Symbol contribution analysis — which syms drove H1 alpha?
  A5: Feature IC stability over time

Phase B: Short failure
  B1: Per-month short selection
  B2: Decile analysis of bottom predictions
  B3: Characteristics of "bad short picks"
  B4: Pump-fade timing test (do picks continue pumping briefly?)
  B5: Inverted long test (does flipping long picks work as shorts?)
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K = 5

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def main():
    t0 = time.time()
    print("=== iter-029: Root cause investigation ===\n", flush=True)

    print("loading data...", flush=True)
    preds = pd.read_parquet(PREDS)
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    cols = ["symbol","open_time","return_pct"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    df = preds.merge(panel.drop(columns=["return_pct"], errors="ignore"),
                     on=["symbol","open_time"], how="left")
    print(f"  preds×panel merged: {len(df):,} rows", flush=True)
    df = df[(df["open_time"] >= H1_START) & (df["open_time"] < H2_END)].copy()
    df["month"] = df["open_time"].dt.to_period("M").astype(str)

    # ============ PHASE A: LONG ALPHA DEGRADATION ============
    print(f"\n\n{'='*70}\n=== PHASE A: Why does long alpha degrade in H2? ===\n{'='*70}", flush=True)

    # --- A1: Per-month long selection alpha ---
    print(f"\n--- A1: Per-month long selection alpha ---")
    monthly = []
    for month, gm in df.groupby("month"):
        cycles = []
        for ot, gc in gm.groupby("open_time"):
            if len(gc) < 2*K: continue
            top = gc.nlargest(K, "pred")
            market_med = gc["return_pct"].median()
            cycles.append({"long_sel": top["return_pct"].mean() - market_med,
                          "market_med": market_med})
        if not cycles: continue
        cd = pd.DataFrame(cycles)
        a = cd["long_sel"].values * 1e4
        m = a.mean(); s = a.std()/np.sqrt(len(a)); t = m/s if s>0 else float("nan")
        monthly.append({"month": month, "long_sel_bps": m, "long_sel_t": t,
                       "market_med_bps": cd["market_med"].mean()*1e4, "n_cycles": len(cd)})
    mdf = pd.DataFrame(monthly).sort_values("month")
    print(f"{'month':<10} {'long_sel bps':>13} {'t':>6} {'market_med':>11} {'n_cycles':>9}")
    for _, r in mdf.iterrows():
        sig = "★" if abs(r["long_sel_t"])>1.96 else " "
        print(f"  {r['month']:<8} {r['long_sel_bps']:>+10.2f}{sig} {r['long_sel_t']:>+5.2f} "
              f"{r['market_med_bps']:>+9.2f}   {int(r['n_cycles']):>6}")

    # --- A2: Cross-sectional dispersion per month ---
    print(f"\n--- A2: Cross-sectional return dispersion per month ---")
    for month, gm in df.groupby("month"):
        cycle_stds = gm.groupby("open_time")["return_pct"].std() * 1e4
        cycle_iqrs = gm.groupby("open_time")["return_pct"].apply(
            lambda x: (x.quantile(0.75) - x.quantile(0.25))) * 1e4
        print(f"  {month}: mean cycle std={cycle_stds.mean():.1f} bps, IQR={cycle_iqrs.mean():.1f} bps")

    # --- A3: Per-decile analysis (predictions sorted into 10 buckets) ---
    print(f"\n--- A3: Per-decile mean return per period ---")
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        sub = sub.copy(); sub["decile"] = sub.groupby("open_time")["pred"].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
        dec = sub.groupby("decile")["return_pct"].agg(["mean","count"])
        dec["mean_bps"] = dec["mean"]*1e4
        print(f"\n  {period_label}:")
        market_med_bps = sub.groupby("open_time")["return_pct"].median().mean()*1e4
        for d, row in dec.iterrows():
            alpha = row["mean_bps"] - market_med_bps
            marker = "←TOP" if d == 9 else ("←BOT" if d == 0 else "")
            print(f"    decile {int(d)}: total {row['mean_bps']:>+7.1f}  alpha vs med {alpha:>+6.1f}  {marker}")

    # --- A4: Per-symbol contribution to long alpha ---
    print(f"\n--- A4: Top-10 syms by long-side contribution (H1 vs H2) ---")
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        contribs = []
        for ot, gc in sub.groupby("open_time"):
            if len(gc) < 2*K: continue
            top = gc.nlargest(K, "pred")
            market_med = gc["return_pct"].median()
            for _, r in top.iterrows():
                contribs.append({"symbol": r["symbol"], "alpha": r["return_pct"] - market_med})
        cdf = pd.DataFrame(contribs).groupby("symbol")["alpha"].agg(["mean","count","sum"])
        cdf["mean_bps"] = cdf["mean"]*1e4
        cdf["sum_bps"] = cdf["sum"]*1e4
        cdf = cdf.sort_values("sum_bps", ascending=False).head(10)
        print(f"\n  {period_label} top-10 sym contributors (when picked as long, alpha vs median):")
        print(f"    {'symbol':<14} {'n picks':>8} {'mean bps':>10} {'cumulative bps':>16}")
        for sym, r in cdf.iterrows():
            print(f"    {sym:<12} {int(r['count']):>6}   {r['mean_bps']:>+8.1f}   {r['sum_bps']:>+12.0f}")

    # --- A5: Feature IC stability across months ---
    print(f"\n--- A5: Feature IC per month (top 5 features by variance of IC) ---")
    feat_ics = {f: [] for f in V0_FEATURES}
    months_list = []
    for month, gm in df.groupby("month"):
        months_list.append(month)
        for f in V0_FEATURES:
            sub = gm[[f, "return_pct"]].dropna()
            if len(sub) < 500:
                feat_ics[f].append(np.nan); continue
            ic, _ = spearmanr(sub[f], sub["return_pct"])
            feat_ics[f].append(ic)
    ic_df = pd.DataFrame(feat_ics, index=months_list).T
    ic_df["mean_abs_ic"] = ic_df.abs().mean(axis=1)
    ic_df["ic_std"] = ic_df.iloc[:, :len(months_list)].std(axis=1)
    print(f"\n  IC of each V0 feature vs forward return (per month):")
    print(f"  {'feature':<28} " + " ".join(f"{m:>8}" for m in months_list))
    for f in V0_FEATURES:
        row = ic_df.loc[f, months_list]
        print(f"  {f:<26} " + " ".join(f"{x:>+8.3f}" if not np.isnan(x) else "       —" for x in row))

    # ============ PHASE B: SHORT SELECTION FAILURE ============
    print(f"\n\n{'='*70}\n=== PHASE B: Why does short selection always fail? ===\n{'='*70}", flush=True)

    # --- B1: Per-month short selection ---
    print(f"\n--- B1: Per-month short selection alpha ---")
    monthly = []
    for month, gm in df.groupby("month"):
        cycles = []
        for ot, gc in gm.groupby("open_time"):
            if len(gc) < 2*K: continue
            bot = gc.nsmallest(K, "pred")
            market_med = gc["return_pct"].median()
            # short selection = market - picks (positive means picks decline MORE than market)
            cycles.append({"short_sel": market_med - bot["return_pct"].mean()})
        if not cycles: continue
        cd = pd.DataFrame(cycles)
        a = cd["short_sel"].values * 1e4
        m = a.mean(); s = a.std()/np.sqrt(len(a)); t = m/s if s>0 else float("nan")
        monthly.append({"month": month, "short_sel_bps": m, "short_sel_t": t, "n_cycles": len(cd)})
    sdf = pd.DataFrame(monthly).sort_values("month")
    print(f"  {'month':<10} {'short_sel bps':>14} {'t':>6} {'n_cycles':>9}")
    for _, r in sdf.iterrows():
        sig = "★" if abs(r["short_sel_t"])>1.96 else " "
        print(f"  {r['month']:<8} {r['short_sel_bps']:>+11.2f}{sig} {r['short_sel_t']:>+5.2f}   {int(r['n_cycles']):>6}")
    pct_pos = (sdf["short_sel_bps"] > 0).mean()*100
    print(f"\n  % months with positive short_selection: {pct_pos:.0f}%")

    # --- B3: Characteristics of model shorts vs ideal shorts ---
    print(f"\n--- B3: Picked short characteristics — model vs IDEAL (lowest realized return) ---")
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        model_short_chars, ideal_short_chars = [], []
        for ot, gc in sub.groupby("open_time"):
            if len(gc) < 2*K: continue
            model_bot = gc.nsmallest(K, "pred")
            ideal_bot = gc.nsmallest(K, "return_pct")  # cheating: pick on realized return
            for c in ["return_1d","funding_rate","rvol_7d","bars_since_high","ret_3d"]:
                model_short_chars.append({"feature": c, "value": model_bot[c].mean()})
                ideal_short_chars.append({"feature": c, "value": ideal_bot[c].mean()})
        m_df = pd.DataFrame(model_short_chars).groupby("feature")["value"].mean()
        i_df = pd.DataFrame(ideal_short_chars).groupby("feature")["value"].mean()
        print(f"\n  {period_label}: features of MODEL shorts vs IDEAL shorts (best in hindsight)")
        print(f"  {'feature':<22} {'model picks':>13} {'ideal picks':>13} {'diff':>9}")
        for f in ["return_1d","funding_rate","rvol_7d","bars_since_high","ret_3d"]:
            mv = m_df[f]; iv = i_df[f]
            print(f"  {f:<22} {mv:>+10.4f}   {iv:>+10.4f}  {mv-iv:>+7.4f}")

    # --- B4: Pump-fade timing test (do picks pump first, then crash?) ---
    print(f"\n--- B4: Recently-pumped names test ---")
    # Are model's short picks "recently pumped"? (high ret_3d means recent pump)
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        picks = []
        for ot, gc in sub.groupby("open_time"):
            if len(gc) < 2*K: continue
            bot = gc.nsmallest(K, "pred")
            top = gc.nlargest(K, "pred")
            picks.append({"period":period_label, "short_ret3d": bot["ret_3d"].mean(),
                         "long_ret3d": top["ret_3d"].mean(),
                         "market_ret3d": gc["ret_3d"].median()})
        pdf = pd.DataFrame(picks)
        print(f"  {period_label}: avg recent-3d return")
        print(f"    short picks ret_3d: {pdf['short_ret3d'].mean()*100:+.2f}%")
        print(f"    long picks ret_3d: {pdf['long_ret3d'].mean()*100:+.2f}%")
        print(f"    market median ret_3d: {pdf['market_ret3d'].mean()*100:+.2f}%")
        if pdf["short_ret3d"].mean() > pdf["market_ret3d"].mean():
            print(f"    → model shorts are RECENTLY PUMPED names (ret_3d > market)")

    # --- B5: Inverted long test — does flipping long picks work better as shorts? ---
    print(f"\n--- B5: Inverted long test — use TOP-K names but go SHORT them ---")
    print(f"  Hypothesis: model's TOP-K (highest predicted alpha) WOULD pump if longs work.")
    print(f"  If we SHORT them instead, we lose money. The TRUE 'good shorts' might")
    print(f"  be NOT identified at all by the cross-sectional alpha model.")
    print()
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        # Compare 4 strategies
        rows = []
        for ot, gc in sub.groupby("open_time"):
            if len(gc) < 2*K: continue
            top = gc.nlargest(K, "pred")["return_pct"].mean()
            bot = gc.nsmallest(K, "pred")["return_pct"].mean()
            market_med = gc["return_pct"].median()
            rows.append({"top": top, "bot": bot, "med": market_med})
        rdf = pd.DataFrame(rows)
        # As-is short: -bot
        as_is_short = -rdf["bot"].mean()*1e4
        # Inverted long short: -top
        inverted_short = -rdf["top"].mean()*1e4
        # Random short (median)
        random_short = -rdf["med"].mean()*1e4
        print(f"  {period_label} short variants (per-cycle bps return on $1 short):")
        print(f"    Model-as-is (short bot-K): {as_is_short:+.2f}")
        print(f"    Inverted-long (short top-K): {inverted_short:+.2f}")
        print(f"    Random short (market median): {random_short:+.2f}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
