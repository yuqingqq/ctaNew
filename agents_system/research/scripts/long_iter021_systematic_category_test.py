"""LONG-PRED iter-021 — SYSTEMATIC category-by-category feature evaluation

For each feature category, train a per-sym Ridge model with V0 + that category
and measure both LONG-K and SHORT-K edges in H1 and H2.

This isolates which category (if any) adds real value to V0 features.

Variants tested:
  V0       — 17 V0 features (baseline)
  +A_xsranks    — V0 + cross-sectional ranks of all V0 features
  +B_momdivs    — V0 + momentum divergences
  +C_volratios  — V0 + vol/funding ratios
  +D_interactions — V0 + 2-way products
  +E_anomalies  — V0 + anomaly z-scores
  +F_compounds  — V0 + multi-signal pump-setup heuristics
  +ALL          — V0 + every engineered feature
  ONLY_new      — only engineered features, no V0
"""
import sys, time, pickle
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter021"; OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
FIT_CUTOFF = pd.Timestamp("2025-10-02",tz="UTC")
K = 5
CYCLES_PER_DAY = 6
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def engineer_all_features(panel: pd.DataFrame):
    print("engineering features (one-time)...", flush=True)
    g = panel.groupby("symbol", group_keys=False)
    for days in [7, 14, 30, 60]:
        bars = days * CYCLES_PER_DAY
        panel[f"ret_{days}d"] = g["return_1d"].transform(lambda x: x.rolling(bars, min_periods=bars//2).sum().shift(1))
    panel["idio_vol_30d"] = g["idio_vol_to_btc_1d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["rvol_30d"] = g["rvol_7d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["funding_30d"] = g["funding_rate"].transform(lambda x: x.rolling(180, min_periods=90).sum().shift(1))

    # Category A: XS ranks
    catA = []
    for feat in V0_FEATURES:
        col = f"xsrank_{feat}"; panel[col] = panel.groupby("open_time")[feat].rank(pct=True); catA.append(col)

    # Category B: Momentum divergences
    panel["mom_div_1d_7d"] = panel["return_1d"] - panel["ret_7d"]/7
    panel["mom_div_3d_30d"] = panel["ret_3d"]/3 - panel["ret_30d"]/30
    panel["mom_div_7d_60d"] = panel["ret_7d"]/7 - panel["ret_60d"]/60
    panel["mom_accel"] = panel["return_1d"] - 0.5*(panel["ret_7d"]/7 + panel["ret_30d"]/30)
    catB = ["mom_div_1d_7d","mom_div_3d_30d","mom_div_7d_60d","mom_accel"]

    # Category C: Vol/funding ratios
    panel["vol_ratio_1h_1d"] = panel["idio_vol_to_btc_1h"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)
    panel["vol_ratio_7d_30d"] = panel["rvol_7d"] / panel["rvol_30d"].replace(0, np.nan)
    panel["vol_ratio_recent_30d"] = panel["idio_vol_to_btc_1d"] / panel["idio_vol_30d"].replace(0, np.nan)
    panel["funding_per_vol"] = panel["funding_rate"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)
    catC = ["vol_ratio_1h_1d","vol_ratio_7d_30d","vol_ratio_recent_30d","funding_per_vol"]

    # Category D: Interactions
    panel["vol_x_mom"] = panel["rvol_7d"] * panel["return_1d"]
    panel["vol_x_neg_mom"] = panel["rvol_7d"] * (-panel["return_1d"])
    panel["funding_x_beta"] = panel["funding_rate"] * panel["beta_to_btc_change_5d"]
    panel["bars_x_vol"] = panel["bars_since_high"] * panel["rvol_7d"]
    catD = ["vol_x_mom","vol_x_neg_mom","funding_x_beta","bars_x_vol"]

    # Category E: Anomaly z-scores
    catE = []
    for feat in ["return_1d","rvol_7d","funding_rate","idio_vol_to_btc_1d"]:
        rm = g[feat].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
        rs = g[feat].transform(lambda x: x.rolling(180, min_periods=90).std().shift(1))
        col = f"z_{feat}"; panel[col] = (panel[feat] - rm) / rs.replace(0, np.nan); catE.append(col)

    # Category F: Multi-signal compounds
    panel["pump_setup_1"] = (-panel["return_1d"].clip(lower=-0.05, upper=0.05)) * \
                            (-panel["funding_rate"]) * panel["vol_ratio_recent_30d"]
    panel["breakout"] = panel["xsrank_bars_since_high"] * panel["vol_ratio_recent_30d"]
    panel["consolidating"] = (1 - panel["xsrank_bars_since_high"]) * (1/panel["vol_ratio_recent_30d"].clip(0.1, 10))
    panel["not_overcrowded"] = -panel["funding_rate"].abs()
    panel["fresh_breakout"] = panel["mom_accel"].clip(lower=0, upper=0.05) * (1 / (1 + panel["funding_rate"].abs()))
    catF = ["pump_setup_1","breakout","consolidating","not_overcrowded","fresh_breakout"]

    print(f"  Categories: A={len(catA)} B={len(catB)} C={len(catC)} D={len(catD)} E={len(catE)} F={len(catF)}", flush=True)
    return panel, {"A": catA, "B": catB, "C": catC, "D": catD, "E": catE, "F": catF}

def fit_preproc(train_df, feat_cols):
    stats = {}
    for c in feat_cols:
        if c not in train_df.columns: continue
        v = train_df[c].dropna()
        if len(v) < 100: stats[c] = {"lo":0.,"hi":1.,"mu":0.,"sd":1.}; continue
        lo, hi = float(v.quantile(0.01)), float(v.quantile(0.99))
        vc = v.clip(lo, hi)
        stats[c] = {"lo":lo,"hi":hi,"mu":float(vc.mean()),"sd":float(vc.std()) or 1.0}
    return stats

def apply_preproc(df, feat_cols, stats):
    X = np.zeros((len(df), len(feat_cols)), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        if c not in stats or c not in df.columns: continue
        s = stats[c]; v = df[c].to_numpy()
        X[:, i] = (np.clip(v, s["lo"], s["hi"]) - s["mu"]) / s["sd"]
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def train_and_predict(panel, feat_cols, label):
    """Train per-sym Ridge, return OOS predictions."""
    print(f"\n  --- {label} ({len(feat_cols)} features) ---", flush=True)
    t0 = time.time()
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel["target_z"].notna()]
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)]
    models, stats_all = {}, {}; n_ok = 0
    chosen_alphas = []
    for sym, gtr in train.groupby("symbol"):
        if len(gtr) < 300: continue
        s = fit_preproc(gtr, feat_cols)
        Xtr = apply_preproc(gtr, feat_cols, s)
        try:
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, gtr["target_z"].to_numpy())
            models[sym] = m; stats_all[sym] = s; n_ok += 1
            chosen_alphas.append(m.alpha_)
        except: pass
    # alpha distribution
    alpha_dist = pd.Series(chosen_alphas).value_counts().sort_index().to_dict()
    preds_list = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models: continue
        Xv = apply_preproc(gv, feat_cols, stats_all[sym])
        pv = models[sym].predict(Xv)
        out = gv[["symbol","open_time","return_pct"]].copy(); out["pred"] = pv
        preds_list.append(out)
    preds = pd.concat(preds_list, ignore_index=True).sort_values(["open_time","symbol"])
    print(f"    fit {n_ok} syms; alpha distribution {alpha_dist} [{time.time()-t0:.0f}s]", flush=True)
    return preds

def measure_edges(preds_df, label):
    """Top-K and bot-K edges in H1, H2."""
    results = {}
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = preds_df[(preds_df["open_time"]>=s) & (preds_df["open_time"]<e)]
        top_means = sub.groupby("open_time").apply(
            lambda g: g.nlargest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        bot_means = sub.groupby("open_time").apply(
            lambda g: -g.nsmallest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        for leg_label, arr in [("long", top_means), ("short", bot_means)]:
            a = arr.values * 1e4
            mean = a.mean(); se = a.std()/np.sqrt(len(a)); t = mean/se if se>0 else float("nan")
            results[f"{period_label}_{leg_label}_mean"] = mean
            results[f"{period_label}_{leg_label}_t"] = t
    return results

def main():
    t0 = time.time()
    print("=== iter-021: Systematic category-by-category evaluation ===\n", flush=True)

    print("loading panel...", flush=True)
    cols = ["symbol","open_time","exit_time","return_pct","target_z","alpha_vs_btc_realized"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    panel, categories = engineer_all_features(panel)

    # Define feature variants
    variants = {
        "V0_baseline":     V0_FEATURES,
        "V0+A_xsranks":    V0_FEATURES + categories["A"],
        "V0+B_momdivs":    V0_FEATURES + categories["B"],
        "V0+C_volratios":  V0_FEATURES + categories["C"],
        "V0+D_interactions": V0_FEATURES + categories["D"],
        "V0+E_anomalies":  V0_FEATURES + categories["E"],
        "V0+F_compounds":  V0_FEATURES + categories["F"],
        "V0+ALL_new":      V0_FEATURES + categories["A"] + categories["B"] + categories["C"] + categories["D"] + categories["E"] + categories["F"],
    }
    all_results = []
    for label, feats in variants.items():
        preds = train_and_predict(panel, feats, label)
        edges = measure_edges(preds, label)
        edges["label"] = label; edges["n_features"] = len(feats)
        all_results.append(edges)
        preds.to_parquet(OUT_DIR/f"preds_{label}.parquet")

    rdf = pd.DataFrame(all_results)
    print(f"\n=== SYSTEMATIC RESULTS ===\n")
    print(f"{'variant':<24} {'n_feat':>7} {'H1_long':>9} {'H1_t':>6} {'H2_long':>9} {'H2_t':>6} {'H1_short':>10} {'H2_short':>10}")
    print("-"*100)
    for _, r in rdf.iterrows():
        sig_long_h2 = "★" if abs(r["H2_long_t"])>1.96 else " "
        sig_short_h2 = "★" if abs(r["H2_short_t"])>1.96 else " "
        print(f"  {r['label']:<24} {int(r['n_features']):>5}  "
              f"{r['H1_long_mean']:>+7.2f}  {r['H1_long_t']:>+5.2f} "
              f"{r['H2_long_mean']:>+7.2f}  {r['H2_long_t']:>+5.2f} {sig_long_h2} "
              f"{r['H1_short_mean']:>+8.2f} {r['H2_short_mean']:>+8.2f} {sig_short_h2}")

    print(f"\n=== VERDICT ===")
    # Best long edge for H2
    best_h2_long = rdf.sort_values("H2_long_mean", ascending=False).iloc[0]
    best_full_long = rdf.sort_values("H1_long_mean", ascending=False).iloc[0]
    print(f"  Best H2 long: {best_h2_long['label']} edge={best_h2_long['H2_long_mean']:+.2f} t={best_h2_long['H2_long_t']:+.2f}")
    print(f"  Best H1 long: {best_full_long['label']} edge={best_full_long['H1_long_mean']:+.2f} t={best_full_long['H1_long_t']:+.2f}")
    # Categories that meaningfully helped H2 long
    baseline_h2 = rdf[rdf["label"]=="V0_baseline"]["H2_long_mean"].iloc[0]
    print(f"\n  Δ vs V0 baseline ({baseline_h2:+.2f}):")
    for _, r in rdf.iterrows():
        d = r["H2_long_mean"] - baseline_h2
        marker = "✓" if d > 1.0 else ("≈" if abs(d)<1.0 else "✗")
        print(f"    {r['label']:<24} Δ={d:+.2f} bps  {marker}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
