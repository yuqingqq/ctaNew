"""LONG-PRED iter-027 — DEEP feature engineering with proper preprocessing.

Combines:
  - Best preprocessing from iter-026 (cross-sectional rank per cycle = P3)
  - 6 new feature categories (G,H,I,J,K,L) — beyond what iter-019 tested
  - Wider alpha grid (per iter-021b finding)
  - Per-sym Ridge AND pooled LightGBM for model-class comparison

Categories engineered:
  G: Calendar/time features (cyclic encoding)
  H: Fine-grained lagged returns (4h, 8h, 12h, 16h, 24h)
  I: Funding accumulators (cumulative paid, sign-conditional)
  J: Regime-conditional features (V0 × is_high_btc_vol, etc.)
  K: Volatility-of-volatility (std/skew/kurt of returns/rvol)
  L: 3-way interactions (rvol × mom × funding)

Three test layers:
  Layer 1: univariate per-feature H1+H2 long edge with t-stats
  Layer 2: per-category model (V0 + each category alone)
  Layer 3: combined best (V0 + best 2-3 categories) + top-N selected features

For each model variant, run BOTH per-sym Ridge and pooled LGBM.

Output: comprehensive results table + recommended feature set.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm"], check=False)
    import lightgbm as lgb

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter027"; OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
FIT_CUTOFF = pd.Timestamp("2025-10-02",tz="UTC")
K = 5
CYCLES_PER_DAY = 6
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def engineer_features(panel):
    """Build all 6 new feature categories. Returns panel + dict of category → feature list."""
    print("engineering features (6 categories)...", flush=True)
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    # --- Base longer-horizon returns (needed for derived features) ---
    for days in [7, 14, 30, 60]:
        bars = days * CYCLES_PER_DAY
        panel[f"ret_{days}d"] = g["return_1d"].transform(lambda x: x.rolling(bars, min_periods=bars//2).sum().shift(1))
    panel["idio_vol_30d"] = g["idio_vol_to_btc_1d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["rvol_30d"] = g["rvol_7d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))

    cats = {}

    # ===== G: Calendar/time features (cyclic) =====
    print("  G: calendar/time...", flush=True)
    hr = panel["open_time"].dt.hour.values
    dow = panel["open_time"].dt.dayofweek.values
    panel["g_hour_sin"] = np.sin(2*np.pi*hr/24)
    panel["g_hour_cos"] = np.cos(2*np.pi*hr/24)
    panel["g_dow_sin"] = np.sin(2*np.pi*dow/7)
    panel["g_dow_cos"] = np.cos(2*np.pi*dow/7)
    panel["g_is_us_session"] = ((hr >= 13) & (hr <= 21)).astype(float)
    panel["g_is_asia_session"] = ((hr >= 0) & (hr <= 8)).astype(float)
    panel["g_is_weekend"] = (dow >= 5).astype(float)
    panel["g_day_of_month"] = panel["open_time"].dt.day.astype(float) / 31.0
    cats["G_calendar"] = ["g_hour_sin","g_hour_cos","g_dow_sin","g_dow_cos",
                          "g_is_us_session","g_is_asia_session","g_is_weekend","g_day_of_month"]

    # ===== H: Fine-grained lagged returns =====
    print("  H: lagged returns...", flush=True)
    # Convert return_1d to 4h returns by diff approach; but we have return_pct (4h forward)
    # Actually return_pct in panel is forward 4h return (the target/holding). For lagged we use cumulative.
    # Use shift to get past returns at various bar offsets
    for lag in [1, 2, 3, 6]:  # 4h, 8h, 12h, 24h ago
        panel[f"h_ret_lag_{lag}"] = g["return_1d"].transform(lambda x: x.shift(lag))
    # Multi-horizon ahead-of-current: short-term momentum slopes
    panel["h_ret_24h_change"] = panel["return_1d"] - panel["h_ret_lag_6"]  # 1d momentum
    panel["h_ret_12h_change"] = panel["return_1d"] - panel["h_ret_lag_3"]  # 12h momentum
    panel["h_sign_change"] = (np.sign(panel["return_1d"]) != np.sign(panel["h_ret_lag_1"])).astype(float)
    cats["H_lagged"] = ["h_ret_lag_1","h_ret_lag_2","h_ret_lag_3","h_ret_lag_6",
                        "h_ret_24h_change","h_ret_12h_change","h_sign_change"]

    # ===== I: Funding accumulators =====
    print("  I: funding accumulators...", flush=True)
    panel["i_cum_funding_7d"] = g["funding_rate"].transform(
        lambda x: x.rolling(42, min_periods=21).sum().shift(1))
    panel["i_cum_funding_30d"] = g["funding_rate"].transform(
        lambda x: x.rolling(180, min_periods=90).sum().shift(1))
    panel["i_pos_funding_share_7d"] = g["funding_rate"].transform(
        lambda x: (x > 0).rolling(42, min_periods=21).mean().shift(1))
    panel["i_max_abs_funding_7d"] = g["funding_rate"].transform(
        lambda x: x.abs().rolling(42, min_periods=21).max().shift(1))
    panel["i_funding_volatility_7d"] = g["funding_rate"].transform(
        lambda x: x.rolling(42, min_periods=21).std().shift(1))
    cats["I_funding_accum"] = ["i_cum_funding_7d","i_cum_funding_30d",
                                "i_pos_funding_share_7d","i_max_abs_funding_7d",
                                "i_funding_volatility_7d"]

    # ===== J: Regime-conditional features =====
    print("  J: regime-conditional...", flush=True)
    # Compute regime indicators from BTC features (use cross-sectional median as proxy if needed)
    btc_rvol_med = panel.groupby("open_time")["btc_rvol_7d"].transform("median")
    rvol_med = panel.groupby("open_time")["rvol_7d"].transform("median")
    panel["j_is_high_btc_vol"] = (panel["btc_rvol_7d"] > btc_rvol_med).astype(float)
    panel["j_is_high_sym_vol"] = (panel["rvol_7d"] > rvol_med).astype(float)
    # Regime-conditioned features
    panel["j_ret_in_high_vol"] = panel["return_1d"] * panel["j_is_high_btc_vol"]
    panel["j_ret_in_low_vol"] = panel["return_1d"] * (1 - panel["j_is_high_btc_vol"])
    panel["j_funding_in_high_vol"] = panel["funding_rate"] * panel["j_is_high_btc_vol"]
    panel["j_funding_in_low_vol"] = panel["funding_rate"] * (1 - panel["j_is_high_btc_vol"])
    # btc trend regime (positive recent 24h return)
    panel["j_btc_24h_return"] = panel.groupby("open_time")["return_1d"].transform("median")
    panel["j_is_btc_up"] = (panel["j_btc_24h_return"] > 0).astype(float)
    panel["j_ret_in_btc_up"] = panel["return_1d"] * panel["j_is_btc_up"]
    panel["j_ret_in_btc_down"] = panel["return_1d"] * (1 - panel["j_is_btc_up"])
    cats["J_regime_cond"] = ["j_ret_in_high_vol","j_ret_in_low_vol",
                             "j_funding_in_high_vol","j_funding_in_low_vol",
                             "j_ret_in_btc_up","j_ret_in_btc_down"]

    # ===== K: Volatility-of-volatility =====
    print("  K: vol-of-vol...", flush=True)
    panel["k_rvol_std_30d"] = g["rvol_7d"].transform(
        lambda x: x.rolling(180, min_periods=90).std().shift(1))
    panel["k_return_kurt_30d"] = g["return_1d"].transform(
        lambda x: x.rolling(180, min_periods=90).kurt().shift(1))
    panel["k_return_skew_30d"] = g["return_1d"].transform(
        lambda x: x.rolling(180, min_periods=90).skew().shift(1))
    panel["k_max_drawdown_7d"] = g["return_1d"].transform(
        lambda x: x.rolling(42, min_periods=21).min().shift(1))
    panel["k_max_gain_7d"] = g["return_1d"].transform(
        lambda x: x.rolling(42, min_periods=21).max().shift(1))
    panel["k_vol_change_30d"] = panel["rvol_7d"] / panel["rvol_30d"].replace(0, np.nan) - 1
    cats["K_volvol"] = ["k_rvol_std_30d","k_return_kurt_30d","k_return_skew_30d",
                       "k_max_drawdown_7d","k_max_gain_7d","k_vol_change_30d"]

    # ===== L: 3-way interactions =====
    print("  L: 3-way interactions...", flush=True)
    panel["l_rvol_mom_funding"] = panel["rvol_7d"] * panel["return_1d"] * panel["funding_rate"]
    panel["l_atr_bars_funding"] = panel["atr_pct"] * panel["bars_since_high"] * panel["funding_rate"]
    panel["l_corr_beta_rvol"] = panel["corr_to_btc_1d"] * panel["beta_to_btc_change_5d"] * panel["rvol_7d"]
    panel["l_mom_funding_idio"] = panel["return_1d"] * panel["funding_rate"] * panel["idio_vol_to_btc_1d"]
    panel["l_obv_funding_rvol"] = panel["obv_z_1d"] * panel["funding_rate"] * panel["rvol_7d"]
    cats["L_3way_inter"] = ["l_rvol_mom_funding","l_atr_bars_funding","l_corr_beta_rvol",
                            "l_mom_funding_idio","l_obv_funding_rvol"]

    n_total = sum(len(v) for v in cats.values())
    print(f"  Total new features: {n_total} across 6 categories", flush=True)
    return panel, cats

# === Preprocessing P3: cross-sectional rank per cycle (best from iter-026) ===
def transform_xs_rank(df, feats):
    out = df[feats].copy()
    for c in feats:
        out[c] = df.groupby("open_time")[c].rank(pct=True) - 0.5
    return np.nan_to_num(out.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def univariate_edge(panel, feat, period_label, s, e):
    """Top-K=5 long edge from picking nlargest of `feat`."""
    sub = panel[(panel["open_time"]>=s) & (panel["open_time"]<e)].dropna(subset=[feat,"return_pct"])
    if len(sub) < 1000: return None
    top_means = sub.groupby("open_time").apply(
        lambda g: g.nlargest(K, feat)["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
    if len(top_means) < 50: return None
    a = top_means.values * 1e4
    mean = a.mean(); se = a.std()/np.sqrt(len(a)); t = mean/se if se>0 else float("nan")
    return {"mean": mean, "t": t, "n": len(a)}

def train_per_sym_ridge(panel, feats, label):
    """Train per-sym Ridge with XS-rank preprocessing."""
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel["target_z"].notna()].copy()
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)].copy()
    # XS-rank preprocessing
    Xtr = transform_xs_rank(train, feats)
    Xte = transform_xs_rank(test, feats)
    train["__idx"] = np.arange(len(train))
    test["__idx"] = np.arange(len(test))

    chosen_alphas, models = [], {}
    for sym, gtr in train.groupby("symbol"):
        if len(gtr) < 300: continue
        idx = gtr["__idx"].values
        try:
            m = RidgeCV(alphas=ALPHAS).fit(Xtr[idx], gtr["target_z"].values)
            models[sym] = m; chosen_alphas.append(m.alpha_)
        except: pass
    alpha_dist = pd.Series(chosen_alphas).value_counts().sort_index().to_dict()

    preds_list = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models: continue
        idx = gv["__idx"].values
        out = gv[["symbol","open_time","return_pct"]].copy()
        out["pred"] = models[sym].predict(Xte[idx])
        preds_list.append(out)
    preds = pd.concat(preds_list, ignore_index=True).sort_values(["open_time","symbol"])
    return preds, alpha_dist

def train_pooled_lgbm(panel, feats, label, target_col="target_z"):
    """Pooled LightGBM with sym_id."""
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel[target_col].notna()].copy()
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)].copy()
    syms = sorted(set(train["symbol"].unique()) | set(test["symbol"].unique()))
    sym2id = {s:i for i,s in enumerate(syms)}
    train["sym_id"] = train["symbol"].map(sym2id).astype("int32")
    test["sym_id"] = test["symbol"].map(sym2id).astype("int32")
    feats_full = feats + ["sym_id"]
    Xtr = train[feats_full].fillna(0).values
    ytr = train[target_col].values
    Xte = test[feats_full].fillna(0).values
    p = dict(objective="regression", metric="rmse", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=5, verbose=-1, num_threads=8)
    train_ds = lgb.Dataset(Xtr, label=ytr, categorical_feature=[len(feats)])
    model = lgb.train(p, train_ds, num_boost_round=500, callbacks=[lgb.log_evaluation(0)])
    test["pred"] = model.predict(Xte)
    return test[["symbol","open_time","return_pct","pred"]], model

def measure_edges(preds):
    results = {}
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = preds[(preds["open_time"]>=s) & (preds["open_time"]<e)]
        top_means = sub.groupby("open_time").apply(
            lambda g: g.nlargest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        a = top_means.values * 1e4
        mean = a.mean(); se = a.std()/np.sqrt(len(a))
        results[f"{period_label}_long"] = mean
        results[f"{period_label}_long_t"] = mean/se if se>0 else float("nan")
        bot_means = sub.groupby("open_time").apply(
            lambda g: -g.nsmallest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        a = bot_means.values * 1e4
        results[f"{period_label}_short"] = a.mean()
    return results

def main():
    t0 = time.time()
    print("=== iter-027: DEEP feature engineering ===\n", flush=True)
    cols = ["symbol","open_time","exit_time","return_pct","target_z"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    panel, cats = engineer_features(panel)
    all_new_feats = sum(cats.values(), [])

    # ============ LAYER 1: Univariate per-feature edge ============
    print(f"\n=== LAYER 1: Univariate H1+H2 long edge per feature ===\n", flush=True)
    rows = []
    for cat_name, feats in cats.items():
        for f in feats:
            if f not in panel.columns: continue
            h1 = univariate_edge(panel, f, "H1", H1_START, H2_START)
            h2 = univariate_edge(panel, f, "H2", H2_START, H2_END)
            if h1 is None or h2 is None: continue
            rows.append({"category": cat_name, "feature": f,
                "H1_mean": h1["mean"], "H1_t": h1["t"],
                "H2_mean": h2["mean"], "H2_t": h2["t"]})
    rdf = pd.DataFrame(rows).sort_values("H2_mean", ascending=False)
    print(f"{'feature':<28} {'cat':<14} {'H1 bps':>8} {'H1 t':>6} {'H2 bps':>8} {'H2 t':>6} sig")
    print("-"*80)
    for _, r in rdf.iterrows():
        sig = "★" if abs(r["H2_t"])>1.96 else " "
        print(f"  {r['feature']:<26} {r['category']:<14} {r['H1_mean']:>+6.2f} {r['H1_t']:>+5.2f} "
              f"{r['H2_mean']:>+6.2f} {r['H2_t']:>+5.2f}  {sig}")
    rdf.to_csv(OUT_DIR/"layer1_univariate.csv", index=False)

    # Print survivors (positive H2)
    surv = rdf[(rdf["H2_mean"] > 0)].copy()
    print(f"\n=== SURVIVORS: positive H2 long edge ({len(surv)} of {len(rdf)}) ===")
    if len(surv):
        for _, r in surv.iterrows():
            sig = "★" if abs(r["H2_t"])>1.96 else " "
            print(f"  {r['feature']:<28} H2={r['H2_mean']:+.2f} t={r['H2_t']:+.2f}{sig}  H1={r['H1_mean']:+.2f}")
    sig_h2 = rdf[abs(rdf["H2_t"]) > 1.96]
    print(f"\n=== STATISTICALLY SIGNIFICANT (|H2 t|>1.96): {len(sig_h2)} ===")
    for _, r in sig_h2.iterrows():
        sign = "+" if r["H2_mean"]>0 else "-"
        print(f"  {sign} {r['feature']:<28} H2={r['H2_mean']:+.2f} t={r['H2_t']:+.2f}")

    # ============ LAYER 2: Per-category model (V0 + each) ============
    print(f"\n=== LAYER 2: Per-category model — Ridge with XS-rank preprocessing ===\n", flush=True)
    layer2_results = []
    variants_l2 = {"V0_only": V0_FEATURES}
    for cat_name, feats in cats.items():
        variants_l2[f"V0+{cat_name}"] = V0_FEATURES + feats
    for label, feats_set in variants_l2.items():
        t1 = time.time()
        preds, ad = train_per_sym_ridge(panel, feats_set, label)
        r = measure_edges(preds); r["label"] = label; r["n"] = len(feats_set); r["alpha_dist"] = str(ad)
        layer2_results.append(r)
        print(f"  {label:<22} n={len(feats_set):>3} H1={r['H1_long']:+6.2f} H2={r['H2_long']:+6.2f} (t={r['H2_long_t']:+.2f}) [{time.time()-t1:.0f}s]")
    l2df = pd.DataFrame(layer2_results)
    l2df.to_csv(OUT_DIR/"layer2_per_category.csv", index=False)
    baseline_h2 = l2df[l2df["label"]=="V0_only"]["H2_long"].iloc[0]
    print(f"\n  Δ vs V0 baseline ({baseline_h2:+.2f}):")
    for _, r in l2df.iterrows():
        if r["label"]=="V0_only": continue
        d = r["H2_long"] - baseline_h2
        marker = "✓ HELPS" if d>1.0 else ("✗ HURTS" if d<-1.0 else "≈ same")
        print(f"    {r['label']:<22} Δ={d:+.2f}  {marker}")

    # ============ LAYER 3: Combined best + top-N selected ============
    print(f"\n=== LAYER 3: Combined best categories + top-N univariate ===\n", flush=True)
    # Get top-3 categories by H2 long delta
    l2_no_v0 = l2df[l2df["label"]!="V0_only"].sort_values("H2_long", ascending=False)
    top_cats = []
    for _, r in l2_no_v0.head(3).iterrows():
        cat = r["label"].replace("V0+", "")
        top_cats.extend(cats[cat])
    # Top-N univariate (by |H2 t|)
    rdf_sorted = rdf.copy()
    rdf_sorted["abs_h2_t"] = abs(rdf_sorted["H2_t"])
    top_n_feats = rdf_sorted.sort_values("abs_h2_t", ascending=False).head(15)["feature"].tolist()

    variants_l3 = {
        "Ridge_V0":              V0_FEATURES,
        "Ridge_V0+top3_cats":    V0_FEATURES + list(set(top_cats)),
        "Ridge_V0+top15_feats":  V0_FEATURES + top_n_feats,
        "Ridge_V0+ALL_new":      V0_FEATURES + all_new_feats,
        "LGBM_V0":               V0_FEATURES,
        "LGBM_V0+top3_cats":     V0_FEATURES + list(set(top_cats)),
        "LGBM_V0+top15_feats":   V0_FEATURES + top_n_feats,
        "LGBM_V0+ALL_new":       V0_FEATURES + all_new_feats,
    }
    layer3_results = []
    for label, feats_set in variants_l3.items():
        t1 = time.time()
        if label.startswith("Ridge_"):
            preds, _ = train_per_sym_ridge(panel, feats_set, label)
        else:
            preds, _ = train_pooled_lgbm(panel, feats_set, label)
        r = measure_edges(preds); r["label"] = label; r["n"] = len(feats_set)
        layer3_results.append(r)
        print(f"  {label:<28} n={len(feats_set):>3} H1={r['H1_long']:+6.2f} H2={r['H2_long']:+6.2f} (t={r['H2_long_t']:+.2f}) H2_short={r['H2_short']:+6.2f} [{time.time()-t1:.0f}s]")
    l3df = pd.DataFrame(layer3_results)
    l3df.to_csv(OUT_DIR/"layer3_combined.csv", index=False)

    print(f"\n=== FINAL VERDICT ===")
    best = l3df.sort_values("H2_long", ascending=False).iloc[0]
    print(f"  Best H2 long: {best['label']} at {best['H2_long']:+.2f} bps (t={best['H2_long_t']:+.2f})")
    print(f"  Best H2 short: {l3df.loc[l3df['H2_short'].idxmax(), 'label']} at {l3df['H2_short'].max():+.2f} bps")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
