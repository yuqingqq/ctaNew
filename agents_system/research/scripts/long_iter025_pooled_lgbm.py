"""LONG-PRED iter-025 — Pooled LightGBM with cross-sym sharing.

Per-sym Ridge has fundamental limitations:
  1. Each sym has limited training data (~3000-5000 rows)
  2. Can't share information across syms
  3. Linear only - no interactions

Pooled LightGBM with sym_id as categorical feature:
  1. Trains on full panel (~1M rows) - much more data
  2. Shares structure across syms via tree splits
  3. Captures non-linear interactions automatically
  4. Regularization via tree depth + leaves

Tests:
  L1: Pooled LGBM with V0 features + sym_id
  L2: Pooled LGBM with V0 + B + C features
  L3: Pooled LGBM with regime features (btc_rvol_7d cross-sym interactions)

Same target (target_z) for fair comparison vs per-sym Ridge.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    print("lightgbm not installed, attempting install...", flush=True)
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm"], check=False)
    import lightgbm as lgb

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter025"; OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
FIT_CUTOFF = pd.Timestamp("2025-10-02",tz="UTC")
K = 5
CYCLES_PER_DAY = 6

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def engineer_BC(panel):
    g = panel.groupby("symbol", group_keys=False)
    for days in [7, 14, 30, 60]:
        bars = days * CYCLES_PER_DAY
        panel[f"ret_{days}d"] = g["return_1d"].transform(lambda x: x.rolling(bars, min_periods=bars//2).sum().shift(1))
    panel["idio_vol_30d"] = g["idio_vol_to_btc_1d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["rvol_30d"] = g["rvol_7d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    # B
    panel["mom_div_1d_7d"] = panel["return_1d"] - panel["ret_7d"]/7
    panel["mom_div_3d_30d"] = panel["ret_3d"]/3 - panel["ret_30d"]/30
    panel["mom_div_7d_60d"] = panel["ret_7d"]/7 - panel["ret_60d"]/60
    panel["mom_accel"] = panel["return_1d"] - 0.5*(panel["ret_7d"]/7 + panel["ret_30d"]/30)
    catB = ["mom_div_1d_7d","mom_div_3d_30d","mom_div_7d_60d","mom_accel"]
    # C
    panel["vol_ratio_1h_1d"] = panel["idio_vol_to_btc_1h"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)
    panel["vol_ratio_7d_30d"] = panel["rvol_7d"] / panel["rvol_30d"].replace(0, np.nan)
    panel["vol_ratio_recent_30d"] = panel["idio_vol_to_btc_1d"] / panel["idio_vol_30d"].replace(0, np.nan)
    panel["funding_per_vol"] = panel["funding_rate"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)
    catC = ["vol_ratio_1h_1d","vol_ratio_7d_30d","vol_ratio_recent_30d","funding_per_vol"]
    return panel, catB, catC

def train_pooled_lgbm(panel, feat_cols, label, params=None):
    """Train pooled LightGBM with sym_id as categorical."""
    print(f"\n--- {label} ({len(feat_cols)} features) ---", flush=True)
    t0 = time.time()
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel["target_z"].notna()].copy()
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)].copy()

    # Encode sym as categorical int
    syms = sorted(set(train["symbol"].unique()) | set(test["symbol"].unique()))
    sym2id = {s:i for i,s in enumerate(syms)}
    train["sym_id"] = train["symbol"].map(sym2id).astype("int32")
    test["sym_id"] = test["symbol"].map(sym2id).astype("int32")

    feat_cols_full = feat_cols + ["sym_id"]
    Xtr = train[feat_cols_full].fillna(0).values
    ytr = train["target_z"].values
    Xte = test[feat_cols_full].fillna(0).values

    p = dict(objective="regression", metric="rmse", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=5, verbose=-1,
             num_threads=8)
    if params: p.update(params)

    train_ds = lgb.Dataset(Xtr, label=ytr, categorical_feature=[len(feat_cols)])
    model = lgb.train(p, train_ds, num_boost_round=500,
                      callbacks=[lgb.log_evaluation(0)])

    preds = model.predict(Xte, num_iteration=model.best_iteration or model.num_trees())
    test["pred"] = preds
    out = test[["symbol","open_time","return_pct","pred"]]
    print(f"  trained {model.num_trees()} trees [{time.time()-t0:.0f}s]", flush=True)
    # Feature importance
    fi = pd.Series(model.feature_importance(importance_type="gain"), index=feat_cols_full).sort_values(ascending=False)
    print(f"  Top 10 features by gain:")
    for f, g in fi.head(10).items(): print(f"    {f:<32} {g:>12,.0f}")

    return out

def measure_edges(preds_df, label):
    results = {}
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = preds_df[(preds_df["open_time"]>=s) & (preds_df["open_time"]<e)]
        top_means = sub.groupby("open_time").apply(
            lambda g: g.nlargest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        a = top_means.values * 1e4
        mean = a.mean(); se = a.std()/np.sqrt(len(a)); t = mean/se if se>0 else float("nan")
        results[f"{period_label}_long_mean"] = mean
        results[f"{period_label}_long_t"] = t
        bot_means = sub.groupby("open_time").apply(
            lambda g: -g.nsmallest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        a = bot_means.values * 1e4
        results[f"{period_label}_short_mean"] = a.mean()
        results[f"{period_label}_short_t"] = a.mean()/(a.std()/np.sqrt(len(a))) if a.std()>0 else float("nan")
    return results

def main():
    t0 = time.time()
    print("=== iter-025: Pooled LightGBM ===\n", flush=True)
    cols = ["symbol","open_time","exit_time","return_pct","target_z"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    panel, catB, catC = engineer_BC(panel)

    variants = {
        "L1_V0_pooled":       V0_FEATURES,
        "L2_V0+BC_pooled":    V0_FEATURES + catB + catC,
    }
    all_r = []
    for label, feats in variants.items():
        preds = train_pooled_lgbm(panel, feats, label)
        r = measure_edges(preds, label); r["label"] = label
        all_r.append(r)
        preds.to_parquet(OUT_DIR/f"preds_{label}.parquet")

    # Compare vs per-sym Ridge baseline
    print(f"\n=== POOLED LGBM RESULTS ===\n")
    print(f"{'variant':<22} {'H1 long':>9} {'H1 t':>6} {'H2 long':>9} {'H2 t':>6} {'H2 short':>10}")
    for r in all_r:
        sig = "★" if abs(r["H2_long_t"])>1.96 else " "
        print(f"  {r['label']:<20}  {r['H1_long_mean']:>+7.2f} {r['H1_long_t']:>+5.2f} "
              f"{r['H2_long_mean']:>+7.2f} {r['H2_long_t']:>+5.2f} {sig}  {r['H2_short_mean']:>+8.2f}")

    print(f"\nReference: per-sym Ridge V0 baseline H2 long ~-9.98 (iter-021) or ~-5.80 (wider alpha)")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
