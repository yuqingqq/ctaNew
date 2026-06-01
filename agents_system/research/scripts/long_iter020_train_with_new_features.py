"""LONG-PRED iter-020 — Properly train models with the new engineered features

This is the deep test the user demanded. We didn't actually train a MODEL with the
new features in iter-019 — we only tested univariate edges. Now we train per-sym
Ridge models with different feature sets and evaluate predictions OOS.

Test 4 model variants:
  M1: V0 baseline (17 V0 features) — control
  M2: V0 + new features (combined) — does adding help?
  M3: ONLY new engineered features — are new features sufficient?
  M4: Top features from iter-019 negative-edge analysis (likely good for shorts)

For each: train per-sym Ridge on data through 2025-10-02, generate predictions
2025-10-04 → 2026-05-26, measure top-K=5 long edge in H1 and H2.

KEY DIFFERENCE FROM iter-019: that tested univariate edge per-feature.
This tests COMBINED predictive power when features are used together by a model.
"""
import sys, time, pickle
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter020"; OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
FIT_CUTOFF = pd.Timestamp("2025-10-02",tz="UTC")  # honest OOS
K = 5
CYCLES_PER_DAY = 6
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def engineer_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the engineered features from iter-019 to the panel."""
    print("  engineering features...", flush=True)
    g = panel.groupby("symbol", group_keys=False)
    # Long-horizon returns
    for days in [7, 14, 30, 60]:
        bars = days * CYCLES_PER_DAY
        panel[f"ret_{days}d"] = g["return_1d"].transform(lambda x: x.rolling(bars, min_periods=bars//2).sum().shift(1))
    panel["idio_vol_30d"] = g["idio_vol_to_btc_1d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["rvol_30d"] = g["rvol_7d"].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
    panel["funding_30d"] = g["funding_rate"].transform(lambda x: x.rolling(180, min_periods=90).sum().shift(1))

    # XS ranks (per cycle)
    for feat in V0_FEATURES:
        panel[f"xsrank_{feat}"] = panel.groupby("open_time")[feat].rank(pct=True)

    # Momentum divergences
    panel["mom_div_1d_7d"] = panel["return_1d"] - panel["ret_7d"]/7
    panel["mom_div_3d_30d"] = panel["ret_3d"]/3 - panel["ret_30d"]/30
    panel["mom_div_7d_60d"] = panel["ret_7d"]/7 - panel["ret_60d"]/60
    panel["mom_accel"] = panel["return_1d"] - 0.5*(panel["ret_7d"]/7 + panel["ret_30d"]/30)

    # Vol/funding ratios
    panel["vol_ratio_1h_1d"] = panel["idio_vol_to_btc_1h"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)
    panel["vol_ratio_7d_30d"] = panel["rvol_7d"] / panel["rvol_30d"].replace(0, np.nan)
    panel["vol_ratio_recent_30d"] = panel["idio_vol_to_btc_1d"] / panel["idio_vol_30d"].replace(0, np.nan)
    panel["funding_per_vol"] = panel["funding_rate"] / panel["idio_vol_to_btc_1d"].replace(0, np.nan)

    # Interactions
    panel["vol_x_mom"] = panel["rvol_7d"] * panel["return_1d"]
    panel["vol_x_neg_mom"] = panel["rvol_7d"] * (-panel["return_1d"])
    panel["funding_x_beta"] = panel["funding_rate"] * panel["beta_to_btc_change_5d"]
    panel["bars_x_vol"] = panel["bars_since_high"] * panel["rvol_7d"]

    # Anomaly z-scores
    for feat in ["return_1d","rvol_7d","funding_rate","idio_vol_to_btc_1d"]:
        rolling_mean = g[feat].transform(lambda x: x.rolling(180, min_periods=90).mean().shift(1))
        rolling_std = g[feat].transform(lambda x: x.rolling(180, min_periods=90).std().shift(1))
        panel[f"z_{feat}"] = (panel[feat] - rolling_mean) / rolling_std.replace(0, np.nan)

    NEW_FEATURES = (
        [f"xsrank_{f}" for f in V0_FEATURES]
        + ["mom_div_1d_7d","mom_div_3d_30d","mom_div_7d_60d","mom_accel"]
        + ["vol_ratio_1h_1d","vol_ratio_7d_30d","vol_ratio_recent_30d","funding_per_vol"]
        + ["vol_x_mom","vol_x_neg_mom","funding_x_beta","bars_x_vol"]
        + ["z_return_1d","z_rvol_7d","z_funding_rate","z_idio_vol_to_btc_1d"]
    )
    print(f"  added {len(NEW_FEATURES)} new features", flush=True)
    return panel, NEW_FEATURES

def fit_preproc(train_df, feat_cols):
    """Winsorize + z-score on train data; save stats."""
    stats = {}
    for c in feat_cols:
        if c not in train_df.columns: continue
        v = train_df[c].dropna()
        if len(v) < 100:
            stats[c] = {"lo":0.,"hi":1.,"mu":0.,"sd":1.}; continue
        lo, hi = float(v.quantile(0.01)), float(v.quantile(0.99))
        vc = v.clip(lo, hi)
        stats[c] = {"lo":lo,"hi":hi,"mu":float(vc.mean()),"sd":float(vc.std()) or 1.0}
    return stats

def apply_preproc(df, feat_cols, stats):
    X = np.zeros((len(df), len(feat_cols)), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        if c not in stats or c not in df.columns: continue
        s = stats[c]
        v = df[c].to_numpy()
        X[:, i] = (np.clip(v, s["lo"], s["hi"]) - s["mu"]) / s["sd"]
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def train_per_sym(panel, feat_cols, label):
    """Train per-sym Ridge."""
    print(f"\n  training {label} (features={len(feat_cols)})...", flush=True)
    t0 = time.time()
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel["target_z"].notna()]
    models, stats_all = {}, {}
    n_ok = 0; n_skip = 0
    for sym, gtr in train.groupby("symbol"):
        if len(gtr) < 300:
            n_skip += 1; continue
        s = fit_preproc(gtr, feat_cols)
        Xtr = apply_preproc(gtr, feat_cols, s)
        try:
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, gtr["target_z"].to_numpy())
            models[sym] = m; stats_all[sym] = s; n_ok += 1
        except: n_skip += 1
    print(f"    fit {n_ok} syms ({n_skip} skipped) [{time.time()-t0:.0f}s]", flush=True)
    return models, stats_all

def predict_oos(panel, feat_cols, models, stats_all):
    """Generate OOS predictions on test window."""
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)]
    preds_list = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models: continue
        Xv = apply_preproc(gv, feat_cols, stats_all[sym])
        pv = models[sym].predict(Xv)
        out = gv[["symbol","open_time","return_pct"]].copy()
        out["pred"] = pv
        preds_list.append(out)
    return pd.concat(preds_list, ignore_index=True).sort_values(["open_time","symbol"])

def measure_long_edge(preds_df, label):
    """Measure top-K=5 long edge in H1 and H2 with significance."""
    print(f"\n  evaluating {label}...", flush=True)
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = preds_df[(preds_df["open_time"]>=s) & (preds_df["open_time"]<e)]
        top_means = sub.groupby("open_time").apply(
            lambda g: g.nlargest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        arr = top_means.values * 1e4
        mean = arr.mean(); se = arr.std()/np.sqrt(len(arr))
        t = mean/se if se>0 else float("nan")
        sig = "★" if abs(t)>1.96 else " "
        print(f"    {period_label}: top-K=5 long edge {mean:+.2f} bps  t={t:+.2f}  sig={sig}  n={len(arr)}")

def main():
    t0 = time.time()
    print("=== iter-020: Train models with new features ===\n", flush=True)

    print("loading panel...", flush=True)
    cols = ["symbol","open_time","exit_time","return_pct","target_z","alpha_vs_btc_realized"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    # Engineer features
    panel, NEW_FEATURES = engineer_features(panel)

    # 3 model variants
    variants = {
        "M1_V0_only": V0_FEATURES,
        "M2_V0_plus_new": V0_FEATURES + NEW_FEATURES,
        "M3_new_only": NEW_FEATURES,
    }

    for label, feats in variants.items():
        models, stats = train_per_sym(panel, feats, label)
        preds = predict_oos(panel, feats, models, stats)
        preds.to_parquet(OUT_DIR/f"preds_{label}.parquet")
        measure_long_edge(preds, label)

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
