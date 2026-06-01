"""LONG-PRED iter-021b — Focused V0 + B + C (best categories only).

Test variants:
  V0_baseline
  V0+B (best individual)
  V0+C (second best)
  V0+B+C (combined best)
  V0+B+C+D (add 3rd best)

This isolates whether B+C captures most of the benefit or whether combining
helps further.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter021b"; OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
FIT_CUTOFF = pd.Timestamp("2025-10-02",tz="UTC")
K = 5
CYCLES_PER_DAY = 6
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]  # wider grid

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def engineer_BCD(panel):
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
    # D
    panel["vol_x_mom"] = panel["rvol_7d"] * panel["return_1d"]
    panel["vol_x_neg_mom"] = panel["rvol_7d"] * (-panel["return_1d"])
    panel["funding_x_beta"] = panel["funding_rate"] * panel["beta_to_btc_change_5d"]
    panel["bars_x_vol"] = panel["bars_since_high"] * panel["rvol_7d"]
    catD = ["vol_x_mom","vol_x_neg_mom","funding_x_beta","bars_x_vol"]
    return panel, catB, catC, catD

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

def train_eval(panel, feat_cols, label):
    print(f"\n--- {label} ({len(feat_cols)} features) ---", flush=True)
    t0 = time.time()
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel["target_z"].notna()]
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)]
    models, stats_all = {}, {}; chosen_alphas = []
    for sym, gtr in train.groupby("symbol"):
        if len(gtr) < 300: continue
        s = fit_preproc(gtr, feat_cols); Xtr = apply_preproc(gtr, feat_cols, s)
        try:
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, gtr["target_z"].to_numpy())
            models[sym] = m; stats_all[sym] = s; chosen_alphas.append(m.alpha_)
        except: pass
    alpha_dist = pd.Series(chosen_alphas).value_counts().sort_index().to_dict()
    print(f"  alpha dist {alpha_dist} [{time.time()-t0:.0f}s]", flush=True)

    preds_list = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models: continue
        Xv = apply_preproc(gv, feat_cols, stats_all[sym])
        pv = models[sym].predict(Xv)
        out = gv[["symbol","open_time","return_pct"]].copy(); out["pred"] = pv
        preds_list.append(out)
    preds = pd.concat(preds_list, ignore_index=True).sort_values(["open_time","symbol"])

    results = {}
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = preds[(preds["open_time"]>=s) & (preds["open_time"]<e)]
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
    print("=== iter-021b: V0+B+C focused ===\n", flush=True)
    cols = ["symbol","open_time","exit_time","return_pct","target_z"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    panel, catB, catC, catD = engineer_BCD(panel)

    variants = {
        "V0":     V0_FEATURES,
        "V0+B":   V0_FEATURES + catB,
        "V0+C":   V0_FEATURES + catC,
        "V0+B+C": V0_FEATURES + catB + catC,
        "V0+B+C+D": V0_FEATURES + catB + catC + catD,
    }
    all_r = []
    for label, feats in variants.items():
        r = train_eval(panel, feats, label); r["label"] = label; r["n"] = len(feats)
        all_r.append(r)
    rdf = pd.DataFrame(all_r)
    print(f"\n=== FOCUSED COMBINED RESULTS ===\n")
    print(f"{'variant':<12} {'n':>3} {'H1 long':>9} {'H1 t':>6} {'H2 long':>9} {'H2 t':>6} {'H2 short':>10}")
    for _, r in rdf.iterrows():
        sig = "★" if abs(r["H2_long_t"])>1.96 else " "
        print(f"  {r['label']:<10} {int(r['n']):>3}  {r['H1_long_mean']:>+7.2f} {r['H1_long_t']:>+5.2f} "
              f"{r['H2_long_mean']:>+7.2f} {r['H2_long_t']:>+5.2f} {sig}  {r['H2_short_mean']:>+8.2f}")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
