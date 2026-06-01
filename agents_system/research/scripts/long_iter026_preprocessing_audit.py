"""LONG-PRED iter-026 — Preprocessing audit + fix.

Test 5 preprocessing variants on V0 features, per-sym Ridge, target_z (baseline):

  P1: current (per-sym clip 1%/99% + z-score) — BASELINE
  P2: cross-sectional z per cycle (across syms in same 4h)
  P3: rank per cycle (replace with pct rank within cycle)
  P4: robust per-sym (median + IQR instead of mean + std)
  P5: no clipping, simple z-score

For each: train per-sym Ridge with wider alpha grid, measure top-K=5 long edge H1+H2.
Report optimal alpha distribution + coefficient magnitudes.

Also report: prediction variance per cycle (low variance = model essentially picks random).
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter026"; OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
FIT_CUTOFF = pd.Timestamp("2025-10-02",tz="UTC")
K = 5
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def preproc_P1(train, test, feats):
    """Current: per-sym clip 1/99 + z-score."""
    stats_per_sym = {}
    for sym, g in train.groupby("symbol"):
        s = {}
        for c in feats:
            v = g[c].dropna()
            if len(v) < 100: s[c] = (0.,1.,0.,1.); continue
            lo, hi = float(v.quantile(0.01)), float(v.quantile(0.99))
            vc = v.clip(lo, hi)
            s[c] = (lo, hi, float(vc.mean()), float(vc.std()) or 1.0)
        stats_per_sym[sym] = s
    return stats_per_sym, "per_sym_clip_z"

def preproc_P4(train, test, feats):
    """Robust per-sym: median + IQR."""
    stats_per_sym = {}
    for sym, g in train.groupby("symbol"):
        s = {}
        for c in feats:
            v = g[c].dropna()
            if len(v) < 100: s[c] = (0.,1.,0.,1.); continue
            q25, med, q75 = float(v.quantile(0.25)), float(v.quantile(0.5)), float(v.quantile(0.75))
            iqr = q75 - q25 or 1.0
            s[c] = (med - 5*iqr, med + 5*iqr, med, iqr)  # clip at median±5×IQR
        stats_per_sym[sym] = s
    return stats_per_sym, "per_sym_robust"

def preproc_P5(train, test, feats):
    """No clipping, simple per-sym z."""
    stats_per_sym = {}
    for sym, g in train.groupby("symbol"):
        s = {}
        for c in feats:
            v = g[c].dropna()
            if len(v) < 100: s[c] = (-1e10, 1e10, 0., 1.); continue
            s[c] = (-1e10, 1e10, float(v.mean()), float(v.std()) or 1.0)
        stats_per_sym[sym] = s
    return stats_per_sym, "per_sym_no_clip"

def apply_per_sym(df, feats, stats_per_sym):
    """Apply per-sym scaling."""
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    sym_arr = df["symbol"].values
    for i, c in enumerate(feats):
        col_v = df[c].to_numpy()
        for j, sym in enumerate(sym_arr):
            if sym not in stats_per_sym: continue
            lo, hi, mu, sd = stats_per_sym[sym].get(c, (0.,1.,0.,1.))
            X[j, i] = (np.clip(col_v[j], lo, hi) - mu) / sd
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def apply_per_sym_vectorized(df, feats, stats_per_sym):
    """Vectorized version - apply per-sym scaling using groupby."""
    out = df[feats].copy()
    for sym, g in df.groupby("symbol"):
        if sym not in stats_per_sym: continue
        idx = g.index
        for c in feats:
            lo, hi, mu, sd = stats_per_sym[sym].get(c, (0.,1.,0.,1.))
            v = df.loc[idx, c].clip(lo, hi)
            out.loc[idx, c] = (v - mu) / sd
    return np.nan_to_num(out.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def preproc_P2_xs_z(train, test, feats):
    """Cross-sectional z per cycle. Compute scaling on-the-fly per cycle."""
    # Compute per-cycle stats on training data (to avoid using future)
    # Then at test time, use cycle-local z-score
    # Simpler approach: just compute cycle-local z on full panel (this uses cross-section only, no time leakage)
    return None, "xs_z_per_cycle"

def transform_P2(df, feats):
    """Apply cross-sectional z per cycle."""
    out = df[feats].copy()
    for c in feats:
        # Z within each cycle (open_time)
        gp = df.groupby("open_time")[c]
        mu = gp.transform("mean")
        sd = gp.transform("std").replace(0, np.nan)
        out[c] = (df[c] - mu) / sd
    return np.nan_to_num(out.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def transform_P3(df, feats):
    """Apply cross-sectional rank per cycle (0 to 1)."""
    out = df[feats].copy()
    for c in feats:
        out[c] = df.groupby("open_time")[c].rank(pct=True) - 0.5  # center around 0
    return np.nan_to_num(out.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def train_eval_with_preproc(panel, feats, preproc_label):
    print(f"\n--- {preproc_label} ---", flush=True)
    t0 = time.time()
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel["target_z"].notna()].copy()
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)].copy()

    if preproc_label == "P1_per_sym_clip_z":
        stats_per_sym, _ = preproc_P1(train, test, feats)
        Xtr = apply_per_sym_vectorized(train, feats, stats_per_sym)
        Xte = apply_per_sym_vectorized(test, feats, stats_per_sym)
    elif preproc_label == "P2_xs_z_per_cycle":
        Xtr = transform_P2(train, feats)
        Xte = transform_P2(test, feats)
    elif preproc_label == "P3_xs_rank_per_cycle":
        Xtr = transform_P3(train, feats)
        Xte = transform_P3(test, feats)
    elif preproc_label == "P4_per_sym_robust":
        stats_per_sym, _ = preproc_P4(train, test, feats)
        Xtr = apply_per_sym_vectorized(train, feats, stats_per_sym)
        Xte = apply_per_sym_vectorized(test, feats, stats_per_sym)
    elif preproc_label == "P5_per_sym_no_clip":
        stats_per_sym, _ = preproc_P5(train, test, feats)
        Xtr = apply_per_sym_vectorized(train, feats, stats_per_sym)
        Xte = apply_per_sym_vectorized(test, feats, stats_per_sym)

    # Train per-sym Ridge
    chosen_alphas, coef_norms, models = [], [], {}
    train["__idx"] = np.arange(len(train))
    test["__idx"] = np.arange(len(test))
    for sym, gtr in train.groupby("symbol"):
        if len(gtr) < 300: continue
        idx = gtr["__idx"].values
        Xtr_sym = Xtr[idx]
        try:
            m = RidgeCV(alphas=ALPHAS).fit(Xtr_sym, gtr["target_z"].values)
            models[sym] = m
            chosen_alphas.append(m.alpha_)
            coef_norms.append(np.linalg.norm(m.coef_))
        except: pass
    alpha_dist = pd.Series(chosen_alphas).value_counts().sort_index().to_dict()
    avg_coef = np.mean(coef_norms) if coef_norms else 0
    print(f"  alpha dist: {alpha_dist}")
    print(f"  mean coef ||w||: {avg_coef:.4f}")

    # Predict
    preds_list = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models: continue
        idx = gv["__idx"].values
        Xv = Xte[idx]
        pv = models[sym].predict(Xv)
        out = gv[["symbol","open_time","return_pct"]].copy(); out["pred"] = pv
        preds_list.append(out)
    preds = pd.concat(preds_list, ignore_index=True).sort_values(["open_time","symbol"])

    # Prediction variance per cycle (low = model picks ~random)
    per_cycle_std = preds.groupby("open_time")["pred"].std().mean()
    print(f"  mean pred-std per cycle: {per_cycle_std:.4f} (low ≈ random selection)")

    # Top-K=5 long edge
    results = {}
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = preds[(preds["open_time"]>=s) & (preds["open_time"]<e)]
        top_means = sub.groupby("open_time").apply(
            lambda g: g.nlargest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        a = top_means.values * 1e4
        mean = a.mean(); se = a.std()/np.sqrt(len(a)); t = mean/se if se>0 else float("nan")
        results[f"{period_label}_long_mean"] = mean
        results[f"{period_label}_long_t"] = t
    print(f"  H1 top-5 long edge: {results['H1_long_mean']:+.2f} (t={results['H1_long_t']:+.2f})")
    print(f"  H2 top-5 long edge: {results['H2_long_mean']:+.2f} (t={results['H2_long_t']:+.2f})")
    print(f"  [{time.time()-t0:.0f}s]")
    return results, avg_coef, per_cycle_std

def main():
    t0 = time.time()
    print("=== iter-026: Preprocessing audit ===\n", flush=True)
    cols = ["symbol","open_time","exit_time","return_pct","target_z"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)

    variants = ["P1_per_sym_clip_z","P2_xs_z_per_cycle","P3_xs_rank_per_cycle",
                "P4_per_sym_robust","P5_per_sym_no_clip"]
    all_r = []
    for label in variants:
        try:
            r, coef, var = train_eval_with_preproc(panel, V0_FEATURES, label)
            r["label"] = label; r["coef"] = coef; r["pred_std"] = var
            all_r.append(r)
        except Exception as e:
            print(f"  {label} FAILED: {e}")

    rdf = pd.DataFrame(all_r)
    print(f"\n=== PREPROCESSING COMPARISON ===\n")
    print(f"{'variant':<24} {'coef ||w||':>10} {'pred_std':>9} {'H1 long':>9} {'H1 t':>6} {'H2 long':>9} {'H2 t':>6}")
    for _, r in rdf.iterrows():
        sig = "★" if abs(r["H2_long_t"])>1.96 else " "
        print(f"  {r['label']:<22}  {r['coef']:>8.4f}  {r['pred_std']:>7.4f}  "
              f"{r['H1_long_mean']:>+7.2f} {r['H1_long_t']:>+5.2f}  "
              f"{r['H2_long_mean']:>+7.2f} {r['H2_long_t']:>+5.2f}  {sig}")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
