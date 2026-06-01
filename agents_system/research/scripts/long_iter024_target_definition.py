"""LONG-PRED iter-024 — Target definition deep test.

Hypothesis: 4h-horizon per-sym-z-score alpha residual is the wrong target for
finding pumpable longs. Test alternative targets while keeping V0 features fixed:

  T1: current (4h forward alpha_vs_btc residualized, per-sym z) [BASELINE]
  T2: forward 24h return (raw, not residualized, not z-scored)
  T3: forward 48h cumulative return
  T4: forward 24h ALPHA-vs-btc (residualized but not z-scored, no per-sym normalization)
  T5: cross-sectional RANK of forward 24h return (per-cycle pct rank)
  T6: forward 4h return RAW (not residualized, not z-scored)

For each target, train per-sym Ridge with V0 features, generate OOS preds,
measure top-K=5 long edge in H1 + H2.

If T2/T3 (longer horizon) significantly improves H2 long edge → target is wrong
If T5 (rank) improves → per-sym z-score normalization is wrong
If T6 (raw) improves → residualizing vs BTC is wrong

This is a deeper test than feature engineering — it questions the prediction target itself.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter024"; OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
FIT_CUTOFF = pd.Timestamp("2025-10-02",tz="UTC")
K = 5
CYCLES_PER_DAY = 6
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

V0_FEATURES = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
               "bars_since_high","autocorr_pctile_7d",
               "corr_to_btc_1d","beta_to_btc_change_5d",
               "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
               "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
               "rvol_7d","ret_3d","btc_rvol_7d"]

def build_targets(panel: pd.DataFrame):
    """Build alternative target columns."""
    print("building alternative targets...", flush=True)
    g = panel.groupby("symbol", group_keys=False)

    # T2: forward 24h raw return (6 bars × 4h)
    # return_pct is realized over the holding period; for 4h hold = 4h return.
    # forward 24h = cumulative of next 6 4h returns
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    panel["forward_4h_raw"] = panel["return_pct"]
    panel["forward_24h_raw"] = g["return_pct"].transform(
        lambda x: x.rolling(6, min_periods=6).sum().shift(-5))  # sum of next 6 bars including current
    panel["forward_48h_raw"] = g["return_pct"].transform(
        lambda x: x.rolling(12, min_periods=12).sum().shift(-11))

    # T4: forward 24h alpha (need BTC's forward 24h return)
    # BTCUSDT might not be in cross-sectional panel; try fallback to median basket return per cycle
    if "BTCUSDT" in panel["symbol"].unique():
        btc = panel[panel["symbol"]=="BTCUSDT"][["open_time","forward_24h_raw"]].rename(
            columns={"forward_24h_raw":"btc_fwd_24h_raw"})
        panel = panel.merge(btc, on="open_time", how="left")
        print("  T4: using BTCUSDT forward as benchmark", flush=True)
    else:
        # Fallback: median forward return across universe as benchmark proxy
        median_fwd = panel.groupby("open_time")["forward_24h_raw"].transform("median")
        panel["btc_fwd_24h_raw"] = median_fwd
        print("  T4: BTCUSDT missing, using cycle-median forward return as proxy", flush=True)
    panel["forward_24h_alpha"] = panel["forward_24h_raw"] - panel["btc_fwd_24h_raw"]

    # T5: cross-sectional RANK of forward 24h return
    panel["forward_24h_rank"] = panel.groupby("open_time")["forward_24h_raw"].rank(pct=True)

    return panel

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

def train_predict_evaluate(panel, feat_cols, target_col, label):
    """Train per-sym Ridge on target_col, generate preds, measure top-K=5 long edge H1+H2."""
    print(f"\n--- {label} (target={target_col}, alphas wider grid) ---", flush=True)
    t0 = time.time()
    train = panel[(panel["exit_time"] < FIT_CUTOFF) & panel[target_col].notna()]
    test = panel[(panel["open_time"] >= H1_START) & (panel["open_time"] <= H2_END)]
    models, stats_all = {}, {}; n_ok = 0
    chosen_alphas = []
    for sym, gtr in train.groupby("symbol"):
        if len(gtr) < 300: continue
        s = fit_preproc(gtr, feat_cols)
        Xtr = apply_preproc(gtr, feat_cols, s)
        y = gtr[target_col].to_numpy()
        try:
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, y)
            models[sym] = m; stats_all[sym] = s; n_ok += 1
            chosen_alphas.append(m.alpha_)
        except: pass
    alpha_dist = pd.Series(chosen_alphas).value_counts().sort_index().to_dict()
    print(f"  fit {n_ok} syms; alpha dist {alpha_dist} [{time.time()-t0:.0f}s]", flush=True)

    preds_list = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models: continue
        Xv = apply_preproc(gv, feat_cols, stats_all[sym])
        pv = models[sym].predict(Xv)
        out = gv[["symbol","open_time","return_pct"]].copy(); out["pred"] = pv
        preds_list.append(out)
    preds = pd.concat(preds_list, ignore_index=True).sort_values(["open_time","symbol"])
    # Evaluate using realized 4h return (so all targets are comparable on the same realized PnL)
    results = {}
    for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
        sub = preds[(preds["open_time"]>=s) & (preds["open_time"]<e)]
        top_means = sub.groupby("open_time").apply(
            lambda g: g.nlargest(K, "pred")["return_pct"].mean() if len(g)>=2*K else np.nan).dropna()
        a = top_means.values * 1e4
        mean = a.mean(); se = a.std()/np.sqrt(len(a)); t = mean/se if se>0 else float("nan")
        results[f"{period_label}_long_mean"] = mean
        results[f"{period_label}_long_t"] = t
    return results

def main():
    t0 = time.time()
    print("=== iter-024: Target definition deep test ===\n", flush=True)

    cols = ["symbol","open_time","exit_time","return_pct","target_z","alpha_vs_btc_realized"] + V0_FEATURES
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    panel = panel.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    panel = build_targets(panel)

    targets = {
        "T1_baseline_z_4h_alpha": "target_z",
        "T2_forward_24h_raw":     "forward_24h_raw",
        "T3_forward_48h_raw":     "forward_48h_raw",
        "T4_forward_24h_alpha":   "forward_24h_alpha",
        "T5_forward_24h_rank":    "forward_24h_rank",
        "T6_forward_4h_raw":      "forward_4h_raw",
    }
    all_results = []
    for label, tgt in targets.items():
        try:
            r = train_predict_evaluate(panel, V0_FEATURES, tgt, label)
            r["label"] = label; r["target"] = tgt
            all_results.append(r)
        except Exception as e:
            print(f"  {label} FAILED: {e}", flush=True)
            all_results.append({"label": label, "target": tgt,
                "H1_long_mean":np.nan,"H1_long_t":np.nan,
                "H2_long_mean":np.nan,"H2_long_t":np.nan})

    rdf = pd.DataFrame(all_results)
    print(f"\n=== TARGET COMPARISON ===\n")
    print(f"{'target':<28} {'H1 long':>9} {'H1 t':>6} {'H2 long':>9} {'H2 t':>6} sig")
    print("-"*70)
    for _, r in rdf.iterrows():
        sig = "★" if abs(r["H2_long_t"])>1.96 else " "
        print(f"  {r['label']:<28} {r['H1_long_mean']:>+7.2f} {r['H1_long_t']:>+5.2f} "
              f"{r['H2_long_mean']:>+7.2f} {r['H2_long_t']:>+5.2f}   {sig}")

    print(f"\n=== INTERPRETATION ===\n")
    baseline_h2 = rdf[rdf["label"]=="T1_baseline_z_4h_alpha"]["H2_long_mean"].iloc[0]
    print(f"  Baseline T1 H2 long edge: {baseline_h2:+.2f} bps")
    for _, r in rdf.iterrows():
        if r["label"]=="T1_baseline_z_4h_alpha": continue
        d = r["H2_long_mean"] - baseline_h2
        marker = "✓ FIX!" if d > 5.0 else ("≈" if abs(d)<5.0 else "✗ WORSE")
        print(f"    {r['label']:<28} Δ={d:+.2f} bps  {marker}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
