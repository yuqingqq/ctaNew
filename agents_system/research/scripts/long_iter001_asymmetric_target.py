"""LONG-PREDICTION iter-001 — Asymmetric Target / Loss

HYPOTHESIS: the forward residual distribution itself is asymmetric in H2 — the upside
tail is compressed/noisy relative to the downside tail. A symmetric MSE-trained Ridge
naturally learns the side with better signal-to-noise, which becomes the downside in
H2. The fix candidate: train a model whose loss specifically targets the upside tail
(via quantile regression or positive-target subset) — this would produce a dedicated
LONG-PREDICTION model that doesn't waste capacity on the working short side.

DEEP DIAGNOSTIC (do NOT jump to the fix without understanding):
  A) Per-cycle forward residual (alpha_A) skewness + tail asymmetry H1 vs H2
  B) Per-pred-decile realized forward distribution: is the top decile noisier?
  C) Per-symbol skew of forward residual H1 vs H2 (which syms have asymmetric distributions?)
  D) Conditional-on-being-picked-by-pred: what's the variance of realized for top-K vs bot-K?

FIX TEST (if asymmetry is real):
  Train an "upside-only" Ridge per sym: target = clip(target_z, 0, +inf) so the model
  learns ONLY positive forward residual relationships. Compare its top-K selection to
  the original model's top-K selection on H2.
"""
from __future__ import annotations
import sys, pickle, importlib.util, json, time, os
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/long_iter001"; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT = REPO/"agents_system/orchestrator/iteration_log.md"

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-001: asymmetric target / loss ===\n", flush=True)

    # ---------------- DEEP DIAGNOSTIC ----------------
    print("loading panel + preds...", flush=True)
    panel = pd.read_parquet(PANEL, columns=["symbol","open_time","alpha_vs_btc_realized","target_z","rstd","rmean"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0)&(panel["open_time"].dt.minute==0)]
    preds = pd.read_parquet(PREDS, columns=["symbol","open_time","pred","return_pct"])
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds = preds[(preds["open_time"].dt.hour%4==0)&(preds["open_time"].dt.minute==0)]

    # ----- A) per-cycle distribution asymmetry of forward residual (alpha_A) -----
    print("\n=== A) Per-cycle distribution asymmetry of forward residual ===")
    for label,(s,e) in [("H1",H1),("H2",H2)]:
        sub = preds[(preds["open_time"]>=s)&(preds["open_time"]<e)]
        per_cyc = sub.groupby("open_time")["return_pct"].agg(
            mean="mean", median="median",
            std="std",
            skew=lambda x: x.skew(),
            kurt=lambda x: x.kurt(),
            p10=lambda x: x.quantile(0.1),
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.9),
        )
        print(f"\n  {label}:")
        print(f"    mean of-cyc-skew      : {per_cyc['skew'].mean():+.3f}  (positive = right tail bigger; negative = left tail bigger)")
        print(f"    median of-cyc-skew    : {per_cyc['skew'].median():+.3f}")
        print(f"    %cycles with neg skew : {100*(per_cyc['skew']<0).mean():.1f}%")
        print(f"    mean of-cyc-kurt      : {per_cyc['kurt'].mean():+.3f}")
        # Tail asymmetry: |p90-median| vs |p10-median|
        up_tail = (per_cyc['p90']-per_cyc['median'])
        dn_tail = (per_cyc['median']-per_cyc['p10'])
        ratio = up_tail / dn_tail.replace(0,np.nan)
        print(f"    upside/downside tail size ratio: mean {ratio.mean():.3f}  median {ratio.median():.3f}")
        print(f"    %cycles upside_tail < downside_tail: {100*(up_tail<dn_tail).mean():.1f}%")

    # ----- B) Per-pred-decile realized distribution (within-decile std and skew) -----
    print("\n=== B) Per-pred-decile realized distribution ===")
    for label,(s,e) in [("H1",H1),("H2",H2)]:
        sub = preds[(preds["open_time"]>=s)&(preds["open_time"]<e)].copy()
        # rank within each cycle by pred
        sub["dec"] = sub.groupby("open_time")["pred"].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
        per_dec = sub.dropna(subset=["dec"]).groupby("dec")["return_pct"].agg(
            mean="mean", std="std",
            skew=lambda x: x.skew(),
            sharpe=lambda x: (x.mean()/x.std()) if x.std()>0 else np.nan,
        )
        print(f"\n  {label}:")
        print(per_dec.round(4).to_string())

    # ----- C) Per-symbol forward residual skew H1 vs H2 -----
    print("\n=== C) Per-symbol forward residual skew H1 vs H2 ===")
    rows = []
    for sym in preds["symbol"].unique():
        a1 = preds[(preds["symbol"]==sym)&(preds["open_time"]>=H1[0])&(preds["open_time"]<H1[1])]["return_pct"]
        a2 = preds[(preds["symbol"]==sym)&(preds["open_time"]>=H2[0])&(preds["open_time"]<H2[1])]["return_pct"]
        if len(a1)>=50 and len(a2)>=50:
            rows.append(dict(sym=sym, skew_h1=a1.skew(), skew_h2=a2.skew(), n1=len(a1), n2=len(a2)))
    sk = pd.DataFrame(rows)
    print(f"  syms n={len(sk)}")
    print(f"  H1 mean per-sym skew: {sk['skew_h1'].mean():+.3f}  median {sk['skew_h1'].median():+.3f}")
    print(f"  H2 mean per-sym skew: {sk['skew_h2'].mean():+.3f}  median {sk['skew_h2'].median():+.3f}")
    print(f"  %syms with H1 skew > 0 (positive tail): {100*(sk['skew_h1']>0).mean():.1f}%")
    print(f"  %syms with H2 skew > 0 (positive tail): {100*(sk['skew_h2']>0).mean():.1f}%")
    print(f"  %syms where skew DROPPED H2 vs H1: {100*(sk['skew_h2']<sk['skew_h1']).mean():.1f}%")
    sk["delta"] = sk["skew_h2"] - sk["skew_h1"]
    print(f"  median skew change H2-H1: {sk['delta'].median():+.3f}")

    # ----- D) Conditional-on-being-picked: variance of realized for top-K vs bot-K -----
    print("\n=== D) Per-cycle realized variance: top-K vs bot-K, H1 vs H2 ===")
    for label,(s,e) in [("H1",H1),("H2",H2)]:
        sub = preds[(preds["open_time"]>=s)&(preds["open_time"]<e)]
        top_stds = []; bot_stds = []
        for ot, g in sub.groupby("open_time"):
            if len(g) < 20: continue
            g = g.sort_values("pred")
            top_stds.append(g.tail(5)["return_pct"].std())
            bot_stds.append(g.head(5)["return_pct"].std())
        print(f"\n  {label}: top-5 within-basket realized std: mean {np.nanmean(top_stds)*1e4:.1f} bps")
        print(f"          bot-5 within-basket realized std: mean {np.nanmean(bot_stds)*1e4:.1f} bps")
        print(f"          top/bot std ratio: {np.nanmean(top_stds)/np.nanmean(bot_stds):.3f}")

    # ---------------- FIX TEST: upside-only Ridge ----------------
    print("\n\n=== FIX TEST: train per-sym Ridge on UPSIDE-ONLY target ===")
    print("Target = max(0, target_z): model learns only positive residual patterns.")
    print("Compare top-K selection to original model.")

    # train upside-only artifact through 2025-10-02 (= original cutoff, for honest OOS)
    from sklearn.linear_model import RidgeCV
    FEAT_COLS = x6.BASE + x6.COHORT_EXTRAS
    panel_full = pd.read_parquet(PANEL)
    panel_full["open_time"] = pd.to_datetime(panel_full["open_time"], utc=True)
    panel_full["exit_time"] = pd.to_datetime(panel_full["exit_time"], utc=True)
    fit_cut = pd.Timestamp("2025-10-02", tz="UTC")
    fit_train = panel_full[(panel_full["exit_time"]<fit_cut) & panel_full["target_z"].notna()].copy()
    fit_train["target_upside"] = fit_train["target_z"].clip(lower=0)   # upside-only target

    models_up, sstats_all, hstats_all = {}, {}, {}
    n_done = 0
    for sym, gtr in fit_train.groupby("symbol"):
        if len(gtr) < 300: continue
        s_, h_ = x6.fit_preproc(gtr, FEAT_COLS)
        Xtr = x6.apply_preproc(gtr, FEAT_COLS, s_, h_)
        try:
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr, gtr["target_upside"].to_numpy())
            models_up[sym] = m; sstats_all[sym] = s_; hstats_all[sym] = h_; n_done += 1
        except: pass
    print(f"  trained upside-only Ridge for {n_done} syms")

    # predict on H2 with upside-only model
    test = panel_full[(panel_full["open_time"]>=H2[0])&(panel_full["open_time"]<=H2[1])].copy()
    test = test[(test["open_time"].dt.hour%4==0)&(test["open_time"].dt.minute==0)]
    pred_rows = []
    for sym, gv in test.groupby("symbol"):
        if sym not in models_up: continue
        Xv = x6.apply_preproc(gv, FEAT_COLS, sstats_all[sym], hstats_all[sym])
        pv = models_up[sym].predict(Xv)
        out = gv[["symbol","open_time","alpha_vs_btc_realized","return_pct"]].copy()
        out.columns = ["symbol","open_time","alpha_A","return_pct"]
        out["pred_up"] = pv
        pred_rows.append(out)
    up_preds = pd.concat(pred_rows, ignore_index=True).sort_values(["open_time","symbol"])
    # Save upside preds
    up_preds.to_parquet(OUT_DIR/"upside_only_h2_preds.parquet")

    # compare top-K selection: upside model vs original
    orig = preds[(preds["open_time"]>=H2[0])&(preds["open_time"]<=H2[1])]
    print("\n  TOP-K edge per cycle (vs all-syms mean):")
    print(f"  {'K':>3}  {'ORIG top-K edge':>16}  {'UPSIDE top-K edge':>18}  {'lift':>8}")
    for k in [1, 2, 3, 5]:
        # original top-K edges
        orig_edges = []
        for ot, g in orig.groupby("open_time"):
            if len(g) < 2*k: continue
            g = g.sort_values("pred"); m = g["return_pct"].mean()
            orig_edges.append(g.tail(k)["return_pct"].mean() - m)
        # upside top-K edges
        up_edges = []
        for ot, g in up_preds.groupby("open_time"):
            if len(g) < 2*k: continue
            g = g.sort_values("pred_up"); m = g["return_pct"].mean()
            up_edges.append(g.tail(k)["return_pct"].mean() - m)
        oe = np.mean(orig_edges)*1e4; ue = np.mean(up_edges)*1e4
        print(f"  {k:>3}  {oe:+12.1f} bps  {ue:+14.1f} bps  {ue-oe:+6.1f}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")
    print(f"\nSaved upside preds: {OUT_DIR/'upside_only_h2_preds.parquet'}")

if __name__ == "__main__": main()
