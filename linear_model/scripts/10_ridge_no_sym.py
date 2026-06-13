"""Step 10: Re-train Ridge WITHOUT sym_id dummies — fixed-effects-free linear.

Hypothesis: removing the 49 sym dummies forces Ridge to rely on the 16
numeric features, which contain the actual cross-sectional signal. Should
fix the decile inversion if the diagnosis is correct.

Two variants:
  V_A: 16 numeric features only (no sym info)
  V_B: 16 numeric features + per-symbol target demean (subtract sym train mean)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

TARGETS  = REPO / "linear_model/data/targets.parquet"
FEATURES = REPO / "linear_model/data/features.parquet"
RESULTS  = REPO / "linear_model/results"

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5
ALL_FOLDS = list(range(10))

NUMERIC_FEATS = ["return_1d","atr_pct","dom_level_vs_bk","dom_change_288b_vs_bk",
                 "bk_ema_slope_4h","corr_change_3d_vs_bk","obv_z_1d","vwap_slope_96",
                 "bars_since_high_xs_rank","idio_vol_1d_vs_bk_xs_rank",
                 "funding_rate","funding_rate_z_7d","corr_to_btc_1d",
                 "idio_vol_to_btc_1h","beta_to_btc_change_5d","funding_rate_1d_change"]


def main():
    print("=== Step 10: Ridge WITHOUT sym dummies ===\n", flush=True)
    t0 = time.time()

    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    feat = pd.read_parquet(FEATURES)
    feat["open_time"] = pd.to_datetime(feat["open_time"], utc=True)
    panel = tgt.merge(feat, on=["symbol","open_time"], how="inner")
    print(f"Panel: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    folds_all = _multi_oos_splits(panel)
    print(f"Folds: {len(folds_all)}", flush=True)

    # Verify all numeric features present
    for f in NUMERIC_FEATS:
        if f not in panel.columns:
            print(f"  MISSING: {f}"); return

    all_preds = []
    coef_log = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        ca = cal[cal["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        te = test.dropna(subset=["target_z"]).copy()
        if len(tr) < 1000 or len(ca) < 200 or len(te) < 100:
            print(f"  fold {fid}: skip"); continue

        Xt = tr[NUMERIC_FEATS].to_numpy(np.float32)
        Xte = te[NUMERIC_FEATS].to_numpy(np.float32)
        yt = tr["target_z"].to_numpy(np.float32)
        mt = ~np.isnan(yt)

        fold_preds = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, mt.sum(), size=mt.sum())
            X_boot = Xt[mt][idx]
            y_boot = yt[mt][idx]
            model = RidgeCV(alphas=ALPHAS, scoring="r2", cv=None, fit_intercept=True)
            model.fit(X_boot, y_boot)
            fold_preds.append(model.predict(Xte).astype(np.float32))
            coef_log.append({"fold":fid, "seed":seed, "alpha":float(model.alpha_),
                              **dict(zip(NUMERIC_FEATS, model.coef_)),
                              "intercept":float(model.intercept_)})

        pred = np.mean(fold_preds, axis=0)
        df_pred = te[["symbol","open_time","exit_time","alpha_beta",
                       "target_z","sigma_idio_ref"]].copy()
        df_pred["pred_z"] = pred
        df_pred["fold"] = fid
        all_preds.append(df_pred)
        cyc_ic = df_pred.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        print(f"  fold {fid}: IC={cyc_ic.mean():+.4f}, "
              f"n_test={len(te):,}, {time.time()-t_fold:.0f}s", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time","symbol"])
    apd.to_parquet(RESULTS / "ridge_no_sym_predictions.parquet", index=False)
    print(f"\nSaved: {RESULTS / 'ridge_no_sym_predictions.parquet'}", flush=True)

    cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    print(f"\nOverall per-cycle IC (Ridge, no sym dummies): {cyc_ic.mean():+.4f}",
          flush=True)
    print(f"  vs Ridge with sym dummies: +0.0135", flush=True)
    print(f"  vs LGBM clean-PIT:         +0.0162", flush=True)

    # Decile analysis
    times = sorted(apd.open_time.unique())
    keep = set(times[::48])
    samp = apd[apd.open_time.isin(keep)].copy()
    samp["dec"] = samp.groupby("open_time")["pred_z"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    samp["alpha_bps"] = samp["alpha_beta"] * 1e4
    print(f"\nDecile analysis (no-sym Ridge):", flush=True)
    dec = samp.groupby("dec")["alpha_bps"].agg(["mean","count"]).round(2)
    for i, row in dec.iterrows():
        print(f"  decile {int(i)}: mean={row['mean']:>+7.2f} bps  "
              f"(n={int(row['count']):,})", flush=True)
    print(f"  decile 9 − 0 = {dec.loc[9,'mean']-dec.loc[0,'mean']:+.2f} bps "
          f"({'POSITIVE (good)' if dec.loc[9,'mean']>dec.loc[0,'mean'] else 'NEGATIVE (inverted)'})",
          flush=True)

    # Coefficients
    coef_df = pd.DataFrame(coef_log)
    coef_summary = coef_df[NUMERIC_FEATS].agg(["mean","std"]).T
    coef_summary["abs_mean"] = coef_summary["mean"].abs()
    coef_summary = coef_summary.sort_values("abs_mean", ascending=False)
    print(f"\nFeature coefficients (no sym):", flush=True)
    print(f"  {'feature':<32} {'mean':>10} {'std':>10}", flush=True)
    for f, row in coef_summary.iterrows():
        print(f"  {f:<32} {row['mean']:+10.4f} {row['std']:>10.4f}", flush=True)

    coef_df.to_csv(RESULTS / "ridge_no_sym_coefs.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
