"""Step 3: Walk-forward 10-fold RidgeCV training with bootstrap-bag ensemble.

Per fold:
  1. Take train rows (filtered by autocorr_pctile_7d ≥ 0.5 per production).
  2. Take cal rows (same filter) for alpha selection.
  3. 5-seed bootstrap: each seed resamples train rows with replacement.
     Fit RidgeCV on bootstrap sample, predict on test rows.
  4. Ensemble = mean of 5 seed predictions.

Output:
  results/predictions.parquet    (symbol, open_time, fold, pred_z, target_z, alpha_beta)
  models/coef_fold{F}_seed{S}.csv  per-fold per-seed coefficients
  results/cv_alphas.csv           per-fold per-seed RidgeCV-selected alpha
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
MODELS   = REPO / "linear_model/models"
RESULTS  = REPO / "linear_model/results"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5    # autocorr_pctile_7d ≥ 0.5 (production filter)
ALL_FOLDS = list(range(10))


def main():
    print("=== Step 3: RidgeCV walk-forward ensemble ===\n", flush=True)
    t0 = time.time()

    print("  Loading targets + features...", flush=True)
    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    feat = pd.read_parquet(FEATURES)
    feat["open_time"] = pd.to_datetime(feat["open_time"], utc=True)
    # Verify alignment
    assert len(tgt) == len(feat), f"length mismatch {len(tgt)} vs {len(feat)}"
    panel = tgt.merge(feat, on=["symbol", "open_time"], how="inner")
    print(f"    panel: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    folds_all = _multi_oos_splits(panel)
    print(f"  Folds: {len(folds_all)}", flush=True)

    feat_cols = [c for c in feat.columns if c not in ("symbol", "open_time")]
    print(f"  Feature columns: {len(feat_cols)} "
          f"(16 numeric + {len(feat_cols)-16} sym dummies)", flush=True)

    cv_alphas = []
    coefs_all = []
    all_preds = []

    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        # Production autocorr filter
        tr = train[train["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        ca = cal[cal["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        te = test.dropna(subset=["target_z"]).copy()
        if len(tr) < 1000 or len(ca) < 200 or len(te) < 100:
            print(f"  fold {fid}: skip (insufficient data)", flush=True)
            continue

        Xt = tr[feat_cols].to_numpy(np.float32)
        Xc = ca[feat_cols].to_numpy(np.float32)
        Xte = te[feat_cols].to_numpy(np.float32)
        yt = tr["target_z"].to_numpy(np.float32)
        yc = ca["target_z"].to_numpy(np.float32)

        # Combine train + cal for RidgeCV's internal CV (alpha selection)
        X_combined = np.vstack([Xt, Xc])
        y_combined = np.concatenate([yt, yc])

        fold_preds = []
        for s_idx, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed)
            # Bootstrap sample train rows
            n = len(Xt)
            idx = rng.integers(0, n, size=n)
            Xb = Xt[idx]
            yb = yt[idx]

            # Use cal as held-out for RidgeCV alpha selection via cv=None
            # Strategy: do RidgeCV on bootstrap train w/ leave-one-out CV-equivalent
            # for alpha selection (sklearn does efficient generalized CV by default)
            model = RidgeCV(alphas=ALPHAS, scoring="r2",
                            cv=None,  # use efficient generalized CV (gcv)
                            fit_intercept=True)
            model.fit(Xb, yb)
            sel_alpha = float(model.alpha_)
            pred_te = model.predict(Xte).astype(np.float32)
            fold_preds.append(pred_te)
            cv_alphas.append({"fold": fid, "seed": seed, "alpha": sel_alpha,
                              "n_train_bootstrap": len(Xb), "n_test": len(te)})
            # Save coefs
            coef_df = pd.DataFrame({"feature": feat_cols,
                                    "coef": model.coef_})
            coef_df["fold"] = fid; coef_df["seed"] = seed
            coef_df["intercept"] = float(model.intercept_)
            coef_df["alpha"] = sel_alpha
            coefs_all.append(coef_df)
            print(f"    fold {fid} seed {s_idx+1}/{len(SEEDS)}: "
                  f"α={sel_alpha:.1f}", flush=True)

        # Ensemble pred
        ensemble_pred = np.mean(fold_preds, axis=0)
        df_pred = te[["symbol", "open_time", "exit_time", "alpha_beta",
                       "target_z", "sigma_idio_ref", "beta_pit"]].copy()
        df_pred["pred_z"] = ensemble_pred
        df_pred["fold"] = fid
        all_preds.append(df_pred)
        # Per-cycle IC (preview)
        cyc_ic = df_pred.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(
                (g["alpha_beta"]).rank()) if len(g) >= 5 else np.nan).dropna()
        print(f"  fold {fid}: n_test={len(te):,}, "
              f"per-cycle IC={cyc_ic.mean():+.4f}, "
              f"time={time.time()-t_fold:.0f}s", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(
        ["open_time", "symbol"]).reset_index(drop=True)
    apd.to_parquet(RESULTS / "predictions.parquet", index=False)
    print(f"\n  Saved predictions: {RESULTS / 'predictions.parquet'} "
          f"({len(apd):,} rows)", flush=True)

    pd.DataFrame(cv_alphas).to_csv(RESULTS / "cv_alphas.csv", index=False)
    pd.concat(coefs_all, ignore_index=True).to_csv(RESULTS / "coefficients.csv",
                                                    index=False)

    # Overall IC summary
    cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    print(f"\n  Overall per-cycle IC (Ridge ensemble): {cyc_ic.mean():+.4f} "
          f"(n cycles = {len(cyc_ic)})", flush=True)
    print(f"  Reference: LGBM WINNER_17 + β-residual IC = +0.0157", flush=True)

    # Alpha selection summary
    df_alphas = pd.DataFrame(cv_alphas)
    print(f"\n  RidgeCV alpha selection per fold (mode across seeds):", flush=True)
    for fid in df_alphas["fold"].unique():
        f_alphas = df_alphas[df_alphas["fold"]==fid]["alpha"]
        print(f"    fold {fid}: alphas {sorted(f_alphas.unique().tolist())} "
              f"(mode={f_alphas.mode().iloc[0]:.1f})", flush=True)

    print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
