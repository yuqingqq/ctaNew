"""Step 12: Optimized Ridge with informed feature engineering.

Based on Step 11 audit, apply per-feature transforms:
  1. Add squared terms for 6 U-shape features (let Ridge capture non-monotonic via quadratic)
  2. Add rank-transformed versions for features with high Spearman/Pearson ratio
  3. Sign-split for asymmetric (high-skew) features
  4. Extended α grid (test smaller values 0.001-1000)
  5. Also try Lasso (L1) and ElasticNet for comparison

Variants:
  R1 = baseline (16 features, z-scored, +49 sym dummies) — original
  R2 = no sym dummies (16 features)
  R3 = R2 + 6 squared terms (U-shape features) = 22 features
  R4 = R3 + rank-transformed versions of high-NL features = 28 features
  R5 = R4 + sign-split for heavy-skew funding features = 32 features
  R6 = R5 + ElasticNet variant (L1+L2 mix)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

TARGETS  = REPO / "linear_model/data/targets.parquet"
PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
RESULTS  = REPO / "linear_model/results"

# Per Step 11 audit
NUMERIC_FEATS = ["return_1d","atr_pct","dom_level_vs_bk","dom_change_288b_vs_bk",
                 "bk_ema_slope_4h","corr_change_3d_vs_bk","obv_z_1d","vwap_slope_96",
                 "bars_since_high_xs_rank","idio_vol_1d_vs_bk_xs_rank",
                 "funding_rate","funding_rate_z_7d","corr_to_btc_1d",
                 "idio_vol_to_btc_1h","beta_to_btc_change_5d","funding_rate_1d_change"]

U_SHAPE_FEATS = ["beta_to_btc_change_5d", "dom_change_288b_vs_bk",
                 "corr_to_btc_1d", "corr_change_3d_vs_bk",
                 "dom_level_vs_bk", "return_1d"]
HEAVY_TAIL = ["funding_rate", "funding_rate_1d_change", "vwap_slope_96",
              "idio_vol_to_btc_1h"]
HIGH_NL = ["idio_vol_1d_vs_bk_xs_rank", "dom_level_vs_bk",
           "corr_to_btc_1d", "vwap_slope_96"]  # rank/pearson ratio > 30

EXTENDED_ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5
ALL_FOLDS = list(range(10))


def winsorize_zscore(s, train_s, p_lo=0.01, p_hi=0.99):
    """Fold-0 train-derived winsorize + z-score."""
    s_train = train_s.dropna()
    lo = s_train.quantile(p_lo); hi = s_train.quantile(p_hi)
    s_w = s_train.clip(lower=lo, upper=hi)
    mu, sd = s_w.mean(), s_w.std()
    if sd < 1e-8: sd = 1.0
    return ((s.clip(lower=lo, upper=hi) - mu) / sd).astype("float32")


def per_symbol_rank(panel, feat):
    """Rank within each (symbol) — converts to 0-1 then z-score equivalent."""
    return panel.groupby("symbol")[feat].rank(pct=True).astype("float32")


def per_cycle_rank(panel, feat):
    """Cross-sectional rank within each open_time."""
    return panel.groupby("open_time")[feat].rank(pct=True).astype("float32")


def build_feature_matrix(panel, train_panel, variant):
    """Build X matrix per variant. train_panel = fold-0 train for transform fitting."""
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"]})

    # Base 16 numeric features (always winsorize + z-score)
    for f in NUMERIC_FEATS:
        X[f] = winsorize_zscore(panel[f], train_panel[f])

    if variant >= 2:
        pass  # variant 2 just removes sym dummies (handled by NOT adding them)

    if variant >= 3:
        # Squared terms for U-shape features
        for f in U_SHAPE_FEATS:
            X[f + "_sq"] = (X[f] ** 2).astype("float32")

    if variant >= 4:
        # Rank-transform for high-NL features (cross-sectional rank within cycle)
        for f in HIGH_NL:
            X[f + "_xsrank"] = per_cycle_rank(panel, f)

    if variant >= 5:
        # Sign-split for asymmetric features (funding_rate has -22 skew)
        for f in ["funding_rate", "funding_rate_1d_change", "vwap_slope_96"]:
            X[f + "_pos"] = X[f].clip(lower=0).astype("float32")
            X[f + "_neg"] = X[f].clip(upper=0).astype("float32")

    # NaN → 0 (mean-impute since z-scored)
    feat_cols = [c for c in X.columns if c not in ("symbol", "open_time")]
    X[feat_cols] = X[feat_cols].fillna(0)
    return X, feat_cols


def train_variant(panel, folds_all, variant, model_type="ridge"):
    """Train one variant across all folds, return preds."""
    train0_idx_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())
    train_panel = panel[train0_idx_mask]
    X, feat_cols = build_feature_matrix(panel, train_panel, variant)
    print(f"    variant {variant} feature count: {len(feat_cols)}", flush=True)

    # Drop raw feature columns from panel before merging X (which has transformed
    # versions with same names) — avoids _x/_y suffix conflicts.
    panel_clean = panel.drop(columns=[c for c in NUMERIC_FEATS if c in panel.columns])
    panel_x = panel_clean.merge(X, on=["symbol", "open_time"], how="left")
    all_preds = []
    alpha_selected = []

    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train, cal, test = _slice(panel_x, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        te = test.dropna(subset=["target_z"]).copy()
        if len(tr) < 1000 or len(te) < 100: continue

        Xt = tr[feat_cols].to_numpy(np.float32)
        Xte = te[feat_cols].to_numpy(np.float32)
        yt = tr["target_z"].to_numpy(np.float32)
        mt = ~np.isnan(yt)

        fold_preds = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, mt.sum(), size=mt.sum())
            Xb = Xt[mt][idx]; yb = yt[mt][idx]
            if model_type == "ridge":
                m = RidgeCV(alphas=EXTENDED_ALPHAS, scoring="r2", cv=None,
                             fit_intercept=True)
            elif model_type == "lasso":
                m = LassoCV(alphas=EXTENDED_ALPHAS, cv=3, max_iter=2000,
                             fit_intercept=True, n_alphas=10)
            elif model_type == "elasticnet":
                m = ElasticNetCV(alphas=EXTENDED_ALPHAS, l1_ratio=[0.1, 0.5, 0.9],
                                  cv=3, max_iter=2000, fit_intercept=True)
            else:
                raise ValueError(model_type)
            m.fit(Xb, yb)
            fold_preds.append(m.predict(Xte).astype(np.float32))
            alpha_selected.append(float(m.alpha_))

        pred = np.mean(fold_preds, axis=0)
        df_pred = te[["symbol","open_time","exit_time","alpha_beta",
                       "target_z","sigma_idio_ref"]].copy()
        df_pred["pred_z"] = pred
        df_pred["fold"] = fid
        all_preds.append(df_pred)
        cyc_ic = df_pred.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        print(f"      fold {fid}: IC={cyc_ic.mean():+.4f} ({time.time()-t_fold:.0f}s, "
              f"α modes={sorted(set([round(a,3) for a in alpha_selected[-5:]]))})",
              flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(
        ["open_time","symbol"]).reset_index(drop=True)
    return apd


def decile_analysis(apd):
    apd_s = apd.copy()
    apd_s["alpha_bps"] = apd_s["alpha_beta"] * 1e4
    times = sorted(apd_s.open_time.unique())
    keep = set(times[::48])
    samp = apd_s[apd_s.open_time.isin(keep)]
    samp = samp.copy()
    samp["dec"] = samp.groupby("open_time")["pred_z"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    dec_means = samp.groupby("dec")["alpha_bps"].mean()
    return dec_means


def main():
    print("=== Step 12: Optimized Ridge with informed transforms ===\n", flush=True)
    t0 = time.time()

    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    # NUMERIC_FEATS only — autocorr_pctile_7d already in targets file
    base = pd.read_parquet(PANEL_BASE, columns=["symbol","open_time"] + NUMERIC_FEATS)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    panel = tgt.merge(base, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    folds_all = _multi_oos_splits(panel)

    variants_to_run = [
        (2, "ridge", "R2: 16 numeric, no transform"),
        (3, "ridge", "R3: + 6 squared (U-shape fix)"),
        (4, "ridge", "R4: + 4 rank-transform (high-NL)"),
        (5, "ridge", "R5: + 3 sign-split (asymmetric)"),
        (5, "elasticnet", "R6: ElasticNet on R5 features"),
    ]

    results = []
    for variant, model_type, desc in variants_to_run:
        print(f"\n  ----- {desc} -----", flush=True)
        apd = train_variant(panel, folds_all, variant, model_type)
        cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        ic = float(cyc_ic.mean())
        dec_means = decile_analysis(apd)
        d0 = float(dec_means.iloc[0]); d9 = float(dec_means.iloc[-1])
        spread = d9 - d0
        # Per-cycle top-3 vs bot-3 spread
        apd_s = apd.copy()
        apd_s["alpha_bps"] = apd_s["alpha_beta"] * 1e4
        times_4h = sorted(apd_s.open_time.unique())
        keep = set(times_4h[::48])
        samp = apd_s[apd_s.open_time.isin(keep)]
        def rank_stats(g):
            g = g.dropna(subset=["pred_z","alpha_bps"]).copy()
            if len(g) < 7: return pd.Series({"top3":np.nan,"bot3":np.nan})
            g = g.sort_values("pred_z", ascending=False)
            return pd.Series({"top3":g.head(3)["alpha_bps"].mean(),
                              "bot3":g.tail(3)["alpha_bps"].mean()})
        per_cycle = samp.groupby("open_time").apply(rank_stats).dropna()
        top3_mean = float(per_cycle["top3"].mean())
        bot3_mean = float(per_cycle["bot3"].mean())
        r = {"variant": desc, "ic": ic, "d0": d0, "d9": d9,
             "spread_d9_d0": spread, "top3_bps": top3_mean,
             "bot3_bps": bot3_mean, "top3_bot3": top3_mean - bot3_mean}
        results.append(r)
        print(f"    IC={ic:+.4f}  d0={d0:+.1f}  d9={d9:+.1f}  "
              f"spread={spread:+.2f}  top3-bot3={r['top3_bot3']:+.2f} bps",
              flush=True)
        apd.to_parquet(RESULTS / f"ridge_opt_v{variant}_{model_type}_preds.parquet",
                        index=False)

    print("\n" + "="*120, flush=True)
    print("  COMPARISON — Ridge variants vs LGBM", flush=True)
    print("="*120, flush=True)
    print(f"  {'variant':<45} {'IC':>10} {'d0_bps':>10} {'d9_bps':>10} "
          f"{'spread':>9} {'top3-bot3':>11}", flush=True)
    for r in results:
        print(f"  {r['variant']:<45} {r['ic']:+10.4f} {r['d0']:+10.1f} "
              f"{r['d9']:+10.1f} {r['spread_d9_d0']:+9.1f} {r['top3_bot3']:+11.1f}",
              flush=True)
    print(f"\n  Reference (LGBM clean-PIT):                 IC=+0.0162, "
          f"d0=-0.1, d9=+8.2, spread=+8.3, top3-bot3=+? (should be positive)", flush=True)

    pd.DataFrame(results).to_csv(RESULTS / "ridge_optimization_summary.csv",
                                  index=False)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
