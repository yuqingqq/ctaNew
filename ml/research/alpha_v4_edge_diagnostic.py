"""Raw edge diagnostic for cross-sectional alpha v4 at h=288.

Step back from cost/deployment economics. The questions:

  A. Feature potential. What rank IC ceiling is implied by the feature
     set? How much of it does LGBM actually capture?
  B. Model learning. Does the LGBM converge to its in-sample optimum?
     Train → cal → OOS IC decay tells us about overfit.
  C. Cross-sectional realization. Even at the same rank IC, the way
     predicted ranks map to realized alpha can be linear-monotone, step,
     or noisy. Quintile profile reveals which.
  D. Top-K selection vs quintile. Are the 5 selected names the best 5
     in the top quintile, or are we drawing roughly uniformly from a
     20% slice (5/25)?
  E. Per-symbol structure. Pooled training only works if alpha
     structure is similar across symbols. Per-symbol IC of OOS predictions
     reveals whether the global model is leaving symbol-specific edges
     unrealized.
  F. Where do errors concentrate? Top-quintile mistakes (predicted top
     but realized bottom) vs bottom-quintile mistakes — and by symbol.

Output: single diagnostic with numbered sections; no LGBM training rerun
needed beyond a single fit on the OOS-holdout fold.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS, XS_FEATURE_COLS_V6, XS_RANK_SOURCES,
    add_xs_rank_features, assemble_universe, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import _stack_xs_panel
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, ACTIVE_FEATURE_COLS, _train,
    _walk_forward_splits, _holdout_split, _slice,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _fit_predict_holdout(panel: pd.DataFrame) -> dict:
    """Fit LGBM ensemble on the OOS holdout split with the same regime
    filter used by alpha_v4_xs_1d.py (train/cal restricted to top-33%
    autocorr_pctile_7d, test unfiltered). Returns predictions on
    train_f, cal_f, and full test."""
    fold = _holdout_split(panel)[0]
    train, cal, test = _slice(panel, fold)
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    log.info("fit_predict: train=%d (filtered=%d), cal=%d (filtered=%d), test=%d",
              len(train), len(train_f), len(cal), len(cal_f), len(test))

    feat_cols = ACTIVE_FEATURE_COLS  # respects FEATURE_SET env var
    X_train = train_f[feat_cols].to_numpy()
    y_train = train_f["demeaned_target"].to_numpy()
    X_cal = cal_f[feat_cols].to_numpy()
    y_cal = cal_f["demeaned_target"].to_numpy()

    models = []
    for seed in ENSEMBLE_SEEDS:
        m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
        models.append(m)

    def _pred(X):
        return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)

    yt_train = _pred(X_train)
    yt_cal = _pred(X_cal)
    yt_test = _pred(test[feat_cols].to_numpy())
    train, cal = train_f, cal_f  # use the filtered frames downstream

    # Feature importance averaged over seeds
    gain = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    importance = pd.Series(dict(zip(feat_cols, gain / gain.sum()))).sort_values(ascending=False)

    return {
        "models": models, "importance": importance,
        "train": train.assign(pred=yt_train),
        "cal": cal.assign(pred=yt_cal),
        "test": test.assign(pred=yt_test),
        "feat_cols": feat_cols,
    }


def _spearman(x, y):
    x = pd.Series(x); y = pd.Series(y)
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 50:
        return np.nan
    return df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def _section_A_feature_potential(train, test, feat_cols):
    """Per-feature IC vs alpha realized, in-sample and OOS. Plus oracle:
    in-sample linear regression of alpha on features, OOS Spearman of fit.
    The linear oracle is a rough upper bound on what a well-fit linear
    model could capture; LGBM should do at least this well."""
    print("\n" + "=" * 80)
    print("A. FEATURE POTENTIAL (per-feature IC vs realized alpha)")
    print("=" * 80)
    rows = []
    for f in feat_cols:
        if f == "sym_id":
            continue
        ic_is = _spearman(train[f], train["alpha_realized"])
        ic_oos = _spearman(test[f], test["alpha_realized"])
        rows.append({"feature": f, "ic_IS": ic_is, "ic_OOS": ic_oos,
                      "decay": ic_oos - ic_is, "abs_oos": abs(ic_oos)})
    df = pd.DataFrame(rows).sort_values("abs_oos", ascending=False)
    print(df.round(4).to_string(index=False))

    # Linear-regression oracle on in-sample. Predict OOS, compare.
    drop = [c for c in feat_cols if c == "sym_id"]
    X_is = train[[c for c in feat_cols if c not in drop]].fillna(0.0).to_numpy()
    y_is = train["alpha_realized"].to_numpy()
    X_oos = test[[c for c in feat_cols if c not in drop]].fillna(0.0).to_numpy()
    y_oos = test["alpha_realized"].to_numpy()
    # Simple OLS via lstsq
    Xc = np.column_stack([np.ones(len(X_is)), X_is])
    beta, *_ = np.linalg.lstsq(Xc, y_is, rcond=None)
    pred_oracle_is = (np.column_stack([np.ones(len(X_is)), X_is]) @ beta)
    pred_oracle_oos = (np.column_stack([np.ones(len(X_oos)), X_oos]) @ beta)
    print(f"\n  Linear oracle (OLS on in-sample, predict OOS):")
    print(f"    IC_IS  = {_spearman(pred_oracle_is, y_is):+.4f}")
    print(f"    IC_OOS = {_spearman(pred_oracle_oos, y_oos):+.4f}")


def _section_B_train_vs_oos(parts):
    """Train, Cal, OOS IC of LGBM predictions. Reports both pooled IC
    (across all (sym,bar) pairs) and per-bar cross-sectional rank IC
    (averaged across bars). The latter is what the portfolio actually
    uses; the former mostly captures cross-bar variance."""
    print("\n" + "=" * 80)
    print("B. MODEL LEARNING — Train / Cal / OOS IC of LGBM predictions")
    print("=" * 80)
    for name in ["train", "cal", "test"]:
        df = parts[name]
        ic_pooled = _spearman(df["pred"], df["alpha_realized"])
        # Per-bar cross-sectional IC averaged over bars (the strategy's IC)
        bar_ics = []
        for t, g in df.groupby("open_time"):
            if len(g) < 5:
                continue
            ic_b = g["pred"].rank().corr(g["alpha_realized"].rank())
            if not np.isnan(ic_b):
                bar_ics.append(ic_b)
        ic_xs = np.mean(bar_ics) if bar_ics else np.nan
        print(f"  {name:<6}  n={len(df):>7}  pooled_IC={ic_pooled:+.4f}  per-bar_xs_IC={ic_xs:+.4f}  ({len(bar_ics)} bars)")
    print("\n  Feature importance (gain share):")
    for f, v in parts["importance"].items():
        print(f"    {f:<28}: {v*100:5.2f}%")


def _section_C_quintile_profile(test):
    """Per-quintile realized alpha. For each bar, rank symbols by predicted
    alpha into 5 quintiles; pool across bars; report mean realized alpha
    per quintile. A linear-monotone profile is what we want; step or noisy
    profiles tell different stories."""
    print("\n" + "=" * 80)
    print("C. QUINTILE PROFILE — predicted-quintile → realized alpha (bps)")
    print("=" * 80)
    df = test[["open_time", "symbol", "pred", "alpha_realized"]].dropna().copy()
    df["q"] = df.groupby("open_time")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop"))
    profile = df.groupby("q").agg(
        n=("alpha_realized", "size"),
        mean_alpha_bps=("alpha_realized", lambda x: x.mean() * 1e4),
        std_alpha_bps=("alpha_realized", lambda x: x.std() * 1e4),
    ).round(3)
    print(profile.to_string())
    if 4 in profile.index and 0 in profile.index:
        spread = profile.loc[4, "mean_alpha_bps"] - profile.loc[0, "mean_alpha_bps"]
        print(f"\n  Q4-Q0 (top-bot) spread: {spread:+.2f} bps/cycle")
        # Linear monotonicity check: regression of mean_alpha on quintile index
        x = profile.index.to_numpy()
        y = profile["mean_alpha_bps"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        # R² for linear
        yhat = slope * x + intercept
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"  Linear fit slope={slope:+.2f} bps/quintile, R²={r2:.3f}  (R²→1 = linear-monotone, R²→0 = step/noisy)")


def _section_D_topK_vs_quintile(test):
    """Compare top-5 selection alpha to mean of top-quintile alpha.
    If top-5 alpha ≫ top-quintile mean, the within-quintile rank carries
    real information. If top-5 alpha ≈ top-quintile mean, the top-5
    pick is essentially a uniform sample from the top quintile."""
    print("\n" + "=" * 80)
    print("D. TOP-K SELECTION QUALITY (within-quintile rank info)")
    print("=" * 80)
    df = test[["open_time", "symbol", "pred", "alpha_realized"]].dropna().copy()
    rows = []
    for k_long in [1, 3, 5]:
        long_alpha = []
        topQ_alpha = []
        for t, g in df.groupby("open_time"):
            if len(g) < 10:
                continue
            sg = g.sort_values("pred")
            top_k = sg.tail(k_long)
            top_q = sg.tail(int(round(0.2 * len(sg))))
            long_alpha.append(top_k["alpha_realized"].mean() * 1e4)
            topQ_alpha.append(top_q["alpha_realized"].mean() * 1e4)
        rows.append({"k_long": k_long,
                      "top_k_mean_bps": np.mean(long_alpha),
                      "top_quintile_mean_bps": np.mean(topQ_alpha),
                      "top_k_minus_quintile": np.mean(long_alpha) - np.mean(topQ_alpha)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\n  If 'top_k_minus_quintile' is significantly positive: within-quintile rank has signal.")
    print("  If ~zero: model can identify the quintile but not refine within it.")


def _section_E_per_symbol_IC(test):
    """Per-symbol Spearman IC of predictions vs realized alpha.
    Pooled training assumes alpha structure is symbol-invariant — if
    per-symbol IC varies wildly (some symbols +0.10, others -0.05), the
    pooled model has unrealized symbol-specific edges."""
    print("\n" + "=" * 80)
    print("E. PER-SYMBOL OOS IC of LGBM predictions vs realized alpha")
    print("=" * 80)
    rows = []
    for s, g in test.groupby("symbol"):
        ic = _spearman(g["pred"], g["alpha_realized"])
        rows.append({"symbol": s, "n": len(g), "ic_oos": ic})
    df = pd.DataFrame(rows).sort_values("ic_oos", ascending=False)
    print(df.round(4).to_string(index=False))
    print(f"\n  Mean IC: {df['ic_oos'].mean():+.4f}, Std IC: {df['ic_oos'].std():.4f}")
    print(f"  Symbols with IC > 0: {(df['ic_oos'] > 0).sum()}/{len(df)}")
    print(f"  Symbols with IC > +0.05: {(df['ic_oos'] > 0.05).sum()}/{len(df)}")
    print(f"  Symbols with IC < -0.05: {(df['ic_oos'] < -0.05).sum()}/{len(df)}")


def _section_F_error_concentration(test):
    """Where do the biggest prediction errors live? Look at the top
    quintile that turned out to be the BOTTOM quintile (false positives
    on long side) and bottom-quintile that turned out to be top
    (false positives on short side). Concentration by symbol/regime."""
    print("\n" + "=" * 80)
    print("F. ERROR CONCENTRATION — where does the model fail OOS?")
    print("=" * 80)
    df = test[["open_time", "symbol", "pred", "alpha_realized", "autocorr_pctile_7d"]].dropna().copy()
    df["pred_q"] = df.groupby("open_time")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop"))
    df["real_q"] = df.groupby("open_time")["alpha_realized"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop"))

    # Confusion: how many top-pred (q=4) end up in real-q ?
    print("\n  Confusion matrix (rows=predicted_q, cols=realized_q), normalized per row:")
    cm = pd.crosstab(df["pred_q"], df["real_q"], normalize="index").round(3)
    print(cm.to_string())

    # Mistake fraction per symbol
    df["miss_long"] = ((df["pred_q"] == 4) & (df["real_q"] <= 1)).astype(int)
    df["miss_short"] = ((df["pred_q"] == 0) & (df["real_q"] >= 3)).astype(int)
    by_sym = df.groupby("symbol").agg(
        n=("miss_long", "size"),
        miss_long_rate=("miss_long", "mean"),
        miss_short_rate=("miss_short", "mean"),
    ).round(4).sort_values("miss_long_rate", ascending=False)
    print("\n  Per-symbol miss rates (top 8 worst long-side false positives):")
    print(by_sym.head(8).to_string())


def _section_G_redundancy(train, feat_cols):
    """Feature correlation matrix. High pairwise correlations point to
    redundancy — LGBM tolerates it but it caps the marginal contribution
    of each feature."""
    print("\n" + "=" * 80)
    print("G. FEATURE REDUNDANCY (Pearson correlation, in-sample)")
    print("=" * 80)
    cols = [c for c in feat_cols if c != "sym_id"]
    corr = train[cols].corr().abs()
    # Show pairs with |corr| > 0.5
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (upper.stack()
                  .reset_index()
                  .rename(columns={"level_0": "f1", "level_1": "f2", 0: "abs_corr"}))
    pairs = pairs.sort_values("abs_corr", ascending=False).head(10)
    print("\n  Top 10 most-correlated feature pairs (|Pearson|):")
    print(pairs.round(3).to_string(index=False))
    # Also: each feature's mean |corr| with all others
    mean_abs_corr = corr.where(~np.eye(len(corr), dtype=bool)).mean(skipna=True)
    print("\n  Per-feature mean |corr| with all other features:")
    print(mean_abs_corr.sort_values(ascending=False).round(3).to_string())


def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d", len(universe))
    pkg = assemble_universe(universe, horizon=HORIZON)
    labels_by_sym = make_xs_alpha_labels(pkg["feats_by_sym"], pkg["basket_close"], HORIZON)

    # If feature set has xs_rank cols, compute them post-stack
    rank_cols = [c for c in ACTIVE_FEATURE_COLS if c.endswith("_xs_rank")]
    src_cols_for_stack = [c for c in ACTIVE_FEATURE_COLS if not c.endswith("_xs_rank")]
    source_features = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    stack_cols = list(set(src_cols_for_stack + source_features))
    panel = _stack_xs_panel(pkg["feats_by_sym"], labels_by_sym, cols=stack_cols)
    if rank_cols:
        panel = add_xs_rank_features(panel,
            sources={s: d for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
        panel = panel.dropna(subset=rank_cols)
    panel = panel.dropna(subset=["autocorr_pctile_7d"])
    log.info("panel: %d rows", len(panel))

    parts = _fit_predict_holdout(panel)

    print("\n" + "#" * 80)
    print(f"# CROSS-SECTIONAL ALPHA v4 EDGE DIAGNOSTIC (h={HORIZON}, OOS holdout)")
    print(f"# Train n={len(parts['train'])}, Cal n={len(parts['cal'])}, OOS n={len(parts['test'])}")
    print("#" * 80)

    _section_A_feature_potential(parts["train"], parts["test"], parts["feat_cols"])
    _section_B_train_vs_oos(parts)
    _section_C_quintile_profile(parts["test"])
    _section_D_topK_vs_quintile(parts["test"])
    _section_E_per_symbol_IC(parts["test"])
    _section_F_error_concentration(parts["test"])
    _section_G_redundancy(parts["train"], parts["feat_cols"])


if __name__ == "__main__":
    main()
