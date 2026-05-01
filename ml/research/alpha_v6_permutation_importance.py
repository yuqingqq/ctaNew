"""Permutation importance for v6 — does each feature actually work IN the model?

Single-feature IC audits ("feature X has OOS |IC| 0.05") test whether a
feature correlates with alpha. They do NOT test whether the model
actually uses the feature in a way that translates to OOS predictive
power. Funding rate features (Phase 4.1) had strong audit IC but added
no portfolio value when integrated — the audit gate is necessary but
not sufficient.

Permutation importance is the sklearn-style direct test: train the model
once with all features, then permute each feature's values in the test
set, re-predict, and measure the drop in OOS rank IC. Features the model
genuinely uses will hurt OOS when permuted; features it ignores won't.

Per-symbol shuffle (preserves marginal distribution within each symbol) is
used so we measure cross-sectional importance, not symbol-distribution effects.

Comparison to LGBM gain importance: the model's INTERNAL view of feature
value during training. Discrepancy between gain (LGBM-believes-useful) and
permutation (actually-OOS-useful) flags potential overfitting.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6, XS_RANK_SOURCES, add_xs_rank_features,
    assemble_universe, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import _stack_xs_panel, portfolio_pnl_turnover_aware
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _train,
    _holdout_split, _slice,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _per_bar_xs_ic(test_df, pred_arr):
    """Mean per-bar Spearman IC of pred vs alpha_realized."""
    df = test_df[["open_time", "alpha_realized"]].copy()
    df["pred"] = pred_arr
    bar_ics = []
    for t, g in df.groupby("open_time"):
        if len(g) < 5: continue
        ic = g["pred"].rank().corr(g["alpha_realized"].rank())
        if not np.isnan(ic): bar_ics.append(ic)
    return np.mean(bar_ics) if bar_ics else np.nan


def _ensemble_predict(models, X):
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)


def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d", len(universe))
    pkg = assemble_universe(universe, horizon=HORIZON)
    labels_by_sym = make_xs_alpha_labels(pkg["feats_by_sym"], pkg["basket_close"], HORIZON)

    # Build v6 panel with xs_rank features
    rank_cols = [c for c in XS_FEATURE_COLS_V6 if c.endswith("_xs_rank")]
    src_cols_for_stack = [c for c in XS_FEATURE_COLS_V6 if not c.endswith("_xs_rank")]
    source_features = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    stack_cols = list(set(src_cols_for_stack + source_features))
    panel = _stack_xs_panel(pkg["feats_by_sym"], labels_by_sym, cols=stack_cols)
    panel = add_xs_rank_features(panel,
        sources={s: d for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    log.info("panel: %d rows", len(panel))

    fold = _holdout_split(panel)[0]
    train, cal, test = _slice(panel, fold)
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    test_f = test
    log.info("train: %d, cal: %d, test: %d", len(train_f), len(cal_f), len(test_f))

    feat_cols = list(XS_FEATURE_COLS_V6)
    perm_features = [f for f in feat_cols if f != "sym_id"]
    X_train = train_f[feat_cols].to_numpy()
    y_train = train_f["demeaned_target"].to_numpy()
    X_cal = cal_f[feat_cols].to_numpy()
    y_cal = cal_f["demeaned_target"].to_numpy()

    log.info("training v6 ensemble (5 seeds)...")
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
        models.append(m)

    # Baseline OOS prediction + IC
    X_test = test_f[feat_cols].to_numpy()
    yt_baseline = _ensemble_predict(models, X_test)
    baseline_ic = _per_bar_xs_ic(test_f, yt_baseline)
    log.info("baseline per-bar XS IC: %+.4f", baseline_ic)

    # LGBM gain importance (model's internal view)
    gains = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    gain_share = gains / gains.sum()

    # Permutation importance: per-symbol shuffle preserves marginal distribution
    rng = np.random.default_rng(42)
    rows = []
    for fi, fname in enumerate(perm_features):
        # Index of fname in feat_cols
        col_idx = feat_cols.index(fname)
        # Per-symbol shuffle of test set: shuffle within each symbol's slice
        ic_drops = []
        for trial in range(3):  # average over 3 shuffles to reduce noise
            test_perm = test_f.copy()
            for s, g in test_f.groupby("symbol"):
                shuffled = rng.permutation(g[fname].to_numpy())
                test_perm.loc[g.index, fname] = shuffled
            X_perm = test_perm[feat_cols].to_numpy()
            yt_perm = _ensemble_predict(models, X_perm)
            ic_perm = _per_bar_xs_ic(test_f, yt_perm)
            ic_drops.append(baseline_ic - ic_perm)
        drop_mean = np.mean(ic_drops)
        rows.append({
            "feature": fname,
            "lgbm_gain_pct": 100 * gain_share[col_idx],
            "permutation_drop_in_ic": drop_mean,
            "baseline_ic": baseline_ic,
            "permuted_ic_mean": baseline_ic - drop_mean,
        })
        log.info("  %s: gain=%5.2f%%  perm_drop=%+.4f", fname,
                  100 * gain_share[col_idx], drop_mean)

    df = pd.DataFrame(rows)
    df["gain_rank"] = df["lgbm_gain_pct"].rank(ascending=False).astype(int)
    df["perm_rank"] = df["permutation_drop_in_ic"].rank(ascending=False).astype(int)
    df["rank_gap"] = df["gain_rank"] - df["perm_rank"]
    df = df.sort_values("permutation_drop_in_ic", ascending=False)

    print("\n" + "=" * 100)
    print("PERMUTATION IMPORTANCE (v6, OOS holdout, 3 shuffles avg)")
    print("=" * 100)
    print(f"\nBaseline per-bar XS IC: {baseline_ic:+.4f}")
    print(f"\nFor each feature: 'perm_drop' = baseline_IC - IC_when_permuted.")
    print(f"  Positive perm_drop = feature genuinely contributes to OOS predictions.")
    print(f"  ~0 perm_drop = feature is ignored by the model OR adds no real OOS info.")
    print(f"  Negative perm_drop = permuting HELPS — the feature was injecting noise.")
    print(f"\nrank_gap = gain_rank - perm_rank.")
    print(f"  ~0 = LGBM correctly identifies feature value.")
    print(f"  >0 = LGBM thinks feature is more important than it actually is OOS (overfit warning).")
    print(f"  <0 = LGBM under-uses a useful feature.\n")
    print(df.round(4).to_string(index=False))

    print("\n--- Features the model uses but DON'T contribute OOS (overfit suspects) ---")
    suspects = df[(df["lgbm_gain_pct"] > 2.0) & (df["permutation_drop_in_ic"].abs() < 0.001)]
    if len(suspects):
        print(suspects.round(4).to_string(index=False))
    else:
        print("  (none)")

    print("\n--- Features the model under-uses (low gain but real OOS impact) ---")
    underused = df[(df["lgbm_gain_pct"] < 1.0) & (df["permutation_drop_in_ic"] > 0.005)]
    if len(underused):
        print(underused.round(4).to_string(index=False))
    else:
        print("  (none)")

    print("\n--- Negative perm_drop (feature is HURTING OOS) ---")
    hurts = df[df["permutation_drop_in_ic"] < -0.002]
    if len(hurts):
        print(hurts.round(4).to_string(index=False))
        print("  These features inject noise OOS — candidates to drop.")
    else:
        print("  (none — all features either help or are neutral)")

    # Summary stats
    print("\n--- Summary ---")
    helpful = (df["permutation_drop_in_ic"] > 0.001).sum()
    neutral = ((df["permutation_drop_in_ic"].abs() <= 0.001)).sum()
    harmful = (df["permutation_drop_in_ic"] < -0.001).sum()
    print(f"  Features that help OOS  (perm_drop > +0.001): {helpful}/{len(df)}")
    print(f"  Features that are neutral (|perm_drop| ≤ 0.001): {neutral}/{len(df)}")
    print(f"  Features that hurt OOS (perm_drop < -0.001):   {harmful}/{len(df)}")
    print(f"\n  Total |perm_drop| sum = {df['permutation_drop_in_ic'].sum():+.4f}")
    print(f"  vs baseline IC of {baseline_ic:+.4f}")
    print(f"  → If sum ≈ baseline: features are roughly orthogonal contributors.")
    print(f"  → If sum >> baseline: features have heavy interactions (good — model uses them together).")
    print(f"  → If sum << baseline: ablating each feature alone barely hurts; features are redundant.")

    from pathlib import Path
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "v6_permutation_importance.csv", index=False)


if __name__ == "__main__":
    main()
