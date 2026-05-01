"""Memory-lean permutation importance for v6 — uses the same panel build
as alpha_v6_leakage_check.py (no funding side-effect, float32 downcast)
to fit in <10GB RAM.

Same diagnostic as alpha_v6_permutation_importance.py: train v6 once on
the holdout fold, permute each feature in the test set, measure drop in
per-bar XS rank IC. Compare to LGBM gain importance.
"""
from __future__ import annotations

import gc
import logging
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_BASE_FEATURES, XS_CROSS_FEATURES, XS_FLOW_FEATURES, XS_RANK_FEATURES,
    XS_FEATURE_COLS_V6, XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _train, _holdout_split, _slice,
)

FEATURE_SET = os.environ.get("FEATURE_SET", "v6").lower()
ACTIVE_COLS = XS_FEATURE_COLS_V6_CLEAN if FEATURE_SET == "v6_clean" else XS_FEATURE_COLS_V6

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_v6_panel_lean():
    """Bypass assemble_universe (which now pulls funding); build minimally."""
    universe = list_universe(min_days=200)
    log.info("universe: %d", len(universe))
    feats_by_sym = {}
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            feats_by_sym[s] = f
    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats_by_sym.keys()))}
    log.info("basket built")

    enriched = {}
    for s, f in feats_by_sym.items():
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f
    log.info("per-sym features built")

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols = [c for c in ACTIVE_COLS if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(ACTIVE_COLS) + ["sym_id", "autocorr_pctile_7d"]
                       + src_cols) - set(rank_cols))
    log.info("labels built; assembling panel from %d cols/symbol", len(needed))

    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        # Downcast float64 -> float32 to halve memory
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
        del f
    del enriched, feats_by_sym, closes, basket_ret, basket_close
    gc.collect()
    log.info("frames ready, concatenating...")
    panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames, labels
    gc.collect()
    log.info("panel concatenated: %d rows", len(panel))
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    panel[rank_cols] = panel[rank_cols].astype("float32")
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    log.info("panel ready: %d rows", len(panel))
    return panel


def _per_bar_xs_ic(test_df, pred_arr):
    df = test_df[["open_time", "alpha_realized"]].copy()
    df["pred"] = pred_arr
    bar_ics = []
    for t, g in df.groupby("open_time"):
        if len(g) < 5:
            continue
        ic = g["pred"].rank().corr(g["alpha_realized"].rank())
        if not np.isnan(ic):
            bar_ics.append(ic)
    return np.mean(bar_ics) if bar_ics else np.nan


def _ensemble_predict(models, X):
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)


def main():
    panel = _build_v6_panel_lean()

    fold = _holdout_split(panel)[0]
    train, cal, test = _slice(panel, fold)
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    test_f = test
    log.info("train: %d, cal: %d, test: %d", len(train_f), len(cal_f), len(test_f))

    feat_cols = list(ACTIVE_COLS)
    perm_features = [f for f in feat_cols if f != "sym_id"]
    X_train = train_f[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_f["demeaned_target"].to_numpy(dtype=np.float32)
    X_cal = cal_f[feat_cols].to_numpy(dtype=np.float32)
    y_cal = cal_f["demeaned_target"].to_numpy(dtype=np.float32)

    log.info("training v6 ensemble (5 seeds)...")
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
        log.info("  seed %d trained, best_iter=%d", seed, m.best_iteration)
        models.append(m)

    # Free training data
    del X_train, y_train, X_cal, y_cal, train, cal, train_f, cal_f
    gc.collect()

    X_test = test_f[feat_cols].to_numpy(dtype=np.float32)
    yt_baseline = _ensemble_predict(models, X_test)
    baseline_ic = _per_bar_xs_ic(test_f, yt_baseline)
    log.info("baseline per-bar XS IC: %+.4f", baseline_ic)

    gains = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    gain_share = gains / gains.sum()

    rng = np.random.default_rng(42)
    rows = []
    sym_idx_map = {s: g.index.values for s, g in test_f.groupby("symbol")}
    for fi, fname in enumerate(perm_features):
        col_idx = feat_cols.index(fname)
        ic_drops = []
        for trial in range(3):
            X_perm = X_test.copy()
            for s, idx in sym_idx_map.items():
                local_pos = test_f.index.get_indexer(idx)
                shuffled = rng.permutation(X_test[local_pos, col_idx])
                X_perm[local_pos, col_idx] = shuffled
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
        log.info("  %2d/%d %s: gain=%5.2f%%  perm_drop=%+.4f", fi + 1, len(perm_features),
                  fname, 100 * gain_share[col_idx], drop_mean)

    df = pd.DataFrame(rows)
    df["gain_rank"] = df["lgbm_gain_pct"].rank(ascending=False).astype(int)
    df["perm_rank"] = df["permutation_drop_in_ic"].rank(ascending=False).astype(int)
    df["rank_gap"] = df["gain_rank"] - df["perm_rank"]
    df = df.sort_values("permutation_drop_in_ic", ascending=False)

    print("\n" + "=" * 100)
    print("PERMUTATION IMPORTANCE (v6, OOS holdout, 3 shuffles avg)")
    print("=" * 100)
    print(f"\nBaseline per-bar XS IC: {baseline_ic:+.4f}")
    print(df.round(4).to_string(index=False))

    print("\n--- Features the model uses (gain >2%) but DON'T contribute OOS (overfit suspects) ---")
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

    print("\n--- Summary ---")
    helpful = (df["permutation_drop_in_ic"] > 0.001).sum()
    neutral = ((df["permutation_drop_in_ic"].abs() <= 0.001)).sum()
    harmful = (df["permutation_drop_in_ic"] < -0.001).sum()
    print(f"  Features that help OOS  (perm_drop > +0.001): {helpful}/{len(df)}")
    print(f"  Features that are neutral (|perm_drop| ≤ 0.001): {neutral}/{len(df)}")
    print(f"  Features that hurt OOS (perm_drop < -0.001):   {harmful}/{len(df)}")
    print(f"\n  Total |perm_drop| sum = {df['permutation_drop_in_ic'].sum():+.4f}")
    print(f"  vs baseline IC of {baseline_ic:+.4f}")

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{FEATURE_SET}_permutation_importance.csv"
    df.to_csv(out_path, index=False)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
