"""Phase 4 salvage: measure feature contributions via LGBM gain importance,
combined with per-fold sign-stability, then output a refined keep-set.

Pipeline:
  1. Train 1 LGBM per fold on the 32-feature dedup set (3 folds = 3 LGBMs)
  2. Extract gain importance from each
  3. Combine: importance_score = mean_normalized_gain * sign_stability_factor
  4. Print ranked table + write keep-set CSV for retrain step
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_vBTC_train_eval import V6_KEEP, NEW_BTC

PANEL_PATH = REPO / "outputs/vBTC_features/panel_with_btc_features.parquet"
OUT_DIR = REPO / "outputs/vBTC_feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC


def _spearman(a, b):
    sub = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(sub) < 1000:
        return np.nan
    return float(sub["a"].rank().corr(sub["b"].rank()))


def main():
    print(f"Loading panel...")
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [c for c in V6_KEEP + NEW_BTC if c in panel.columns]
    print(f"  {len(panel):,} rows, {len(feat_set)} features")

    folds = _multi_oos_splits(panel)
    fold_idx = [len(folds) // 5, len(folds) // 2, 4 * len(folds) // 5]
    folds_to_eval = [folds[i] for i in fold_idx]
    print(f"  Multi-OOS folds {fold_idx}, fids {[f['fid'] for f in folds_to_eval]}")

    # === Train one LGBM per fold + extract gain importance ===
    importance_per_fold = {}    # {fold_id: {feature: gain}}
    test_ic_per_fold = {}       # {fold_id: {feature: test_IC}}
    for i, fold in enumerate(folds_to_eval):
        t0 = time.time()
        print(f"\n  Fold {fold['fid']}: training LGBM with all {len(feat_set)} features...")
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        yt = tr["btc_target"].to_numpy(np.float32)
        yc = ca["btc_target"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        model = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=42)
        gains = model.feature_importance(importance_type="gain")
        gains_norm = gains / max(gains.sum(), 1e-9)
        importance_per_fold[fold["fid"]] = dict(zip(feat_set, gains_norm))

        # Also test IC for stability
        test_filt = test[test["autocorr_pctile_7d"] >= THRESHOLD]
        test_ic_per_fold[fold["fid"]] = {f: _spearman(test_filt[f], test_filt["btc_target"]) for f in feat_set}
        print(f"    LGBM trained in {time.time()-t0:.0f}s, best_iter={model.best_iteration}")

    # === Aggregate ===
    print("\n" + "=" * 110)
    print("FEATURE IMPORTANCE — combined gain + sign-stability ranking")
    print("=" * 110)

    rows = []
    for f in feat_set:
        gains = [importance_per_fold[fid].get(f, 0.0) for fid in importance_per_fold]
        ics = [test_ic_per_fold[fid].get(f, 0.0) for fid in test_ic_per_fold]
        signs = [1 if i > 0 else -1 if i < 0 else 0 for i in ics if not np.isnan(i)]
        if not signs:
            sign_stab = 0.0
        else:
            sign_stab = max(signs.count(1), signs.count(-1)) / len(signs)
        mean_gain = float(np.mean(gains))
        # Combined score: gain × (sign_stab^2). Squaring penalizes mid-stab features.
        score = mean_gain * (sign_stab ** 2)
        tag = "v6" if f in V6_KEEP else "new"
        rows.append({
            "feature": f, "tag": tag,
            "gain_f0": gains[0], "gain_f1": gains[1], "gain_f2": gains[2],
            "ic_f0": ics[0], "ic_f1": ics[1], "ic_f2": ics[2],
            "mean_gain": mean_gain,
            "mean_ic": float(np.nanmean(ics)),
            "sign_stab": sign_stab,
            "score": score,
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False)

    print(f"\n  {'feature':<32} {'tag':<4} {'g_f0':>6} {'g_f1':>6} {'g_f2':>6} "
          f"{'mean_g':>7} {'mn_IC':>7} {'stab':>5} {'score':>7}")
    for _, r in df.iterrows():
        print(f"  {r['feature']:<32} {r['tag']:<4} "
              f"{r['gain_f0']*100:>5.2f}% {r['gain_f1']*100:>5.2f}% {r['gain_f2']*100:>5.2f}% "
              f"{r['mean_gain']*100:>6.2f}% {r['mean_ic']:>+7.4f} {r['sign_stab']:>5.2f} "
              f"{r['score']*100:>6.3f}%")

    # === Pick keep set: top-K by score, with floors ===
    # Strategy: any feature with score ≥ 0.005 (=0.5% of total gain × full stab)
    # OR top-15 by score (whichever is fewer/larger). Then deduplicate.
    SCORE_THRESHOLD = 0.005   # 0.5% gain at full stability
    by_score = df[df["score"] >= SCORE_THRESHOLD]["feature"].tolist()
    top15 = df.head(15)["feature"].tolist()
    keep = sorted(set(by_score) | set(top15), key=lambda f: -df[df["feature"] == f]["score"].iloc[0])

    print(f"\n  KEEP SET ({len(keep)} features, sorted by score):")
    for f in keep:
        r = df[df["feature"] == f].iloc[0]
        print(f"    {f:<32} mean_gain={r['mean_gain']*100:.2f}% stab={r['sign_stab']:.2f} score={r['score']*100:.3f}%")

    drop = sorted([f for f in feat_set if f not in keep])
    print(f"\n  DROPPED ({len(drop)} features):")
    for f in drop:
        r = df[df["feature"] == f].iloc[0]
        print(f"    {f:<32} mean_gain={r['mean_gain']*100:.2f}% stab={r['sign_stab']:.2f} score={r['score']*100:.3f}%")

    df.to_csv(OUT_DIR / "feature_score_table.csv", index=False)
    pd.Series(keep).to_csv(OUT_DIR / "keep_features.csv", index=False, header=["feature"])
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
