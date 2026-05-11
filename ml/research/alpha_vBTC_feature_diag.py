"""Phase 4 post-mortem: per-feature diagnostic on btc_target vs basket-target.

Computes for each of 32 features in the dedup set:
  - IC vs btc_target (full sample)
  - IC vs alpha_realized (basket-target, v6_clean reference)
  - Per-fold IC vs btc_target on test sets (sign stability)

Goal: explain whether negative Sharpe in Phase 4 is a feature problem
(features predict basket-target but not BTC-target) or a strategy problem
(features predict, but C/D/B mechanics fail).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_vBTC_train_eval import V6_KEEP, NEW_BTC

PANEL_PATH = REPO / "outputs/vBTC_features/panel_with_btc_features.parquet"
OUT_DIR = REPO / "outputs/vBTC_feature_diag"
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

    # === Full-sample IC vs both targets ===
    print("\n  === Full-sample IC vs btc_target and alpha_realized ===")
    pool = panel[panel["autocorr_pctile_7d"] >= THRESHOLD].copy()
    rows = []
    for f in feat_set:
        ic_btc = _spearman(pool[f], pool["btc_target"])
        ic_basket = _spearman(pool[f], pool["alpha_realized"])
        flag = "v6" if f in V6_KEEP else "new"
        rows.append({"feature": f, "tag": flag,
                     "ic_btc": ic_btc, "ic_basket": ic_basket,
                     "ratio": ic_btc / ic_basket if ic_basket and abs(ic_basket) > 1e-6 else np.nan})
    df_full = pd.DataFrame(rows).sort_values("ic_btc", key=lambda x: -x.abs())
    print(f"  {'feature':<32} {'tag':<4} {'IC_btc':>8} {'IC_bskt':>8} {'ratio':>7}")
    for _, r in df_full.iterrows():
        print(f"  {r['feature']:<32} {r['tag']:<4} {r['ic_btc']:>+8.4f} {r['ic_basket']:>+8.4f} {r['ratio']:>+7.2f}")

    # === Per-fold IC stability ===
    print("\n  === Per-fold IC vs btc_target on TEST sets (sign-stability check) ===")
    folds = _multi_oos_splits(panel)
    fold_idx = [len(folds) // 5, len(folds) // 2, 4 * len(folds) // 5]
    folds_to_eval = [folds[i] for i in fold_idx]
    print(f"  Evaluating folds {fold_idx}, fids {[f['fid'] for f in folds_to_eval]}")

    per_fold_ic = {f: [] for f in feat_set}
    for fold in folds_to_eval:
        _, _, test = _slice(panel, fold)
        test_filt = test[test["autocorr_pctile_7d"] >= THRESHOLD]
        for f in feat_set:
            ic = _spearman(test_filt[f], test_filt["btc_target"])
            per_fold_ic[f].append(ic)

    fold_ids = [f["fid"] for f in folds_to_eval]
    print(f"  {'feature':<32} {'fold' + str(fold_ids[0]):>8} {'fold' + str(fold_ids[1]):>8} "
          f"{'fold' + str(fold_ids[2]):>8} {'mean':>8} {'sign_stab':>10}")
    rows2 = []
    for f, ics in per_fold_ic.items():
        signs = [1 if i > 0 else -1 if i < 0 else 0 for i in ics if not np.isnan(i)]
        sign_stab = max(signs.count(1), signs.count(-1)) / max(len(signs), 1) if signs else 0
        mean = np.nanmean(ics)
        rows2.append({"feature": f, "ic_f0": ics[0], "ic_f1": ics[1], "ic_f2": ics[2],
                      "ic_mean": mean, "sign_stab": sign_stab})
        print(f"  {f:<32} {ics[0]:>+8.4f} {ics[1]:>+8.4f} {ics[2]:>+8.4f} "
              f"{mean:>+8.4f} {sign_stab:>10.2f}")

    df_fold = pd.DataFrame(rows2)

    # === Aggregate diagnostics ===
    print("\n" + "=" * 90)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 90)

    # 1. How many features have stronger IC against basket than BTC target?
    df_full["btc_weaker"] = df_full["ic_btc"].abs() < df_full["ic_basket"].abs()
    n_btc_weaker = int(df_full["btc_weaker"].sum())
    print(f"\n  1. Features with weaker |IC| against btc_target than alpha_realized: "
          f"{n_btc_weaker} of {len(df_full)}")
    print(f"     → if >50%, it's a TARGET problem (features don't transfer to BTC residual)")

    # 2. Median absolute IC ratio (btc / basket)
    valid_ratios = df_full["ratio"].dropna()
    if len(valid_ratios) > 0:
        print(f"\n  2. Median IC ratio btc/basket: {valid_ratios.abs().median():+.2f}")
        print(f"     → < 0.7 means btc-target IC is much weaker; ~1.0 means equally strong")

    # 3. Sign-stability across folds
    flips = df_fold[df_fold["sign_stab"] < 0.67]
    print(f"\n  3. Features with sign-instability (< 2 of 3 folds same sign): "
          f"{len(flips)} of {len(df_fold)}")
    if len(flips) > 0:
        print(f"     Worst offenders:")
        for _, r in flips.head(5).iterrows():
            print(f"       {r['feature']:<32} ICs: {r['ic_f0']:+.4f} {r['ic_f1']:+.4f} {r['ic_f2']:+.4f}")
        print(f"     → if >30% are unstable, model is fitting noise that doesn't generalize")

    # 4. Top-5 features for each target
    print(f"\n  4. Top-5 features by |IC| vs each target:")
    print(f"     Against btc_target:")
    for _, r in df_full.head(5).iterrows():
        print(f"       {r['feature']:<32} IC={r['ic_btc']:+.4f}")
    print(f"     Against alpha_realized (basket):")
    df_basket_top = df_full.iloc[df_full["ic_basket"].abs().argsort()[::-1][:5]]
    for _, r in df_basket_top.iterrows():
        print(f"       {r['feature']:<32} IC={r['ic_basket']:+.4f}")

    df_full.to_csv(OUT_DIR / "feature_ic_full_sample.csv", index=False)
    df_fold.to_csv(OUT_DIR / "feature_ic_per_fold.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
