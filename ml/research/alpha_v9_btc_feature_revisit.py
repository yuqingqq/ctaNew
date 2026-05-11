"""Phase 1: rank v6_clean features by IC against BTC-β-adjusted target.

For each of the 28 v6_clean features, compute:
  - Spearman rank correlation against btc_beta_target on TRAINING data (cycle-level)
  - Single-feature LGBM IC (quick 1-fold training, predict-and-correlate)

Output a ranked list with recommendation: keep / drop / consider.

This is intentionally fast (no ensembles, no multi-OOS) — we just want a
ranking to inform Phase 3's feature subset decision.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_btc_beta_target import add_btc_beta_target

OUT_DIR = REPO / "outputs/btc_feature_revisit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC


def main():
    print("Building panel + adding BTC target...")
    panel = build_wide_panel()
    panel = add_btc_beta_target(panel)
    print(f"  panel: {len(panel):,} rows, btc_beta_target non-NaN: {panel['btc_beta_target'].notna().sum():,}")

    # Compute IC on TRAINING data (use first fold's train set, ~80% of panel)
    folds = _multi_oos_splits(panel)
    fold0 = folds[0]
    train, _, _ = _slice(panel, fold0)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    tr = tr.dropna(subset=["btc_beta_target"])
    print(f"\n  Training subset for IC: {len(tr):,} rows (fold 0 train)")

    # Reference: also compute IC against basket-target for comparison
    v6_features = [c for c in XS_FEATURE_COLS_V6_CLEAN if c in panel.columns]
    print(f"  Computing IC for {len(v6_features)} v6_clean features...")

    rows = []
    for feat in v6_features:
        sub = tr[[feat, "btc_beta_target", "demeaned_target"]].dropna()
        if len(sub) < 1000:
            continue
        # Spearman rank IC vs each target
        ic_btc = float(sub[feat].rank().corr(sub["btc_beta_target"].rank()))
        ic_basket = float(sub[feat].rank().corr(sub["demeaned_target"].rank()))
        rows.append({
            "feature": feat,
            "ic_btc": ic_btc,
            "ic_basket": ic_basket,
            "btc_minus_basket": ic_btc - ic_basket,
            "abs_ic_btc": abs(ic_btc),
            "abs_ic_basket": abs(ic_basket),
        })

    df = pd.DataFrame(rows).sort_values("abs_ic_btc", ascending=False)

    print("\n" + "=" * 100)
    print(f"FEATURE IC RANKING (sorted by |IC vs BTC target|, fold 0 train data)")
    print("=" * 100)
    print(f"  {'rank':>4}  {'feature':<35}  {'IC_btc':>9}  {'IC_basket':>10}  "
          f"{'Δ (btc-basket)':>15}  {'recommendation':>14}")
    for i, r in enumerate(df.itertuples(), 1):
        # Recommendation thresholds:
        # |IC| > 0.04: KEEP (strong)
        # |IC| 0.02-0.04: CONSIDER (mid)
        # |IC| < 0.02: DROP (weak)
        abs_btc = r.abs_ic_btc
        if abs_btc > 0.04: rec = "KEEP (strong)"
        elif abs_btc > 0.02: rec = "CONSIDER"
        else: rec = "DROP (weak)"
        print(f"  {i:>4}  {r.feature:<35}  {r.ic_btc:>+9.4f}  {r.ic_basket:>+10.4f}  "
              f"{r.btc_minus_basket:>+15.4f}  {rec:>14}")

    # Summary stats
    print(f"\n  Summary:")
    print(f"    Mean |IC| against BTC:    {df['abs_ic_btc'].mean():.4f}")
    print(f"    Mean |IC| against basket: {df['abs_ic_basket'].mean():.4f}")
    print(f"    Features with |IC_btc| > 0.04 (KEEP):     {(df['abs_ic_btc'] > 0.04).sum()}")
    print(f"    Features with |IC_btc| 0.02-0.04 (CONSIDER): {((df['abs_ic_btc'] > 0.02) & (df['abs_ic_btc'] <= 0.04)).sum()}")
    print(f"    Features with |IC_btc| < 0.02 (DROP):      {(df['abs_ic_btc'] <= 0.02).sum()}")
    print(f"    Cross-rank correlation (IC_btc vs IC_basket): "
          f"{df['ic_btc'].rank().corr(df['ic_basket'].rank()):.3f}")

    # Save
    df.to_csv(OUT_DIR / "feature_ic_ranking.csv", index=False)
    keep_features = df[df["abs_ic_btc"] > 0.02]["feature"].tolist()
    print(f"\n  KEEP+CONSIDER list ({len(keep_features)} features): {keep_features}")
    with open(OUT_DIR / "keep_features.txt", "w") as f:
        for feat in keep_features:
            f.write(f"{feat}\n")
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
