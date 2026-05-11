"""Check redundancy between v6_clean keep features and new BTC features.

Strategy:
  1. Load combined feature set (37 features) on training data
  2. Compute pairwise Spearman rank correlation matrix
  3. For each highly-correlated pair (|ρ| > 0.85), drop the one with lower
     |IC| against btc_target
  4. Output deduplicated final feature set

Threshold: |ρ| = 0.85 — high but not perfect collinearity. Keeps features
that share signal direction but differ in transformation/window.
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
OUT_DIR = REPO / "outputs/vBTC_dedup"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC
CORR_THRESHOLD = 0.85


def main():
    print(f"Loading panel...")
    panel = pd.read_parquet(PANEL_PATH)
    folds = _multi_oos_splits(panel)
    fold0 = folds[0]
    train, _, _ = _slice(panel, fold0)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    tr = tr.dropna(subset=["btc_target"])
    print(f"  Training subset: {len(tr):,} rows")

    all_feats = [c for c in V6_KEEP + NEW_BTC if c in panel.columns]
    print(f"  Combined feature set: {len(all_feats)} features ({len(V6_KEEP)} v6 + {len(NEW_BTC)} new)")

    # Compute IC against btc_target for each feature
    print("\n  Computing IC for each feature...")
    ic_lookup = {}
    for f in all_feats:
        sub = tr[[f, "btc_target"]].dropna()
        if len(sub) < 1000:
            ic_lookup[f] = 0.0; continue
        ic_lookup[f] = abs(float(sub[f].rank().corr(sub["btc_target"].rank())))

    # Correlation matrix on training data
    print("  Computing pairwise rank correlation...")
    feat_data = tr[all_feats].dropna()
    print(f"    Non-NaN rows for corr: {len(feat_data):,}")
    # Use Spearman rank correlation (more robust than Pearson)
    rank_data = feat_data.rank()
    corr = rank_data.corr()

    # Find high-corr pairs
    print(f"\n  Pairs with |ρ| > {CORR_THRESHOLD}:")
    pairs = []
    for i, fi in enumerate(all_feats):
        for fj in all_feats[i+1:]:
            if fi not in corr.index or fj not in corr.columns: continue
            r = corr.loc[fi, fj]
            if not np.isfinite(r): continue
            if abs(r) >= CORR_THRESHOLD:
                pairs.append((fi, fj, r, ic_lookup[fi], ic_lookup[fj]))

    if pairs:
        print(f"  {'feat A':<32} {'feat B':<32} {'ρ':>7} {'|IC_A|':>8} {'|IC_B|':>8} {'drop':>10}")
        for fi, fj, r, ic_a, ic_b in pairs:
            drop = fj if ic_a >= ic_b else fi
            print(f"  {fi:<32} {fj:<32} {r:>+7.3f} {ic_a:>+8.4f} {ic_b:>+8.4f} {drop:>10}")
    else:
        print("  (none)")

    # Greedy dedup: iterate pairs, drop the lower-IC member of each pair
    drops = set()
    for fi, fj, r, ic_a, ic_b in pairs:
        if fi in drops or fj in drops: continue
        if ic_a >= ic_b:
            drops.add(fj)
        else:
            drops.add(fi)
    kept = [f for f in all_feats if f not in drops]

    print(f"\n  Drops: {sorted(drops)}")
    print(f"  Kept ({len(kept)}/{len(all_feats)}):")
    for f in kept:
        flag = "v6" if f in V6_KEEP else "new"
        print(f"    {f:<32} ({flag}, |IC|={ic_lookup[f]:.4f})")

    # Save
    pd.Series(kept).to_csv(OUT_DIR / "deduped_feature_set.csv", index=False, header=["feature"])
    pd.DataFrame(pairs, columns=["feat_a", "feat_b", "rho", "ic_a", "ic_b"]).to_csv(
        OUT_DIR / "high_corr_pairs.csv", index=False
    )
    # Also save full correlation matrix for inspection
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
