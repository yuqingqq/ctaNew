"""Phase H: feature redundancy analysis within WINNER_21.

For each feature, compute:
  - univariate Spearman IC vs alpha_A
  - R² explained by all OTHER features in WINNER_21 (linear)
  - Pairwise Pearson correlation matrix (heatmap-ready)
  - Hierarchical clustering on 1-|r| distance

Redundant features = high R² (>0.5) explained by other features.
Bad-redundant features = high R² AND low marginal IC.

Output:
  outputs/vBTC_feature_redundancy/r2_and_ic.csv
  outputs/vBTC_feature_redundancy/correlation_matrix.csv
  outputs/vBTC_feature_redundancy/cluster_assignment.csv
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT = REPO / "outputs/vBTC_feature_redundancy"
OUT.mkdir(parents=True, exist_ok=True)

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING


def main():
    print("=== Phase H: Feature redundancy analysis on WINNER_21 ===\n", flush=True)

    print("  Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL)
    feats = [f for f in WINNER_21 if f in panel.columns]
    skipped = set(WINNER_21) - set(feats)
    if skipped:
        print(f"  Skipped (not in panel): {sorted(skipped)}", flush=True)
    print(f"  Using {len(feats)} features", flush=True)

    sub = panel[feats + ["alpha_A"]].dropna()
    print(f"  N obs (after dropna): {len(sub):,}\n", flush=True)

    # 1. Pairwise Pearson correlation
    print("--- (1) Pairwise Pearson correlation ---", flush=True)
    corr = sub[feats].corr()
    corr.to_csv(OUT / "correlation_matrix.csv")

    # Top redundant pairs (|r| > 0.5)
    rows = []
    for i, fi in enumerate(feats):
        for j, fj in enumerate(feats):
            if j <= i: continue
            r = corr.loc[fi, fj]
            if abs(r) >= 0.5:
                rows.append({"feat_a": fi, "feat_b": fj, "pearson_r": r})
    pairs_df = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    print(f"  Pairs with |r|>=0.5: {len(pairs_df)}\n", flush=True)
    if len(pairs_df) > 0:
        for _, row in pairs_df.iterrows():
            print(f"    {row['feat_a']:<32} vs {row['feat_b']:<32}  r={row['pearson_r']:+.3f}",
                  flush=True)
    pairs_df.to_csv(OUT / "high_correlation_pairs.csv", index=False)

    # 2. Per-feature R² explained by OTHERS
    print("\n--- (2) Each feature's R² explained by the other 20 ---", flush=True)
    rows = []
    for f in feats:
        others = [o for o in feats if o != f]
        X = sub[others].values
        y = sub[f].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        # Univariate Spearman IC
        ic = sub[f].rank().corr(sub["alpha_A"].rank())
        # Residual IC: orthogonalize f to others, then IC
        resid = y - reg.predict(X)
        resid_ic = pd.Series(resid).rank().corr(sub["alpha_A"].rank())
        rows.append({
            "feature": f, "univariate_IC": ic, "R2_by_others": r2,
            "residual_IC": resid_ic, "ic_loss_from_redundancy": abs(ic) - abs(resid_ic),
        })
    df = pd.DataFrame(rows).sort_values("R2_by_others", ascending=False)
    df.to_csv(OUT / "r2_and_ic.csv", index=False)

    print(f"  {'feature':<32}  {'univIC':>8}  {'R²_by_others':>14}  "
          f"{'residIC':>8}  {'IC_lost_to_redund':>18}", flush=True)
    for _, row in df.iterrows():
        tag = ""
        if row["R2_by_others"] >= 0.7:
            tag = "  ← HIGH"
        elif row["R2_by_others"] >= 0.5:
            tag = "  ← MOD"
        print(f"  {row['feature']:<32}  {row['univariate_IC']:>+8.4f}  "
              f"{row['R2_by_others']:>+14.3f}  {row['residual_IC']:>+8.4f}  "
              f"{row['ic_loss_from_redundancy']:>+18.4f}{tag}", flush=True)

    # 3. Hierarchical clustering on 1-|r| distance
    print("\n--- (3) Hierarchical clustering on 1-|r| distance ---", flush=True)
    abs_corr = corr.abs().values.copy()
    np.fill_diagonal(abs_corr, 1.0)
    dist = 1 - abs_corr
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    for k in [10, 12, 15, 17]:
        labels = fcluster(Z, k, criterion="maxclust")
        print(f"\n  K={k}:", flush=True)
        groups = {}
        for f, lab in zip(feats, labels):
            groups.setdefault(int(lab), []).append(f)
        # Show clusters with size>=2 sorted by size descending
        clusters_with_dup = [(c, m) for c, m in groups.items() if len(m) >= 2]
        clusters_with_dup.sort(key=lambda x: -len(x[1]))
        for c, members in clusters_with_dup:
            # Find the highest |IC| feature in this cluster
            ics_in_c = df.set_index("feature").loc[members]
            best = ics_in_c["univariate_IC"].abs().idxmax()
            best_ic = ics_in_c.loc[best, "univariate_IC"]
            print(f"    cluster {c:>2}  (n={len(members)})  best={best}  bestIC={best_ic:+.4f}",
                  flush=True)
            for m in sorted(members):
                marker = " *" if m == best else "  "
                ic = ics_in_c.loc[m, "univariate_IC"]
                r2 = ics_in_c.loc[m, "R2_by_others"]
                print(f"      {marker} {m:<30} IC={ic:+.4f}  R²_oth={r2:.2f}", flush=True)

    # 4. Recommendations
    print(f"\n--- (4) Drop candidates (R²_by_others >= 0.5 AND |residual_IC| < 0.005) ---",
          flush=True)
    candidates = df[(df["R2_by_others"] >= 0.5) & (df["residual_IC"].abs() < 0.005)]
    if len(candidates) == 0:
        print(f"  None — no feature is BOTH highly redundant AND noise-only", flush=True)
    else:
        for _, row in candidates.iterrows():
            print(f"  {row['feature']:<32}  R²={row['R2_by_others']:.3f}  "
                  f"residIC={row['residual_IC']:+.4f}", flush=True)

    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
