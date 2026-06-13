"""Step 11: Per-feature audit — check if features are suitable for linear model.

For each of the 16 numeric features:
  1. Distribution: mean, std, skew, kurt, percentiles, range
  2. Linear (Pearson) correlation with target_bps
  3. Spearman rank correlation with target_bps
  4. Decile mean target — is the relationship monotonic?
  5. Per-symbol vs cross-sectional behavior
  6. Recommended transform: identity, log, sign-split, polynomial, rank

Output suggests which features need:
  - Winsorize (heavy tails)
  - Log transform (multiplicative)
  - Sign split (different effect for + and -)
  - Polynomial (non-monotonic — quadratic adds Ridge representational power)
  - Rank transform (monotonic but non-linear)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

TARGETS  = REPO / "linear_model/data/targets.parquet"
PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT = REPO / "linear_model/results"

NUMERIC_FEATS = ["return_1d","atr_pct","dom_level_vs_bk","dom_change_288b_vs_bk",
                 "bk_ema_slope_4h","corr_change_3d_vs_bk","obv_z_1d","vwap_slope_96",
                 "bars_since_high_xs_rank","idio_vol_1d_vs_bk_xs_rank",
                 "funding_rate","funding_rate_z_7d","corr_to_btc_1d",
                 "idio_vol_to_btc_1h","beta_to_btc_change_5d","funding_rate_1d_change"]


def shape_diagnose(decile_targets):
    """Classify monotonicity:
      'monotonic_up'     : d0→d9 increasing (good for linear positive coef)
      'monotonic_down'   : d0→d9 decreasing (good for linear negative coef)
      'u_shape'          : convex (high at both tails, low middle)
      'inverted_u'       : concave (high middle, low tails) — DEADLY for linear
      'noisy'            : no clear pattern
    """
    d = decile_targets.values
    if len(d) < 10: return "insufficient"
    # Spearman of (rank, value)
    rho = stats.spearmanr(range(10), d).statistic
    # Second derivative proxy: middle vs tails
    mid = np.mean(d[3:7])
    tails = np.mean(d[[0,1,2,7,8,9]])
    if rho > 0.7:   return "monotonic_up"
    elif rho < -0.7: return "monotonic_down"
    elif mid > tails + abs(np.std(d)) * 0.5: return "inverted_u (deadly for linear)"
    elif mid < tails - abs(np.std(d)) * 0.5: return "u_shape"
    else: return "noisy"


def main():
    print("=== Step 11: Per-feature audit ===\n", flush=True)

    # Load targets + raw features
    tgt = pd.read_parquet(TARGETS, columns=["symbol","open_time","alpha_beta","target_bps_raw"])
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    base = pd.read_parquet(PANEL_BASE, columns=["symbol","open_time"] + NUMERIC_FEATS +
                            ["autocorr_pctile_7d"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    df = tgt.merge(base, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(df):,} rows × {df.shape[1]} cols\n", flush=True)

    # Take fold-0 training slice for unbiased feature inspection
    folds = _multi_oos_splits(df)
    train0, _, _ = _slice(df, folds[0])
    train0 = train0[train0["autocorr_pctile_7d"] >= 0.5].dropna(subset=["target_bps_raw"])
    # Drop very extreme target outliers for cleaner correlation signal
    target = train0["target_bps_raw"].clip(-1000, 1000)
    print(f"Fold-0 training (post-filter): {len(train0):,} rows", flush=True)
    print(f"  target_bps mean={target.mean():+.2f}  std={target.std():.2f}  "
          f"range [{target.min():+.0f}, {target.max():+.0f}]\n", flush=True)

    print(f"{'='*120}", flush=True)
    print(f"  PER-FEATURE AUDIT", flush=True)
    print(f"{'='*120}", flush=True)

    feature_diag = []
    for f in NUMERIC_FEATS:
        s = train0[f].dropna()
        n_drop = train0[f].isna().sum()
        # Match index for target alignment
        idx = s.index
        y = target.loc[idx]
        valid = ~y.isna()
        s, y = s[valid], y[valid]
        if len(s) < 1000:
            print(f"\n  {f}: insufficient data ({len(s)} rows)")
            continue

        # Distribution
        skew = stats.skew(s)
        kurt = stats.kurtosis(s)
        p1, p50, p99 = s.quantile([0.01, 0.5, 0.99]).values
        # Linear vs rank correlation
        pearson_r = stats.pearsonr(s, y).statistic
        spearman_r = stats.spearmanr(s, y).statistic
        # Decile analysis
        try:
            quantiles = pd.qcut(s, 10, labels=False, duplicates="drop")
            decile_means = y.groupby(quantiles).mean()
            shape = shape_diagnose(decile_means)
            d0 = float(decile_means.iloc[0]); d9 = float(decile_means.iloc[-1])
            mid = float(decile_means.iloc[3:7].mean())
        except Exception:
            decile_means = pd.Series(dtype=float)
            shape = "binning_failed"
            d0 = d9 = mid = np.nan

        feature_diag.append({
            "feature": f,
            "n_valid": len(s),
            "n_dropped_nan": int(n_drop),
            "mean": s.mean(),
            "std": s.std(),
            "skew": skew,
            "kurt": kurt,
            "p1": p1,
            "p50": p50,
            "p99": p99,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "d0_target_bps": d0,
            "d9_target_bps": d9,
            "mid_target_bps": mid,
            "shape": shape,
        })

        print(f"\n  ===== {f} =====", flush=True)
        print(f"    dist: mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"skew={skew:+.2f}  kurt={kurt:+.2f}", flush=True)
        print(f"    pctiles: p1={p1:+.4f}  p50={p50:+.4f}  p99={p99:+.4f}",
              flush=True)
        print(f"    pearson_r={pearson_r:+.4f}  spearman_r={spearman_r:+.4f}",
              flush=True)
        if not decile_means.empty:
            print(f"    decile means (bps): {[f'{v:+.1f}' for v in decile_means.values]}",
                  flush=True)
            print(f"    d0={d0:+.1f}  mid(3-6)={mid:+.1f}  d9={d9:+.1f}  "
                  f"→ shape: {shape}", flush=True)

    df_diag = pd.DataFrame(feature_diag).sort_values("spearman_r", key=abs,
                                                      ascending=False)
    df_diag.to_csv(OUT / "feature_audit.csv", index=False)
    print(f"\n  Saved per-feature audit: {OUT / 'feature_audit.csv'}", flush=True)

    # ===== SUMMARY =====
    print(f"\n{'='*120}", flush=True)
    print(f"  SUMMARY — features sorted by |spearman_r|", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"  {'feature':<32} {'|pearson|':>10} {'|spearman|':>11} "
          f"{'skew':>7} {'kurt':>7} {'shape':<35}", flush=True)
    for _, r in df_diag.iterrows():
        print(f"  {r['feature']:<32} {abs(r['pearson_r']):>10.4f} "
              f"{abs(r['spearman_r']):>11.4f} {r['skew']:>+7.2f} "
              f"{r['kurt']:>+7.2f} {r['shape']:<35}", flush=True)

    # Transforms recommendation
    print(f"\n{'='*120}", flush=True)
    print(f"  RECOMMENDED TRANSFORMS PER FEATURE", flush=True)
    print(f"{'='*120}", flush=True)
    rec_rows = []
    for _, r in df_diag.iterrows():
        f = r["feature"]
        recs = []
        # Heavy tails (high abs kurtosis)
        if abs(r["kurt"]) > 5:
            recs.append("rank or winsorize (kurt={:+.1f})".format(r["kurt"]))
        # Asymmetric
        if abs(r["skew"]) > 1.5:
            recs.append("sign-split (skew={:+.1f})".format(r["skew"]))
        # Non-monotonic shape
        if "inverted_u" in r["shape"] or "u_shape" in r["shape"]:
            recs.append("polynomial(2) [non-monotonic]")
        # Weak linear vs strong rank: signal is monotonic but non-linear
        ratio = abs(r["spearman_r"]) / max(abs(r["pearson_r"]), 1e-6)
        if ratio > 1.5 and abs(r["spearman_r"]) > 0.005:
            recs.append("rank-transform (rank/pearson = {:.1f}×)".format(ratio))
        rec_rows.append({"feature": f, "shape": r["shape"],
                         "recommendations": "; ".join(recs) if recs else "keep as-is"})
        print(f"  {f:<32} {r['shape']:<35} → {'; '.join(recs) if recs else 'keep as-is'}",
              flush=True)
    pd.DataFrame(rec_rows).to_csv(OUT / "feature_transforms_recommended.csv", index=False)

    # ===== Overall takeaways =====
    print(f"\n{'='*120}", flush=True)
    print(f"  OVERALL TAKEAWAYS", flush=True)
    print(f"{'='*120}", flush=True)
    n_inv = sum(1 for r in feature_diag if "inverted_u" in r.get("shape",""))
    n_u   = sum(1 for r in feature_diag if "u_shape" in r.get("shape","") and "inverted" not in r.get("shape",""))
    n_mono = sum(1 for r in feature_diag if "monotonic" in r.get("shape",""))
    n_noisy = sum(1 for r in feature_diag if r.get("shape","") == "noisy")
    print(f"  features with inverted-U shape (deadly for linear):  {n_inv}", flush=True)
    print(f"  features with U-shape (also bad for linear):         {n_u}", flush=True)
    print(f"  features with monotonic shape (good for linear):     {n_mono}", flush=True)
    print(f"  features with noisy shape:                           {n_noisy}", flush=True)
    print(f"\n  Linear viability: {n_mono}/{len(feature_diag)} features are clean-monotonic",
          flush=True)


if __name__ == "__main__":
    main()
