"""Step 18: Per-feature audit of R7's 35 features.

For each feature:
  1. Distribution: mean, std, skew, kurt, p1/p99 (post winsorize+z-score)
  2. Per-cycle IC vs realized α_β (Spearman)
  3. Ridge coefficient (mean across 10 folds × 5 seeds bootstrap)
  4. Std of coefficient across seeds (stability)
  5. Per-feature contribution to pred_z: |coef| × feat_std (since features
     are z-scored, std ≈ 1, so |coef| approximates contribution)
  6. Group by family: W17 base / R3 squared / new monotonic / new u-shape

Output: ranked table showing which features contribute, which are noise.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

TARGETS = REPO / "linear_model/data/targets.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT = REPO / "linear_model/results"

# R7 feature structure
BASE_16 = ["return_1d","atr_pct","dom_level_vs_bk","dom_change_288b_vs_bk",
           "bk_ema_slope_4h","corr_change_3d_vs_bk","obv_z_1d","vwap_slope_96",
           "bars_since_high_xs_rank","idio_vol_1d_vs_bk_xs_rank",
           "funding_rate","funding_rate_z_7d","corr_to_btc_1d",
           "idio_vol_to_btc_1h","beta_to_btc_change_5d","funding_rate_1d_change"]
U_SHAPE_R3 = ["beta_to_btc_change_5d", "dom_change_288b_vs_bk",
              "corr_to_btc_1d", "corr_change_3d_vs_bk",
              "dom_level_vs_bk", "return_1d"]
NEW_MONOTONIC = ["xs_alpha_iqr_12b", "xs_alpha_mean_48b",
                 "idio_ret_to_btc_48b", "idio_max_abs_12b",
                 "idio_vol_to_btc_1d"]
NEW_USHAPE = ["idio_kurt_1d", "idio_ret_to_btc_12b",
              "beta_to_btc", "idio_skew_1d"]

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5


def winsorize_zscore(s, train_s, p_lo=0.01, p_hi=0.99):
    s_train = train_s.dropna()
    lo, hi = s_train.quantile(p_lo), s_train.quantile(p_hi)
    s_w = s_train.clip(lower=lo, upper=hi)
    mu, sd = s_w.mean(), s_w.std()
    if sd < 1e-8: sd = 1.0
    return ((s.clip(lower=lo, upper=hi) - mu) / sd).astype("float32"), lo, hi, mu, sd


def main():
    print("=== Step 18: R7 per-feature audit ===\n", flush=True)
    t0 = time.time()

    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    feats_to_load = list(set(BASE_16 + NEW_MONOTONIC + NEW_USHAPE))
    base = pd.read_parquet(PANEL, columns=["symbol","open_time"] + feats_to_load)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    panel = tgt.merge(base, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    folds_all = _multi_oos_splits(panel)
    train_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())
    train_panel = panel[train_mask]

    # Build R7 feature matrix
    print("Building R7 feature matrix (35 features)...", flush=True)
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"],
                      "alpha_beta": panel["alpha_beta"],
                      "target_z": panel["target_z"],
                      "autocorr_pctile_7d": panel["autocorr_pctile_7d"]})
    feature_meta = []
    # 16 base
    for f in BASE_16:
        v, lo, hi, mu, sd = winsorize_zscore(panel[f], train_panel[f])
        X[f] = v
        feature_meta.append({"feature": f, "family": "W17_base", "lo":lo, "hi":hi,
                            "mu":mu, "sd":sd, "raw_skew": stats.skew(train_panel[f].dropna()),
                            "raw_kurt": stats.kurtosis(train_panel[f].dropna())})
    # 6 R3 squared
    for f in U_SHAPE_R3:
        X[f + "_sq"] = (X[f] ** 2).astype("float32")
        feature_meta.append({"feature": f + "_sq", "family": "R3_squared",
                            "lo":np.nan, "hi":np.nan, "mu":0, "sd":1,
                            "raw_skew": np.nan, "raw_kurt": np.nan})
    # 5 new monotonic
    for f in NEW_MONOTONIC:
        v, lo, hi, mu, sd = winsorize_zscore(panel[f], train_panel[f])
        X[f] = v
        feature_meta.append({"feature": f, "family": "new_monotonic", "lo":lo, "hi":hi,
                            "mu":mu, "sd":sd, "raw_skew": stats.skew(train_panel[f].dropna()),
                            "raw_kurt": stats.kurtosis(train_panel[f].dropna())})
    # 4 new u-shape + squared
    for f in NEW_USHAPE:
        v, lo, hi, mu, sd = winsorize_zscore(panel[f], train_panel[f])
        X[f] = v
        feature_meta.append({"feature": f, "family": "new_ushape_base",
                            "lo":lo, "hi":hi, "mu":mu, "sd":sd,
                            "raw_skew": stats.skew(train_panel[f].dropna()),
                            "raw_kurt": stats.kurtosis(train_panel[f].dropna())})
        X[f + "_sq"] = (X[f] ** 2).astype("float32")
        feature_meta.append({"feature": f + "_sq", "family": "new_ushape_sq",
                            "lo":np.nan, "hi":np.nan, "mu":0, "sd":1,
                            "raw_skew": np.nan, "raw_kurt": np.nan})
    feat_cols = [c for c in X.columns if c not in ("symbol","open_time","alpha_beta",
                                                     "target_z","autocorr_pctile_7d")]
    X[feat_cols] = X[feat_cols].fillna(0)
    print(f"  feature count: {len(feat_cols)}", flush=True)

    df_meta = pd.DataFrame(feature_meta).set_index("feature")

    # ===== Compute per-feature IC + shape on training data =====
    print("\nComputing per-feature stats (IC, shape)...", flush=True)
    train_X = X[train_mask].dropna(subset=["alpha_beta"]).reset_index(drop=True)
    target = train_X["alpha_beta"].clip(-0.1, 0.1).values

    stats_rows = []
    for f in feat_cols:
        v = train_X[f].values
        valid = ~np.isnan(v) & ~np.isnan(target)
        v, t = v[valid], target[valid]
        if len(v) < 1000:
            continue
        # Per-cycle IC (CS Spearman)
        df_tmp = pd.DataFrame({"f":v, "y":t, "open_time":train_X.loc[valid, "open_time"].values})
        cyc_ic = df_tmp.groupby("open_time").apply(
            lambda g: g["f"].rank().corr(g["y"].rank()) if len(g)>=5 else np.nan
        ).dropna()
        # Decile shape
        try:
            q = pd.qcut(pd.Series(v), 10, labels=False, duplicates="drop")
            dec = pd.Series(t).groupby(q).mean()
            d0, d9 = float(dec.iloc[0]) * 1e4, float(dec.iloc[-1]) * 1e4
        except Exception:
            d0 = d9 = np.nan
        stats_rows.append({"feature":f,
                            "f_mean": v.mean(), "f_std": v.std(),
                            "cs_ic_mean": float(cyc_ic.mean()),
                            "cs_ic_abs": abs(float(cyc_ic.mean())),
                            "decile_0_bps": d0, "decile_9_bps": d9,
                            "spread_bps": d9 - d0})
    df_stats = pd.DataFrame(stats_rows).set_index("feature")
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # ===== Extract Ridge coefficients across all folds × seeds =====
    print("\nTraining R7 across folds × seeds to extract coefs...", flush=True)
    coef_rows = []
    for fid in range(10):
        if fid >= len(folds_all): continue
        train, cal, test = _slice(X, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        if len(tr) < 1000: continue
        Xt = tr[feat_cols].to_numpy(np.float32)
        yt = tr["target_z"].to_numpy(np.float32)
        mt = ~np.isnan(yt)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, mt.sum(), size=mt.sum())
            m = RidgeCV(alphas=ALPHAS, scoring="r2", cv=None, fit_intercept=True)
            m.fit(Xt[mt][idx], yt[mt][idx])
            for fi, f in enumerate(feat_cols):
                coef_rows.append({"fold":fid, "seed":seed, "feature":f,
                                  "coef":float(m.coef_[fi]),
                                  "alpha":float(m.alpha_)})
        print(f"  fold {fid} done", flush=True)
    coef_df = pd.DataFrame(coef_rows)
    coef_summary = coef_df.groupby("feature")["coef"].agg(
        ["mean", "std", "min", "max"]).rename(
        columns={"mean":"coef_mean", "std":"coef_std",
                  "min":"coef_min", "max":"coef_max"})
    coef_summary["coef_abs"] = coef_summary["coef_mean"].abs()
    # Stability: |mean| / std (high = stable + non-zero)
    coef_summary["coef_stability"] = (coef_summary["coef_abs"] /
                                       coef_summary["coef_std"].replace(0, np.nan))

    # ===== Merge all stats =====
    full = coef_summary.join(df_stats, how="outer").join(df_meta, how="outer")
    full = full.sort_values("coef_abs", ascending=False)
    full.to_csv(OUT / "r7_feature_audit_full.csv")

    # ===== Display =====
    print(f"\n{'='*130}", flush=True)
    print(f"  R7 FEATURE AUDIT — sorted by |coef|", flush=True)
    print(f"{'='*130}", flush=True)
    print(f"  {'feature':<32} {'family':<18} {'coef':>+8} {'coef_std':>8} "
          f"{'stab':>5} {'|IC|':>8} {'d0_bps':>9} {'d9_bps':>9} {'kurt':>7}",
          flush=True)
    for f, r in full.iterrows():
        coef_mean = r.get("coef_mean", np.nan)
        if pd.isna(coef_mean): continue
        stab = r.get("coef_stability", np.nan)
        ic = r.get("cs_ic_abs", np.nan)
        d0 = r.get("decile_0_bps", np.nan)
        d9 = r.get("decile_9_bps", np.nan)
        kurt = r.get("raw_kurt", np.nan)
        print(f"  {str(f):<32} {str(r.get('family','')):<18} "
              f"{coef_mean:>+8.4f} {r.get('coef_std',np.nan):>8.4f} "
              f"{(stab if not pd.isna(stab) else 0):>5.1f} "
              f"{(ic if not pd.isna(ic) else 0):>8.4f} "
              f"{(d0 if not pd.isna(d0) else 0):>+9.1f} "
              f"{(d9 if not pd.isna(d9) else 0):>+9.1f} "
              f"{(kurt if not pd.isna(kurt) else 0):>+7.1f}", flush=True)

    # Family-level summary
    print(f"\n{'='*120}", flush=True)
    print(f"  FAMILY-LEVEL COEFFICIENT MASS", flush=True)
    print(f"{'='*120}", flush=True)
    fam_summary = full.groupby("family").agg(
        n_features=("coef_abs", "count"),
        sum_abs_coef=("coef_abs", "sum"),
        max_abs_coef=("coef_abs", "max"),
        mean_stab=("coef_stability", "mean"),
        mean_ic_abs=("cs_ic_abs", "mean"),
    ).sort_values("sum_abs_coef", ascending=False)
    print(fam_summary.round(4).to_string(), flush=True)

    # Top contributors
    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 10 CONTRIBUTORS (highest |coef| × stability)", flush=True)
    print(f"{'='*120}", flush=True)
    full["impact"] = full["coef_abs"] * full["coef_stability"].fillna(0)
    top = full.sort_values("impact", ascending=False).head(10)
    for f, r in top.iterrows():
        print(f"  {str(f):<32} {str(r.get('family','')):<18} "
              f"coef={r['coef_mean']:+.4f}±{r['coef_std']:.4f}  "
              f"stab={r['coef_stability']:.1f}  |IC|={r['cs_ic_abs']:.4f}", flush=True)

    # Bottom contributors (potential noise)
    print(f"\n{'='*120}", flush=True)
    print(f"  BOTTOM 10 (likely noise — small coef or unstable)", flush=True)
    print(f"{'='*120}", flush=True)
    bot = full.sort_values("impact", ascending=True).head(10)
    for f, r in bot.iterrows():
        print(f"  {str(f):<32} {str(r.get('family','')):<18} "
              f"coef={r['coef_mean']:+.4f}±{r['coef_std']:.4f}  "
              f"stab={r['coef_stability']:.1f}  |IC|={r['cs_ic_abs']:.4f}", flush=True)

    print(f"\nSaved: {OUT / 'r7_feature_audit_full.csv'}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
