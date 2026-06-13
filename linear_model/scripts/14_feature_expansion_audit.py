"""Step 14: Audit candidate features for linear-model expansion.

For each new feature (not in WINNER_17):
  1. NaN rate (panel-wide + per-symbol)
  2. Distribution: mean, std, skew, kurt, p1/p99
  3. Linear (Pearson) and rank (Spearman) corr with target_bps
  4. Decile shape (monotonic / U-shape / inverted-U / noisy)
  5. Correlation with existing R3 features (collinearity check)

Then propose: which new features to add, with what transforms.
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

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
TARGETS = REPO / "linear_model/data/targets.parquet"
OUT = REPO / "linear_model/results"

# Current R3 set
R3_FEATS = ["return_1d","atr_pct","dom_level_vs_bk","dom_change_288b_vs_bk",
            "bk_ema_slope_4h","corr_change_3d_vs_bk","obv_z_1d","vwap_slope_96",
            "bars_since_high_xs_rank","idio_vol_1d_vs_bk_xs_rank",
            "funding_rate","funding_rate_z_7d","corr_to_btc_1d",
            "idio_vol_to_btc_1h","beta_to_btc_change_5d","funding_rate_1d_change"]

# Candidates NOT in R3 — try to add
CANDIDATES = {
    "BTC_residual_target_horizon": [
        "idio_ret_to_btc_48b",   # 4h backward β-residual — MATCHES TARGET
        "idio_ret_to_btc_12b",   # 1h backward β-residual
        "idio_vol_to_btc_1d",    # 1d idio vol (cleaner than W17's 1h)
    ],
    "BTC_regime": [
        "btc_ret_48b",           # 4h BTC momentum
        "btc_ret_12b",           # 1h BTC momentum
        "btc_ret_288b",          # 1d BTC momentum
        "btc_ema_slope_4h",      # BTC EMA slope
    ],
    "BTC_residual_distributional": [
        "idio_skew_1d",
        "idio_kurt_1d",
        "idio_max_abs_12b",      # 1h max abs idio (jump indicator)
    ],
    "xs_alpha_context": [
        "xs_alpha_mean_48b",     # 4h cross-sectional mean alpha
        "xs_alpha_dispersion_48b",  # 4h dispersion (regime indicator)
        "xs_alpha_iqr_12b",      # 1h IQR
    ],
    "BTC_relationship_extras": [
        "beta_to_btc",           # β LEVEL (W17 only has 5d change)
    ],
    "name_factor": [
        "name_factor_loading_1d",
        "name_idio_share_1d",
    ],
}
ALL_CAND = [f for fs in CANDIDATES.values() for f in fs]


def shape_diag(decile_targets):
    d = decile_targets.values
    if len(d) < 10: return "insufficient"
    rho = stats.spearmanr(range(10), d).statistic
    mid = np.mean(d[3:7])
    tails = np.mean(d[[0,1,2,7,8,9]])
    if rho > 0.7:   return "monotonic_up"
    elif rho < -0.7: return "monotonic_down"
    elif mid > tails + abs(np.std(d)) * 0.5: return "inverted_u (BAD)"
    elif mid < tails - abs(np.std(d)) * 0.5: return "u_shape (sq fix)"
    else: return "noisy"


def main():
    print("=== Step 14: Feature expansion audit ===\n", flush=True)

    tgt = pd.read_parquet(TARGETS, columns=["symbol","open_time","target_bps_raw"])
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    panel = pd.read_parquet(PANEL,
        columns=["symbol","open_time","autocorr_pctile_7d"] + R3_FEATS + ALL_CAND)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    df = tgt.merge(panel, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(df):,} rows × {df.shape[1]} cols", flush=True)

    folds = _multi_oos_splits(df)
    train0, _, _ = _slice(df, folds[0])
    train0 = train0[train0["autocorr_pctile_7d"] >= 0.5].dropna(subset=["target_bps_raw"])
    target = train0["target_bps_raw"].clip(-1000, 1000)
    print(f"Fold-0 training: {len(train0):,} rows\n", flush=True)

    print("="*120, flush=True)
    print("CANDIDATE FEATURE AUDIT", flush=True)
    print("="*120, flush=True)

    audit_rows = []
    for fam, feats in CANDIDATES.items():
        print(f"\n  --- {fam} ---", flush=True)
        for f in feats:
            if f not in df.columns:
                print(f"    MISSING: {f}")
                continue
            s = train0[f]
            n_total = len(s)
            n_valid = s.notna().sum()
            nan_pct = (1 - n_valid/n_total) * 100
            # per-symbol coverage
            psyn = train0.groupby("symbol")[f].apply(lambda x: x.notna().mean() > 0.5).sum()

            valid = s.notna() & ~target.isna()
            ss = s[valid]; yy = target[valid]
            if len(ss) < 1000:
                print(f"    {f:<30} insufficient (n={len(ss)})")
                continue
            skew = stats.skew(ss); kurt = stats.kurtosis(ss)
            p1, p50, p99 = ss.quantile([0.01, 0.5, 0.99]).values
            pr = stats.pearsonr(ss, yy).statistic
            sr = stats.spearmanr(ss, yy).statistic
            try:
                q = pd.qcut(ss, 10, labels=False, duplicates="drop")
                dec = yy.groupby(q).mean()
                shape = shape_diag(dec)
                d0, d9 = float(dec.iloc[0]), float(dec.iloc[-1])
            except Exception:
                shape = "binning_fail"; d0=d9=np.nan

            audit_rows.append({"family": fam, "feature": f, "n_valid": int(n_valid),
                              "nan_pct": nan_pct, "psyn_cov_50pct": int(psyn),
                              "skew": skew, "kurt": kurt,
                              "p1": p1, "p99": p99,
                              "pearson_r": pr, "spearman_r": sr,
                              "d0_bps": d0, "d9_bps": d9, "shape": shape})
            print(f"    {f:<30} NaN={nan_pct:5.1f}%  syms>50%={psyn}/50  "
                  f"|p|={abs(pr):.4f} |s|={abs(sr):.4f} "
                  f"shape={shape}", flush=True)

    df_audit = pd.DataFrame(audit_rows).sort_values("spearman_r", key=abs, ascending=False)
    df_audit.to_csv(OUT / "step14_candidate_feature_audit.csv", index=False)

    print("\n" + "="*120, flush=True)
    print("RANKED BY |spearman_r| (top candidates first)", flush=True)
    print("="*120, flush=True)
    print(f"  {'feature':<30} {'|pearson|':>9} {'|spearman|':>10} "
          f"{'skew':>7} {'kurt':>8} {'nan%':>6} {'syms':>5} {'shape':<22}", flush=True)
    for _, r in df_audit.iterrows():
        print(f"  {r['feature']:<30} {abs(r['pearson_r']):>9.4f} "
              f"{abs(r['spearman_r']):>10.4f} {r['skew']:>+7.2f} "
              f"{r['kurt']:>+8.1f} {r['nan_pct']:>5.1f}% "
              f"{r['psyn_cov_50pct']:>5}/50 {r['shape']:<22}", flush=True)

    # Recommended additions
    print("\n" + "="*120, flush=True)
    print("RECOMMENDED ADDITIONS — filter rules:", flush=True)
    print("  - |spearman| >= 0.005 (some signal)")
    print("  - NaN < 30% (data quality)")
    print("  - 40+ syms covered (universe completeness)")
    print("="*120, flush=True)
    keep = df_audit[(df_audit["spearman_r"].abs() >= 0.005)
                    & (df_audit["nan_pct"] < 30)
                    & (df_audit["psyn_cov_50pct"] >= 40)]
    print(f"\n  {len(keep)} features pass filters:", flush=True)
    for _, r in keep.iterrows():
        rec = []
        if "u_shape" in r["shape"]: rec.append("squared")
        if "inverted_u" in r["shape"]: rec.append("BAD-deadly")
        if abs(r["kurt"]) > 5: rec.append("winsorize-tight")
        if abs(r["skew"]) > 1.5: rec.append("sign-split")
        ratio = abs(r["spearman_r"]) / max(abs(r["pearson_r"]), 1e-6)
        if ratio > 2: rec.append(f"rank-xform(r/p={ratio:.1f})")
        print(f"    {r['feature']:<30} → {'+ '.join(rec) if rec else 'as-is'}",
              flush=True)


if __name__ == "__main__":
    main()
