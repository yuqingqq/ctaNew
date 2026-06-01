"""X71 — Feature redundancy + orthogonality analysis of V5_mv3 (29 features).

PART A — Redundancy:
  1. Per-feature univariate IC (corr with target_z)
  2. Per-feature R² regressed on the other 28 (linear redundancy)
  3. Pairwise |corr| > 0.7 flags
  4. Combine: redundant = high R² (>0.5) AND low |IC| (<0.01)
  5. Drop-flagged retraining (test if removing redundant features hurts)

PART B — Orthogonality (what new info to add):
  - Build a few cheap candidate features from existing klines/funding
  - Measure their univariate IC + R² vs existing 29 (orthogonality)
  - Candidates: ret_7d, ret_14d, funding_rate_change_3d, vol_of_vol,
    rsi_14_xs_rank, volume_ratio_xs_rank, dist_ema_50

Runs on canonical HL-50 panel. Light CPU (mostly pandas + small Ridge).
"""
from __future__ import annotations
import sys, importlib.util, warnings
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    print("=== X71 feature analysis of V5_mv3 (29 features) ===\n", flush=True)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70_v5_full.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    canonical = sorted(set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
        columns=["symbol"])["symbol"].unique()) - {"BTCUSDT"})
    panel = panel[panel["symbol"].isin(canonical)]
    panel = x6.build_target_z(panel)
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                            .rank(pct=True).astype("float32"))

    cx_7 = ["bn_perp_okx_perp_z","bn_perp_okx_spot_z","okx_perp_spot_z",
            "bn_perp_cb_spot_z","okx_cb_spot_z","bn_spot_okx_spot_z","bn_spot_cb_spot_z"]
    aggT = ["signed_volume_4h","tfi_4h","aggr_ratio_4h","buy_count_4h","avg_trade_size_4h"]
    feats = [f for f in dict.fromkeys(x6.BASE + x6.COHORT_EXTRAS + aggT + cx_7) if f in panel.columns]
    print(f"V5_mv3 features ({len(feats)})\n")

    # Build a clean matrix (drop rows with any NaN in features+target)
    target = "alpha_A" if "alpha_A" in panel.columns else "target_z"
    sub = panel[feats + [target]].replace([np.inf,-np.inf], np.nan).dropna()
    print(f"Clean rows for analysis: {len(sub):,}\n")
    X = sub[feats].values
    y = sub[target].values
    # Standardize
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

    # === PART A1: univariate IC ===
    print("=" * 70)
    print("A1. Per-feature univariate IC (|corr| with target) + A2. R² on others")
    print("=" * 70)
    print(f"  {'feature':<26} {'uni_IC':>8} {'R2_others':>10} {'verdict':>12}")
    ics, r2s = {}, {}
    for j, f in enumerate(feats):
        ic = np.corrcoef(Xs[:, j], y)[0, 1]
        ics[f] = ic
        # R² of feature j on others
        others = np.delete(Xs, j, axis=1)
        lr = LinearRegression().fit(others, Xs[:, j])
        r2 = lr.score(others, Xs[:, j])
        r2s[f] = r2
        redundant = (r2 > 0.5) and (abs(ic) < 0.010)
        verdict = "REDUNDANT" if redundant else ("low-IC" if abs(ic) < 0.005 else "keep")
        print(f"  {f:<26} {ic:>+8.4f} {r2:>10.3f} {verdict:>12}", flush=True)

    # === PART A3: pairwise high correlations ===
    print("\n" + "=" * 70)
    print("A3. Pairwise |corr| > 0.7 (collinear pairs)")
    print("=" * 70)
    corr = np.corrcoef(Xs.T)
    found = False
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            if abs(corr[i, j]) > 0.7:
                print(f"  {feats[i]:<26} ~ {feats[j]:<26} corr={corr[i,j]:+.3f}")
                found = True
    if not found: print("  (none — no pair exceeds 0.7)")

    # === PART A4: candidate prune list ===
    prune = [f for f in feats if r2s[f] > 0.5 and abs(ics[f]) < 0.010]
    print(f"\nPrune candidates (R²>0.5 AND |IC|<0.010): {prune}")

    # === PART A5: drop-flagged retraining ===
    if prune:
        print("\n" + "=" * 70)
        print("A5. Retraining test: V5_mv3 minus prune candidates")
        print("=" * 70)
        folds = x6.get_folds(panel)
        x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
        # baseline
        apd0 = x6.train_per_sym_ridge(panel, folds, feats, label="x71_base")
        apd0.to_parquet(CACHE/"x71_base_preds.parquet", index=False)
        m0 = x6.run_sleeve_on_preds(CACHE/"x71_base_preds.parquet", "x71_base")
        print(f"  baseline (29 feats): Sharpe={m0.get('sharpe',0):+.2f}")
        pruned_feats = [f for f in feats if f not in prune]
        apd1 = x6.train_per_sym_ridge(panel, folds, pruned_feats, label="x71_pruned")
        apd1.to_parquet(CACHE/"x71_pruned_preds.parquet", index=False)
        m1 = x6.run_sleeve_on_preds(CACHE/"x71_pruned_preds.parquet", "x71_pruned")
        print(f"  pruned ({len(pruned_feats)} feats): Sharpe={m1.get('sharpe',0):+.2f} "
              f"(Δ {(m1.get('sharpe',0) or 0)-(m0.get('sharpe',0) or 0):+.2f})")

    # === PART B: orthogonality of candidate NEW features ===
    print("\n" + "=" * 70)
    print("B. Orthogonality of candidate NEW features (IC + R² vs existing 29)")
    print("=" * 70)
    # Build candidates from existing panel columns where possible
    cand_cols = {}
    # funding momentum
    if "funding_rate_z_7d" in panel.columns and "funding_rate" in panel.columns:
        cand_cols["funding_x_corr"] = panel["funding_rate_z_7d"] * panel.get("corr_to_btc_1d", 0)
    # vol-of-vol proxy
    if "rvol_7d" in panel.columns:
        cand_cols["rvol_7d_chg"] = panel.groupby("symbol")["rvol_7d"].diff()
    # aggT interaction
    if "tfi_4h" in panel.columns and "signed_volume_4h" in panel.columns:
        cand_cols["tfi_x_svol"] = panel["tfi_4h"] * np.sign(panel["signed_volume_4h"])
    # crossX dispersion (spread of the 7 crossX features)
    cx_present = [c for c in cx_7 if c in panel.columns]
    if len(cx_present) >= 3:
        cand_cols["crossX_dispersion"] = panel[cx_present].std(axis=1)
        cand_cols["crossX_mean"] = panel[cx_present].mean(axis=1)

    if cand_cols:
        cand_df = pd.DataFrame(cand_cols)
        cand_df[target] = panel[target].values
        cand_df["__key"] = np.arange(len(cand_df))
        # Align to clean sub for R² vs existing
        print(f"  {'candidate':<22} {'uni_IC':>8} {'R2_vs_29':>10} {'orthogonal?':>12}")
        for c in cand_cols:
            cc = pd.concat([panel[feats].reset_index(drop=True),
                            cand_df[[c, target]].reset_index(drop=True)], axis=1)
            cc = cc.replace([np.inf,-np.inf], np.nan).dropna()
            if len(cc) < 1000:
                print(f"  {c:<22} (insufficient clean rows)")
                continue
            cv = cc[c].values
            cv_s = (cv - cv.mean())/(cv.std()+1e-9)
            ic = np.corrcoef(cv_s, cc[target].values)[0,1]
            Xe = cc[feats].values
            Xe_s = (Xe - Xe.mean(0))/(Xe.std(0)+1e-9)
            lr = LinearRegression().fit(Xe_s, cv_s)
            r2 = lr.score(Xe_s, cv_s)
            orth = "YES" if (r2 < 0.5 and abs(ic) > 0.005) else "no"
            print(f"  {c:<22} {ic:>+8.4f} {r2:>10.3f} {orth:>12}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
