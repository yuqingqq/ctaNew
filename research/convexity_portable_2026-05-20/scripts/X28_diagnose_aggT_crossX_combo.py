"""X28 — Diagnose why aggT+crossX combination HURTS Per-sym Ridge.

X26 found:
  V0 baseline (BASE+cohort, 17 feats):    +2.01 (IC +0.0062)
  V1 +aggT (22 feats):                    +1.62 (IC +0.0044)
  V2 +crossX (22 feats):                  +1.90 (IC +0.0062)
  V3 +aggT+crossX (27 feats):             +0.60 (IC +0.0045)  ← combo HURTS

The combination drops -1.41 Sharpe vs baseline. Why?

Diagnostics:
  A. Per-sym IC of each feature group on target
  B. Ridge coefficient comparison V0 vs V3 — does cohort get under-weighted?
  C. Per-fold α choice variance
  D. Coverage interaction — syms missing one or both
  E. Prediction distribution comparison

Outputs:
  - Per-sym IC matrix
  - Mean Ridge coefficient norms by feature group
  - Prediction stability comparison
"""
from __future__ import annotations
import sys, time, warnings, importlib.util, gc
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def load_panel():
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    cross_df = pd.read_parquet(REPO / "data/ml/cache/cross_exchange_features.parquet")
    cross_df["open_time"] = pd.to_datetime(cross_df["open_time"], utc=True)
    cross_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cross_df[["symbol", "open_time"] + cross_z_cols],
                        on=["symbol", "open_time"], how="left")
    panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    return panel, cross_z_cols


def main():
    t0 = time.time()
    print("=== X28 diagnose why aggT+crossX combo hurts Per-sym Ridge ===\n", flush=True)
    panel, cross_z_cols = load_panel()
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms")

    target = "target_z"
    BASE = x6.BASE
    COHORT = x6.COHORT_EXTRAS
    AGGT = x6.AGGT_EXTRAS
    CROSSX = cross_z_cols

    # === A. Per-sym IC of each feature group ===
    print(f"\n=== A. Per-sym IC of each feature group on target_z ===")
    per_sym_ic = []
    for sym, g in panel.groupby("symbol"):
        row = {"symbol": sym, "n_rows": len(g)}
        for grp_name, feats in [("BASE", BASE), ("cohort", COHORT), ("aggT", AGGT), ("crossX", CROSSX)]:
            cor_vals = []
            for f in feats:
                if f not in g.columns: continue
                valid = g[f].notna() & g[target].notna()
                if valid.sum() < 100: continue
                cor_vals.append(g.loc[valid, f].corr(g.loc[valid, target]))
            if cor_vals:
                row[f"mean_|IC|_{grp_name}"] = np.mean(np.abs(cor_vals))
            row[f"coverage_{grp_name}"] = (g[feats[0]].notna().mean() if feats and feats[0] in g.columns else 0)
        per_sym_ic.append(row)
    psdf = pd.DataFrame(per_sym_ic)
    print(f"\nMean per-sym |IC| by group:")
    for c in [c for c in psdf.columns if c.startswith("mean_|IC|_")]:
        print(f"  {c}: mean={psdf[c].mean():.4f}, median={psdf[c].median():.4f}")
    print(f"\nCoverage by group (fraction of bars with non-null first feature):")
    for c in [c for c in psdf.columns if c.startswith("coverage_")]:
        print(f"  {c}: mean={psdf[c].mean():.3f}, syms_with_>50%={(psdf[c]>0.5).sum()}")
    psdf.to_csv(OUT / "X28_per_sym_ic.csv", index=False)
    print(f"\nSaved per-sym IC → {OUT/'X28_per_sym_ic.csv'}")

    # === B. Ridge coefficient comparison V0 vs V3 ===
    print(f"\n=== B. Ridge coefficient norms per feature group, V0 vs V3 ===")
    folds = x6.get_folds(panel)
    f_idx, ts, te, ec = folds[5]  # mid-sample fold
    train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
    test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
    print(f"\n  Using fold {f_idx} (mid-sample): n_train={len(train_all):,}")

    # Per-sym coefficients for V0 (BASE+cohort) vs V3 (BASE+cohort+aggT+crossX)
    v0_feats = BASE + COHORT
    v3_feats = BASE + COHORT + AGGT + CROSSX
    coef_stats = {"V0_BASE_norm": [], "V0_cohort_norm": [],
                  "V3_BASE_norm": [], "V3_cohort_norm": [], "V3_aggT_norm": [], "V3_crossX_norm": [],
                  "V0_alpha": [], "V3_alpha": []}
    syms_done = 0
    for sym, gtr in train_all.groupby("symbol"):
        if len(gtr) < 300: continue
        if syms_done >= 20: break  # sample of 20 syms
        for fset_name, feats in [("V0", v0_feats), ("V3", v3_feats)]:
            try:
                sstats, hstats = x6.fit_preproc(gtr, feats)
                Xtr = x6.apply_preproc(gtr, feats, sstats, hstats)
                ytr = gtr["target_z"].to_numpy()
                m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
                coef_stats[f"{fset_name}_alpha"].append(float(m.alpha_))
                # Group coefficient norms
                idx = 0
                for grp_name, grp_feats in [("BASE", BASE), ("cohort", COHORT),
                                              ("aggT", AGGT), ("crossX", CROSSX)]:
                    if grp_name not in fset_name or fset_name == "V0" and grp_name in ("aggT", "crossX"):
                        # not in this feature set
                        continue
                    # find offset
                    if grp_name == "BASE": start = 0
                    elif grp_name == "cohort": start = len(BASE)
                    elif grp_name == "aggT": start = len(BASE) + len(COHORT)
                    elif grp_name == "crossX": start = len(BASE) + len(COHORT) + len(AGGT)
                    end = start + len(grp_feats)
                    if end > len(m.coef_): continue
                    norm = float(np.linalg.norm(m.coef_[start:end]))
                    coef_stats[f"{fset_name}_{grp_name}_norm"].append(norm)
            except Exception: pass
        syms_done += 1

    print(f"\n  Coefficient L2 norms (mean across {syms_done} syms):")
    for k in ["V0_BASE_norm", "V0_cohort_norm", "V3_BASE_norm", "V3_cohort_norm",
              "V3_aggT_norm", "V3_crossX_norm"]:
        vals = coef_stats[k]
        if vals:
            print(f"    {k}: {np.mean(vals):.3f} (std {np.std(vals):.3f})")
    print(f"  Median α picked:")
    for k in ["V0_alpha", "V3_alpha"]:
        vals = coef_stats[k]
        if vals:
            print(f"    {k}: {np.median(vals):.2f}")

    # Key comparison
    if coef_stats["V0_cohort_norm"] and coef_stats["V3_cohort_norm"]:
        v0_c = np.mean(coef_stats["V0_cohort_norm"])
        v3_c = np.mean(coef_stats["V3_cohort_norm"])
        print(f"\n  *** Cohort norm V0→V3: {v0_c:.3f} → {v3_c:.3f} "
              f"({(v3_c-v0_c)/v0_c*100:+.0f}% change) ***")
        if v3_c < v0_c * 0.7:
            print(f"  → Cohort coefficients SHRUNK by adding aggT/crossX (under-weighted)")

    # === C. Predictions comparison (V0 vs V3) ===
    print(f"\n=== C. Per-cycle prediction spread (V0 vs V3) ===")
    v0_path = CACHE / "x26_V0_BASE_cohort_preds.parquet"
    v3_path = CACHE / "x26_V3_BASE_cohort_aggT_crossX_preds.parquet"
    if v0_path.exists() and v3_path.exists():
        v0 = pd.read_parquet(v0_path)
        v3 = pd.read_parquet(v3_path)
        v0_spread = v0.groupby("open_time")["pred"].std()
        v3_spread = v3.groupby("open_time")["pred"].std()
        print(f"  Within-cycle pred std (median):")
        print(f"    V0: {v0_spread.median():.4f}")
        print(f"    V3: {v3_spread.median():.4f}")
        # Per (sym, time) correlation between V0 and V3 predictions
        merged = v0.merge(v3[["symbol", "open_time", "pred"]],
                           on=["symbol", "open_time"], suffixes=("_v0", "_v3"))
        corr = merged["pred_v0"].corr(merged["pred_v3"])
        print(f"  V0-V3 prediction correlation: {corr:.4f}")
        print(f"  V0 IC: {merged['pred_v0'].corr(merged['alpha_A']):+.4f}")
        print(f"  V3 IC: {merged['pred_v3'].corr(merged['alpha_A']):+.4f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
