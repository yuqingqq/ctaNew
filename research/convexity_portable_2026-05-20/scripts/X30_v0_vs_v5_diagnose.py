"""X30 — Re-diagnose V0 vs V5 (vs V3) with PANEL V2.

X28 used OLD panel — conclusions confounded by aggT NaN.
X29 found V5 (BASE+cohort+ALL, 31 feats, v2) = +1.66, 7/9, 26% conc — beats V0 robustness.

This script:
  A. Per-sym IC of each feature group with v2 panel
  B. Ridge coefficient norms V0 vs V5 (does each group contribute meaningfully?)
  C. Prediction comparison V0 vs V5 vs V3
  D. Fold-by-fold Sharpe breakdown (V0 fragile? V5 robust?)
  E. Per-cycle prediction agreement: V0 ∩ V5 picks
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


def load_panel_v2():
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS + x6.V3_EXTRAS)
    panel_path = REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet"
    panel = pd.read_parquet(panel_path, columns=list(set(needed)))
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
    print("=== X30 V0 vs V5 (vs V3) diagnostic with PANEL V2 ===\n", flush=True)
    panel, cross_z_cols = load_panel_v2()
    print(f"  panel v2: {len(panel):,} rows × {panel['symbol'].nunique()} syms")
    target = "target_z"
    BASE = x6.BASE
    COHORT = x6.COHORT_EXTRAS
    AGGT = x6.AGGT_EXTRAS
    CROSSX = cross_z_cols
    V3_IDIO = x6.V3_EXTRAS

    # === A. Per-sym IC by feature group ===
    print(f"\n=== A. Per-sym mean |IC| by feature group (v2 panel) ===")
    rows = []
    for sym, g in panel.groupby("symbol"):
        row = {"symbol": sym, "n": len(g)}
        for grp_name, feats in [("BASE", BASE), ("cohort", COHORT), ("aggT", AGGT),
                                 ("crossX", CROSSX), ("v3", V3_IDIO)]:
            ics = []
            for f in feats:
                if f not in g.columns: continue
                valid = g[f].notna() & g[target].notna()
                if valid.sum() < 100: continue
                ics.append(g.loc[valid, f].corr(g.loc[valid, target]))
            if ics:
                row[f"mean_|IC|_{grp_name}"] = np.mean(np.abs(ics))
                row[f"max_|IC|_{grp_name}"] = np.max(np.abs(ics))
                row[f"cov_{grp_name}"] = (g[feats[0]].notna().mean() if feats and feats[0] in g.columns else 0)
        rows.append(row)
    psdf = pd.DataFrame(rows)
    print(f"\n{'Group':<10} {'mean_|IC|':>12} {'med_|IC|':>12} {'syms_>50%cov':>14}")
    for grp in ["BASE", "cohort", "aggT", "crossX", "v3"]:
        c = f"mean_|IC|_{grp}"
        cov_c = f"cov_{grp}"
        if c in psdf.columns:
            print(f"{grp:<10} {psdf[c].mean():>12.4f} {psdf[c].median():>12.4f} "
                  f"{(psdf[cov_c]>0.5).sum() if cov_c in psdf.columns else '?':>14}")
    psdf.to_csv(OUT / "X30_per_sym_ic_v2.csv", index=False)
    print(f"\nSaved → {OUT / 'X30_per_sym_ic_v2.csv'}")

    # === B. Ridge coefficient norms V0 vs V5 ===
    print(f"\n=== B. Ridge coefficient norms V0 vs V5 (fold 5, 20 sample syms) ===")
    folds = x6.get_folds(panel)
    f_idx, ts, te, ec = folds[5]
    train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
    print(f"\n  fold {f_idx} (mid-sample): n_train={len(train_all):,}")

    v0_feats = BASE + COHORT
    v5_feats = list(dict.fromkeys(BASE + COHORT + AGGT + CROSSX + V3_IDIO))
    print(f"  V0: {len(v0_feats)} feats, V5: {len(v5_feats)} feats")

    coef_stats = {f"V0_{g}_norm": [] for g in ["BASE", "cohort"]}
    coef_stats.update({f"V5_{g}_norm": [] for g in ["BASE", "cohort", "aggT", "crossX", "v3"]})
    alpha_stats = {"V0": [], "V5": []}

    syms_done = 0
    for sym, gtr in train_all.groupby("symbol"):
        if len(gtr) < 300: continue
        if syms_done >= 30: break

        # Find group boundaries (preserving v5_feats dedup order)
        v5_group_idx = {}
        i = 0
        for grp_name, grp_feats in [("BASE", BASE), ("cohort", COHORT), ("aggT", AGGT),
                                      ("crossX", CROSSX), ("v3", V3_IDIO)]:
            v5_group_idx[grp_name] = (i, i + len(grp_feats))
            i += len(grp_feats)

        # V0
        try:
            sstats, hstats = x6.fit_preproc(gtr, v0_feats)
            Xtr = x6.apply_preproc(gtr, v0_feats, sstats, hstats)
            ytr = gtr["target_z"].to_numpy()
            m0 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
            coef_stats["V0_BASE_norm"].append(float(np.linalg.norm(m0.coef_[:len(BASE)])))
            coef_stats["V0_cohort_norm"].append(float(np.linalg.norm(m0.coef_[len(BASE):])))
            alpha_stats["V0"].append(float(m0.alpha_))
        except Exception: continue

        # V5
        try:
            sstats, hstats = x6.fit_preproc(gtr, v5_feats)
            Xtr = x6.apply_preproc(gtr, v5_feats, sstats, hstats)
            m5 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
            for grp_name, (start, end) in v5_group_idx.items():
                if end > len(m5.coef_): continue
                coef_stats[f"V5_{grp_name}_norm"].append(float(np.linalg.norm(m5.coef_[start:end])))
            alpha_stats["V5"].append(float(m5.alpha_))
        except Exception: pass
        syms_done += 1

    print(f"\n  Coefficient L2 norms (mean across {syms_done} syms):")
    print(f"  Group               V0         V5      Δ")
    for grp in ["BASE", "cohort"]:
        v0 = np.mean(coef_stats[f"V0_{grp}_norm"]) if coef_stats[f"V0_{grp}_norm"] else 0
        v5 = np.mean(coef_stats[f"V5_{grp}_norm"]) if coef_stats[f"V5_{grp}_norm"] else 0
        delta_pct = (v5 - v0) / v0 * 100 if v0 > 0 else 0
        print(f"  {grp:<18} {v0:>8.3f} {v5:>8.3f}  {delta_pct:>+6.0f}%")
    for grp in ["aggT", "crossX", "v3"]:
        v5 = np.mean(coef_stats[f"V5_{grp}_norm"]) if coef_stats[f"V5_{grp}_norm"] else 0
        print(f"  {grp:<18} {'n/a':>8} {v5:>8.3f}")
    print(f"\n  α picked: V0 median={np.median(alpha_stats['V0']):.0f}, V5 median={np.median(alpha_stats['V5']):.0f}")

    # === C. Prediction comparison V0 vs V5 ===
    print(f"\n=== C. Prediction comparison V0 vs V5 (and V3 if available) ===")
    v0_p = CACHE / "x29_V0_BASE_cohort_v2_preds.parquet"
    v3_p = CACHE / "x29_V3_BASE_cohort_aggT_crossX_v2_preds.parquet"
    v5_p = CACHE / "x29_V5_BASE_cohort_ALL_v2_preds.parquet"
    if all(p.exists() for p in [v0_p, v3_p, v5_p]):
        v0 = pd.read_parquet(v0_p)
        v3 = pd.read_parquet(v3_p)
        v5 = pd.read_parquet(v5_p)
        m = v0.merge(v3[["symbol","open_time","pred"]], on=["symbol","open_time"], suffixes=("_v0","_v3"))
        m = m.merge(v5[["symbol","open_time","pred"]].rename(columns={"pred":"pred_v5"}),
                     on=["symbol","open_time"])
        print(f"  Pred correlations:")
        print(f"    V0-V3: {m['pred_v0'].corr(m['pred_v3']):.4f}")
        print(f"    V0-V5: {m['pred_v0'].corr(m['pred_v5']):.4f}")
        print(f"    V3-V5: {m['pred_v3'].corr(m['pred_v5']):.4f}")
        print(f"  Per-cycle pred std (within-time):")
        print(f"    V0: {v0.groupby('open_time')['pred'].std().median():.4f}")
        print(f"    V3: {v3.groupby('open_time')['pred'].std().median():.4f}")
        print(f"    V5: {v5.groupby('open_time')['pred'].std().median():.4f}")
        print(f"  IC vs alpha_A:")
        print(f"    V0: {m['pred_v0'].corr(m['alpha_A']):+.4f}")
        print(f"    V3: {m['pred_v3'].corr(m['alpha_A']):+.4f}")
        print(f"    V5: {m['pred_v5'].corr(m['alpha_A']):+.4f}")

    # === D. Per-fold Sharpe (load sleeve results) ===
    print(f"\n=== D. Per-fold Sharpe (from sleeve output files) ===")
    for label, name in [("V0", "x29_V0_BASE_cohort_v2"), ("V3", "x29_V3_BASE_cohort_aggT_crossX_v2"),
                         ("V5", "x29_V5_BASE_cohort_ALL_v2")]:
        sleeve_dir = OUT / f"_x6_sleeve_{name}"
        if sleeve_dir.exists():
            # find per-fold sharpe file
            for f in sleeve_dir.glob("*.csv"):
                if "fold" in f.name.lower():
                    try:
                        df = pd.read_csv(f)
                        if "sharpe" in df.columns and len(df) <= 12:
                            print(f"  {label} per-fold:")
                            print(df[["fold","sharpe"]].to_string(index=False) if "fold" in df.columns else df.head())
                    except Exception: pass

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
