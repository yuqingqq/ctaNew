"""X33 — Compile master comparison table from all Phase 1 + Phase 2 result CSVs.

Outputs:
  - MASTER_RESULTS.md: human-readable per-dimension tables
  - X33_master_results.csv: long-format CSV of all tests
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"


def load_csv(name):
    """Load a result CSV and return DataFrame or None."""
    fp = OUT / f"{name}.csv"
    if not fp.exists(): return None
    try:
        return pd.read_csv(fp)
    except Exception as e:
        print(f"  ERR loading {name}: {e}")
        return None


def main():
    print("=== X33 compile master results table ===\n")

    # =====================================================================
    # Part A: Core 36-cell matrix (X6 baseline)
    # =====================================================================
    print("\n## Part A: Core Phase 1 matrix (X6, 36 cells: 6 archs × 6 feature sets)\n")
    x6 = load_csv("X6_controlled_matrix")
    if x6 is not None:
        # Take first occurrence of each (model, arch, feature_set) — earliest entry
        x6_cells = x6.drop_duplicates(subset=["model", "arch", "feature_set"], keep="first")
        # Pivot to a matrix
        pivot = x6_cells.pivot_table(values="sharpe", index="arch", columns="feature_set",
                                       aggfunc="mean")
        print(pivot.round(2).to_string())
        print()

    # =====================================================================
    # Part B: Phase 2 corrections (X22, X29 — with panel v2)
    # =====================================================================
    print("\n## Part B: Phase 2 corrections with panel v2 (X22, X29)\n")
    rows = []
    x22 = load_csv("X22_rerun_matrix_clean")
    if x22 is not None:
        for _, r in x22.iterrows():
            rows.append({"test": "X22", "variant": r["cell"],
                          "sharpe_panel_v2": r.get("sharpe"),
                          "sharpe_ref": r.get("ref_sharpe"),
                          "note": r.get("desc", "")})
    x29 = load_csv("X29_cohort_combos_panel_v2")
    if x29 is not None:
        for _, r in x29.iterrows():
            rows.append({"test": "X29", "variant": r["variant"],
                          "sharpe_panel_v2": r.get("sharpe"),
                          "note": r.get("desc", "")})
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False, max_colwidth=40))

    # =====================================================================
    # Part C: Regularization variants
    # =====================================================================
    print("\n\n## Part C: Regularization variants (X8, X8b, X8c, X8e, X10)\n")
    rows = []
    for name, label in [("X8_ridge_reg_sweep", "X8 Ridge wider α"),
                         ("X8c_symid_norm_sweep", "X8c sym_id normalization"),
                         ("X8d_cohort_collinearity", "X8d cohort fix"),
                         ("X8e_lgbm_reg_sweep", "X8e LGBM reg")]:
        df = load_csv(name)
        if df is not None:
            for _, r in df.iterrows():
                rows.append({"test": label, "variant": r.get("variant", r.get("cell", "?")),
                              "sharpe": r.get("sharpe")})
    x10 = load_csv("X10_c1_norm_symid_results")
    if x10 is not None:
        for _, r in x10.iterrows():
            rows.append({"test": "X10 C1-norm applied", "variant": r["cell"],
                          "sharpe": r.get("sharpe")})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, max_colwidth=45))

    # =====================================================================
    # Part D: Universe variants (X11, X20, X23, X32)
    # =====================================================================
    print("\n\n## Part D: Universe variants (X11, X23, X32 — Ridge Per-sym + cohort)\n")
    rows = []
    x11 = load_csv("X11_universe_stress")
    if x11 is not None:
        for _, r in x11.iterrows():
            rows.append({"test": "X11", "universe": r["universe"], "n_syms": r["n_syms"],
                          "sharpe": r.get("sharpe")})
    x23 = load_csv("X23_universe_sweep_fixed")
    if x23 is not None:
        for _, r in x23.iterrows():
            rows.append({"test": "X23", "universe": r["universe"], "n_syms": r["n_syms"],
                          "sharpe": r.get("sharpe")})
    x32 = load_csv("X32_hl70_universe")
    if x32 is not None:
        for _, r in x32.iterrows():
            rows.append({"test": "X32 (HL-70)", "universe": r["universe"], "n_syms": r["n_syms"],
                          "sharpe": r.get("sharpe")})
    if rows:
        df = pd.DataFrame(rows)
        df = df[df["sharpe"].notna()].sort_values("sharpe", ascending=False)
        print(df.to_string(index=False, max_colwidth=35))

    # =====================================================================
    # Part E: Cluster variants (X24, X25)
    # =====================================================================
    print("\n\n## Part E: Cluster-based universes (X24 hand-crafted, X25 data-driven)\n")
    rows = []
    x24 = load_csv("X24_cluster_universes")
    if x24 is not None:
        for _, r in x24.iterrows():
            rows.append({"test": "X24", "universe": r["universe"], "n_syms": r["n_syms"],
                          "sharpe": r.get("sharpe")})
    x25 = load_csv("X25_datadriven_cluster_universes")
    if x25 is not None:
        for _, r in x25.iterrows():
            rows.append({"test": "X25", "universe": r["universe"], "n_syms": r["n_syms"],
                          "sharpe": r.get("sharpe")})
    if rows:
        df = pd.DataFrame(rows)
        df = df[df["sharpe"].notna()].sort_values("sharpe", ascending=False)
        print(df.to_string(index=False, max_colwidth=35))

    # =====================================================================
    # Part F: Other key experiments
    # =====================================================================
    print("\n\n## Part F: Other Phase 2 experiments\n")
    rows = []
    for name, label in [("X12_apply_to_v31", "X12 V3.1 augment"),
                         ("X13_ensemble_vs_groupalpha", "X13 orthogonality"),
                         ("X14b_crossX_5m_rerun", "X14b crossX 5m (LEAKY)"),
                         ("X14d_basis_ffill_rerun", "X14d crossX 5m basis-ffill"),
                         ("X19_preproc_sweep", "X19 preprocessing"),
                         ("X20_universe_nstress", "X20 (drifted)"),
                         ("X26_cohort_combos", "X26 (OLD panel)"),
                         ("X27_pergroup_alpha_xuniv", "X27 per-group α cross-univ"),
                         ("X29_cohort_combos_panel_v2", "X29 (v2 panel)")]:
        df = load_csv(name)
        if df is not None:
            for _, r in df.iterrows():
                # Pick the best identifier column
                v = (r.get("variant") or r.get("cell") or r.get("universe")
                     or r.get("pair") or r.get("desc", "?"))
                sh = r.get("sharpe") or r.get("validate_pergroup_sharpe")
                rows.append({"test": label, "variant": str(v)[:40], "sharpe": sh})
    if rows:
        df = pd.DataFrame(rows)
        df = df[df["sharpe"].notna()]
        print(df.to_string(index=False, max_colwidth=45))

    # Save consolidated CSV
    print("\n\n=== Saving long-format master CSV ===")
    master_rows = []

    def add(test, variant, sharpe, n_syms=None, feature_set=None, model=None, arch=None,
            panel="v1", note=""):
        master_rows.append({
            "test": test, "variant": variant, "sharpe": sharpe,
            "n_syms": n_syms, "feature_set": feature_set,
            "model": model, "arch": arch, "panel": panel, "note": note
        })

    # X6 matrix
    if x6 is not None:
        for _, r in x6.iterrows():
            add("X6 matrix", r["cell"], r.get("sharpe"),
                feature_set=r.get("feature_set"),
                model=r.get("model"), arch=r.get("arch"), panel="v1")

    # X22 / X29 panel_v2
    if x22 is not None:
        for _, r in x22.iterrows():
            add("X22 panel_v2", r["cell"], r.get("sharpe"),
                model=r.get("model"), arch=r.get("arch"), panel="v2",
                note=r.get("desc", ""))
    if x29 is not None:
        for _, r in x29.iterrows():
            add("X29 panel_v2", r["variant"], r.get("sharpe"),
                feature_set=r.get("desc", ""), panel="v2")

    # Universe (X23/X32)
    if x23 is not None:
        for _, r in x23.iterrows():
            add("X23 universe", r["universe"], r.get("sharpe"), n_syms=r["n_syms"],
                feature_set="BASE+cohort", model="Ridge", arch="per-sym", panel="v1")
    if x32 is not None:
        for _, r in x32.iterrows():
            add("X32 HL-70", r["universe"], r.get("sharpe"), n_syms=r["n_syms"],
                feature_set="BASE+cohort", model="Ridge", arch="per-sym",
                panel="hl70")

    # Clusters
    if x24 is not None:
        for _, r in x24.iterrows():
            add("X24 cluster", r["universe"], r.get("sharpe"), n_syms=r["n_syms"],
                feature_set="BASE+cohort", model="Ridge", arch="per-sym", panel="v1")

    # Reg sweeps
    if x10 is not None:
        for _, r in x10.iterrows():
            add("X10 C1-norm", r["cell"], r.get("sharpe"),
                model="Ridge", arch="pool+symid_C1norm", panel="v1",
                feature_set=r.get("feature_set", ""))

    master_df = pd.DataFrame(master_rows)
    master_df = master_df[master_df["sharpe"].notna()].copy()
    out_csv = OUT / "X33_master_results.csv"
    master_df.to_csv(out_csv, index=False)
    print(f"Saved {len(master_df)} tests → {out_csv}")
    print(f"\nTop 20 by Sharpe:")
    print(master_df.nlargest(20, "sharpe")[["test", "variant", "sharpe", "n_syms"]].to_string(index=False))


if __name__ == "__main__":
    main()
