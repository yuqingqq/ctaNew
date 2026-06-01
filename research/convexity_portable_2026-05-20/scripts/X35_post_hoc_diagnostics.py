"""X35 — Post-hoc diagnostics on existing V0/V5 predictions.

Combined script with sections:
  X35a: LOFO (leave-one-fold-out) — drop each fold one at a time, recompute Sharpe
  X35b: AI cluster sym dropout — drop TAO/VIRTUAL/VVV individually, see impact
  X35c: Per-fold Sharpe breakdown for V0 and V5
  X35d: Block bootstrap Sharpe CI for V0 and V5
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CACHE = REPO / "research/convexity_portable_2026-05-20/results/_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def load_preds(name):
    fp = CACHE / f"{name}_preds.parquet"
    if not fp.exists():
        print(f"  MISSING: {fp}")
        return None
    return pd.read_parquet(fp)


def rerun_sleeve_on_subset(apd, label="tmp"):
    """Save apd to tmp parquet and run sleeve eval."""
    tmp_path = CACHE / f"_tmp_{label}_preds.parquet"
    apd.to_parquet(tmp_path, index=False)
    return x6.run_sleeve_on_preds(tmp_path, f"_tmp_{label}")


def main():
    print("=" * 70)
    print("X35 POST-HOC DIAGNOSTICS")
    print("=" * 70)

    # Load V0 (BASE+cohort) and V5 (ALL) predictions from X29
    print("\nLoading prediction files...")
    v0 = load_preds("x29_V0_BASE_cohort_v2")
    v5 = load_preds("x29_V5_BASE_cohort_ALL_v2")
    if v0 is None or v5 is None:
        print("Missing prediction files; abort")
        return
    print(f"  V0: {len(v0):,} rows, folds {sorted(v0['fold'].unique())}")
    print(f"  V5: {len(v5):,} rows, folds {sorted(v5['fold'].unique())}")

    # =====================================================================
    # X35a: LOFO (drop each fold, recompute Sharpe)
    # =====================================================================
    print("\n" + "=" * 70)
    print("X35a: LOFO — drop one fold at a time, recompute Sharpe")
    print("=" * 70)

    for label, apd in [("V0", v0), ("V5", v5)]:
        print(f"\n[{label}]")
        # Full
        m_full = rerun_sleeve_on_subset(apd, f"x35a_{label}_full")
        full_sh = m_full.get("sharpe", 0)
        print(f"  All folds: Sharpe={full_sh:+.2f}")

        # Drop one fold at a time
        for drop_f in sorted(apd["fold"].unique()):
            apd_drop = apd[apd["fold"] != drop_f]
            m = rerun_sleeve_on_subset(apd_drop, f"x35a_{label}_drop{drop_f}")
            sh = m.get("sharpe", 0)
            print(f"  drop fold {drop_f}: Sharpe={sh:+.2f} (Δ {sh-full_sh:+.2f})")

    # =====================================================================
    # X35b: AI cluster sym dropout
    # =====================================================================
    print("\n" + "=" * 70)
    print("X35b: AI cluster sym dropout (TAO/VIRTUAL/VVV)")
    print("=" * 70)

    AI_SYMS = ["TAOUSDT", "VIRTUALUSDT", "VVVUSDT"]
    for label, apd in [("V0", v0), ("V5", v5)]:
        print(f"\n[{label}]")
        for sym in AI_SYMS:
            apd_drop = apd[apd["symbol"] != sym]
            m = rerun_sleeve_on_subset(apd_drop, f"x35b_{label}_drop{sym[:5]}")
            sh = m.get("sharpe", 0)
            print(f"  drop {sym}: Sharpe={sh:+.2f}")
        # drop all 3
        apd_drop_all = apd[~apd["symbol"].isin(AI_SYMS)]
        m = rerun_sleeve_on_subset(apd_drop_all, f"x35b_{label}_dropAI")
        print(f"  drop ALL AI: Sharpe={m.get('sharpe', 0):+.2f}")

    # =====================================================================
    # X35c: Per-fold Sharpe breakdown
    # =====================================================================
    print("\n" + "=" * 70)
    print("X35c: Per-fold Sharpe breakdown")
    print("=" * 70)

    for label, apd in [("V0", v0), ("V5", v5)]:
        print(f"\n[{label}] per-fold")
        for f in sorted(apd["fold"].unique()):
            apd_only = apd[apd["fold"] == f]
            m = rerun_sleeve_on_subset(apd_only, f"x35c_{label}_only{f}")
            sh = m.get("sharpe", 0)
            pnl = m.get("totPnL", 0)
            print(f"  fold {f}: Sharpe={sh:+.2f}, PnL={pnl}")

    # =====================================================================
    # X35d: Block bootstrap CI
    # =====================================================================
    print("\n" + "=" * 70)
    print("X35d: Block bootstrap Sharpe CI (resample folds with replacement)")
    print("=" * 70)
    np.random.seed(20260521)
    n_boot = 100
    for label, apd in [("V0", v0), ("V5", v5)]:
        folds = sorted(apd["fold"].unique())
        bootstrap_sharpes = []
        for _ in range(n_boot):
            sampled_folds = np.random.choice(folds, size=len(folds), replace=True)
            # Concat sampled fold predictions
            parts = []
            for sf in sampled_folds:
                parts.append(apd[apd["fold"] == sf])
            apd_boot = pd.concat(parts, ignore_index=True)
            # Renumber folds to avoid duplicate keys in sleeve
            apd_boot["fold"] = np.arange(len(apd_boot)) // (len(apd_boot) // len(folds) + 1)
            m = rerun_sleeve_on_subset(apd_boot, f"x35d_{label}_boot")
            bootstrap_sharpes.append(m.get("sharpe", 0))
        sh_arr = np.array(bootstrap_sharpes)
        print(f"\n[{label}] bootstrap (n={n_boot}):")
        print(f"  mean ± std: {sh_arr.mean():+.2f} ± {sh_arr.std():.2f}")
        print(f"  median: {np.median(sh_arr):+.2f}")
        print(f"  95% CI: [{np.percentile(sh_arr, 2.5):+.2f}, {np.percentile(sh_arr, 97.5):+.2f}]")
        print(f"  prob Sharpe > 0: {(sh_arr > 0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
