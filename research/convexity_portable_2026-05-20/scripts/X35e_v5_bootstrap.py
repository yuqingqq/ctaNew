"""X35e — V5-only bootstrap CI (50 iterations, faster than X35d).

X35d hit 2-hour timeout. This is the focused V5 follow-up.
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


def rerun_sleeve(apd, label):
    tmp = CACHE / f"_tmp_{label}_preds.parquet"
    apd.to_parquet(tmp, index=False)
    return x6.run_sleeve_on_preds(tmp, f"_tmp_{label}")


def main():
    v5 = pd.read_parquet(CACHE / "x29_V5_BASE_cohort_ALL_v2_preds.parquet")
    folds = sorted(v5["fold"].unique())
    print(f"V5: {len(v5):,} rows, folds {folds}")

    np.random.seed(20260521)
    n_boot = 50
    sharpes = []
    for i in range(n_boot):
        sampled = np.random.choice(folds, size=len(folds), replace=True)
        parts = [v5[v5["fold"] == sf] for sf in sampled]
        apd_boot = pd.concat(parts, ignore_index=True)
        apd_boot["fold"] = np.arange(len(apd_boot)) // (len(apd_boot) // len(folds) + 1)
        m = rerun_sleeve(apd_boot, "x35e_V5_boot")
        sh = m.get("sharpe", 0)
        sharpes.append(sh)
        if (i + 1) % 5 == 0:
            arr = np.array(sharpes)
            print(f"  iter {i+1:3d}/{n_boot}: mean={arr.mean():+.2f} std={arr.std():.2f} "
                  f"latest={sh:+.2f}", flush=True)

    arr = np.array(sharpes)
    print(f"\n[V5] bootstrap (n={n_boot}):")
    print(f"  mean ± std: {arr.mean():+.2f} ± {arr.std():.2f}")
    print(f"  median: {np.median(arr):+.2f}")
    print(f"  95% CI: [{np.percentile(arr, 2.5):+.2f}, {np.percentile(arr, 97.5):+.2f}]")
    print(f"  prob Sharpe > 0: {(arr > 0).mean()*100:.0f}%")
    print(f"  prob Sharpe > +1: {(arr > 1).mean()*100:.0f}%")

    print(f"\nReference: V0 bootstrap (X35d, n=100): mean +0.20 ± 1.15, "
          f"95% CI [-2.07, +2.06], P>0=56%")


if __name__ == "__main__":
    main()
