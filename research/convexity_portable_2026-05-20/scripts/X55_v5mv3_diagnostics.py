"""X55 — Diagnostics for V5_minus_v3_7cx on both HL-50 and HL-70.

Sections:
  A. LOFO (drop one fold at a time) on both universes
  B. Block bootstrap CI (50 iter) on both
  C. Cost sensitivity (V5_minus_v3_7cx at cost ∈ {1, 2, 3, 4.5, 6, 9, 12})
  D. AI cluster dropout (TAO/VIRTUAL/VVV)
  E. Per-sym dropout (top sym contributors)
"""
from __future__ import annotations
import sys, importlib.util, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CACHE = REPO / "research/convexity_portable_2026-05-20/results/_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
import phase_ah_sleeve as sleeve


def rerun_sleeve(apd, label):
    tmp = CACHE / f"_tmp_{label}_preds.parquet"
    apd.to_parquet(tmp, index=False)
    return x6.run_sleeve_on_preds(tmp, f"_tmp_{label}")


def main():
    print("=== X55 V5_minus_v3_7cx full diagnostics ===\n")

    # Load predictions
    hl50_apd = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    hl70_apd = pd.read_parquet(CACHE / "x53_V5_minus_v3_preds.parquet")
    print(f"HL-50: {len(hl50_apd):,} rows, folds {sorted(hl50_apd['fold'].unique())}")
    print(f"HL-70: {len(hl70_apd):,} rows, folds {sorted(hl70_apd['fold'].unique())}")

    cases = [("HL50", hl50_apd, +1.74), ("HL70", hl70_apd, +1.67)]

    # === A. LOFO ===
    print("\n" + "=" * 70)
    print("A. LOFO (drop one fold at a time)")
    print("=" * 70)
    for label, apd, ref in cases:
        print(f"\n[{label} baseline ref: {ref:+.2f}]")
        for f in sorted(apd["fold"].unique()):
            apd_d = apd[apd["fold"] != f]
            m = rerun_sleeve(apd_d, f"x55a_{label}_drop{f}")
            sh = m.get("sharpe", 0) or 0
            print(f"  drop fold {f}: {sh:+.2f} (Δ {sh-ref:+.2f})", flush=True)

    # === B. Block bootstrap CI (50 iter each) ===
    print("\n" + "=" * 70)
    print("B. Block bootstrap (n=50 each)")
    print("=" * 70)
    np.random.seed(20260521)
    for label, apd, ref in cases:
        folds = sorted(apd["fold"].unique())
        sharpes = []
        for i in range(50):
            sampled = np.random.choice(folds, size=len(folds), replace=True)
            parts = [apd[apd["fold"] == sf] for sf in sampled]
            apd_boot = pd.concat(parts, ignore_index=True)
            apd_boot["fold"] = np.arange(len(apd_boot)) // (len(apd_boot)//len(folds)+1)
            m = rerun_sleeve(apd_boot, f"x55b_{label}_boot")
            sharpes.append(m.get("sharpe", 0) or 0)
            if (i+1) % 10 == 0:
                arr = np.array(sharpes)
                print(f"  [{label}] iter {i+1}/50: mean={arr.mean():+.2f} std={arr.std():.2f}",
                      flush=True)
        arr = np.array(sharpes)
        print(f"\n[{label}] V5_minus_v3_7cx bootstrap (n=50):")
        print(f"  mean ± std: {arr.mean():+.2f} ± {arr.std():.2f}")
        print(f"  median: {np.median(arr):+.2f}")
        print(f"  95% CI: [{np.percentile(arr,2.5):+.2f}, {np.percentile(arr,97.5):+.2f}]")
        print(f"  P(>0): {(arr>0).mean()*100:.0f}%  P(>+1): {(arr>1).mean()*100:.0f}%")

    # === C. Cost sensitivity ===
    print("\n" + "=" * 70)
    print("C. Cost sensitivity")
    print("=" * 70)
    pred_paths = [
        ("HL50", CACHE / "x54_V5_minus_v3_7cx_preds.parquet"),
        ("HL70", CACHE / "x53_V5_minus_v3_preds.parquet"),
    ]
    costs = [1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0]
    for label, path in pred_paths:
        print(f"\n[{label}] V5_minus_v3_7cx cost sweep:")
        print(f"  {'cost':<6} {'Sharpe':>8} {'folds+':>8} {'conc':>8}")
        for c in costs:
            sleeve.COST_PER_LEG = c
            sleeve.COST_PER_UNIT_ABS_DELTA = 0.5 * c
            m = x6.run_sleeve_on_preds(path, f"x55c_{label}_c{c}")
            sh = m.get("sharpe")
            fp = m.get("folds_pos", "?")
            conc = m.get("concentration", "?")
            sh_str = f"{sh:+.2f}" if isinstance(sh,(int,float)) else "?"
            print(f"  {c:<6.1f} {sh_str:>8} {str(fp):>8} {str(conc):>8}", flush=True)

    # === D. AI cluster dropout ===
    print("\n" + "=" * 70)
    print("D. AI cluster dropout (TAO/VIRTUAL/VVV)")
    print("=" * 70)
    AI = ["TAOUSDT", "VIRTUALUSDT", "VVVUSDT"]
    for label, apd, ref in cases:
        print(f"\n[{label} baseline ref: {ref:+.2f}]")
        for sym in AI:
            apd_d = apd[apd["symbol"] != sym]
            m = rerun_sleeve(apd_d, f"x55d_{label}_drop{sym[:5]}")
            sh = m.get("sharpe", 0) or 0
            print(f"  drop {sym}: {sh:+.2f} (Δ {sh-ref:+.2f})", flush=True)
        apd_d = apd[~apd["symbol"].isin(AI)]
        m = rerun_sleeve(apd_d, f"x55d_{label}_dropAI")
        sh = m.get("sharpe", 0) or 0
        print(f"  drop ALL AI: {sh:+.2f} (Δ {sh-ref:+.2f})", flush=True)


if __name__ == "__main__":
    main()
