"""X56 — Ensemble V0 + V5_minus_v3_7cx predictions.

Test if averaging the two best Per-sym Ridge cells gives a better Sharpe
than either alone. Tests on both canonical HL-50 and HL-70.
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


def normalize(s):
    return (s - s.mean()) / s.std() if s.std() > 0 else s


def ensemble(v0, vbest, weight_v0=0.5):
    """Average normalized predictions per (sym, time)."""
    m = v0[["symbol","open_time","pred","alpha_A","return_pct","exit_time","fold"]].copy()
    m = m.merge(vbest[["symbol","open_time","pred"]],
                 on=["symbol","open_time"], suffixes=("_v0", "_vbest"))
    # Normalize each per-fold so they have comparable scale
    m["pred_v0_n"] = m.groupby("fold")["pred_v0"].transform(normalize)
    m["pred_vbest_n"] = m.groupby("fold")["pred_vbest"].transform(normalize)
    m["pred"] = weight_v0 * m["pred_v0_n"] + (1 - weight_v0) * m["pred_vbest_n"]
    return m[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]]


def main():
    # V0 = BASE+cohort on canonical HL-50
    v0_hl50 = pd.read_parquet(CACHE / "x29_V0_BASE_cohort_v2_preds.parquet")
    # V5_minus_v3_7cx on canonical HL-50
    vbest_hl50 = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    # V5_minus_v3 on HL-70 (X53 produced this for HL-70 directly)
    vbest_hl70 = pd.read_parquet(CACHE / "x53_V5_minus_v3_preds.parquet")
    # V0 on HL-70 — from X32
    v0_hl70 = pd.read_parquet(CACHE / "x32_HL70_full_preds.parquet")

    print(f"V0 HL-50: {len(v0_hl50):,} rows")
    print(f"V5_minus_v3_7cx HL-50: {len(vbest_hl50):,} rows")
    print(f"V0 HL-70: {len(v0_hl70):,} rows")
    print(f"V5_minus_v3 HL-70: {len(vbest_hl70):,} rows")

    for label_hl, v0, vbest in [("HL50", v0_hl50, vbest_hl50), ("HL70", v0_hl70, vbest_hl70)]:
        print(f"\n=== {label_hl} ensemble sweep ===")
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            apd = ensemble(v0, vbest, w)
            tmp = CACHE / f"x56_{label_hl}_w{w}_preds.parquet"
            apd.to_parquet(tmp, index=False)
            m = x6.run_sleeve_on_preds(tmp, f"x56_{label_hl}_w{w}")
            sh = m.get("sharpe")
            fp = m.get("folds_pos", "?")
            conc = m.get("concentration", "?")
            sh_str = f"{sh:+.2f}" if isinstance(sh,(int,float)) else "?"
            print(f"  w_v0={w}: Sharpe={sh_str} folds={fp} conc={conc}", flush=True)


if __name__ == "__main__":
    main()
