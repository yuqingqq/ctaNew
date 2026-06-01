"""X69 — How does the HL-70-trained model perform when trading only HL-50 syms?

Two transport questions:
  A. Take HL-70-trained V5_mv3 predictions, trade only canonical HL-50 subset.
     (Does training on the broader 70-sym panel produce better/worse HL-50 picks?)
  B. Take HL-50-trained V5_mv3 predictions (native), for reference (+1.74).
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
CACHE = REPO / "research/convexity_portable_2026-05-20/results/_cache"
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    print("=== X69 HL-70-trained model traded on HL-50 subset ===\n")
    # Canonical HL-50 syms
    canonical = sorted(set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
        columns=["symbol"])["symbol"].unique()) - {"BTCUSDT"})
    print(f"Canonical HL-50: {len(canonical)} syms")

    # A. HL-70-trained predictions, filtered to HL-50 syms
    hl70_preds = pd.read_parquet(CACHE / "x53_V5_minus_v3_preds.parquet")
    hl70_on_50 = hl70_preds[hl70_preds["symbol"].isin(canonical)]
    print(f"HL-70 preds filtered to HL-50: {len(hl70_on_50):,} rows × {hl70_on_50['symbol'].nunique()} syms")
    tmp = CACHE / "x69_hl70trained_on_hl50_preds.parquet"
    hl70_on_50.to_parquet(tmp, index=False)
    m_a = x6.run_sleeve_on_preds(tmp, "x69_hl70_on_hl50")

    # B. Native HL-50-trained (reference)
    m_b = x6.run_sleeve_on_preds(CACHE / "x54_V5_minus_v3_7cx_preds.parquet", "x69_native_hl50")

    print(f"\n=== Results ===")
    print(f"  A. HL-70-trained, traded on HL-50 subset: Sharpe={m_a.get('sharpe', 0):+.2f} "
          f"folds={m_a.get('folds_pos','?')} conc={m_a.get('concentration','?')}")
    print(f"  B. HL-50-trained native (reference):       Sharpe={m_b.get('sharpe', 0):+.2f} "
          f"folds={m_b.get('folds_pos','?')} conc={m_b.get('concentration','?')}")
    a_sh = m_a.get('sharpe', 0) or 0
    b_sh = m_b.get('sharpe', 0) or 0
    print(f"\n  Transport lift (A - B): {a_sh - b_sh:+.2f}")
    print(f"  Interpretation: if A ≈ B, training universe doesn't matter for HL-50 picks.")
    print(f"                  if A < B, broader-panel training degrades HL-50 picks (overfit dilution).")


if __name__ == "__main__":
    main()
