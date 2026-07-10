"""Thorough per-regime STABILITY / concentration re-eval on clean books (2026-07-10).

For EACH regime x era: monthly net concentration (top-2-month share of positive net), positive-month
rate, and the per-fold/sub-period spread — to characterize how EVENT-CONCENTRATED and FRAGILE each
regime edge is (the comprehensive version of #1 era-fragility + #5 thin-alpha). Committed generator.
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load, attribute
import warnings; warnings.filterwarnings("ignore")

def regime_concentration(df, era):
    print(f"\n=== {era}: per-regime monthly concentration (clean, pinned cost) ===")
    print(f"  {'regime':<9}{'cyc':>5}{'mo':>4} | {'net':>7} {'mean':>6} | {'pos-mo':>7} | {'top2-share':>11} | {'top month':>16}")
    df = df.copy(); df["month"] = pd.to_datetime(df["t"]).dt.to_period("M").astype(str)
    for rg in ["side","bear","bull","deepbull"]:
        s = df[df.reg == rg]
        if len(s) < 5: continue
        m = s.groupby("month")["net_resid"].sum().sort_values(ascending=False)
        tot = m.sum(); pos = m.clip(lower=0).sum()
        top2 = m.head(2).sum()
        share = f"{top2/pos*100:.0f}% of pos" if pos > 0 else "net<0"
        top = f"{m.index[0]} {m.iloc[0]:+.0f}"
        print(f"  {rg:<9}{len(s):>5}{len(m):>4} | {tot:>+7.0f} {s.net_resid.mean():>+6.1f} | {(m>0).mean()*100:>5.0f}% | {share:>11} | {top:>16}")

def main():
    reg = btc_reg()
    for era, bp, lp in (("OOS 2023-25", "hl_v4base_oos_cleanfix", "hl_v4long_oos_cleanfix"),
                        ("RECENT 2025-10+", "hl_tgt_res_base_cleanfix", "hl_tgt_res_long_cleanfix")):
        base, long = load(bp, lp); df = attribute(base, long, reg)
        regime_concentration(df, era)
    print("\nSTABILITYDONE")

if __name__ == "__main__":
    main()
