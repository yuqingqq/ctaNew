"""COMMITTED re-derivation of limitation #5 (thin/event-concentrated SIDE alpha) on CLEAN books.

The V4_LIMITATIONS "76% of side net from 2 dispersion months" figure had NO committed generator and
didn't reproduce (audit). This computes it properly on the clean (_cleanfix) OOS + recent books: the
side-regime 1L/2S book NET per cycle (pinned 0.5x9 cost, residual frame), aggregated by month, and the
concentration (top-K months' share of total positive side net + monthly positive-rate). Committed so
the limitation is a reproducible number, not folklore.
"""
import numpy as np, pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load, attribute
import warnings; warnings.filterwarnings("ignore")

def month_concentration(df, era):
    s = df[df.reg == "side"].copy()
    s["month"] = pd.to_datetime(s["t"]).dt.to_period("M").astype(str)
    m = s.groupby("month")["net_resid"].agg(["sum","mean","count"]).sort_values("sum", ascending=False)
    tot = m["sum"].sum(); pos = m["sum"].clip(lower=0).sum()
    top2 = m["sum"].head(2).sum()
    print(f"\n=== {era}: SIDE-regime monthly net concentration (clean books, pinned cost) ===")
    print(f"  {len(s)} side cycles over {len(m)} months; total side net {tot:+.0f} bps")
    print(f"  positive months: {(m['sum']>0).sum()}/{len(m)} ({(m['sum']>0).mean()*100:.0f}%)")
    print(f"  top-2 months = {top2:+.0f} bps = {top2/tot*100 if tot>0 else float('nan'):.0f}% of total net"
          f"  ({top2/pos*100 if pos>0 else float('nan'):.0f}% of positive net)")
    print(f"  top-3 months share of total: {m['sum'].head(3).sum()/tot*100 if tot>0 else float('nan'):.0f}%")
    print("  top months:")
    for mo, r in m.head(4).iterrows():
        print(f"     {mo}: net {r['sum']:+7.0f} bps ({int(r['count'])} cyc, mean {r['mean']:+.1f})")
    return top2/tot if tot>0 else np.nan

def main():
    reg = btc_reg()
    for era, bp, lp in (("OOS 2023-25", "hl_v4base_oos_cleanfix", "hl_v4long_oos_cleanfix"),
                        ("RECENT 2025-10+", "hl_tgt_res_base_cleanfix", "hl_tgt_res_long_cleanfix")):
        base, long = load(bp, lp); df = attribute(base, long, reg)
        month_concentration(df, era)
    print("\nLIM5DONE")

if __name__ == "__main__":
    main()
