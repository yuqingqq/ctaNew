"""HOW to use L2 for a multi-day book: EXECUTION/CAPACITY, not alpha. Quantify per-name tradeable depth + market
impact from bookDepth (reconstruct one-side depth at 0.2%/1%/5% from l2_liq1 [notional within 1%], l2_touch
[0.2%/1% ratio], l2_slope [5%/1% ratio]), and the impact (price % you cross) to trade $100k and $500k one-side.
Shows the liquidity cross-section (majors deep vs alts thin) = why the illiquidity-bound edge collapses at size, and
what depth-aware sizing/gating would act on.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
DEPLOYED = set("ADA APT ARB ATOM AVAX BCH BNB BTC DOGE DOT ETH FIL INJ LINK LTC NEAR OP RUNE SEI SOL SUI TIA UNI WLD XRP".split())

def impact_pct(N, d02, d1, d5):
    """price FRACTION you cross to fill N one-side, piecewise-linear through cum-depth at 0.2%/1%/5% (levels as fractions)."""
    pts = [(0.0, 0.0), (0.002, d02), (0.01, d1), (0.05, d5)]
    for (p0, c0), (p1, c1) in zip(pts, pts[1:]):
        if N <= c1:
            return p0 + (p1 - p0) * (N - c0) / max(c1 - c0, 1e-9)
    return 0.05 * N / max(d5, 1e-9)   # beyond 5%: extrapolate (very thin)

def main():
    rec = pd.Timestamp("2025-10-01", tz="UTC"); rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet")]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f)
        d = d[pd.to_datetime(d.index, utc=True) >= rec]
        if not len(d): continue
        L = np.exp(d["l2_liq1"]).median()                       # total notional within 1% of mid ($)
        touch = d["l2_touch"].median(); slope = d["l2_slope"].median()
        if not np.isfinite(L): continue
        d1 = L / 2; d02 = touch * L / 2; d5 = slope * L / 2       # one-side depth within 0.2/1/5%
        rows.append({"sym": sym.replace("USDT", ""), "depth1pct_$": L, "one_side_1pct_$": d1,
                     "imp_100k_pct": impact_pct(100_000, d02, d1, d5), "imp_500k_pct": impact_pct(500_000, d02, d1, d5),
                     "deployed": sym.replace("USDT", "") in DEPLOYED})
    T = pd.DataFrame(rows)
    print(f"per-name recent-median tradeable depth + impact ({len(T)} names)\n")
    print("=== DEPLOYED-25 universe (what the book actually trades), sorted by $100k impact ===")
    dep = T[T.deployed].sort_values("imp_100k_pct")
    for _, r in dep.iterrows():
        print(f"  {r['sym']:8s} depth±1%=${r['depth1pct_$']/1e6:6.1f}M | impact 100k {r['imp_100k_pct']*100:5.2f}% | 500k {r['imp_500k_pct']*100:5.2f}%")
    print(f"\n  DEPLOYED impact-to-trade-$100k: median {dep.imp_100k_pct.median()*100:.2f}% | "
          f"cheapest {dep.imp_100k_pct.min()*100:.2f}% ({dep.loc[dep.imp_100k_pct.idxmin(),'sym']}) | "
          f"priciest {dep.imp_100k_pct.max()*100:.2f}% ({dep.loc[dep.imp_100k_pct.idxmax(),'sym']})")
    print(f"  spread = {dep.imp_100k_pct.max()/max(dep.imp_100k_pct.min(),1e-9):.0f}x across the tradeable universe")
    print(f"\n=== FULL covered universe ({len(T)}) impact-to-$100k percentiles ===")
    for q in [10, 50, 90, 99]:
        print(f"  p{q}: {np.percentile(T.imp_100k_pct, q)*100:.2f}%")
    print(f"  names where $100k already costs >1% (too thin to size up): {int((T.imp_100k_pct>0.01).sum())}/{len(T)}")
    print("IMPACTDONE")

if __name__ == "__main__":
    main()
