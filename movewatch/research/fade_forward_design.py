"""Run: python3 research/fade_forward_design.py

Design numbers for forward hypothesis H3 (docs/FORWARD_TEST_v2_amend_1.md), from consumed
replay windows only (2020-01..2026-09, all already read for pump questions; decides nothing about
the forward result). Population as H3: non-meme heavy pumps (vol_persist > A2 heavy cut), short
+4 h -> +24 h.

Prints:
- the research cost actually charged (fee + tiered slippage at stress 1, per side), so the flat
  forward cost can be set to it;
- per-trade net at the flat forward cost WITHOUT funding (the live recorder has no funding), and
  the funding the research added, by year;
- F1's primary net (flat cost + funding), the cluster-week standard error per trade, and the power of the declared pass rule (mean > 0
  and 95% CI lower bound > 0) at several sample sizes, for the 2022-25 and the 2026 effect sizes
  (normal approximation: SE(n) = SE_full * sqrt(n_full / n))."""
import sys
from math import erf, sqrt
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.btc_hedge as bh
import research.heavy_fade as hf
import research.premium_classes as pc

FLAT_COST = 2 * (bh.FEE + 0.0015)          # declared forward cost: 0.40% round trip
NS = (150, 300, 400, 600, 800)


def phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


if __name__ == "__main__":
    p = pc.frame(require="vol_persist")
    x = p[(p["vol_persist"] > hf.HEAVY) & (~p["meme"]) & p["r_S24"].notna()].copy()
    x["slip"] = [bh.slip(v) for v in x["liq"].to_numpy(float)]
    x["gross"] = -x["r_S24"]
    x["net_flat"] = x["gross"] - FLAT_COST
    x["fund"] = x["fund_S24"]
    x["yr"] = x["time"].dt.year
    print(f"non-meme heavy pumps with a 24 h short: n {len(x)}; flat forward cost {100 * FLAT_COST:.2f}% round trip")
    print(f"research slippage charged at stress 1, per side: mean {100 * x['slip'].mean():.3f}%, "
          f"median {100 * x['slip'].median():.3f}%; research round trip mean {100 * (2 * (bh.FEE + x['slip'])).mean():.3f}%")
    print(f"\n{'year':7} {'n':>5} {'gross':>7} {'net flat, no funding':>21} {'funding added':>14}")
    yr = x["yr"]
    for lab, g in list(x.groupby(yr)) + [("2022-25", x[(yr >= 2022) & (yr <= 2025)]), ("all", x)]:
        print(f"{str(lab):7} {len(g):5d} {100 * g['gross'].mean():+6.2f}% {100 * g['net_flat'].mean():+20.2f}% {100 * g['fund'].mean():+13.3f}%")
    x["net_f1"] = x["net_flat"] + x["fund"]       # F1's primary: flat cost, funding from the exchange
    x = x[x["net_f1"].notna()]
    yr = x["yr"]
    print(f"F1 primary net (flat cost + funding): 2022-25 {100 * x.loc[(yr >= 2022) & (yr <= 2025), 'net_f1'].mean():+.2f}%, "
          f"2026 {100 * x.loc[yr == 2026, 'net_f1'].mean():+.2f}%, all {100 * x['net_f1'].mean():+.2f}% (n {len(x)})")
    full = hf.stats(x["net_f1"].to_numpy(), x["time"])
    se = (full["ci"][1] - full["ci"][0]) / (2 * 1.96)
    print(f"\ncluster-week SE of the mean at n {full['n']}: {100 * se:.2f}% (CI [{100 * full['ci'][0]:+.2f}, {100 * full['ci'][1]:+.2f}])")
    effects = {"2022-25": x.loc[(yr >= 2022) & (yr <= 2025), "net_f1"].mean(), "2026": x.loc[yr == 2026, "net_f1"].mean()}
    print("power of 'mean > 0 and CI lower bound > 0' (normal approximation):")
    for lab, mu in effects.items():
        cells = []
        for n in NS:
            s = se * sqrt(full["n"] / n)
            cells.append(f"n {n}: {100 * phi(mu / s - 1.96):3.0f}%")
        print(f"  true mean {100 * mu:+.2f}% ({lab}): " + " | ".join(cells))
