"""Research cycle 3b: inverse-vol WEIGHTING within legs (distinct from timing, which was null).
Crypto lit (2024-25): "to address extreme vol dispersion, apply inverse-vol weighting so highly volatile
tokens get smaller allocations." Our edge IS a low-vol edge, so risk-balancing the legs (down-weight the
high-vol names, esp. the high-vol SHORT leg) may raise risk-adjusted return.

Schemes within each quintile leg (PIT rvol_7d): EQUAL (baseline) vs INV-VOL (w propto 1/rvol) vs
INV-VOL-WINSOR (rvol winsorized per bar 10/90 before inverting -> bounded weight ratio). Same leg
membership across schemes -> per-bar aligned -> paired block-bootstrap CI on Sharpe(scheme)-Sharpe(equal).
Run: python3 -u -m live.build_vol_weight
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

PYR = 6 * 365.0
RNG = np.random.default_rng(0)


def leg_returns(d, wcol):
    """Per-bar weighted long-short return; weights normalized within (bar, leg). Returns sorted (times, ls)."""
    dd = d[d["leg"] != 0].copy()
    dd["w"] = dd[wcol].to_numpy()
    denom = dd.groupby(["open_time", "leg"])["w"].transform("sum")
    dd["contrib"] = dd["leg"] * (dd["w"] / denom) * dd["return_pct"]
    g = dd.groupby("open_time")
    per = pd.DataFrame({"ls": g["contrib"].sum(),
                        "nl": g["leg"].apply(lambda x: int((x == 1).sum())),
                        "ns": g["leg"].apply(lambda x: int((x == -1).sum()))})
    per = per[(per["nl"] >= 3) & (per["ns"] >= 3)].sort_index()
    return per.index.to_numpy(), per["ls"].to_numpy()


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR)


def block_ci(a, b, block=30, nb=3000):
    """Paired block-bootstrap CI on Sharpe(b) - Sharpe(a) (per-bar aligned)."""
    n = len(a); nblk = int(np.ceil(n / block)); diffs = np.empty(nb)
    for i in range(nb):
        starts = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[i] = sh(b[idx]) / np.sqrt(PYR) - sh(a[idx]) / np.sqrt(PYR)
    d = diffs * np.sqrt(PYR)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct", "rvol_7d"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna(subset=["pred", "return_pct", "rvol_7d"])
        d = d[d["rvol_7d"] > 0].copy()
        r = d.groupby("open_time")["pred"].rank(pct=True)
        d["leg"] = np.where(r >= 0.8, 1, np.where(r <= 0.2, -1, 0))
        # weight columns
        d["w_eq"] = 1.0
        d["w_iv"] = 1.0 / d["rvol_7d"]
        lo = d.groupby("open_time")["rvol_7d"].transform(lambda s: s.quantile(0.10))
        hi = d.groupby("open_time")["rvol_7d"].transform(lambda s: s.quantile(0.90))
        d["w_ivw"] = 1.0 / d["rvol_7d"].clip(lower=lo, upper=hi)
        print(f"===== {era} =====", flush=True)
        t0, eq = leg_returns(d, "w_eq")
        _, iv = leg_returns(d, "w_iv")
        _, ivw = leg_returns(d, "w_ivw")
        print(f"  EQUAL         gross {eq.mean()*1e4:+.2f}bps  Sharpe {sh(eq):+.2f}", flush=True)
        for name, x in [("INV-VOL", iv), ("INV-VOL-WINSOR", ivw)]:
            lo_ci, hi_ci = block_ci(eq, x)
            v = "improves" if lo_ci > 0 else ("hurts" if hi_ci < 0 else "CI spans 0 (null)")
            print(f"  {name:<14}gross {x.mean()*1e4:+.2f}bps  Sharpe {sh(x):+.2f}  | dSh vs EQUAL "
                  f"{sh(x)-sh(eq):+.2f} 95% CI [{lo_ci:+.2f}, {hi_ci:+.2f}] -> {v}", flush=True)
        print("", flush=True)
    print("VOLWEIGHTDONE", flush=True)


if __name__ == "__main__":
    main()
