"""Detail-review loop — iteration 6 (D4): concentration, now that BOTH premises behind diversification are dead.

The book holds ~26 names because two things were believed:
  (a) breadth helps  -> C2 measured the realised IR multiplier at 0.88x against 2.1x theory. FALSE here.
  (b) cost punishes concentration -> C1 measured true cost at $1-2.5k clips as 1.2-3.3 bps, 3-5x below what
      was charged. So the penalty for holding fewer, larger positions is much smaller than assumed.
Both premises are now falsified and the concentration question has never been revisited under the corrected
numbers. (The repo did sweep top-K historically and landed on K=2-3, but at the OLD cost.)

Test: vary K per side from a broad quintile down to K=2, at the MEASURED cost, on the combined two-sleeve
book, under the chronological hard split. Report net Sharpe, drawdown at MATCHED volatility (the leverage
artifact caught twice already), and the dollar position size at $250k-$1M so the answer is implementable.

Gate: some K materially beats the incumbent quintile on held-out net, paired block CI > 0.
Falsifier: no K beats it -> the current breadth is already right and concentration is a dead lever too.
Run: python3 -u -m live.dr_iter6_concentration
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import block_ci, paired_block_ci, pit_adv, tag_ci
from live.cl_iter4_capacity import build, cost_tiers
from live.dr_iter1_shortonly import load
from live.dp_phase1_consolidate import sharpe_d
from live.mc_oi_universe import topn, N as NTOP

SEL = (pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HO = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"))
COST_BPS = 2.0
KS = [None, 10, 5, 3, 2]          # None = the incumbent quintile
VOL_TARGET = 0.10


def book(d, K, cost_bps=COST_BPS):
    """Top-K per side (or quintile when K is None) on the incumbent preds, band-smoothed."""
    x = d.dropna(subset=["pred", "alpha_A"]).copy()
    if K is None:
        x["rk"] = x.groupby("open_time")["pred"].rank(pct=True)
        x["pos"] = np.where(x["rk"] >= 0.8, 1.0, np.where(x["rk"] <= 0.2, -1.0, 0.0))
    else:
        hi = x.groupby("open_time")["pred"].rank(ascending=False, method="first")
        lo = x.groupby("open_time")["pred"].rank(ascending=True, method="first")
        x["pos"] = np.where(hi <= K, 1.0, np.where(lo <= K, -1.0, 0.0))
    A = x.pivot_table(index="symbol", columns="open_time", values="alpha_A").fillna(0.0)
    P = x.pivot_table(index="symbol", columns="open_time", values="pos", fill_value=0.0).reindex_like(A)
    pos, neg = P.clip(lower=0), P.clip(upper=0)
    W = pos.div(pos.sum().replace(0, np.nan), axis=1).fillna(0.0) \
        + neg.div(neg.sum().abs().replace(0, np.nan), axis=1).fillna(0.0)
    g = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    net = (g - 0.25 * dW.sum(axis=0) * cost_bps * 2 / 1e4).iloc[1:]
    net.index = pd.to_datetime(net.index, utc=True)
    names = float((W.abs() > 1e-9).sum(axis=0).mean())
    return net.groupby(net.index.floor("1D")).sum(), names


def dd_at_vol(s, target=VOL_TARGET):
    v = s.std() * np.sqrt(365)
    x = (s * (target / v)).to_numpy() if v > 0 else s.to_numpy()
    eq = np.cumsum(x)
    return float((eq - np.maximum.accumulate(eq)).min())


def main():
    P = load()
    print(f"{'window':<9}{'K':<8}{'names':>7}{'Sharpe':>9}{'net CI':>21}{'vol%':>7}"
          f"{'DD@10vol':>10}", flush=True)
    store = {}
    for wname, (t0, t1) in (("SELECT", SEL), ("HOLDOUT", HO)):
        w = topn(P[(P.open_time >= t0) & (P.open_time < t1)], "tadv", NTOP)
        for K in KS:
            s, names = book(w, K)
            if len(s) < 60:
                continue
            store[(wname, K)] = s
            lo, hi = block_ci(s.to_numpy(), block=7)
            v = s.std() * np.sqrt(365)
            lab = "quintile" if K is None else f"top-{K}"
            print(f"{wname:<9}{lab:<8}{names:>7.1f}{sharpe_d(s):>9.2f}"
                  f"{f'[{lo:+.2f},{hi:+.2f}] {tag_ci(lo,hi)}':>21}{v*100:>7.1f}{dd_at_vol(s)*100:>10.2f}",
                  flush=True)
        print("", flush=True)

    print("=== paired Δ vs the incumbent quintile, HELD-OUT ===", flush=True)
    base = store.get(("HOLDOUT", None))
    for K in KS[1:]:
        v = store.get(("HOLDOUT", K))
        if base is None or v is None:
            continue
        idx = base.index.intersection(v.index)
        dd, lo, hi = paired_block_ci(base.loc[idx].to_numpy(), v.loc[idx].to_numpy(), block=7)
        print(f"  top-{K:<3} Δ {dd:+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}", flush=True)

    print("\n=== implementability at $250k-$1M (10% vol target) ===", flush=True)
    for K in KS:
        s = store.get(("HOLDOUT", K))
        if s is None:
            continue
        v = s.std() * np.sqrt(365)
        scale = VOL_TARGET / v if v > 0 else np.nan
        n = 2 * (K if K else 8)
        lab = "quintile" if K is None else f"top-{K}"
        print(f"  {lab:<9} scale {scale:.2f}x  ~{n} positions  "
              f"${250_000*scale*2/n:>8,.0f} per position at $250k  "
              f"${1_000_000*scale*2/n:>9,.0f} at $1M", flush=True)
    print("\nDRITER6DONE", flush=True)


if __name__ == "__main__":
    main()
