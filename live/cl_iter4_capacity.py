"""Cost/turnover loop — iteration 4 (D2 + D3 + D4): the deployability / capacity frontier.

The decision-relevant question after iters 1-3: is there ANY configuration of the incumbent whose NET Sharpe
is significantly > 0 in BOTH eras, and at what execution clip size does it die?

Pre-registered grid (small, to limit multiplicity — ALL 27 cells reported, no cherry-picking):
  universe tier   : top20 / top40 / all         (PIT trailing-ADV)
  turnover control: none / EWMA λ=0.7 / band K=3,M=8 (enter top-K, hold until it exits top-(K+M))
  cost tier       : cost_10k / cost_50k / cost_100k  (the calibrated depth model — the capacity frontier)
Signal: the incumbent per-symbol-Ridge `fast` only (the only thing with significant gross in iters 1-3).
Books on per-name BTC-residual returns; 7d-block CI on the net Sharpe; a config must pass in BOTH eras and
be the SAME config in both — a per-era winner is a selection artifact, not a result.

Gates: G1 >=1 config net CI>0 both eras @cost_10k; G2 that same config also @cost_50k (capacity >= $50k).
Falsifier: no config passes G1 -> the honest terminal read for this branch is that the strategy is not
net-deployable at retail-accessible clip sizes on free data.
Run: python3 -u -m live.cl_iter4_capacity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import (
    ERAS, CACHE, REPO, block_ci, build_panel, get_preds, maxdd, restrict_topn, sharpe, tag_ci,
)
from live.build_deployed_band import band_topk

TIERS = [(20, "top20"), (40, "top40"), (999, "all")]
CONTROLS = ["none", "ewma0.7", "band"]
CLIPS = ["cost_10k", "cost_50k", "cost_100k"]
K, M, Q = 3, 8, 0.2


def cost_tiers():
    c = pd.read_csv(REPO / "live/state/v3loop/persym_cost_cal.csv").set_index("symbol")
    return {k: (c[k], float(c[k].median())) for k in CLIPS}


def _norm(P):
    pos = P.clip(lower=0); neg = P.clip(upper=0)
    return pos.div(pos.sum().replace(0, np.nan), axis=1).fillna(0.0) \
        + neg.div(neg.sum().abs().replace(0, np.nan), axis=1).fillna(0.0)


def build(d: pd.DataFrame, control: str):
    """Returns (W, A) names x bars on per-name residual returns."""
    x = d.dropna(subset=["pred", "alpha_A"]).copy()
    if control == "band":
        x["rhi"] = x.groupby("open_time")["pred"].rank(ascending=False, method="first")
        x["rlo"] = x.groupby("open_time")["pred"].rank(ascending=True, method="first")
        x = x.sort_values(["symbol", "open_time"])
        x["pos"] = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                   for _, g in x.groupby("symbol", sort=True)])
    else:
        x["rk"] = x.groupby("open_time")["pred"].rank(pct=True)
        x["pos"] = np.where(x["rk"] >= 1 - Q, 1.0, np.where(x["rk"] <= Q, -1.0, 0.0))
    A = x.pivot_table(index="symbol", columns="open_time", values="alpha_A").fillna(0.0)
    P = x.pivot_table(index="symbol", columns="open_time", values="pos", fill_value=0.0).reindex_like(A)
    W = _norm(P)
    if control.startswith("ewma"):
        lam = float(control.replace("ewma", ""))
        mask = x.pivot_table(index="symbol", columns="open_time", values="alpha_A").notna().astype(float)
        W = _norm(W.T.ewm(alpha=1 - lam, adjust=False).mean().T * mask.reindex_like(W).fillna(0.0))
    return W, A


def main():
    CT = cost_tiers()
    PAN = build_panel()
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    rows = []
    for era in ERAS:
        p = get_preds(era)
        if "alpha_A" not in p.columns:
            p = p.merge(lab, on=["symbol", "open_time"], how="left")
        for n, tl in TIERS:
            d = restrict_topn(p, n)
            for ctl in CONTROLS:
                W, A = build(d, ctl)
                g = (W * A).sum(axis=0)
                dW = W.diff(axis=1).abs()
                turn = 0.25 * dW.sum(axis=0)
                r = dict(era=era, tier=tl, ctl=ctl, bars=len(g) - 1,
                         names=float((W.abs() > 1e-9).sum(axis=0).mean()),
                         turn=float(turn.iloc[1:].mean()), gross=sharpe(g.iloc[1:]),
                         maxdd=maxdd(g.iloc[1:]) * 1e4)
                r["g_lo"], r["g_hi"] = block_ci(g.iloc[1:].to_numpy())
                for clip in CLIPS:
                    c, med = CT[clip]
                    cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
                    ch = 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4
                    net = (g - ch).iloc[1:]
                    lo, hi = block_ci(net.to_numpy())
                    r[f"net_{clip}"] = sharpe(net); r[f"lo_{clip}"] = lo; r[f"hi_{clip}"] = hi
                    r[f"bps_{clip}"] = float(ch.iloc[1:].mean() * 1e4)
                rows.append(r)
                print(f"  [{era}/{tl}/{ctl}] built", flush=True)
    T = pd.DataFrame(rows); T.to_csv(CACHE / "iter4_capacity.csv", index=False)

    for clip in CLIPS:
        print(f"\n============ NET SHARPE @ {clip} (per-name residual books, 7d-block CI) ============",
              flush=True)
        for era in ERAS:
            print(f"\n----- {era} -----", flush=True)
            print(f"  {'tier':<8}{'control':<10}{'turn':<7}{'names':<7}{'cost bps/bar':<14}"
                  f"{'grossSh':<9}{'net [CI]':<28}{'maxDD':<8}", flush=True)
            for _, tl in TIERS:
                for ctl in CONTROLS:
                    x = T[(T.era == era) & (T.tier == tl) & (T.ctl == ctl)].iloc[0]
                    cell = (f"{x[f'net_{clip}']:+.2f} [{x[f'lo_{clip}']:+.2f},{x[f'hi_{clip}']:+.2f}] "
                            f"{tag_ci(x[f'lo_{clip}'], x[f'hi_{clip}'])}")
                    print(f"  {tl:<8}{ctl:<10}{x.turn:<7.3f}{x.names:<7.1f}{x[f'bps_{clip}']:<14.2f}"
                          f"{x.gross:<+9.2f}{cell:<28}{x.maxdd:<8.0f}", flush=True)

    print("\n============ GATE READ (same config must pass in BOTH eras) ============", flush=True)
    for clip in CLIPS:
        winners = []
        for _, tl in TIERS:
            for ctl in CONTROLS:
                ok = all(T[(T.era == e) & (T.tier == tl) & (T.ctl == ctl)].iloc[0][f"lo_{clip}"] > 0
                         for e in ERAS)
                if ok:
                    winners.append(f"{tl}/{ctl}")
        print(f"  {clip:<12} configs passing both-era net CI>0: {len(winners)}/9 "
              f"{winners if winners else '(none)'}", flush=True)
    print("\n  MULTIPLICITY NOTE: 9 configs tested per clip tier. The both-era requirement is the guard;", flush=True)
    print("  1/9 passing is weak evidence, not a result. Report all cells (done above).", flush=True)
    print("\nITER4DONE", flush=True)


if __name__ == "__main__":
    main()
