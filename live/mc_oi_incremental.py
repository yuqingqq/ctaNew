"""Gauntlet for the one both-era lead from live/mc_oi_universe.py.

Finding to test: the OI/ADV rank (open interest relative to turnover) has same-sign rank-IC in BOTH eras
(+0.0235 OOS / +0.0442 RECENT, long-high) and blending it into the incumbent prediction ADDS rank-IC in both
(Δ +0.0117 [+0.0057,+0.0177] OOS, +0.0224 [+0.0159,+0.0293] RECENT).

Two ways this dies, both checked here — in falsification order:

  C1 ORTHOGONALITY. High OI/ADV = positions held long relative to churn = calm names; low = frantically
     traded racy names. That is close to the low-vol axis `docs/CONCLUSION_2026-08-03.md` already identifies
     as THE edge. If residualizing OI/ADV on the vol rank kills its IC, it is the known factor in a new hat,
     not new information.
  C2 CONVERSION. Iteration 1's lesson: rank-IC parity does not imply book parity. Build the blend at book
     level (top40/band, per-symbol cost) under iteration 5's hard split — select 2023-06→2024-12, hold out
     2025-01→2026-06 — and require the held-out paired Δnet to clear 0.

Gates: C1 residualized IC keeps sign + CI excludes 0 in BOTH eras; C2 held-out Δnet@10k CI > 0.
Falsifier: either fails -> record as another IC that does not convert, adopt nothing.
Run: python3 -u -m live.mc_oi_incremental
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import (
    ERAS, CACHE, block_ci, build_panel, get_preds, paired_block_ci, pit_adv, sharpe, tag_ci,
)
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import load_oi, topn, SPAN0, SPAN1, HO0, SEL0, SEL1, N

RNG = np.random.default_rng(41)


def perbar_ic(d, sig, tgt="alpha_A"):
    return d.groupby("open_time").apply(
        lambda x: spearmanr(x[sig], x[tgt]).correlation if len(x) >= 10 else np.nan).dropna()


def day_ci(s, nb=3000):
    gg = [x.to_numpy() for _, x in s.groupby(pd.to_datetime(s.index, utc=True).floor("1D"))]
    b = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def xs_resid(d, ycol, xcol):
    """Per-bar cross-sectional residual of ycol on xcol (both already pct-ranks)."""
    def f(g):
        x = g[xcol].to_numpy(); y = g[ycol].to_numpy()
        if len(g) < 10 or np.std(x) == 0:
            return pd.Series(np.nan, index=g.index)
        b = np.polyfit(x, y, 1)[0]
        return pd.Series(y - b * x - (y - b * x).mean(), index=g.index)
    return d.groupby("open_time", group_keys=False).apply(f)


def main():
    CT = cost_tiers()
    OI, _ = load_oi()
    PAN = build_panel()
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized", "rvol_7d"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    P = P.drop(columns=[c for c in ("alpha_A",) if c in P.columns]).merge(
        lab, on=["symbol", "open_time"], how="left")
    P = P.merge(OI, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left")
    P = P.dropna(subset=["pred", "alpha_A", "oi_usd", "tadv", "rvol_7d"])
    P["oi_adv"] = P["oi_usd"] / P["tadv"]

    print("============ C1 — is OI/ADV distinct from the known low-vol factor? ============", flush=True)
    for era, (t0, t1) in (("OOS", (SPAN0, HO0)), ("RECENT", (HO0, SPAN1))):
        d = topn(P[(P.open_time >= t0) & (P.open_time < t1)], "tadv", N).copy()
        d["oiadv_rank"] = d.groupby("open_time")["oi_adv"].rank(pct=True)
        d["vol_rank"] = d.groupby("open_time")["rvol_7d"].rank(pct=True)
        rho = d.groupby("open_time").apply(
            lambda g: spearmanr(g["oiadv_rank"], g["vol_rank"]).correlation).dropna()
        d["oiadv_resid"] = xs_resid(d, "oiadv_rank", "vol_rank")
        raw = perbar_ic(d, "oiadv_rank"); res = perbar_ic(d.dropna(subset=["oiadv_resid"]), "oiadv_resid")
        rlo, rhi = day_ci(raw); slo, shi = day_ci(res)
        vol = perbar_ic(d, "vol_rank"); vlo, vhi = day_ci(vol)
        print(f"\n----- {era} -----", flush=True)
        print(f"  xs corr(OI/ADV rank, vol rank)      {rho.mean():+.3f}", flush=True)
        print(f"  IC  vol_rank (long high)            {vol.mean():+.4f} [{vlo:+.4f},{vhi:+.4f}]", flush=True)
        print(f"  IC  OI/ADV raw (long high)          {raw.mean():+.4f} [{rlo:+.4f},{rhi:+.4f}]", flush=True)
        print(f"  IC  OI/ADV residualized on vol      {res.mean():+.4f} [{slo:+.4f},{shi:+.4f}]  "
              f"{tag_ci(slo, shi)}", flush=True)

    print("\n============ C2 — does the blend convert at book level, out of selection? ============",
          flush=True)
    P["oiadv_rank"] = P.groupby("open_time")["oi_adv"].rank(pct=True)
    series = {}
    for wname, (t0, t1) in (("SELECT", (SEL0, SEL1)), ("HOLDOUT", (HO0, SPAN1))):
        w = topn(P[(P.open_time >= t0) & (P.open_time < t1)], "tadv", N).copy()
        zp = w.groupby("open_time")["pred"].transform(lambda x: (x - x.mean()) / (x.std() or 1))
        zs = w.groupby("open_time")["oiadv_rank"].transform(lambda x: (x - x.mean()) / (x.std() or 1))
        for name, sig in (("pred", zp), ("blend", zp + zs), ("oiadv_only", zs)):
            v = w.copy(); v["pred"] = sig
            W, Aa = build(v, "band")
            g = (W * Aa).sum(axis=0); dW = W.diff(axis=1).abs()
            c, med = CT["cost_10k"]
            cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
            net = (g - 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4).iloc[1:]
            series[(wname, name)] = net
            lo, hi = block_ci(net.to_numpy())
            print(f"  {wname:<8}{name:<12} gross {sharpe(g.iloc[1:]):+.2f}  net@10k {sharpe(net):+.2f} "
                  f"[{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}  turn "
                  f"{(0.25*dW.sum(axis=0)).iloc[1:].mean():.3f}", flush=True)
    print("", flush=True)
    for wname in ("SELECT", "HOLDOUT"):
        a, b = series[(wname, "pred")], series[(wname, "blend")]
        idx = a.index.intersection(b.index)
        dd, lo, hi = paired_block_ci(a.loc[idx].to_numpy(), b.loc[idx].to_numpy())
        print(f"  {wname:<8} Δ(blend − pred) net@10k {dd:+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}",
              flush=True)
    a, b = series[("HOLDOUT", "pred")], series[("HOLDOUT", "blend")]
    idx = a.index.intersection(b.index)
    _, lo, _ = paired_block_ci(a.loc[idx].to_numpy(), b.loc[idx].to_numpy())
    print(f"\n  C2 GATE (held-out Δnet CI>0): {'PASS' if lo > 0 else 'FAIL'}", flush=True)
    print("\nMCOIINCDONE", flush=True)


if __name__ == "__main__":
    main()
