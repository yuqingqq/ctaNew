"""Small-capital research — C2: does removing the COST-DRIVEN universe restriction convert into IR?

C1 established that at $1-2.5k clips the true cost is 1.2-3.3 bps, versus the 5.8-9.4 bps `cost_10k` charged
throughout this repo — a 3-5x overcharge at a $250k-$1M capital scale. The top-40 ADV restriction and the
$3M/day liquidity floor were adopted *specifically* to solve that cost problem. If cost is 3-5x smaller, the
restriction may be unnecessary, and lifting it takes the universe from 40 to ~176 names.

Theory: breadth x4.4 -> IR x2.1 (fundamental law). This repo has repeatedly found nominal and EFFECTIVE
breadth diverge sharply, so the point of this test is to measure the realised multiplier, not assume it.

COST EXTRAPOLATION — the honest weak point, stated up front. Only 31 symbols have >=1100 days of aggTrades
and hence a measured cost. For the remaining ~145 we regress log(total cost @$2k) on log(ADV) over the
measured set and extrapolate. The whole question is whether the UNMEASURED thin names are tradeable, so this
is exactly where an over-optimistic answer could enter. The fit quality is reported, and a conservative
variant (2x the extrapolated cost for unmeasured names) is run alongside.

Gates: G1 net Sharpe rises with N under the hard split; G2 the realised IR multiplier from 40 -> full is
positive with a paired block CI excluding 0; G3 the conclusion survives the conservative cost variant.
Falsifier: G2 fails -> breadth does not convert, and per the plan's arithmetic 3.5 is not reachable this way.
Run: python3 -u -m live.sc_c2_universe
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import (CACHE, ERAS, REPO, block_ci, build_panel, get_preds,
                                    paired_block_ci, pit_adv, sharpe, tag_ci)
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build

SEL = (pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HO = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"))
TIERS = [20, 40, 80, 999]
PYR = 6 * 365.0


def measured_costs() -> pd.DataFrame:
    """Per-symbol TRUE cost at a $1-2.5k clip = own half-spread + own walk (C1 + markout study)."""
    C = pd.read_parquet(CACHE / "cost_curve.parquet")
    M = pd.read_parquet(CACHE / "markout.parquet")
    hs = M.groupby("symbol").apply(
        lambda g: float((g["half_spread"] * g["notional"]).sum() / g["notional"].sum())).rename("hs")
    walk = C[C["bucket"] == "1-2.5k"].set_index("symbol")["mean_slip"].rename("walk")
    D = pd.concat([hs, walk], axis=1).dropna()
    D["cost2k"] = D["hs"] + D["walk"]
    return D


def cost_map(adv_by_sym: pd.Series, conservative=False) -> tuple[pd.Series, float, float]:
    """Measured cost where available; log-ADV extrapolation elsewhere. Returns (map, R2, n_measured)."""
    M = measured_costs()
    j = pd.concat([M["cost2k"], np.log(adv_by_sym.rename("ladv"))], axis=1).dropna()
    b, a = np.polyfit(j["ladv"], np.log(j["cost2k"]), 1)
    pred = np.exp(a + b * np.log(adv_by_sym))
    r2 = float(np.corrcoef(j["ladv"], np.log(j["cost2k"]))[0, 1] ** 2)
    out = pred.copy()
    out.loc[M.index.intersection(out.index)] = M["cost2k"].reindex(out.index.intersection(M.index))
    if conservative:
        unmeasured = out.index.difference(M.index)
        out.loc[unmeasured] = out.loc[unmeasured] * 2.0
    return out.clip(lower=0.2, upper=40.0), r2, len(j)


def book_net(P, n, costs):
    d = P.copy()
    if n < 999:
        d["ar"] = d.groupby("open_time")["tadv"].rank(ascending=False, method="first")
        d = d[d["ar"] <= n]
    if d.empty:
        return None, None
    W, A = build(d, "band")
    g = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    kv = pd.Series([float(costs.get(s, costs.median())) for s in W.index], index=W.index)
    ch = 0.25 * dW.mul(kv, axis=0).sum(axis=0) / 1e4
    return g.iloc[1:], (g - ch).iloc[1:]


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    lab = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"})[["symbol", "open_time", "alpha_A"]]
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    P = P.drop(columns=[c for c in ("alpha_A",) if c in P.columns]).merge(
        lab, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left").dropna(subset=["tadv", "alpha_A"])
    adv = P.groupby("symbol")["tadv"].median()

    costs, r2, nmeas = cost_map(adv)
    cons, _, _ = cost_map(adv, conservative=True)
    print(f"cost model: {nmeas} symbols MEASURED, {len(costs)-nmeas} extrapolated from log(ADV) "
          f"(R2={r2:.2f})", flush=True)
    print(f"  measured mean {measured_costs()['cost2k'].mean():.2f} bps | "
          f"full-universe mean {costs.mean():.2f} bps | conservative variant {cons.mean():.2f} bps",
          flush=True)
    print(f"  universe size by tier: " + ", ".join(
        f"N={n if n<999 else 'all'}" for n in TIERS), flush=True)

    for label, cm in (("MEASURED/EXTRAPOLATED", costs), ("CONSERVATIVE (2x unmeasured)", cons)):
        print(f"\n================ {label} ================", flush=True)
        store = {}
        for wname, (t0, t1) in (("SELECT", SEL), ("HOLDOUT", HO)):
            w = P[(P.open_time >= t0) & (P.open_time < t1)]
            print(f"\n  --- {wname} ---", flush=True)
            print(f"    {'universe':<10}{'names/bar':>11}{'gross':>8}{'net':>8}{'net CI':>22}", flush=True)
            for n in TIERS:
                g, net = book_net(w, n, cm)
                if net is None or len(net) < 50:
                    continue
                store[(wname, n)] = net
                lo, hi = block_ci(net.to_numpy())
                nb = w if n >= 999 else w[w.groupby("open_time")["tadv"].rank(
                    ascending=False, method="first") <= n]
                npb = nb.groupby("open_time").size().mean()
                lab_n = "all" if n >= 999 else str(n)
                print(f"    {lab_n:<10}{npb:>11.0f}{sharpe(g):>+8.2f}{sharpe(net):>+8.2f}"
                      f"{f'[{lo:+.2f},{hi:+.2f}] {tag_ci(lo,hi)}':>22}", flush=True)
        base = store.get(("HOLDOUT", 40))
        print(f"\n  paired Δ vs top-40 on the HELD-OUT window:", flush=True)
        for n in TIERS:
            if n == 40 or ("HOLDOUT", n) not in store:
                continue
            v = store[("HOLDOUT", n)]
            idx = base.index.intersection(v.index)
            dd, lo, hi = paired_block_ci(base.loc[idx].to_numpy(), v.loc[idx].to_numpy())
            lab_n = "all" if n >= 999 else str(n)
            print(f"    N={lab_n:<6} Δnet {dd:+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}", flush=True)
        if base is not None and ("HOLDOUT", 999) in store:
            full = store[("HOLDOUT", 999)]
            mult = sharpe(full) / sharpe(base) if sharpe(base) else np.nan
            print(f"\n  realised IR multiplier 40 -> full: {mult:.2f}x   (theory from breadth x4.4: 2.1x)",
                  flush=True)
    print("\nC2DONE", flush=True)


if __name__ == "__main__":
    main()
