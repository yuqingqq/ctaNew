"""Cost/turnover loop — iteration 1 (direction D1).

H1: because cost scales with turnover and ~90% of the per-symbol-Ridge rank-IC sits in a PERSISTENT per-name
tilt, a slow book keeps most of the gross alpha and converts materially more of it to net than the 4h book.

Signals (all PIT, same walk-forward preds):
  fast       incumbent pred                              (turnover ~0.40/bar)
  stat       shift-1 expanding mean of own past preds    (the persistent tilt)
  stat_ew    30d-halflife EWMA of own past preds         (slow-dynamic control)
  stat_froz  tilt frozen after era's first 90d           (A1 staleness control; evaluated post-freeze only)
  rvol_slow  −expanding mean of xs-rank(rvol_7d)         (A4: is it just a low-vol sort?)
  fast_ewma  incumbent pred with λ=0.7 weight smoothing  (validated incumbent turnover control, for reference)

Universes: PIT trailing-ADV top-40 (the deployable one) and full eligible (reference).
Costs: per-symbol calibrated depth cost ($10k clip) + flat grid 6/8/12/24 bps.
Hedge: ERA-LOCKED beta (fit on the other era) — never in-era.
Stats: 7d-block bootstrap on Sharpe; PAIRED block bootstrap on Δ(variant − fast) on common bars.

Gates (pre-registered, live/COST_TURNOVER_LOOP.md):
  G1 slow-book gross Sharpe CI excludes 0 in BOTH eras
  G2 Δ(slow − fast) net Sharpe @8bps CI excludes 0 in BOTH eras on top-40
  G3 if rvol_slow >= stat, the honest read is "factor sort", not "the ML tilt"
Falsifier: G1 fails in either era, or G2 spans 0 in both → NULL.

Run: python3 -u -m live.cl_iter1_static
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import (
    ERAS, CACHE, add_slow_signals, block_ci, book, build_panel, cost_map, get_preds,
    hedge_beta, hedged, maxdd, net_series, paired_block_ci, restrict_topn, sharpe, smooth_ewma, tag_ci,
)

SIGS = ["fast", "stat", "stat_ew", "stat_froz", "rvol_slow"]
TIERS = [40, 999]
FLAT = [6, 8, 12, 24]
OTHER = {"RECENT": "OOS", "OOS": "RECENT"}


def prep(era: str, rvol: pd.DataFrame) -> pd.DataFrame:
    p = add_slow_signals(get_preds(era))
    p = p.merge(rvol, on=["symbol", "open_time"], how="left")
    p["vrank"] = p.groupby("open_time")["rvol_7d"].rank(pct=True)
    p["rvol_slow"] = -p.sort_values(["symbol", "open_time"]).groupby("symbol")["vrank"].transform(
        lambda s: s.shift(1).expanding(min_periods=30).mean())
    p["fast"] = p["pred"]
    return p


def main():
    pc = cost_map()
    PAN = build_panel()
    rvol = PAN[["symbol", "open_time", "rvol_7d"]].copy()
    store, meta = {}, {}
    for era in ERAS:
        p = prep(era, rvol)
        print(f"[{era}] rows {len(p):,} syms {p.symbol.nunique()} bars {p.open_time.nunique()}", flush=True)
        for n in TIERS:
            d = restrict_topn(p, n)
            for sig in SIGS:
                W, R, mask = book(d, sig)
                store[(era, n, sig)] = net_series(W, R, mask, persym_cost=pc)
                meta[(era, n, sig)] = dict(names=float((W.abs() > 0).sum(axis=0).mean()),
                                           hold=float((W.abs() > 0).mean(axis=1).max()))
            W, R, mask = book(d, "fast")
            store[(era, n, "fast_ewma")] = net_series(smooth_ewma(W, mask, 0.7), R, mask, persym_cost=pc)
            meta[(era, n, "fast_ewma")] = meta[(era, n, "fast")]
        print(f"[{era}] books built", flush=True)

    rows = []
    for era in ERAS:
        for n in TIERS:
            for sig in SIGS + ["fast_ewma"]:
                j = store[(era, n, sig)]
                beta = hedge_beta(store[(OTHER[era], n, sig)])          # era-locked
                al = hedged(j, beta)
                glo, ghi = block_ci(al)
                r = dict(era=era, tier=("top40" if n < 999 else "all"), sig=sig, bars=len(j),
                         turn=float(j["t"].mean()), beta=beta, gross=sharpe(al), g_lo=glo, g_hi=ghi,
                         names=meta[(era, n, sig)]["names"], maxdd_bps=maxdd(al) * 1e4)
                ps = (al - j["c_ps"])
                lo, hi = block_ci(ps)
                r.update(net_ps=sharpe(ps), ps_lo=lo, ps_hi=hi,
                         cost_ps_bps=float(j["c_ps"].mean() * 1e4))
                for c in FLAT:
                    net = al - j["t"] * c / 1e4
                    r[f"net{c}"] = sharpe(net)
                rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(CACHE / "iter1_results.csv", index=False)

    print("\n================ LEVELS (era-locked hedge, 7d-block CI) ================", flush=True)
    for era in ERAS:
        for n in TIERS:
            tier = "top40" if n < 999 else "all"
            print(f"\n----- {era} / {tier} -----", flush=True)
            print(f"  {'signal':<10}{'turn':<7}{'names':<7}{'grossSh [CI]':<26}{'net@8':<8}{'net@24':<8}"
                  f"{'net@ps [CI]':<26}{'psCost':<8}{'maxDD':<9}", flush=True)
            for sig in SIGS + ["fast_ewma"]:
                x = T[(T.era == era) & (T.tier == tier) & (T.sig == sig)].iloc[0]
                print(f"  {sig:<10}{x.turn:<7.3f}{x.names:<7.1f}"
                      f"{f'{x.gross:+.2f} [{x.g_lo:+.2f},{x.g_hi:+.2f}] {tag_ci(x.g_lo, x.g_hi)}':<26}"
                      f"{x.net8:<+8.2f}{x.net24:<+8.2f}"
                      f"{f'{x.net_ps:+.2f} [{x.ps_lo:+.2f},{x.ps_hi:+.2f}] {tag_ci(x.ps_lo, x.ps_hi)}':<26}"
                      f"{x.cost_ps_bps:<8.2f}{x.maxdd_bps:<9.0f}", flush=True)

    print("\n================ PAIRED Δ vs fast (common bars, paired block boot) ================", flush=True)
    prows = []
    for era in ERAS:
        for n in TIERS:
            tier = "top40" if n < 999 else "all"
            jf = store[(era, n, "fast")]
            bf = hedge_beta(store[(OTHER[era], n, "fast")])
            print(f"\n----- {era} / {tier} -----", flush=True)
            print(f"  {'variant':<10}{'Δgross [CI]':<28}{'Δnet@8 [CI]':<28}{'Δnet@24 [CI]':<28}"
                  f"{'Δnet@ps [CI]':<28}", flush=True)
            for sig in SIGS[1:] + ["fast_ewma"]:
                jv = store[(era, n, sig)]
                bv = hedge_beta(store[(OTHER[era], n, sig)])
                idx = jf.index.intersection(jv.index)
                a, b = jf.loc[idx], jv.loc[idx]
                fa, va = hedged(a, bf), hedged(b, bv)
                cells, rec = [], dict(era=era, tier=tier, sig=sig, bars=len(idx))
                for name, fx, vx in (("gross", fa, va),
                                     ("net8", fa - a["t"] * 8e-4, va - b["t"] * 8e-4),
                                     ("net24", fa - a["t"] * 24e-4, va - b["t"] * 24e-4),
                                     ("netps", fa - a["c_ps"], va - b["c_ps"])):
                    dd, lo, hi = paired_block_ci(fx.to_numpy(), vx.to_numpy())
                    cells.append(f"{dd:+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}")
                    rec.update({f"d_{name}": dd, f"d_{name}_lo": lo, f"d_{name}_hi": hi})
                prows.append(rec)
                print(f"  {sig:<10}" + "".join(f"{c:<28}" for c in cells), flush=True)
    pd.DataFrame(prows).to_csv(CACHE / "iter1_paired.csv", index=False)

    print("\n================ GATE READ ================", flush=True)
    for sig in ("stat", "stat_ew", "rvol_slow"):
        g1 = all(T[(T.era == e) & (T.tier == "top40") & (T.sig == sig)].iloc[0].g_lo > 0 for e in ERAS)
        P = pd.DataFrame(prows)
        g2 = all(P[(P.era == e) & (P.tier == "top40") & (P.sig == sig)].iloc[0].d_net8_lo > 0 for e in ERAS)
        print(f"  {sig:<10} G1(gross CI>0 both eras, top40) {'PASS' if g1 else 'FAIL'}   "
              f"G2(Δnet@8 CI>0 both eras, top40) {'PASS' if g2 else 'FAIL'}", flush=True)
    print("\nITER1DONE", flush=True)


if __name__ == "__main__":
    main()
