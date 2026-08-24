"""Cost/turnover loop — iteration 5: chronological HARD SPLIT (fixes iter4's power + selection problems).

iter4 could not distinguish "does not work in RECENT" from "RECENT (1,400 bars) cannot measure it", and its
winners were selected on the same OOS window they were scored on. Fix: select on one window, evaluate on a
disjoint later one.

  SELECT   2023-06-01 -> 2024-12-31   pick the single best config by net Sharpe @cost_10k
  HOLDOUT  2025-01-01 -> 2026-06-30   evaluate that one config; nothing here informs the choice

Grid is the same pre-registered 3x3 (tier x turnover-control) as iter4. All 9 reported on both windows for
transparency; only the pre-committed selection rule counts. Books on per-name BTC-residual returns.

Gates: G1 held-out net CI>0 @cost_10k; G2 same @cost_50k; G3 (texture) sign-consistent by calendar year.
Falsifier: G1 fails -> iter4's OOS positives do not survive honest selection; deployability branch closes negative.
Run: python3 -u -m live.cl_iter5_hardsplit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import (
    ERAS, CACHE, block_ci, build_panel, get_preds, maxdd, restrict_topn, sharpe, tag_ci,
)
from live.cl_iter4_capacity import TIERS, CONTROLS, CLIPS, build, cost_tiers

SEL0, SEL1 = pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")


def evaluate(d: pd.DataFrame, ctl: str, CT: dict) -> dict:
    W, A = build(d, ctl)
    g = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    out = dict(bars=len(g) - 1, names=float((W.abs() > 1e-9).sum(axis=0).mean()),
               turn=float((0.25 * dW.sum(axis=0)).iloc[1:].mean()),
               gross=sharpe(g.iloc[1:]), maxdd=maxdd(g.iloc[1:]) * 1e4)
    out["g_lo"], out["g_hi"] = block_ci(g.iloc[1:].to_numpy())
    for clip in CLIPS:
        c, med = CT[clip]
        cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
        net = (g - 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4).iloc[1:]
        out[f"net_{clip}"] = sharpe(net)
        out[f"lo_{clip}"], out[f"hi_{clip}"] = block_ci(net.to_numpy())
        out[f"series_{clip}"] = net
    return out


def main():
    CT = cost_tiers()
    PAN = build_panel()
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    parts = []
    for era in ERAS:
        p = get_preds(era)
        if "alpha_A" not in p.columns:
            p = p.merge(lab, on=["symbol", "open_time"], how="left")
        parts.append(p)
    P = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "open_time"]).sort_values(
        ["symbol", "open_time"])
    print(f"pooled walk-forward preds: {len(P):,} rows, {P.open_time.min().date()} -> "
          f"{P.open_time.max().date()}, {P.symbol.nunique()} syms", flush=True)

    res = {}
    for wname, (t0, t1) in (("SELECT", (SEL0, SEL1)), ("HOLDOUT", (HO0, HO1))):
        w = P[(P.open_time >= t0) & (P.open_time < t1)]
        print(f"\n[{wname}] {t0.date()} -> {t1.date()}: {w.open_time.nunique()} bars", flush=True)
        for n, tl in TIERS:
            d = restrict_topn(w, n)
            for ctl in CONTROLS:
                res[(wname, tl, ctl)] = evaluate(d, ctl, CT)

    for wname in ("SELECT", "HOLDOUT"):
        print(f"\n============ {wname} — net Sharpe by clip tier (7d-block CI) ============", flush=True)
        print(f"  {'tier':<8}{'control':<10}{'bars':<7}{'turn':<7}{'gross':<8}"
              f"{'net@10k [CI]':<27}{'net@50k [CI]':<27}{'net@100k':<9}{'maxDD':<8}", flush=True)
        for _, tl in TIERS:
            for ctl in CONTROLS:
                r = res[(wname, tl, ctl)]
                c10 = f"{r['net_cost_10k']:+.2f} [{r['lo_cost_10k']:+.2f},{r['hi_cost_10k']:+.2f}] " \
                      f"{tag_ci(r['lo_cost_10k'], r['hi_cost_10k'])}"
                c50 = f"{r['net_cost_50k']:+.2f} [{r['lo_cost_50k']:+.2f},{r['hi_cost_50k']:+.2f}] " \
                      f"{tag_ci(r['lo_cost_50k'], r['hi_cost_50k'])}"
                print(f"  {tl:<8}{ctl:<10}{r['bars']:<7}{r['turn']:<7.3f}{r['gross']:<+8.2f}"
                      f"{c10:<27}{c50:<27}{r['net_cost_100k']:<+9.2f}{r['maxdd']:<8.0f}", flush=True)

    # ---- pre-committed selection rule ----
    cands = [(tl, ctl) for _, tl in TIERS for ctl in CONTROLS]
    best = max(cands, key=lambda k: res[("SELECT", k[0], k[1])]["net_cost_10k"])
    tl, ctl = best
    sel = res[("SELECT", tl, ctl)]; ho = res[("HOLDOUT", tl, ctl)]
    print(f"\n============ PRE-COMMITTED SELECTION ============", flush=True)
    print(f"  selected on 2023-06..2024-12 by net@10k: {tl}/{ctl}  "
          f"(select net@10k {sel['net_cost_10k']:+.2f}, gross {sel['gross']:+.2f}, turn {sel['turn']:.3f})",
          flush=True)
    print(f"\n  HELD-OUT 2025-01..2026-06 ({ho['bars']} bars, {ho['names']:.1f} names/bar):", flush=True)
    print(f"    gross      {ho['gross']:+.2f} [{ho['g_lo']:+.2f},{ho['g_hi']:+.2f}] "
          f"{tag_ci(ho['g_lo'], ho['g_hi'])}", flush=True)
    for clip in CLIPS:
        print(f"    net@{clip[5:]:<6} {ho[f'net_{clip}']:+.2f} [{ho[f'lo_{clip}']:+.2f},"
              f"{ho[f'hi_{clip}']:+.2f}] {tag_ci(ho[f'lo_{clip}'], ho[f'hi_{clip}'])}", flush=True)

    print("\n  G3 texture — held-out net@10k by calendar year:", flush=True)
    s = ho["series_cost_10k"]
    yr = s.groupby(s.index.year)
    for y, v in yr:
        print(f"    {y}: Sharpe {sharpe(v):+.2f}  mean {v.mean()*1e4:+.2f} bps/bar  n={len(v)}", flush=True)
    signs = [np.sign(sharpe(v)) for _, v in yr]

    g1 = ho["lo_cost_10k"] > 0
    g2 = ho["lo_cost_50k"] > 0
    g3 = len(set(signs)) == 1
    print(f"\n============ GATE READ ============", flush=True)
    print(f"  G1 held-out net@10k CI>0 : {'PASS' if g1 else 'FAIL'}", flush=True)
    print(f"  G2 held-out net@50k CI>0 : {'PASS' if g2 else 'FAIL'}", flush=True)
    print(f"  G3 year-sign consistency : {'PASS' if g3 else 'FAIL'}", flush=True)

    rows = []
    for (w, t, c), r in res.items():
        rows.append({k: v for k, v in r.items() if not k.startswith("series_")}
                    | {"window": w, "tier": t, "ctl": c})
    pd.DataFrame(rows).to_csv(CACHE / "iter5_hardsplit.csv", index=False)
    print("\nITER5DONE", flush=True)


if __name__ == "__main__":
    main()
