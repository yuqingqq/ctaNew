"""Cost/turnover loop — iteration 1b: WHY did the persistent tilt pass at IC level but fail at portfolio level?

iter1 result: `stat` (persistent tilt) rank-IC ~= `fast` in BOTH eras (measured over the full cross-section),
but as a top-40 quintile book its OOS gross Sharpe is +0.26 vs fast +2.59 (Δ −2.32, CI excludes 0).
Two competing explanations — this script separates them:

  E1 BREADTH. IC is unchanged within the traded universe, but a static book makes ~one long-lived bet instead
     of thousands of independent ones. Same IC, far less breadth -> lower Sharpe. (Predicts: IC(stat) ~= IC(fast)
     inside top-40 too, and stat's book returns are strongly autocorrelated.)
  E2 UNIVERSE. The persistent tilt's information is BETWEEN tiers (calm majors vs racy thin alts). Restricted to
     the top-40 majors there is nothing left to rank in OOS. (Predicts: IC(stat) collapses inside top-40 OOS
     while IC(fast) survives.)

Also reports the stat book's yearly sub-period returns (is the RECENT win one episode?) and lag-1..6
autocorrelation of each book's per-bar return (the breadth proxy).
Run: python3 -u -m live.cl_iter1b_diag
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import (
    ERAS, CACHE, add_slow_signals, block_ci, book, build_panel, cost_map, get_preds,
    hedge_beta, hedged, net_series, restrict_topn, sharpe,
)

SIGS = ["fast", "stat", "dyn", "rvol_slow"]
OTHER = {"RECENT": "OOS", "OOS": "RECENT"}
RNG = np.random.default_rng(5)


def perbar_ic(d, sig):
    return d.groupby("open_time").apply(
        lambda g: spearmanr(g[sig], g["alpha_A"]).correlation if len(g) >= 10 else np.nan).dropna()


def day_ci_diff(a, b, nb=3000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["d"] = j["b"] - j["a"]
    gg = [x["d"].to_numpy() for _, x in j.groupby(pd.to_datetime(j.index, utc=True).floor("1D"))]
    boot = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(j["d"].mean()), *np.percentile(boot, [2.5, 97.5])


def main():
    pc = cost_map()
    PAN = build_panel()
    rvol = PAN[["symbol", "open_time", "rvol_7d"]]
    print("=========== IC INSIDE vs OUTSIDE the traded universe (E1 vs E2) ===========", flush=True)
    store = {}
    for era in ERAS:
        p = add_slow_signals(get_preds(era)).merge(rvol, on=["symbol", "open_time"], how="left")
        p["vrank"] = p.groupby("open_time")["rvol_7d"].rank(pct=True)
        p["rvol_slow"] = -p.sort_values(["symbol", "open_time"]).groupby("symbol")["vrank"].transform(
            lambda s: s.shift(1).expanding(min_periods=30).mean())
        p["fast"] = p["pred"]
        for n, lab in ((40, "top40"), (999, "all")):
            d = restrict_topn(p, n).dropna(subset=SIGS)
            ics = {s: perbar_ic(d, s) for s in SIGS}
            store[(era, lab)] = (d, ics)
            print(f"\n----- {era} / {lab} ({d.symbol.nunique()} syms, {d.open_time.nunique()} bars, "
                  f"{d.groupby('open_time').size().mean():.0f} names/bar) -----", flush=True)
            for s in SIGS:
                line = f"  {s:<10} rank-IC {ics[s].mean():+.4f} (se {ics[s].std()/np.sqrt(len(ics[s])):.4f})"
                if s != "fast":
                    dd, lo, hi = day_ci_diff(ics["fast"], ics[s])
                    line += f"   Δ vs fast {dd:+.4f} [{lo:+.4f},{hi:+.4f}]" \
                            f" {'WORSE' if hi < 0 else ('BETTER' if lo > 0 else 'tie')}"
                print(line, flush=True)

    print("\n=========== BOOK-RETURN AUTOCORRELATION (breadth proxy) ===========", flush=True)
    books = {}
    for era in ERAS:
        for lab, n in (("top40", 40), ("all", 999)):
            d = store[(era, lab)][0]
            for s in ("fast", "stat"):
                W, R, mask = book(d, s)
                books[(era, lab, s)] = net_series(W, R, mask, persym_cost=pc)
    for era in ERAS:
        for lab in ("top40", "all"):
            for s in ("fast", "stat"):
                j = books[(era, lab, s)]
                al = hedged(j, hedge_beta(books[(OTHER[era], lab, s)]))
                ac = [float(al.autocorr(k)) for k in (1, 2, 3, 6, 12)]
                # Sharpe deflation from autocorrelation: independent-bar equivalent
                rho1 = ac[0]
                infl = np.sqrt((1 + rho1) / (1 - rho1)) if abs(rho1) < 0.99 else np.nan
                print(f"  {era:<7}{lab:<7}{s:<6} Sh {sharpe(al):+.2f}  ac(1,2,3,6,12) "
                      + " ".join(f"{a:+.3f}" for a in ac) + f"   var-inflation x{infl:.2f}", flush=True)

    print("\n=========== stat BOOK BY CALENDAR YEAR (is the win one episode?) ===========", flush=True)
    for era in ERAS:
        for lab in ("top40", "all"):
            j = books[(era, lab, "stat")]
            al = hedged(j, hedge_beta(books[(OTHER[era], lab, "stat")]))
            yr = al.groupby(al.index.year).agg(["mean", "count"])
            cells = "  ".join(f"{int(y)}: {r['mean']*1e4:+.1f}bps/bar n={int(r['count'])}"
                              for y, r in yr.iterrows())
            print(f"  {era:<7}{lab:<7} {cells}", flush=True)

    print("\n=========== fast BOOK: same, for reference ===========", flush=True)
    for era in ERAS:
        for lab in ("top40", "all"):
            j = books[(era, lab, "fast")]
            al = hedged(j, hedge_beta(books[(OTHER[era], lab, "fast")]))
            yr = al.groupby(al.index.year).agg(["mean", "count"])
            cells = "  ".join(f"{int(y)}: {r['mean']*1e4:+.1f}bps/bar n={int(r['count'])}"
                              for y, r in yr.iterrows())
            print(f"  {era:<7}{lab:<7} {cells}", flush=True)
    print("\nITER1BDONE", flush=True)


if __name__ == "__main__":
    main()
