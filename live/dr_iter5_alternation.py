"""Detail-review loop — iteration 5 (D5): does the sleeve alternation have a PREDICTABLE driver?

"Everything alternates between windows" has now shown up three independent times:
  sleeve A +2.21 -> +0.69 while sleeve B +0.45 -> +1.04
  by half-year: A +1.05 / +1.60 / -0.69   B +1.00 / -0.26 / +2.57
  long_only best in SELECT, worst in HOLDOUT
I have twice concluded "you cannot tell which will work, so combine" — WITHOUT ever testing whether you can.
If the alternation has a state driver, timing it is worth more than either sleeve.

TEST. Build both sleeves' daily net series over the full 2023-06 -> 2026-07 span, build PIT state variables
from the same panel, and ask whether the A-minus-B return spread is conditionally predictable:
  A  spread by state tercile, both eras, day-clustered CI
  B  a state-conditional allocation rule (tilt to whichever sleeve the state favours) vs fixed 50/50,
     with the rule FROZEN on 2023-06..2025-01 and evaluated on 2025-01..2026-07

State variables, all trailing and shifted (no look-ahead):
  mkt_vol_30d    realised vol of the equal-weight basket
  xs_disp_20d    mean cross-sectional dispersion of daily returns
  mkt_trend_30d  trailing basket return
  avg_corr_30d   basket vol / mean single-name vol (a co-movement proxy)

Gates: G1 the A-B spread differs across state terciles with day-clustered CI excluding 0, in BOTH eras;
G2 the frozen conditional rule beats fixed 50/50 held-out, paired block CI > 0.
Falsifier: G1 fails -> the alternation is genuinely unpredictable, fixed weights are correct, and the
"just combine" conclusion is right for the right reason rather than by default.
Run: python3 -u -m live.dr_iter5_alternation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import CACHE, block_ci, paired_block_ci, tag_ci
from live.dr_iter1_shortonly import load
from live.dr_iter2_delist import CACHE as _C  # noqa: F401  (keeps cache dir import consistent)
from live.dp_phase1_consolidate import sharpe_d
from live.dr_iter1_shortonly import legs, series, est_beta
from live.mc_oi_universe import topn, N as NTOP
from live.bx_iter3_horizon import daily_panel

SPAN = (pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"))
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
RNG = np.random.default_rng(91)


def sleeve_series(P):
    """Daily net series for the reversal sleeve (A) and the momentum sleeve (B) over the whole span."""
    from live.dp_phase1_consolidate import sleeve_momentum
    from live.dp_phase1_review import sleeve_reversal_basis
    import live.dp_phase1_consolidate as C
    import live.dp_phase1_review as R
    out = {}
    for nm, (t0, t1) in (("early", (SPAN[0], SPLIT)), ("late", (SPLIT, SPAN[1]))):
        C.HO0, C.HO1 = t0, t1
        R.HO0, R.HO1 = t0, t1
        a = sleeve_reversal_basis(2.0, "raw")
        b = sleeve_momentum(2.0)
        out[nm] = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    return pd.concat([out["early"], out["late"]]).sort_index()


def states(d):
    """PIT state variables from the daily panel."""
    r = d.pivot_table(index="date", columns="symbol", values="ret_1d")
    basket = r.mean(axis=1)
    S = pd.DataFrame(index=r.index)
    S["mkt_vol_30d"] = basket.rolling(30, min_periods=15).std().shift(1)
    S["xs_disp_20d"] = r.std(axis=1).rolling(20, min_periods=10).mean().shift(1)
    S["mkt_trend_30d"] = basket.rolling(30, min_periods=15).sum().shift(1)
    mean_name_vol = r.std(axis=0)  # placeholder, replaced below
    nv = r.rolling(30, min_periods=15).std().mean(axis=1)
    S["avg_corr_30d"] = (basket.rolling(30, min_periods=15).std() / nv.replace(0, np.nan)).shift(1)
    return S


def day_ci(x, nb=3000):
    a = np.asarray(x, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return (np.nan, np.nan)
    b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(nb)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    print("building both sleeves over the full span...", flush=True)
    P = load()
    J = sleeve_series(P)
    J.index = pd.to_datetime(J.index, utc=True)
    J["spread"] = J["A"] - J["B"]
    print(f"  {len(J)} days, {J.index.min().date()} -> {J.index.max().date()}", flush=True)
    print(f"  A Sharpe {sharpe_d(J['A']):+.2f} | B {sharpe_d(J['B']):+.2f} | corr {J['A'].corr(J['B']):+.3f}",
          flush=True)

    d = daily_panel()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    S = states(d)
    S.index = pd.to_datetime(S.index, utc=True)
    X = J.join(S, how="inner").dropna()
    print(f"  joined with state: {len(X)} days\n", flush=True)

    print("=== A — A-minus-B spread by state tercile (day-clustered CI) ===", flush=True)
    svars = ["mkt_vol_30d", "xs_disp_20d", "mkt_trend_30d", "avg_corr_30d"]
    good = []
    for v in svars:
        print(f"  --- {v} ---", flush=True)
        ok_both = True
        for era, m in (("early", X.index < SPLIT), ("late", X.index >= SPLIT)):
            e = X[m]
            if len(e) < 60:
                continue
            q = pd.qcut(e[v], 3, labels=["low", "mid", "high"], duplicates="drop")
            cells = []
            for lab in ["low", "high"]:
                s = e.loc[q == lab, "spread"]
                if len(s) < 20:
                    continue
                lo, hi = day_ci(s)
                cells.append(f"{lab} {s.mean()*1e4:+.1f}bps[{lo*1e4:+.1f},{hi*1e4:+.1f}]")
            hi_s = e.loc[q == "high", "spread"]; lo_s = e.loc[q == "low", "spread"]
            diff = hi_s.mean() - lo_s.mean()
            dlo, dhi = day_ci(np.concatenate([hi_s.to_numpy(), -lo_s.to_numpy()]))
            sig = (dlo > 0) or (dhi < 0)
            ok_both &= sig
            print(f"    {era:<6}" + "  ".join(cells) +
                  f"   high-low {diff*1e4:+.1f}bps {'SIG' if sig else 'ns'}", flush=True)
        if ok_both:
            good.append(v)
    print(f"\n  G1 state vars significant in BOTH eras: {good if good else 'NONE'}", flush=True)

    print("\n=== B — frozen conditional allocation vs fixed 50/50 (held-out) ===", flush=True)
    early, late = X[X.index < SPLIT], X[X.index >= SPLIT]
    fixed = 0.5 * late["A"] + 0.5 * late["B"]
    lo, hi = block_ci(fixed.to_numpy(), block=7)
    print(f"  fixed 50/50            Sharpe {sharpe_d(fixed):+.2f} [{lo:+.2f},{hi:+.2f}]", flush=True)
    for v in svars:
        thr = early[v].median()                     # rule frozen on the early window
        favours_A = early.loc[early[v] > thr, "spread"].mean() > 0
        w = np.where(late[v] > thr, 0.75 if favours_A else 0.25, 0.25 if favours_A else 0.75)
        cond = w * late["A"] + (1 - w) * late["B"]
        dd, dlo, dhi = paired_block_ci(fixed.to_numpy(), cond.to_numpy(), block=7)
        print(f"  tilt on {v:<15} Sharpe {sharpe_d(cond):+.2f}   Δ vs fixed {dd:+.2f} "
              f"[{dlo:+.2f},{dhi:+.2f}] {tag_ci(dlo, dhi)}", flush=True)
    print("\nDRITER5DONE", flush=True)


if __name__ == "__main__":
    main()
