"""Beyond-cross-section loop — iteration 4 (B6): test the slow signal with the RIGHT instrument.

B3 found OOS rank-IC rising monotonically with horizon (+0.026 at 1d -> +0.069 at 30d, all CIs excluding 0)
and then failed it on a book Sharpe measured with 27 blocks. That conflated two different things: the IC is
estimated from thousands of name-observations; only the PORTFOLIO SHARPE was underpowered. Writing off the
horizon on that basis was my error.

For a slow signal on limited history the right instrument is the IC with non-overlapping-block CIs, a
chronological hard split, and an explicit IC -> implied-Sharpe translation with breadth stated — not a
directly-measured Sharpe that the sample cannot support.

  G1  hard split: pick H on 2023-06..2024-12 by IC, evaluate on 2025-01..2026-06, block CI must exclude 0
  G2  same sign in both eras at the selected H
  G3  attribution: drop-one over features. If removing mom_s (the short-horizon reversal the incumbent
      already trades) kills the IC, this is the existing edge re-measured, not a new one
  G4  implied Sharpe = IC * sqrt(independent bets/yr), breadth stated, plus the history length that would be
      needed to confirm it directly. Reported as an implication, never as a validated Sharpe.
Falsifier: G1 fails -> the horizon pattern is in-sample and the axis closes.
Run: python3 -u -m live.bx_iter4_slowsignal
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from live.cost_loop_harness import CACHE, pit_adv, tag_ci
from live.build_alpha_beta_decomp import x6
from live.bx_iter3_horizon import daily_panel, build_features, FEATS, walk, mean_block_ci

HS = {"7d": 7, "14d": 14, "30d": 30}
SEL = (pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HO = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"))
ERAS = {"OOS": ("2023-06-01", "2025-10-01"), "RECENT": ("2025-10-01", "2026-07-01")}
NTOP = 40


def preds_for(x, H, t0, t1, feats, tag):
    fp = CACHE / f"bx4_{tag}_{H}_{t0.date()}.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["date"] = pd.to_datetime(d["date"], utc=True); return d
    cuts = pd.date_range(t0, t1, freq="3MS", tz="UTC")
    if len(cuts) < 2:
        return pd.DataFrame()
    import live.bx_iter3_horizon as m
    old = m.FEATS
    m.FEATS = feats
    try:
        P = walk(x, H, cuts)
    finally:
        m.FEATS = old
    if P.empty:
        return P
    P["date"] = pd.to_datetime(P["date"], utc=True)
    P.to_parquet(fp, index=False)
    return P


def ic_of(P, adv=None):
    p = P.dropna(subset=["pred", "y"])
    if adv is not None:
        p = p.merge(adv, on=["symbol", "date"], how="left").dropna(subset=["tadv"])
        p["r"] = p.groupby("date")["tadv"].rank(ascending=False, method="first")
        p = p[p["r"] <= NTOP]
    return p.groupby("date").apply(
        lambda g: spearmanr(g["pred"], g["y"]).correlation if len(g) >= 10 else np.nan).dropna()


def main():
    d = daily_panel()
    A = pit_adv()
    print("=== G1 — hard split: select H on 2023-06..2024-12, evaluate 2025-01..2026-06 ===", flush=True)
    xs = {h: build_features(d, H) for h, H in HS.items()}
    sel_ic, ho_ic = {}, {}
    for hname, H in HS.items():
        x = xs[hname]
        Ps = preds_for(x, H, *SEL, FEATS, "sel")
        Ph = preds_for(x, H, *HO, FEATS, "ho")
        if Ps.empty or Ph.empty:
            print(f"  {hname}: insufficient", flush=True); continue
        s, h_ = ic_of(Ps), ic_of(Ph)
        sel_ic[hname], ho_ic[hname] = s, h_
        slo, shi = mean_block_ci(s.to_numpy(), block=H)
        hlo, hhi = mean_block_ci(h_.to_numpy(), block=H)
        print(f"  {hname:<5} SELECT IC {s.mean():+.4f} [{slo:+.4f},{shi:+.4f}]   "
              f"HELD-OUT IC {h_.mean():+.4f} [{hlo:+.4f},{hhi:+.4f}] {tag_ci(hlo, hhi)}  "
              f"(n={len(h_)} dates)", flush=True)
    if not sel_ic:
        print("no horizons produced predictions"); return
    best = max(sel_ic, key=lambda k: sel_ic[k].mean())
    H = HS[best]
    hlo, hhi = mean_block_ci(ho_ic[best].to_numpy(), block=H)
    g1 = hlo > 0
    print(f"\n  selected by SELECT-window IC: {best}", flush=True)
    print(f"  HELD-OUT IC {ho_ic[best].mean():+.4f} [{hlo:+.4f},{hhi:+.4f}]  -> G1 "
          f"{'PASS' if g1 else 'FAIL'}", flush=True)

    print(f"\n=== G2 — both eras at H={best} ===", flush=True)
    x = xs[best]
    era_ic = {}
    for era, (t0, t1) in ERAS.items():
        P = preds_for(x, H, pd.Timestamp(t0, tz="UTC"), pd.Timestamp(t1, tz="UTC"), FEATS, f"era{era}")
        if P.empty:
            continue
        ic = ic_of(P); era_ic[era] = ic
        lo, hi = mean_block_ci(ic.to_numpy(), block=H)
        ic40 = ic_of(P, A)
        lo4, hi4 = mean_block_ci(ic40.to_numpy(), block=H)
        print(f"  {era:<8} full-universe IC {ic.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {tag_ci(lo, hi)}   "
              f"top-40 IC {ic40.mean():+.4f} [{lo4:+.4f},{hi4:+.4f}] {tag_ci(lo4, hi4)}", flush=True)
    g2 = len(era_ic) == 2 and np.sign(era_ic["OOS"].mean()) == np.sign(era_ic["RECENT"].mean())
    print(f"  G2 {'PASS' if g2 else 'FAIL'}", flush=True)

    print(f"\n=== G3 — attribution: is it just the short-horizon reversal? (H={best}) ===", flush=True)
    base = era_ic.get("OOS")
    for drop in FEATS:
        sub = [f for f in FEATS if f != drop]
        P = preds_for(x, H, pd.Timestamp(ERAS["OOS"][0], tz="UTC"),
                      pd.Timestamp(ERAS["OOS"][1], tz="UTC"), sub, f"drop_{drop}")
        if P.empty:
            continue
        ic = ic_of(P)
        j = pd.concat([base.rename("a"), ic.rename("b")], axis=1).dropna()
        dd = float((j["b"] - j["a"]).mean())
        lo, hi = mean_block_ci((j["b"] - j["a"]).to_numpy(), block=H)
        tagd = "CARRIES" if hi < 0 else ("drop-helps" if lo > 0 else "neutral")
        print(f"  −{drop:<10} IC {ic.mean():+.4f}   Δ {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  {tagd}", flush=True)

    print(f"\n=== G4 — honest IC -> implied Sharpe translation (H={best}) ===", flush=True)
    icv = era_ic["OOS"].mean()
    nbar = 40
    reb = 365.0 / H
    br = nbar * reb
    print(f"  IC {icv:+.4f}, breadth ~{nbar} names x {reb:.1f} rebalances/yr = {br:.0f} bets/yr", flush=True)
    print(f"  naive fundamental law implied Sharpe = IC*sqrt(BR) = {icv*np.sqrt(br):+.2f}", flush=True)
    print("  CAVEAT: the law assumes INDEPENDENT bets. This repo measured n_eff ~97 of 175 names per bar for", flush=True)
    print("  the 4h target, but consecutive slow bets overlap heavily, so true breadth is far below the", flush=True)
    print(f"  nominal {br:.0f}. Treat this as an upper bound, not a forecast.", flush=True)
    need = (2.0 / max(icv, 1e-6)) ** 2 / nbar * H / 365.0
    print(f"  years of history needed to confirm a Sharpe directly at this IC: ~{need:.1f}y "
          f"(we have 5.4y)", flush=True)
    print("\nBXITER4DONE", flush=True)


if __name__ == "__main__":
    main()
