"""Detail-review loop — iteration 7: audit the FOUNDATION, not the constructions on top of it.

Every number in this session — the +1.26 combined book, the cost curve's payoff, the concentration sweep,
the sleeve correlation — rests on `get_preds()`, a cached set of walk-forward per-symbol Ridge predictions I
have used as ground truth and never verified. This repo's own history is a catalogue of look-ahead found
late: a liquidity gate reading same-day volume (~0.3 Sharpe), VPIN bucket sizing off the full dataset, a
label shifted by 1 instead of the horizon, a venue premium inflated by bar misalignment. Auditing the
constructions while trusting the base is the wrong order.

FIVE CHECKS, cheapest and most decisive first:

  F1 LOOK-AHEAD IN THE PREDICTION. CLAUDE.md's own rule: anything with |IC| > 0.10 against a forward return
     is suspicious. Test pred against (a) the true forward label, (b) the label shifted +1 bar — a model with
     no leakage should score ~0 on (b) — and (c) the CONTEMPORANEOUS and PAST return, which it must not know.
  F2 PURGE / EMBARGO. Verify directly that no training row's exit_time reaches into any test window, and
     that the 1-day embargo is actually enforced in the cached preds' fold structure.
  F3 CACHE INTEGRITY. Recompute one era's predictions from scratch and compare to the cache. If the cache was
     written by an older version of the pipeline, everything downstream is measuring something else.
  F4 LABEL SANITY. alpha_A should be ~zero-mean cross-sectionally and its scale should match a 4h residual.
     A drifting or mis-scaled label silently changes every Sharpe.
  F5 SURVIVORSHIP. How many symbols in the panel stop trading mid-sample, and does the book hold them into
     the gap? Delisted names that vanish from the panel bias a short-heavy book upward.

Run: python3 -u -m live.dr_iter7_foundation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import CACHE, ERAS, CUTS, build_panel, get_preds
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.v0_feature_ablation import V0

EMB = pd.Timedelta(days=1)


def perbar_ic(d, a, b):
    return d.groupby("open_time").apply(
        lambda g: spearmanr(g[a], g[b]).correlation if len(g) >= 10 else np.nan).dropna().mean()


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    lab = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"})[
        ["symbol", "open_time", "exit_time", "alpha_A"]]

    print("=== F1 — look-ahead in the cached predictions ===", flush=True)
    print("    rule (CLAUDE.md): |IC| > 0.10 against any forward return is suspicious;", flush=True)
    print("    a clean model scores ~0 against the +1-bar-shifted label and against PAST returns\n", flush=True)
    for era in ERAS:
        # get_preds already carries alpha_A and return_pct; merging `lab` would duplicate them
        P = get_preds(era).copy()
        for c in ("alpha_A", "return_pct"):
            if c not in P.columns:
                src = lab if c == "alpha_A" else RP
                P = P.merge(src[["symbol", "open_time", c]], on=["symbol", "open_time"], how="left")
        P = P.sort_values(["symbol", "open_time"])
        g = P.groupby("symbol")
        P["fwd1"] = g["alpha_A"].shift(-1)      # the NEXT bar's label — must be ~0
        P["lag1"] = g["alpha_A"].shift(1)       # the PREVIOUS bar's label — must be ~0
        P["contemp"] = P["return_pct"]          # same-bar raw return (the model must not know it)
        P["past_ret"] = g["return_pct"].shift(1)
        row = []
        for c in ("alpha_A", "fwd1", "lag1", "past_ret"):
            s = P.dropna(subset=["pred", c])
            row.append(f"{c} {perbar_ic(s, 'pred', c):+.4f}")
        print(f"  {era:<8}" + "   ".join(row), flush=True)
    print("    (alpha_A is the true target; the other three should all be near zero)", flush=True)

    print("\n=== F2 — purge / embargo actually enforced? ===", flush=True)
    for era in ERAS:
        cuts = CUTS[era]
        worst = None
        for i in range(len(cuts) - 1):
            c0 = cuts[i]; fc = c0 - EMB
            tr = PAN[(PAN.exit_time < fc)]
            if tr.empty:
                continue
            leak = (tr["exit_time"].max() - c0).total_seconds() / 3600
            worst = leak if worst is None else max(worst, leak)
        print(f"  {era:<8} max train exit_time minus test start: {worst:+.1f} h "
              f"({'OK — purged' if worst is not None and worst < 0 else 'LEAK'})", flush=True)

    print("\n=== F3 — cache integrity: recompute vs cached ===", flush=True)
    era = "RECENT"
    cached = get_preds(era)[["symbol", "open_time", "pred"]].rename(columns={"pred": "cached"})
    fresh = gen_pred(PAN, list(V0), CUTS[era])
    fresh["open_time"] = pd.to_datetime(fresh["open_time"], utc=True)
    m = cached.merge(fresh.rename(columns={"pred": "fresh"}), on=["symbol", "open_time"], how="inner")
    if len(m):
        c = float(np.corrcoef(m["cached"], m["fresh"])[0, 1])
        md = float((m["cached"] - m["fresh"]).abs().max())
        print(f"  {era}: {len(m):,} overlapping preds  corr {c:.6f}  max abs diff {md:.2e}  "
              f"{'MATCH' if c > 0.9999 and md < 1e-6 else 'DIFFERS — cache is stale'}", flush=True)
    else:
        print("  no overlap to compare", flush=True)

    print("\n=== F4 — label sanity ===", flush=True)
    a = PAN.groupby("open_time")["alpha_vs_btc_realized"]
    xs_mean = a.transform("mean")
    print(f"  cross-sectional mean of alpha_A: avg {float(xs_mean.mean())*1e4:+.3f} bps "
          f"(should be ~0 by construction of a residual)", flush=True)
    print(f"  alpha_A sd {float(PAN['alpha_vs_btc_realized'].std())*1e4:.0f} bps per 4h bar "
          f"| skew {float(PAN['alpha_vs_btc_realized'].skew()):+.2f}", flush=True)
    print(f"  z_res sd {float(PAN['z_res'].std()):.3f} (should be ~1)  "
          f"| clipped at +/-10: {100*float((PAN['z_res'].abs() >= 9.99).mean()):.3f}%", flush=True)

    print("\n=== F5 — survivorship: do symbols vanish mid-sample? ===", flush=True)
    last = PAN.groupby("symbol")["open_time"].max()
    end = PAN["open_time"].max()
    dead = last[last < end - pd.Timedelta(days=30)]
    print(f"  {len(dead)} of {PAN.symbol.nunique()} symbols stop >30d before the panel end", flush=True)
    if len(dead):
        print(f"    earliest exits: " + ", ".join(f"{s} {t.date()}" for s, t in dead.nsmallest(5).items()),
              flush=True)
        # do the dead names carry unusual final returns? (a short book profits if they collapse then vanish)
        tailret = []
        for s, t in dead.items():
            w = PAN[(PAN.symbol == s) & (PAN.open_time > t - pd.Timedelta(days=10))]
            if len(w) > 5:
                tailret.append(float(w["alpha_vs_btc_realized"].mean()))
        if tailret:
            print(f"    mean alpha_A in their final 10 days: {np.mean(tailret)*1e4:+.1f} bps/bar "
                  f"(n={len(tailret)}) — strongly negative would flatter a short book", flush=True)
    print("\nDRITER7DONE", flush=True)


if __name__ == "__main__":
    main()
