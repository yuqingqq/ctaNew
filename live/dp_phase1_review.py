"""Adversarial review of the Phase 1 consolidation result (+1.60 Sharpe, P(Sh<0) 3.4%).

Four problems I found reviewing my own work. Three are testable; this tests them.

  R1  INCONSISTENT P&L ACCOUNTING between the sleeves. Sleeve A is marked on `alpha_A` — the BTC-beta
      RESIDUAL return — which silently assumes a beta hedge that is free to run. Sleeve B is marked on RAW
      daily returns, dollar-neutral. Combining them mixes two different accounting bases. Rebuild sleeve A on
      raw returns and re-measure.
  R2  THE PASSIVE COST WAS MEASURED ON THE HOLDOUT ITSELF. `maker_exec_probe.py` ran on 2025-01..2026-07 —
      the same window Phase 1 evaluates. Using it to set the 4.2 bps assumption leaks holdout information
      into the cost model. Report the whole cost grid instead of one number.
  R3  RETURN CONCENTRATION. A 39% -vol book over 15 months can be carried by a handful of days. Drop the best
      N days and see what survives.
  R4  (not testable, stated in the writeup) SLEEVE SELECTION. The decision to combine these two sleeves was
      made after seeing which of five survived. That is a selection effect no in-sample statistic can undo.

Run: python3 -u -m live.dp_phase1_review
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import ERAS, build_panel, get_preds, pit_adv, tag_ci
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP
from live.dp_phase1_consolidate import (HO0, HO1, sleeve_momentum, sharpe_d, maxdd, boot_sharpe)

RNG = np.random.default_rng(202)


def sleeve_reversal_basis(cost_bps, basis="residual"):
    """Sleeve A marked on either the BTC-beta residual (as Phase 1 did) or RAW returns (R1)."""
    CT = cost_tiers(); c10, cmed = CT["cost_10k"]
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    lab = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"})[["symbol", "open_time", "alpha_A"]]
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    P = P.drop(columns=[c for c in ("alpha_A", "return_pct") if c in P.columns]).merge(
        lab, on=["symbol", "open_time"], how="left").merge(RP, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left")
    ho = topn(P[(P.open_time >= HO0) & (P.open_time < HO1)].dropna(
        subset=["tadv", "alpha_A", "return_pct"]), "tadv", NTOP)
    W, Aa = build(ho, "band")
    if basis == "raw":
        R = ho.pivot_table(index="symbol", columns="open_time", values="return_pct").reindex_like(Aa).fillna(0.0)
        g = (W * R).sum(axis=0)
    else:
        g = (W * Aa).sum(axis=0)
    dW = W.diff(axis=1).abs()
    if cost_bps is None:
        kv = pd.Series([float(c10.get(s, cmed)) for s in W.index], index=W.index)
        ch = 0.25 * dW.mul(kv, axis=0).sum(axis=0) / 1e4
    else:
        ch = 0.25 * dW.sum(axis=0) * cost_bps / 1e4
    net = (g - ch).iloc[1:]
    net.index = pd.to_datetime(net.index, utc=True)
    return net.groupby(net.index.floor("1D")).sum()


def stat(s, label):
    b = boot_sharpe(s.to_numpy())
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"  {label:<34}{sharpe_d(s):>+7.2f}  [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi):<8}"
          f"maxDD {maxdd(s)*100:>7.1f}%   boot P(<0) {float((b < 0).mean())*100:>5.1f}%", flush=True)
    return sharpe_d(s)


def main():
    print("=== R1 — does the result survive CONSISTENT P&L accounting? ===", flush=True)
    print("    Phase 1 marked sleeve A on the BTC-beta residual (assumes a free beta hedge);", flush=True)
    print("    sleeve B on raw dollar-neutral returns. Rebuild A on raw returns to match.\n", flush=True)
    mom = sleeve_momentum(4.2)
    for basis in ("residual", "raw"):
        rev = sleeve_reversal_basis(4.2, basis)
        j = pd.concat([rev.rename("rev"), mom.rename("mom")], axis=1).dropna()
        w = 1.0 / j.std(); w = w / w.sum()
        comb = (j * w).sum(axis=1)
        print(f"  --- sleeve A marked on {basis.upper()} ---", flush=True)
        stat(j["rev"], "A alone")
        stat(comb, "A+B equal-risk")
        print(f"      sleeve correlation {float(j['rev'].corr(j['mom'])):+.3f}\n", flush=True)

    print("=== R2 — cost sensitivity (the 4.2 bps figure was measured ON the holdout) ===", flush=True)
    rev_r = {c: sleeve_reversal_basis(c, "raw") for c in (0.0, 4.2, 6.0, 8.35, 12.0)}
    for c in (0.0, 4.2, 6.0, 8.35, 12.0):
        m = sleeve_momentum(c if c > 0 else 0.01)
        j = pd.concat([rev_r[c].rename("rev"), m.rename("mom")], axis=1).dropna()
        w = 1.0 / j.std(); w = w / w.sum()
        stat((j * w).sum(axis=1), f"A+B @ {c:>5.2f} bps/unit")

    print("\n=== R3 — return concentration: drop the best N days ===", flush=True)
    rev = sleeve_reversal_basis(4.2, "raw")
    j = pd.concat([rev.rename("rev"), mom.rename("mom")], axis=1).dropna()
    w = 1.0 / j.std(); w = w / w.sum()
    comb = (j * w).sum(axis=1)
    for n in (0, 1, 3, 5, 10):
        s = comb.sort_values(ascending=False).iloc[n:] if n else comb
        print(f"  drop best {n:<3} days -> Sharpe {sharpe_d(s):+.2f}   "
              f"(mean {s.mean()*1e4:+.1f} bps/day, n={len(s)})", flush=True)
    print(f"  worst 5 days: {', '.join(f'{v*100:+.1f}%' for v in comb.nsmallest(5))}", flush=True)
    print(f"  best  5 days: {', '.join(f'{v*100:+.1f}%' for v in comb.nlargest(5))}", flush=True)
    print(f"  skew {float(comb.skew()):+.2f}   share of total P&L from best 5 days: "
          f"{float(comb.nlargest(5).sum() / comb.sum())*100:.0f}%", flush=True)
    print("\nREVIEWDONE", flush=True)


if __name__ == "__main__":
    main()
