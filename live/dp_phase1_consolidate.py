"""Deployment plan — Phase 1: what is the honest OUTCOME DISTRIBUTION of the best book we can actually build?

Every prior test measured ONE component in isolation. This assembles the measured, validated components into
a single specification and asks the only question that decides whether to trade:

    given everything we know, what is the distribution of outcomes — not the point estimate?

Components (each already validated separately, none newly fitted here):
  universe     PIT trailing-ADV top-40            (cost 8.35 -> ~3 bps/unit; ordering replicates across the split)
  construction K+M hysteresis band                (turnover 0.40 -> 0.26 at equal gross)
  execution    taker 8.35 bps vs passive ~4.2 bps (measured on 22,542 real trades, live/maker_exec_probe.py)
  sleeve A     per-symbol Ridge XS reversal, 4h   (held-out net +0.85, CI spans 0)
  sleeve B     14d skip-recent momentum           (held-out IC +0.055 SIG; realised ~+1.2)

All parameters were frozen on 2023-06..2024-12 in earlier iterations; this evaluates ONLY on the held-out
window 2025-01..2026-06. Nothing here is selected on the evaluation data.

Reports, for each sleeve and the equal-risk combination: net Sharpe with 7d-block CI, max drawdown,
P(true Sharpe < 0) from the bootstrap, and the correlation between sleeves.
Run: python3 -u -m live.dp_phase1_consolidate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import CACHE, ERAS, block_ci, build_panel, get_preds, pit_adv, tag_ci
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP
from live.bx_iter3_horizon import daily_panel, build_features, FEATS
from live.bx_iter4_slowsignal import preds_for

HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
H_MOM = 14
PASSIVE_BPS = 4.2          # measured: 60m passive, 1-5bp inside, incl. chased non-fills
RNG = np.random.default_rng(101)


def sharpe_d(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 2 and x.std() > 0 else np.nan


def maxdd(x):
    eq = np.cumsum(np.asarray(x, float))
    return float((eq - np.maximum.accumulate(eq)).min())


def boot_sharpe(x, block=7, nb=5000):
    a = np.asarray(x, float); n = len(a)
    nb_blk = int(np.ceil(n / block)); out = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nb_blk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        out[i] = sharpe_d(a[idx])
    return out


def sleeve_reversal(cost_bps_override=None):
    """4h XS reversal book -> DAILY net return series on the held-out window."""
    CT = cost_tiers(); c10, cmed = CT["cost_10k"]
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    lab = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"})[["symbol", "open_time", "alpha_A"]]
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    P = P.drop(columns=[c for c in ("alpha_A",) if c in P.columns]).merge(
        lab, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left")
    ho = topn(P[(P.open_time >= HO0) & (P.open_time < HO1)].dropna(subset=["tadv", "alpha_A"]), "tadv", NTOP)
    W, Aa = build(ho, "band")
    g = (W * Aa).sum(axis=0)
    dW = W.diff(axis=1).abs()
    if cost_bps_override is None:
        kv = pd.Series([float(c10.get(s, cmed)) for s in W.index], index=W.index)
        ch = 0.25 * dW.mul(kv, axis=0).sum(axis=0) / 1e4
    else:
        ch = 0.25 * dW.sum(axis=0) * cost_bps_override / 1e4
    net = (g - ch).iloc[1:]
    net.index = pd.to_datetime(net.index, utc=True)
    return net.groupby(net.index.floor("1D")).sum()


def sleeve_momentum(cost_bps_override=None):
    """14d skip-recent momentum -> DAILY net return series on the held-out window."""
    CT = cost_tiers(); c10, cmed = CT["cost_10k"]
    d = daily_panel()
    x = build_features(d, H_MOM)
    P = preds_for(x, H_MOM, HO0, HO1, FEATS, "ho")
    if P.empty:
        return pd.Series(dtype=float)
    P["date"] = pd.to_datetime(P["date"], utc=True)
    A = pit_adv()
    P = P.merge(A, on=["symbol", "date"], how="left").dropna(subset=["tadv", "pred"])
    P["ar"] = P.groupby("date")["tadv"].rank(ascending=False, method="first")
    P = P[P["ar"] <= NTOP]
    dates = np.sort(P["date"].unique())
    blk = {dt: i // H_MOM for i, dt in enumerate(dates)}
    P["blk"] = P["date"].map(blk)
    first = P.groupby(["blk", "symbol"])["pred"].transform("first")
    P["rk"] = P.assign(_f=first).groupby("date")["_f"].rank(pct=True)
    P["pos"] = np.where(P["rk"] >= 0.8, 1.0, np.where(P["rk"] <= 0.2, -1.0, 0.0))
    # daily P&L from the held positions, dollar-neutral
    px = d[["symbol", "date", "ret_1d"]]
    P = P.merge(px, on=["symbol", "date"], how="left").dropna(subset=["ret_1d"])
    daily = P[P["pos"] != 0].groupby("date").apply(
        lambda g: (g.loc[g.pos > 0, "ret_1d"].mean() - g.loc[g.pos < 0, "ret_1d"].mean())
        if (g.pos > 0).any() and (g.pos < 0).any() else np.nan).dropna()
    names = P[P["pos"] != 0].groupby("blk")["symbol"].apply(set)
    churn = float(np.mean([len(names.iloc[i] - names.iloc[i - 1]) / max(len(names.iloc[i]), 1)
                           for i in range(1, len(names))])) if len(names) > 1 else 1.0
    cbps = cost_bps_override if cost_bps_override is not None else float(
        np.mean([c10.get(s, cmed) for s in P["symbol"].unique()]))
    daily = daily - churn * 2 * cbps / 1e4 / H_MOM          # amortise the rebalance across the block
    return daily


def report(name, s):
    if s is None or len(s) < 30:
        print(f"  {name:<26} insufficient data", flush=True); return None
    b = boot_sharpe(s.to_numpy())
    lo, hi = np.percentile(b, [2.5, 97.5])
    pneg = float((b < 0).mean())
    print(f"  {name:<26}{sharpe_d(s):>+7.2f}  [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi):<8}"
          f"maxDD {maxdd(s)*100:>7.1f}%   P(Sh<0) {pneg*100:>5.1f}%   n={len(s)}", flush=True)
    return dict(sh=sharpe_d(s), lo=lo, hi=hi, pneg=pneg, dd=maxdd(s))


def main():
    print("PHASE 1 — held-out 2025-01..2026-06, all parameters frozen on 2023-06..2024-12\n", flush=True)
    for label, cost in (("TAKER (calibrated per-symbol)", None), ("PASSIVE (4.2 bps measured)", PASSIVE_BPS)):
        print(f"================ {label} ================", flush=True)
        a = sleeve_reversal(cost)
        b = sleeve_momentum(cost)
        j = pd.concat([a.rename("rev"), b.rename("mom")], axis=1).dropna()
        print(f"  {'sleeve':<26}{'Sharpe':>7}  {'95% CI':<19}{'':<8}{'maxDD':<14}{'P(Sh<0)':<14}", flush=True)
        ra = report("A: XS reversal (4h)", j["rev"])
        rb = report("B: 14d skip-momentum", j["mom"])
        if len(j) > 30:
            corr = float(j["rev"].corr(j["mom"]))
            w = 1.0 / j.std()
            w = w / w.sum()
            comb = (j * w).sum(axis=1)
            rc = report("A+B equal-risk", comb)
            print(f"\n  sleeve correlation {corr:+.3f}   weights rev {w['rev']:.2f} / mom {w['mom']:.2f}",
                  flush=True)
            if ra and rb and rc:
                print(f"  combination vs best single: {rc['sh']:+.2f} vs "
                      f"{max(ra['sh'], rb['sh']):+.2f}", flush=True)
        print("", flush=True)

    print("================ WHAT THIS MEANS ================", flush=True)
    print("  A CI spanning zero is not 'no edge' — it is 'the data cannot distinguish this from zero'.", flush=True)
    print("  P(Sh<0) is the bootstrap probability the true Sharpe is negative; it is the number that", flush=True)
    print("  should drive sizing, not the point estimate. Phase 2 asks the literature what fraction of", flush=True)
    print("  capital to commit at that probability.", flush=True)
    print("\nPHASE1DONE", flush=True)


if __name__ == "__main__":
    main()
