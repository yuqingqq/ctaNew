"""Second adversarial pass on Phase 1 — checking what the first review did NOT.

The first review corrected the headline (+1.60 -> +1.26) and found the concentration problem. It did not
check four further things, and one of them tests the ONE claim I said survives.

  S1  Does the DIVERSIFICATION claim replicate on the SELECT window? I asserted "combining halves drawdown"
      from a single window. If it does not replicate on 2023-06..2025-01, it is an artifact of one sample and
      the only surviving claim dies with it.
  S2  Are the 5 dominant days CLUSTERED? 50% of P&L from 5 of 453 days is one thing if they are scattered and
      quite another if they are one episode — an episode is a single event, not a repeatable edge.
  S3  Is the P&L concentrated in a few SYMBOLS as well as a few days?
  S4  The bootstrap P(Sh<0) I quoted is the sampling distribution of the ESTIMATOR, not a posterior over the
      parameter. Stating it as "probability the true Sharpe is negative" is wrong. Restate correctly and give
      the honest frequentist reading.

Run: python3 -u -m live.dp_phase1_review2
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import ERAS, build_panel, get_preds, pit_adv, tag_ci
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP
from live.bx_iter3_horizon import daily_panel, build_features, FEATS
from live.bx_iter4_slowsignal import preds_for
from live.dp_phase1_consolidate import sharpe_d, maxdd, boot_sharpe

SELW = (pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HOW = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"))
COST = 4.2
H_MOM = 14


def rev_sleeve(t0, t1, cost=COST):
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
    w = topn(P[(P.open_time >= t0) & (P.open_time < t1)].dropna(
        subset=["tadv", "alpha_A", "return_pct"]), "tadv", NTOP)
    if w.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    W, Aa = build(w, "band")
    R = w.pivot_table(index="symbol", columns="open_time", values="return_pct").reindex_like(Aa).fillna(0.0)
    contrib = (W * R)                                   # per-symbol contribution, for S3
    g = contrib.sum(axis=0)
    dW = W.diff(axis=1).abs()
    ch = 0.25 * dW.sum(axis=0) * cost / 1e4
    net = (g - ch).iloc[1:]
    net.index = pd.to_datetime(net.index, utc=True)
    return net.groupby(net.index.floor("1D")).sum(), contrib


def mom_sleeve(t0, t1, cost=COST):
    CT = cost_tiers(); c10, cmed = CT["cost_10k"]
    d = daily_panel()
    x = build_features(d, H_MOM)
    tag = "sel" if t0 == SELW[0] else "ho"
    P = preds_for(x, H_MOM, t0, t1, FEATS, tag)
    if P.empty:
        return pd.Series(dtype=float), pd.DataFrame()
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
    P = P.merge(d[["symbol", "date", "ret_1d"]], on=["symbol", "date"], how="left").dropna(subset=["ret_1d"])
    daily = P[P["pos"] != 0].groupby("date").apply(
        lambda g: (g.loc[g.pos > 0, "ret_1d"].mean() - g.loc[g.pos < 0, "ret_1d"].mean())
        if (g.pos > 0).any() and (g.pos < 0).any() else np.nan).dropna()
    names = P[P["pos"] != 0].groupby("blk")["symbol"].apply(set)
    churn = float(np.mean([len(names.iloc[i] - names.iloc[i - 1]) / max(len(names.iloc[i]), 1)
                           for i in range(1, len(names))])) if len(names) > 1 else 1.0
    daily = daily - churn * 2 * cost / 1e4 / H_MOM
    per = P[P["pos"] != 0].assign(c=lambda z: z["pos"] * z["ret_1d"])
    return daily, per


def combo(a, b):
    j = pd.concat([a.rename("rev"), b.rename("mom")], axis=1).dropna()
    w = 1.0 / j.std(); w = w / w.sum()
    return j, (j * w).sum(axis=1)


def main():
    print("=== S1 — does the DIVERSIFICATION claim replicate on the SELECT window? ===", flush=True)
    print("    (the one claim I said survives review; asserted from a single window)\n", flush=True)
    for wname, (t0, t1) in (("SELECT 2023-06..2025-01", SELW), ("HOLDOUT 2025-01..2026-07", HOW)):
        a, _ = rev_sleeve(t0, t1)
        b, _ = mom_sleeve(t0, t1)
        if len(a) < 60 or len(b) < 60:
            print(f"  {wname}: insufficient", flush=True); continue
        j, c = combo(a, b)
        print(f"  --- {wname} ({len(j)} days) ---", flush=True)
        for nm, s in (("A: XS reversal", j["rev"]), ("B: 14d momentum", j["mom"]), ("A+B", c)):
            print(f"    {nm:<18}Sharpe {sharpe_d(s):+6.2f}   maxDD {maxdd(s)*100:+7.1f}%", flush=True)
        dd_single = min(maxdd(j["rev"]), maxdd(j["mom"]))
        print(f"    correlation {float(j['rev'].corr(j['mom'])):+.3f}   "
              f"maxDD: worst single {dd_single*100:+.1f}% -> combined {maxdd(c)*100:+.1f}%  "
              f"({(1 - maxdd(c)/dd_single)*100:+.0f}% change)\n", flush=True)

    print("=== S2 — are the dominant days CLUSTERED (one episode) or scattered? ===", flush=True)
    a, rc = rev_sleeve(*HOW)
    b, mc = mom_sleeve(*HOW)
    j, c = combo(a, b)
    top = c.nlargest(10)
    print("  top-10 days:", flush=True)
    for dt, v in top.items():
        print(f"    {pd.Timestamp(dt).date()}  {v*100:+6.2f}%   "
              f"(rev {j.loc[dt,'rev']*100:+.2f}%, mom {j.loc[dt,'mom']*100:+.2f}%)", flush=True)
    ds = pd.DatetimeIndex(top.index).sort_values()
    gaps = np.diff(ds.values).astype("timedelta64[D]").astype(int)
    print(f"  gaps between top-10 days (days): {list(gaps)}", flush=True)
    print(f"  span: {ds.min().date()} -> {ds.max().date()}; distinct months: {len(set(ds.to_period('M')))}",
          flush=True)
    wor = c.nsmallest(10)
    print(f"  worst-10 span: {pd.DatetimeIndex(wor.index).min().date()} -> "
          f"{pd.DatetimeIndex(wor.index).max().date()}", flush=True)

    print("\n=== S3 — is P&L concentrated in a few SYMBOLS? ===", flush=True)
    if not rc.empty:
        sc = rc.sum(axis=1).sort_values()
        tot = sc.sum()
        print(f"  sleeve A: {len(sc)} symbols traded; total {tot*100:+.1f}%", flush=True)
        print(f"    top 3 contributors: {', '.join(f'{k} {v*100:+.1f}%' for k, v in sc.tail(3).items())}",
              flush=True)
        print(f"    bot 3 contributors: {', '.join(f'{k} {v*100:+.1f}%' for k, v in sc.head(3).items())}",
              flush=True)
        print(f"    share of gross P&L from top 3 names: "
              f"{float(sc.tail(3).sum() / sc.abs().sum())*100:.0f}% of gross absolute", flush=True)

    print("\n=== S4 — correct statistical reading of the bootstrap ===", flush=True)
    bs = boot_sharpe(c.to_numpy())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  combined Sharpe {sharpe_d(c):+.2f}, block-bootstrap 95% interval [{lo:+.2f},{hi:+.2f}]",
          flush=True)
    print(f"  fraction of bootstrap replicates below zero: {float((bs < 0).mean())*100:.1f}%", flush=True)
    print("  CORRECTION: that fraction approximates the sampling distribution of the ESTIMATOR under", flush=True)
    print("  resampling — it is NOT the posterior probability that the true Sharpe is negative, which is", flush=True)
    print("  what I called it. The correct frequentist statement is simply that the interval contains", flush=True)
    print("  zero, so the null of no edge is not rejected at 5%.", flush=True)
    print("\n=== S5 — multiple comparisons across the whole programme ===", flush=True)
    print("  19 iterations, each testing 2-27 cells. Order 200+ configurations have been examined on", flush=True)
    print("  substantially overlapping data. A single surviving configuration at p~0.05-0.10 is what", flush=True)
    print("  a search of that size produces under the null. This cannot be corrected after the fact —", flush=True)
    print("  only out-of-sample forward data discriminates.", flush=True)
    print("\nREVIEW2DONE", flush=True)


if __name__ == "__main__":
    main()
