"""Due diligence on the deployment plan — the details that change the decision, not the headline.

Four things the plan asserts or assumes but never checked:

  D1  POSITION OVERLAP, not just return correlation. Two sleeves can have ~0 return correlation while holding
      the same names at the same time. If they do, the combined book is more concentrated than the
      correlation implies and the diversification argument is weaker than it looks.
  D2  IS SLEEVE A WORTH DEPLOYING? A decayed +2.21 -> +0.69 (-69%) between windows while B went +0.45 ->
      +1.04 (+130%). The plan treats them symmetrically. Check sub-period stability inside the holdout: if A
      is still decaying, A+B going forward may be worse than B alone.
  D3  TAIL SHAPE. Skew is -0.60. Is the loss tail a SHORT-SQUEEZE pattern (short leg blowing up)? That is the
      documented failure mode of every prior book in this repo. Decompose the worst days by leg.
  D4  PRACTICALITY at a 10% vol target: gross notional, per-name position size, and whether the book is
      implementable at plausible account sizes.

Run: python3 -u -m live.dp_phase3_diligence
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import ERAS, build_panel, get_preds, pit_adv
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP
from live.bx_iter3_horizon import daily_panel, build_features, FEATS
from live.bx_iter4_slowsignal import preds_for
from live.dp_phase1_consolidate import sharpe_d, maxdd
from live.dp_phase1_review2 import SELW, HOW, COST, H_MOM

VOL_TARGET = 0.10


def rev_positions(t0, t1):
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
    W, Aa = build(w, "band")
    R = w.pivot_table(index="symbol", columns="open_time", values="return_pct").reindex_like(Aa).fillna(0.0)
    longs = (W.clip(lower=0) * R).sum(axis=0)
    shorts = (W.clip(upper=0) * R).sum(axis=0)
    dW = W.diff(axis=1).abs()
    ch = 0.25 * dW.sum(axis=0) * COST / 1e4
    net = (longs + shorts - ch).iloc[1:]
    for s in (longs, shorts, net):
        s.index = pd.to_datetime(s.index, utc=True)
    d = pd.DataFrame({"net": net,
                      "long": longs.reindex(net.index), "short": shorts.reindex(net.index)})
    daily = d.groupby(d.index.floor("1D")).sum()
    # daily position snapshot (sign per symbol, last bar of the day)
    Wt = W.T.copy(); Wt.index = pd.to_datetime(Wt.index, utc=True)
    pos = Wt.groupby(Wt.index.floor("1D")).last()
    return daily, pos


def mom_positions(t0, t1):
    d = daily_panel()
    x = build_features(d, H_MOM)
    tag = "sel" if t0 == SELW[0] else "ho"
    P = preds_for(x, H_MOM, t0, t1, FEATS, tag)
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
    act = P[P["pos"] != 0]
    daily = pd.DataFrame({
        "long": act[act.pos > 0].groupby("date")["ret_1d"].mean(),
        "short": -act[act.pos < 0].groupby("date")["ret_1d"].mean(),
    }).dropna()
    names = act.groupby("blk")["symbol"].apply(set)
    churn = float(np.mean([len(names.iloc[i] - names.iloc[i - 1]) / max(len(names.iloc[i]), 1)
                           for i in range(1, len(names))])) if len(names) > 1 else 1.0
    daily["net"] = daily["long"] + daily["short"] - churn * 2 * COST / 1e4 / H_MOM
    pos = act.pivot_table(index="date", columns="symbol", values="pos", fill_value=0.0)
    return daily, pos


def main():
    print("=== D1 — POSITION overlap (not return correlation) ===", flush=True)
    ra, pa = rev_positions(*HOW)
    rb, pb = mom_positions(*HOW)
    common = pa.index.intersection(pb.index)
    cols = pa.columns.intersection(pb.columns)
    A_ = np.sign(pa.loc[common, cols].to_numpy())
    B_ = np.sign(pb.loc[common, cols].to_numpy())
    both = (A_ != 0) & (B_ != 0)
    agree = both & (A_ == B_)
    oppose = both & (A_ == -B_)
    print(f"  days {len(common)}, shared symbols {len(cols)}", flush=True)
    print(f"  name-days held by BOTH sleeves: {both.sum():,}", flush=True)
    print(f"    same direction (reinforcing): {agree.sum():,} ({100*agree.sum()/max(both.sum(),1):.0f}%)",
          flush=True)
    print(f"    opposite  (cancelling):       {oppose.sum():,} ({100*oppose.sum()/max(both.sum(),1):.0f}%)",
          flush=True)
    nA = (A_ != 0).sum(); nB = (B_ != 0).sum()
    print(f"  overlap as share of sleeve-A positions {100*both.sum()/max(nA,1):.0f}%, "
          f"of sleeve-B {100*both.sum()/max(nB,1):.0f}%", flush=True)
    print(f"  -> net directional reinforcement = {100*(agree.sum()-oppose.sum())/max(both.sum(),1):+.0f}%",
          flush=True)

    print("\n=== D2 — is sleeve A still decaying INSIDE the holdout? ===", flush=True)
    j = pd.concat([ra["net"].rename("A"), rb["net"].rename("B")], axis=1).dropna()
    j["AB"] = 0.5 * (j["A"] / j["A"].std() + j["B"] / j["B"].std()) * j[["A", "B"]].std().mean()
    halves = [("2025 H1", "2025-01-01", "2025-07-01"), ("2025 H2", "2025-07-01", "2026-01-01"),
              ("2026 H1", "2026-01-01", "2026-07-01")]
    print(f"  {'period':<10}{'A Sharpe':>10}{'B Sharpe':>10}{'A+B':>10}{'days':>7}", flush=True)
    for nm, a0, a1 in halves:
        s = j[(j.index >= pd.Timestamp(a0, tz="UTC")) & (j.index < pd.Timestamp(a1, tz="UTC"))]
        if len(s) < 40:
            continue
        print(f"  {nm:<10}{sharpe_d(s['A']):>10.2f}{sharpe_d(s['B']):>10.2f}"
              f"{sharpe_d(s['AB']):>10.2f}{len(s):>7}", flush=True)
    print(f"  full SELECT->HOLDOUT decay: A -69%, B +130% (from review 2)", flush=True)

    print("\n=== D3 — tail shape: is the loss tail a SHORT-SQUEEZE? ===", flush=True)
    comb = j["AB"]
    worst = comb.nsmallest(8)
    print(f"  {'date':<12}{'combined':>10}{'A long':>9}{'A short':>9}{'B long':>9}{'B short':>9}", flush=True)
    for dt in worst.index:
        la = ra.loc[dt, "long"] if dt in ra.index else np.nan
        sa = ra.loc[dt, "short"] if dt in ra.index else np.nan
        lb = rb.loc[dt, "long"] if dt in rb.index else np.nan
        sb = rb.loc[dt, "short"] if dt in rb.index else np.nan
        print(f"  {str(pd.Timestamp(dt).date()):<12}{comb.loc[dt]*100:>9.2f}%{la*100:>8.2f}%"
              f"{sa*100:>8.2f}%{lb*100:>8.2f}%{sb*100:>8.2f}%", flush=True)
    sh_neg = sum(1 for dt in worst.index
                 if (ra.loc[dt, "short"] if dt in ra.index else 0) < (ra.loc[dt, "long"] if dt in ra.index else 0))
    print(f"  worst days where the SHORT leg was the bigger loser: {sh_neg}/8", flush=True)
    print(f"  leg Sharpes over the holdout — A long {sharpe_d(ra['long']):+.2f} / A short "
          f"{sharpe_d(ra['short']):+.2f} | B long {sharpe_d(rb['long']):+.2f} / B short "
          f"{sharpe_d(rb['short']):+.2f}", flush=True)

    print("\n=== D4 — practicality at a 10% vol target ===", flush=True)
    vol = comb.std() * np.sqrt(365)
    scale = VOL_TARGET / vol
    nA_avg = float((np.abs(pa.to_numpy()) > 1e-9).sum(axis=1).mean())
    nB_avg = float((np.abs(pb.to_numpy()) > 1e-9).sum(axis=1).mean())
    print(f"  combined book vol at unit gross {vol*100:.1f}%/yr -> scale {scale:.2f}x for {VOL_TARGET*100:.0f}%",
          flush=True)
    print(f"  names held: sleeve A {nA_avg:.0f}, sleeve B {nB_avg:.0f}", flush=True)
    for cap in (100_000, 1_000_000, 10_000_000):
        gross = cap * scale * 2                     # long 1 + short 1 per unit
        per = gross / max(nA_avg + nB_avg, 1)
        print(f"    ${cap:>10,} account -> gross ${gross:>12,.0f}, ~${per:>9,.0f} per position", flush=True)
    print("\nDILIGENCEDONE", flush=True)


if __name__ == "__main__":
    main()
