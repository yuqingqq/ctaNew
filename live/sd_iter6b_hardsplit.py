"""Signal-diversity loop — iteration 6b: hard-split resolution of the factor-count experiment.

iter6 gave a genuinely split verdict rather than a flat null:
  OOS  gross rises monotonically with K to an interior optimum, exactly the shape the SOTA paper reports:
       K=0 +1.47 -> K=2 +1.69 -> K=5 +1.98 -> K=10 +1.89   (net +0.67/+0.88/+1.12/+1.03)
       and K=5 of 176 names is what their K=30 of ~1000 scales to.
  RECENT the opposite: gross +0.88 -> +0.05 -> +0.06 -> -0.50.

Every paired delta spans 0. RECENT is 1596 bars with CI widths of ~4 Sharpe, so it cannot resolve a +/-1
difference — this loop's own method note says the both-era gate conflates "unstable" with "underpowered",
and that a chronological HARD SPLIT is the right instrument for a level question. That is this script.

  SELECT   2023-06-01 -> 2025-01-01   pick K by net@10k
  HOLDOUT  2025-01-01 -> 2026-07-01   evaluate that K; nothing here informs the choice

Uses the preds already cached by iter6 (no refitting). All K reported on both windows for transparency; only
the pre-committed selection rule counts.

Gate: selected K's held-out net@10k beats K=0 with paired 7d-block CI > 0.
Falsifier: fails -> the weak-factor result does not survive honest selection in a 4h crypto cross-section.
Run: python3 -u -m live.sd_iter6b_hardsplit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import (
    CACHE, ERAS, block_ci, build_panel, paired_block_ci, pit_adv, sharpe, tag_ci,
)
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP
from live.sd_iter6_factorK import KS

SEL0, SEL1 = pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
HO0, HO1 = SEL1, pd.Timestamp("2026-07-01", tz="UTC")


def load_preds(k: int) -> pd.DataFrame:
    parts = []
    for era in ERAS:
        fp = CACHE / f"sd6_z_k{k}_{era}.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "open_time"])


def book_net(P, cost10, cmed):
    W, A = build(P, "band")
    g = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    kv = pd.Series([float(cost10.get(s, cmed)) for s in W.index], index=W.index)
    net = (g - 0.25 * dW.mul(kv, axis=0).sum(axis=0) / 1e4).iloc[1:]
    return net, g.iloc[1:], (0.25 * dW.sum(axis=0)).iloc[1:]


def main():
    CT = cost_tiers(); cost10, cmed = CT["cost_10k"]
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    lab = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"})[["symbol", "open_time", "alpha_A"]]
    A = pit_adv()

    ser = {}
    for k in KS:
        P = load_preds(k)
        if P.empty:
            print(f"  K={k}: no cached preds", flush=True); continue
        P = P.merge(lab, on=["symbol", "open_time"], how="left").dropna()
        P["date"] = P["open_time"].dt.floor("1D")
        P = P.merge(A, on=["symbol", "date"], how="left").dropna(subset=["tadv"])
        for wname, (t0, t1) in (("SELECT", (SEL0, SEL1)), ("HOLDOUT", (HO0, HO1))):
            w = topn(P[(P.open_time >= t0) & (P.open_time < t1)], "tadv", NTOP)
            if w.empty:
                continue
            net, g, tu = book_net(w, cost10, cmed)
            ser[(wname, k)] = net
            lo, hi = block_ci(net.to_numpy())
            print(f"  {wname:<8} K={k:<3} bars {len(net):<5} gross {sharpe(g):+.2f}  "
                  f"net@10k {sharpe(net):+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}  turn {tu.mean():.3f}",
                  flush=True)
        print("", flush=True)

    avail = [k for k in KS if ("SELECT", k) in ser and ("HOLDOUT", k) in ser]
    best = max(avail, key=lambda k: sharpe(ser[("SELECT", k)]))
    print("============ PRE-COMMITTED SELECTION ============", flush=True)
    print(f"  selected on 2023-06..2024-12 by net@10k: K={best} "
          f"(select net {sharpe(ser[('SELECT', best)]):+.2f})", flush=True)
    hb, h0 = ser[("HOLDOUT", best)], ser[("HOLDOUT", 0)]
    lo, hi = block_ci(hb.to_numpy())
    print(f"  HELD-OUT K={best}: net@10k {sharpe(hb):+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}", flush=True)
    lo0, hi0 = block_ci(h0.to_numpy())
    print(f"  HELD-OUT K=0   : net@10k {sharpe(h0):+.2f} [{lo0:+.2f},{hi0:+.2f}] {tag_ci(lo0, hi0)}",
          flush=True)
    idx = hb.index.intersection(h0.index)
    dd, dlo, dhi = paired_block_ci(h0.loc[idx].to_numpy(), hb.loc[idx].to_numpy())
    print(f"\n  GATE  Δ(K={best} − K=0) held-out net@10k {dd:+.2f} [{dlo:+.2f},{dhi:+.2f}] "
          f"{tag_ci(dlo, dhi)}  -> {'PASS' if dlo > 0 else 'FAIL'}", flush=True)
    print("\nSDITER6BDONE", flush=True)


if __name__ == "__main__":
    main()
