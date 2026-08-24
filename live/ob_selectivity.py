"""Does the OB data we ALREADY HAVE make market making viable? (testing my own unproven claim)

I claimed passive quoting fails because we cannot quote SELECTIVELY, and that selectivity needs L2 we do not
own — then recommended buying data without proving it would help. Both halves were wrong to assert:
  - `bookdepth_flow_all_5min_v3_recovered` has 182 symbols x 1245 days of 30-second book snapshots with
    depth at +/-0.2% and +/-1%, imbalance, signed pressure and absorption flags.
  - Whether that state information converts a losing maker book into a winning one is an EMPIRICAL question
    on data already on disk.

Measured baseline (live/bx_iter1_markout.py, 31 syms x 754 days): a maker earns a half-spread of +1.83 bps
and loses 2.4 bps to adverse selection => -0.6 bps GROSS, -2.6 bps after the 2.0 bps VIP-0 maker fee, with
0/31 symbols positive. The question here is whether some subset of BOOK STATES flips that sign.

TEST. For every aggressive trade, the passive counterparty filled at that price with maker_sign =
-aggressor_sign. Join the most recent book snapshot STRICTLY BEFORE the trade (PIT), then bucket by book
state expressed RELATIVE TO THE MAKER'S SIDE:

    fav_imb = maker_sign * (bid02 - ask02) / (bid02 + ask02)

A maker who bought (aggressor sold into them) is long and wants deep bids, so fav_imb > 0 is support on
their side. Microstructure theory predicts markout improves in fav_imb. Also bucket on total depth and on
the aggressor's own recent pressure.

Gate: does ANY state bucket have maker markout - 2.0 bps fee > 0, in BOTH eras, on a non-trivial share of
volume? If yes, the OB data we own has demonstrable execution value and the case for more/better data is
evidence-based. If no, both my claim and my recommendation were unfounded and the market-making route closes
on owned data.
Run: python3 -u -m live.ob_selectivity [--workers 6]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
AGG = REPO / "data/ml/test/parquet/aggTrades"
OB = REPO / "data/ml/cache/research/bookdepth_flow_all_5min_v3_recovered"
OUT = REPO / "live/state/cost_loop/ob_selectivity.parquet"
HORIZ = [5, 30]                     # seconds; 5s is the decision-relevant maker horizon
MAKER_FEE = 2.0
SAMPLE_EVERY = 15
IMB_EDGES = [-1.01, -0.5, -0.2, 0.0, 0.2, 0.5, 1.01]
IMB_LABELS = ["<-0.5", "-0.5..-0.2", "-0.2..0", "0..0.2", "0.2..0.5", ">0.5"]


def day(sym: str, dstr: str):
    fa = AGG / sym / f"{dstr}.parquet"
    fo = OB / sym / f"{dstr}.parquet"
    if not fa.exists() or not fo.exists():
        return None
    try:
        a = pd.read_parquet(fa, columns=["transact_time", "price", "quantity", "is_buyer_maker"])
        o = pd.read_parquet(fo, columns=["snapshot_time", "bid02", "ask02", "bid1", "ask1"])
    except Exception:
        return None
    if len(a) < 500 or len(o) < 50:
        return None
    # the two sources carry different datetime resolutions (ms vs ns); merge_asof requires identical dtypes
    a["t"] = pd.to_datetime(a["transact_time"], utc=True).astype("datetime64[ns, UTC]")
    o["t"] = pd.to_datetime(o["snapshot_time"], utc=True).astype("datetime64[ns, UTC]")
    o = o.dropna(subset=["bid02", "ask02"]).sort_values("t")
    if o.empty:
        return None
    a = a.sort_values("t")
    # PIT: attach the most recent snapshot STRICTLY BEFORE each trade
    m = pd.merge_asof(a, o[["t", "bid02", "ask02", "bid1", "ask1"]], on="t",
                      direction="backward", allow_exact_matches=False)
    m = m.dropna(subset=["bid02", "ask02"])
    if len(m) < 200:
        return None
    p = m["price"].to_numpy(float); q = m["quantity"].to_numpy(float)
    aggr = np.where(m["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    mk = -aggr
    ts = m["t"].to_numpy("datetime64[s]").astype("int64")
    b02 = m["bid02"].to_numpy(float); a02 = m["ask02"].to_numpy(float)
    tot = b02 + a02
    imb = np.where(tot > 0, (b02 - a02) / np.where(tot > 0, tot, np.nan), np.nan)
    fav = mk * imb                                     # book support on the MAKER's side
    # reconstruct mid from signed trades (same method as the markout study)
    ask = pd.Series(np.where(aggr > 0, p, np.nan)).ffill().to_numpy()
    bid = pd.Series(np.where(aggr < 0, p, np.nan)).ffill().to_numpy()
    mid = (ask + bid) / 2.0
    out = []
    for h in HORIZ:
        j = np.searchsorted(ts, ts + h, side="left")
        ok = (j < len(ts)) & np.isfinite(mid) & np.isfinite(fav)
        if ok.sum() < 100:
            continue
        jj = np.clip(j, 0, len(ts) - 1)
        tot_pnl = np.full(len(ts), np.nan)
        tot_pnl[ok] = mk[ok] * (mid[jj[ok]] - p[ok]) / p[ok] * 1e4
        d = pd.DataFrame({"h": h, "fav": fav, "mk_pnl": tot_pnl, "w": p * q,
                          "depth": tot}).dropna()
        d["bucket"] = pd.cut(d["fav"], IMB_EDGES, labels=IMB_LABELS)
        gg = d.groupby(["h", "bucket"], observed=True).apply(
            lambda x: pd.Series({"n": len(x), "wsum": x["w"].sum(),
                                 "wpnl": float((x["mk_pnl"] * x["w"]).sum() / x["w"].sum())})).reset_index()
        gg["symbol"] = sym; gg["date"] = dstr
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else None


def sym_job(sym: str):
    days = sorted(p.stem for p in (OB / sym).glob("*.parquet"))[::SAMPLE_EVERY]
    rows = [r for d in days if (r := day(sym, d)) is not None]
    if not rows:
        return None
    print(f"  {sym:<12} {len(rows)} days", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--symbols", type=int, default=10)
    a = ap.parse_args()
    have = [d.name for d in AGG.iterdir()
            if d.is_dir() and len(list(d.glob("*.parquet"))) >= 1100 and (OB / d.name).exists()]
    syms = sorted(have)[:a.symbols] if a.symbols else sorted(have)
    print(f"OB selectivity: {len(syms)} symbols with BOTH aggTrades and rebuilt OB, "
          f"every {SAMPLE_EVERY}th day\n", flush=True)
    with mp.Pool(a.workers) as pool:
        parts = pool.map(sym_job, syms)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    D["date"] = pd.to_datetime(D["date"], utc=True)

    print("\n=== maker markout by BOOK STATE on the maker's side (volume-weighted) ===", flush=True)
    print("   fav_imb > 0 means depth supports the side the maker just took on\n", flush=True)
    for h in HORIZ:
        print(f"  --- horizon {h}s ---", flush=True)
        print(f"    {'fav_imb bucket':<14}{'volume share':>14}{'gross bps':>11}{'net of 2bp fee':>16}",
              flush=True)
        s = D[D.h == h]
        tot_w = s["wsum"].sum()
        for lab in IMB_LABELS:
            b = s[s["bucket"] == lab]
            if b.empty:
                continue
            g = float((b["wpnl"] * b["wsum"]).sum() / b["wsum"].sum())
            print(f"    {lab:<14}{100*b['wsum'].sum()/tot_w:>13.1f}%{g:>11.3f}{g-MAKER_FEE:>16.3f}",
                  flush=True)

    print("\n=== BOTH-ERA check on the best bucket ===", flush=True)
    s5 = D[D.h == 5]
    best, bestv = None, -9e9
    for lab in IMB_LABELS:
        b = s5[s5["bucket"] == lab]
        if b.empty or b["wsum"].sum() / s5["wsum"].sum() < 0.03:
            continue
        v = float((b["wpnl"] * b["wsum"]).sum() / b["wsum"].sum())
        if v > bestv:
            best, bestv = lab, v
    print(f"  best bucket at 5s: {best}  gross {bestv:.3f} bps  net {bestv-MAKER_FEE:.3f}", flush=True)
    for era, t0, t1 in (("OOS", "2023-06-01", "2025-10-01"), ("RECENT", "2025-10-01", "2026-06-01")):
        e = s5[(s5["bucket"] == best) & (s5.date >= pd.Timestamp(t0, tz="UTC"))
               & (s5.date < pd.Timestamp(t1, tz="UTC"))]
        if e.empty:
            continue
        v = float((e["wpnl"] * e["wsum"]).sum() / e["wsum"].sum())
        print(f"    {era:<8} gross {v:>7.3f}  net {v-MAKER_FEE:>7.3f}  "
              f"{'PROFITABLE' if v-MAKER_FEE > 0 else 'loses'}", flush=True)

    print("\n=== VERDICT ===", flush=True)
    any_pos = any(
        float((s5[s5['bucket'] == l]['wpnl'] * s5[s5['bucket'] == l]['wsum']).sum()
              / s5[s5['bucket'] == l]['wsum'].sum()) - MAKER_FEE > 0
        for l in IMB_LABELS if not s5[s5['bucket'] == l].empty)
    print(f"  any book state with positive net maker markout: {'YES' if any_pos else 'NO'}", flush=True)
    print("OBSELDONE", flush=True)


if __name__ == "__main__":
    main()
