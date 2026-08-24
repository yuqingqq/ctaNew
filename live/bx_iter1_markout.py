"""Beyond-cross-section loop — iteration 1 (B1): is LIQUIDITY PROVISION profitable as a business?

Every prior iteration in this repo tried to earn a PREDICTION premium and treated maker fills only as a way
to pay less cost. This measures the other side: what a passive counterparty EARNS.

Binance aggTrades carry `is_buyer_maker`, so the aggressor side is known exactly — no Lee-Ready needed. For
each trade the passive counterparty filled at P with maker_sign = -aggressor_sign, and their gross P&L per
unit notional over horizon h is the MARKOUT:

    markout_h = -aggressor_sign * (P_{t+h} - P_t) / P_t

Volume-weighted, that is the gross revenue of providing liquidity before fees. Net of the Binance USDM VIP-0
maker fee (2.0 bps) it says whether the spread earned exceeds adverse selection.

WHAT THIS DOES NOT MODEL (stated up front, not buried): queue priority — we cannot verify we would have been
filled; inventory risk; the fact that we observe only executed aggressive trades; and that a real maker
chooses when to quote. This BOUNDS the opportunity, it does not simulate a business. A positive result is a
reason to build a fill-verified test, not a P&L claim.

Gates (live/BEYOND_XS_LOOP.md): G1 net markout positive at >=1 horizon in BOTH eras; G2 positive on >=1/3 of
symbols; G3 horizon profile monotone-decaying. A1 sign convention verified before anything else.
Run: python3 -u -m live.bx_iter1_markout [--workers 6]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
AGG = REPO / "data/ml/test/parquet/aggTrades"
OUT = REPO / "live/state/cost_loop/markout.parquet"
HORIZONS = [1, 5, 15, 60, 300]          # seconds
MAKER_FEE_BPS = 2.0
MIN_FILES = 1100
ERAS = {"OOS": ("2023-06-01", "2025-10-01"), "RECENT": ("2025-10-01", "2026-06-01")}
SAMPLE_EVERY = 7                         # every 7th day — 178 days/era-span per symbol, plenty for a mean


def symbols():
    return sorted(d.name for d in AGG.iterdir()
                  if d.is_dir() and len(list(d.glob("*.parquet"))) >= MIN_FILES)


def day_markout(fp: Path) -> dict | None:
    try:
        d = pd.read_parquet(fp, columns=["transact_time", "price", "quantity", "is_buyer_maker"])
    except Exception:
        return None
    if len(d) < 500:
        return None
    t = pd.to_datetime(d["transact_time"], utc=True).to_numpy("datetime64[ns]").astype("int64") / 1e9
    p = d["price"].to_numpy(dtype="float64")
    q = d["quantity"].to_numpy(dtype="float64")
    # is_buyer_maker True  => the BUYER was passive => the AGGRESSOR was a seller => aggressor_sign = -1
    aggr = np.where(d["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    # A maker fills AT the trade price, so measuring drift from that price credits ZERO spread and reports
    # pure adverse selection. The standard decomposition needs the MID. With trades only, reconstruct it:
    # last buy-aggressor price ~ the ask, last sell-aggressor price ~ the bid.
    ask = pd.Series(np.where(aggr > 0, p, np.nan)).ffill().to_numpy()
    bid = pd.Series(np.where(aggr < 0, p, np.nan)).ffill().to_numpy()
    mid = (ask + bid) / 2.0
    mk = -aggr                                     # the passive counterparty's side
    valid = np.isfinite(mid)
    out = {"n": len(d), "notional": float((p * q).sum())}
    w0 = (p * q)
    # immediate credit the maker earns at the fill = effective half-spread
    hs = mk * (mid - p) / p * 1e4
    m0 = valid & np.isfinite(hs)
    out["half_spread"] = float(np.sum(hs[m0] * w0[m0]) / w0[m0].sum())
    for h in HORIZONS:
        j = np.searchsorted(t, t + h, side="left")
        ok = (j < len(t)) & valid
        if ok.sum() < 100:
            continue
        jj = np.clip(j, 0, len(t) - 1)
        # total maker P&L per unit notional: fill at p, mark at the mid h seconds later
        tot = np.full(len(t), np.nan)
        imp = np.full(len(t), np.nan)
        tot[ok] = mk[ok] * (mid[jj[ok]] - p[ok]) / p[ok] * 1e4
        imp[ok] = mk[ok] * (mid[jj[ok]] - mid[ok]) / p[ok] * 1e4      # adverse selection component
        m = np.isfinite(tot)
        w = w0[m]
        out[f"mk_{h}"] = float(np.sum(tot[m] * w) / w.sum())          # spread earned + adverse drift
        out[f"imp_{h}"] = float(np.sum(imp[m] * w) / w.sum())
        out[f"aggr_{h}"] = -out[f"imp_{h}"]                            # A1 check: aggressor's own markout
        out[f"w_{h}"] = float(w.sum())
    return out


def sym_job(sym: str) -> pd.DataFrame | None:
    files = sorted((AGG / sym).glob("*.parquet"))[::SAMPLE_EVERY]
    rows = []
    for fp in files:
        r = day_markout(fp)
        if r:
            r["date"] = pd.Timestamp(fp.stem, tz="UTC")
            r["symbol"] = sym
            rows.append(r)
    if not rows:
        return None
    print(f"  {sym:<12} {len(rows)} days", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--symbols", type=int, default=0)
    a = ap.parse_args()
    syms = symbols()
    if a.symbols:
        syms = syms[:a.symbols]
    print(f"markout study: {len(syms)} symbols, every {SAMPLE_EVERY}th day, horizons {HORIZONS}s", flush=True)
    with mp.Pool(a.workers) as pool:
        parts = pool.map(sym_job, syms)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    print(f"\n{len(D):,} symbol-days\n", flush=True)

    print("=== effective half-spread earned at the fill (volume-weighted) ===", flush=True)
    hw = D.dropna(subset=["half_spread"])
    print(f"    {float((hw['half_spread']*hw['notional']).sum()/hw['notional'].sum()):+.3f} bps\n", flush=True)
    print("=== A1 SIGN CHECK: aggressor markout must be POSITIVE at short horizon ===", flush=True)
    print("    (trades push price in the aggressor's direction; if negative, the study is inverted)",
          flush=True)
    for h in HORIZONS:
        c, w = f"aggr_{h}", f"w_{h}"
        s = D.dropna(subset=[c])
        v = float((s[c] * s[w]).sum() / s[w].sum())
        print(f"    aggressor markout @{h:>3}s: {v:+7.3f} bps", flush=True)
    ok1 = float((D.dropna(subset=["aggr_1"])["aggr_1"] * D.dropna(subset=["aggr_1"])["w_1"]).sum() /
                D.dropna(subset=["aggr_1"])["w_1"].sum()) > 0
    print(f"    A1 {'PASS' if ok1 else 'FAIL — sign convention inverted, abort'}\n", flush=True)

    print("=== MAKER MARKOUT by era, volume-weighted (gross, then net of 2.0 bps maker fee) ===", flush=True)
    D["date"] = pd.to_datetime(D["date"], utc=True)
    res = {}
    for era, (t0, t1) in ERAS.items():
        e = D[(D.date >= pd.Timestamp(t0, tz="UTC")) & (D.date < pd.Timestamp(t1, tz="UTC"))]
        print(f"\n----- {era} ({e.date.nunique()} days, {e.symbol.nunique()} syms) -----", flush=True)
        print(f"  {'horizon':<10}{'gross bps':<12}{'net of fee':<12}{'verdict':<10}", flush=True)
        for h in HORIZONS:
            c, w = f"mk_{h}", f"w_{h}"
            s = e.dropna(subset=[c])
            if s.empty:
                continue
            g = float((s[c] * s[w]).sum() / s[w].sum())
            n = g - MAKER_FEE_BPS
            res[(era, h)] = n
            print(f"  {h:>3}s{'':<6}{g:<+12.3f}{n:<+12.3f}{'PROFITABLE' if n > 0 else 'loses':<10}",
                  flush=True)

    print("\n=== G2 — per-symbol dispersion at the best horizon ===", flush=True)
    best_h = max(HORIZONS, key=lambda h: min(res.get(("OOS", h), -9), res.get(("RECENT", h), -9)))
    c, w = f"mk_{best_h}", f"w_{best_h}"
    ps = D.dropna(subset=[c]).groupby("symbol").apply(
        lambda g: float((g[c] * g[w]).sum() / g[w].sum()) - MAKER_FEE_BPS).sort_values()
    print(f"  horizon {best_h}s, net of fee, per symbol:", flush=True)
    print("   worst 5:", ", ".join(f"{k} {v:+.2f}" for k, v in ps.head(5).items()), flush=True)
    print("   best  5:", ", ".join(f"{k} {v:+.2f}" for k, v in ps.tail(5).items()), flush=True)
    frac = float((ps > 0).mean())
    print(f"  symbols with positive net markout: {int((ps > 0).sum())}/{len(ps)} ({frac*100:.0f}%)",
          flush=True)

    print("\n=== GATE READ ===", flush=True)
    g1 = any(res.get(("OOS", h), -9) > 0 and res.get(("RECENT", h), -9) > 0 for h in HORIZONS)
    g2 = frac >= 1 / 3
    prof = [res.get(("OOS", h), np.nan) for h in HORIZONS]
    g3 = all(np.nan_to_num(prof[i], nan=-9) >= np.nan_to_num(prof[i + 1], nan=-9) for i in range(len(prof) - 1))
    print(f"  G1 net markout > 0 at >=1 horizon in BOTH eras : {'PASS' if g1 else 'FAIL'}", flush=True)
    print(f"  G2 positive on >=1/3 of symbols                 : {'PASS' if g2 else 'FAIL'}", flush=True)
    print(f"  G3 horizon profile monotone-decaying            : {'PASS' if g3 else 'FAIL'}", flush=True)
    print("\nBXITER1DONE", flush=True)


if __name__ == "__main__":
    main()
