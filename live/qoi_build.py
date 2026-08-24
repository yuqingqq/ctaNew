"""Signal-diversity loop iteration 7 — build the clock-phase order-imbalance series (Kim & Hansen 2026).

The paper's claim: order imbalance in the FIRST SECONDS of each quarter-hour boundary (minute in
{0,15,30,45}) carries information about subsequent returns, because scheduled/algorithmic execution clusters
there. Every flow feature this repo has ever built averages over exactly that phase and destroys it —
`fl_tfi`/`fl_vpin` are 5-minute resamples, the OB-flow family is 5-minute, bookDepth is 30-second, and every
`.dt.minute` reference in the codebase is the 4h-grid filter. So this is not a re-slice of a tested signal.

Builds, per symbol, one row per quarter-hour boundary window:
    qoi      = sum(q * sign) / sum(q)   over trades with second < WIN_SEC and minute in {0,15,30,45}
               sign = +1 aggressive buy (is_buyer_maker False), -1 aggressive sell
    n        = trade count in the window   (A0 measurability: empty windows are the abort condition)
    vol      = traded quantity in the window
Also builds the same at 30s and 60s so the window length can be FROZEN on the selection window (G3) rather
than chosen after seeing held-out results.

Output: live/state/cost_loop/qoi_windows.parquet  (symbol, wtime, qoi_10s/30s/60s, n_10s/30s/60s, vol_10s)
Run: python3 -u -m live.qoi_build [--workers 6] [--symbols N]
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
AGG = REPO / "data/ml/test/parquet/aggTrades"
OUT = REPO / "live/state/cost_loop/qoi_windows.parquet"
PHASES = {0, 15, 30, 45}
WINDOWS = [10, 30, 60]
MIN_FILES = 1100


def symbols() -> list[str]:
    out = []
    for d in sorted(AGG.iterdir()):
        if d.is_dir() and len(list(d.glob("*.parquet"))) >= MIN_FILES:
            out.append(d.name)
    return out


def build_sym(sym: str) -> pd.DataFrame | None:
    files = sorted((AGG / sym).glob("*.parquet"))
    if len(files) < MIN_FILES:
        return None
    parts = []
    for fp in files:
        try:
            d = pd.read_parquet(fp, columns=["transact_time", "quantity", "is_buyer_maker"])
        except Exception:
            continue
        if d.empty:
            continue
        t = pd.to_datetime(d["transact_time"], utc=True)
        mn = t.dt.minute.to_numpy()
        keep = np.isin(mn, list(PHASES))
        if not keep.any():
            continue
        d = d.loc[keep]; t = t.loc[keep]
        sec = t.dt.second.to_numpy() + t.dt.microsecond.to_numpy() / 1e6
        wt = t.dt.floor("15min")
        q = d["quantity"].to_numpy(dtype="float64")
        sgn = np.where(d["is_buyer_maker"].to_numpy(), -1.0, 1.0)
        base = pd.DataFrame({"wtime": wt.to_numpy(), "sec": sec, "q": q, "sq": q * sgn})
        agg = None
        for w in WINDOWS:
            sub = base[base["sec"] < w]
            if sub.empty:
                continue
            g = sub.groupby("wtime", sort=True)
            a = pd.DataFrame({f"sq_{w}": g["sq"].sum(), f"q_{w}": g["q"].sum(), f"n_{w}": g.size()})
            agg = a if agg is None else agg.join(a, how="outer")
        if agg is not None and len(agg):
            parts.append(agg)
    if not parts:
        return None
    D = pd.concat(parts).groupby(level=0).sum()
    for w in WINDOWS:
        if f"sq_{w}" in D.columns:
            D[f"qoi_{w}s"] = D[f"sq_{w}"] / D[f"q_{w}"].replace(0, np.nan)
            D[f"n_{w}s"] = D[f"n_{w}"]
    D["vol_10s"] = D.get("q_10", np.nan)
    cols = [c for c in D.columns if c.startswith("qoi_") or c.startswith("n_") or c == "vol_10s"]
    D = D[cols].reset_index().rename(columns={"index": "wtime"})
    D["symbol"] = sym
    return D


def _w(sym):
    t0 = time.time()
    try:
        r = build_sym(sym)
        print(f"  {sym:<12} {0 if r is None else len(r):>7} windows [{time.time()-t0:.0f}s]", flush=True)
        return r
    except Exception as e:
        print(f"  {sym:<12} ERR {str(e)[:70]}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--symbols", type=int, default=0)
    a = ap.parse_args()
    syms = symbols()
    if a.symbols:
        syms = syms[:a.symbols]
    print(f"clock-phase order imbalance: {len(syms)} symbols with >={MIN_FILES} daily aggTrades files",
          flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        parts = pool.map(_w, syms)
    parts = [p for p in parts if p is not None and len(p)]
    D = pd.concat(parts, ignore_index=True)
    D["wtime"] = pd.to_datetime(D["wtime"], utc=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(D):,} windows, {D.symbol.nunique()} syms, "
          f"{D.wtime.min()} -> {D.wtime.max()}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- A0: measurability. Expected 96 windows/day/symbol; count how many are missing or empty.
    print("\n=== A0 MEASURABILITY (abort if >5% of windows empty at 10s) ===", flush=True)
    span = D.groupby("symbol")["wtime"].agg(["min", "max"])
    rows = []
    for sym, r in span.iterrows():
        exp = int(((r["max"] - r["min"]).total_seconds() / 900) + 1)
        got = int((D.symbol == sym).sum())
        sub = D[D.symbol == sym]
        empty = int((sub["n_10s"].fillna(0) == 0).sum())
        rows.append(dict(symbol=sym, expected=exp, present=got,
                         miss_pct=100 * (1 - got / max(exp, 1)), empty_pct=100 * empty / max(got, 1)))
    A0 = pd.DataFrame(rows).sort_values("miss_pct", ascending=False)
    print(A0.head(12).to_string(index=False), flush=True)
    bad = A0[A0["miss_pct"] > 5.0]
    print(f"\n  symbols with >5% missing 10s windows: {len(bad)}/{len(A0)}  "
          f"-> G0 {'PASS' if len(A0) - len(bad) >= 25 else 'FAIL'} (need >=25 clean)", flush=True)
    A0.to_csv(REPO / "live/state/cost_loop/qoi_A0.csv", index=False)
    print("QOIBUILDDONE", flush=True)


if __name__ == "__main__":
    main()
