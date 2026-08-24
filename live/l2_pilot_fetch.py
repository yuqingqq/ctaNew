"""Fetch + SAVE the imbalance across book depths (imb1/imb2/imb3/imb5) for a both-era pilot, so the window x depth
sweep can run offline (no repeated re-fetch). 50 both-era symbols x 3mo/era (enough for a 90-bar rolling window).
Saves the OBSERVATION-bar series (obs_bar); the sweep computes rolling mu/sd per window then shifts +4h for PIT.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_loader import load_symbol
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
LVL = ["imb1", "imb2", "imb3", "imb5"]

def pilot_syms(n=50):
    ok = []
    for f in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet"):
        if "BTCUSDT" in f: continue
        ix = pd.to_datetime(pd.read_parquet(f, columns=["l2_imb1"]).index, utc=True)
        if ((ix >= "2024-04-01") & (ix < "2024-07-01")).sum() > 80 and ((ix >= "2026-04-01") & (ix < "2026-07-01")).sum() > 80:
            ok.append(Path(f).stem[3:])
    return sorted(ok)[:n]

def main():
    syms = pilot_syms(50)
    days = pd.date_range("2026-04-01", "2026-06-30").append(pd.date_range("2024-04-01", "2024-06-30"))
    print(f"fetching imb1/2/3/5 for {len(syms)} syms x {len(days)} days...", flush=True)
    rows = []
    for i, sym in enumerate(syms, 1):
        out = load_symbol(sym, days)
        if out is None: continue
        cols = [f"l2_{c}" for c in LVL if f"l2_{c}" in out.columns]
        o = out[cols].copy(); o.columns = [c[3:] for c in cols]
        o["symbol"] = sym; o["obs_bar"] = pd.to_datetime(o.index, utc=True)
        rows.append(o.reset_index(drop=True))
        if i % 10 == 0: print(f"  {i}/{len(syms)}", flush=True)
    L = pd.concat(rows, ignore_index=True)
    L.to_parquet(SD / "pilot_imbdepth.parquet")
    print(f"saved {len(L)} rows, {L.symbol.nunique()} syms, cols {list(L.columns)}", flush=True)
    print("PILOTFETCHDONE")

if __name__ == "__main__":
    main()
