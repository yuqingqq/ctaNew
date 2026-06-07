"""Bounded wait for the just-closed boundary 5m bar to propagate, BEFORE features/decide.

The decision's cross-sectional xs-rank ranks over the whole ~174 panel cohort, so the build needs the
boundary 5m bar (the bar AT the 4h boundary) for ~all symbols. That bar closes at boundary+5m and the WS/FAPI
deliver it within seconds-to-tens-of-seconds — but the cycle fires at ~boundary+5m+3s, before the slower
(thin/illiquid) symbols arrive, so ~40-60 are missing it and the cohort guard aborts. The old 89s backfill
hid this by waiting long enough for FAPI to catch up; the fast cycle has to wait explicitly.

Poll the boundary-date day files until >= MIN symbols have the boundary bar (or timeout), then return — the
cycle's backfill then fills any genuine remaining gaps via FAPI (which has caught up by then).

Usage: python3 live/wait_boundary_bar.py <boundary> [min=155] [timeout_s=35]
"""
import sys, time, glob
from pathlib import Path
import pandas as pd
REPO = Path("/home/yuqing/ctaNew")
KLINES = REPO / "data/ml/test/parquet/klines"


def _have_bar(sym, day, B):
    p = KLINES / sym / "5m" / f"{day}.parquet"
    if not p.exists():
        return False
    try:
        m = pd.to_datetime(pd.read_parquet(p, columns=["open_time"])["open_time"], utc=True)
        return bool((m == B).any())
    except Exception:
        return False


def main():
    B = pd.Timestamp(sys.argv[1])
    if B.tzinfo is None:
        B = B.tz_localize("UTC")
    MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 155
    TIMEOUT = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0
    syms = REPO.joinpath("live/state/collector_syms.txt").read_text().split()
    day = B.strftime("%Y-%m-%d")
    t0 = time.time()
    while True:
        have = sum(_have_bar(s, day, B) for s in syms)
        if have >= MIN:
            print(f"[wait_boundary] {have}/{len(syms)} have {B:%m-%d %H:%M} bar after {time.time()-t0:.0f}s → proceed", flush=True)
            return 0
        if time.time() - t0 >= TIMEOUT:
            print(f"[wait_boundary] TIMEOUT {time.time()-t0:.0f}s — only {have}/{len(syms)} have {B:%m-%d %H:%M} bar (backfill will fetch the rest)", flush=True)
            return 0
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
