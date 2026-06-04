"""Event-driven boundary trigger — block until the next 4h boundary's CLOSING 5m kline has actually flushed
to the collector files for a quorum of reference symbols, then return. Replaces the real-time loop's fixed
grace so it fires as soon as the data is in (≈ the irreducible flush latency, usually a few seconds) instead
of a guessed delay. Decoupled from the collector by polling its parquet files (no IPC).

Prints the boundary + how many seconds past the close it took (the unavoidable data latency).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
KL = REPO / "data/ml/test/parquet/klines"
_syms = [s for s in (REPO / "live/state/collector_syms.txt").read_text().split() if s != "BTCUSDT"]
REF = ["BTCUSDT"] + _syms[:12]          # BTC + a 12-sym sample = cheap quorum proxy for "klines have flushed"
POLL_S = 4
CAP_MIN = 4                              # hard cap so a stalled feed can't hang the loop forever


def _latest_5m(sym):
    fs = sorted((KL / sym / "5m").glob("*.parquet"))
    if not fs:
        return None
    try:
        return pd.to_datetime(pd.read_parquet(fs[-1], columns=["open_time"])["open_time"], utc=True).max()
    except Exception:
        return None


def main():
    now = pd.Timestamp.utcnow()
    nb = now.floor("4h") + pd.Timedelta(hours=4)            # next 4h boundary
    s = (nb - now).total_seconds()
    if s > 0:
        time.sleep(s)
    need = nb - pd.Timedelta(minutes=5)                     # the 5m bar that CLOSES at nb (open_time nb-5m)
    quorum = max(1, int(0.8 * len(REF)))
    cap = nb + pd.Timedelta(minutes=CAP_MIN)
    ready = 0
    while pd.Timestamp.utcnow() < cap:
        ready = sum(1 for x in (_latest_5m(s) for s in REF) if x is not None and x >= need)
        if ready >= quorum:
            break
        time.sleep(POLL_S)
    waited = (pd.Timestamp.utcnow() - nb).total_seconds()
    print(f"[wait_bar_ready] boundary {nb} → {ready}/{len(REF)} ref syms have the close bar, {waited:.0f}s past close")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # bypass pyarrow/pandas atexit teardown that segfaults here → guarantees exit 0 for the supervisor
