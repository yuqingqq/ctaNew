"""Read-only probe: does the Binance routed kline WS deliver the SAME closed bar more than once?

Subscribes to @kline_5m for ~120 symbols across 2 connections (mirroring the collector's 60/conn) on the
routed /market/ws endpoint, listens through one 5m close, and counts how many times each (symbol, kline_open)
closed bar (x=true) is received. dups = any (sym,t) seen >1. Pure measurement; touches no collector state.
"""
import asyncio, json, time, collections, sys
from pathlib import Path
import websockets

WS = "wss://fstream.binance.com/market/ws"
REPO = Path("/home/yuqing/ctaNew")
syms = REPO.joinpath("live/state/collector_syms.txt").read_text().split()[:120]
RUN_S = 380

counts = collections.Counter()          # (sym, kline_open_ms) -> times received
arrivals = {}                           # (sym, t) -> [recv_times]


async def conn(chunk, cid):
    streams = [f"{s.lower()}@kline_5m" for s in chunk]
    async with websockets.connect(WS, ping_interval=180, max_queue=None, open_timeout=20) as ws:
        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": cid}))
        t_end = time.time() + RUN_S
        while time.time() < t_end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            d = json.loads(msg)
            if "data" in d:
                d = d["data"]
            k = d.get("k")
            if not k or not k.get("x"):          # only CLOSED bars
                continue
            key = (d["s"], int(k["t"]))
            counts[key] += 1
            arrivals.setdefault(key, []).append(round(time.time(), 2))


async def main():
    chunks = [syms[:60], syms[60:120]]
    await asyncio.gather(*[conn(c, i) for i, c in enumerate(chunks)], return_exceptions=True)
    closed = sum(counts.values())
    distinct = len(counts)
    dups = {k: v for k, v in counts.items() if v > 1}
    print(f"\n=== WS DUP PROBE ({len(syms)} syms, {RUN_S}s) ===")
    print(f"closed-bar frames received: {closed}")
    print(f"distinct (sym, bar) pairs:  {distinct}")
    print(f"DUPLICATED pairs (>1):      {len(dups)}")
    if dups:
        for (s, t), n in sorted(dups.items(), key=lambda x: -x[1])[:15]:
            gaps = arrivals[(s, t)]
            print(f"   {s} bar@{t}: received {n}x at {gaps} (spread {gaps[-1]-gaps[0]:.1f}s)")
    else:
        print("   → NO duplicates: every closed bar arrived exactly once.")


if __name__ == "__main__":
    asyncio.run(main())
