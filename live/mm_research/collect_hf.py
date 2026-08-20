"""
HF market-making data collector (E0 of the MM research program, live/mm_research/).

Streams per symbol on Binance USDM futures:
  <s>@bookTicker      best bid/ask, realtime      (~50-300 msg/s on majors)
  <s>@depth20@100ms   top-20 partial book snapshot (10 msg/s, self-contained — no
                      diff-book reconstruction needed; futures partial payloads carry
                      e="depthUpdate" with b/a = full top-20 arrays)
  <s>@trade           raw per-match trades, realtime (NOTE: @aggTrade delivers NOTHING on
                      the URL-subscribed fstream endpoints — verified 2026-08-19; @trade
                      works and is finer-grained. Historical aggTrades come from Vision.)

CRITICAL — endpoint pitfall (verified 2026-08-19): on fstream, bookTicker and
depth20@100ms do NOT push via SUBSCRIBE on the routed /market/ws path (ACK arrives,
zero data — inverse of the kline pitfall in data_collectors/binance_ws_collector.py,
where the bare /ws is the silent one). They DO push with URL-based subscription:
combined /stream?streams=a/b/c (payload wrapped {"stream":...,"data":{...}}) or raw
/ws/<stream>. This collector therefore uses the COMBINED endpoint for all streams.

Storage: append-only CSV per (stream, symbol, UTC hour) under data/mm_hf/raw/,
gzipped on hour rotation. Rough steady-state: ~0.5-1 GB/day gzipped for the default
16-symbol pilot (bookTicker on BTC/ETH dominates). data/ is gitignored.

Rows (recv_ns = local monotonic-free wall clock ns at parse time; E/T = exchange
event/transaction ms — recv_ns/1e6 − E estimates one-way latency + clock offset):
  bookTicker: recv_ns,E,T,u,bid,bid_qty,ask,ask_qty
  depth20:    recv_ns,E,T,u,bids,asks         (bids/asks = "p@q|p@q|..." 20 levels)
  trade:      recv_ns,E,T,trade_id,price,qty,is_buyer_maker

Run:  nohup python3 live/mm_research/collect_hf.py > data/mm_hf/collector.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import shutil
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
import websockets

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/mm_hf"
RAW = ROOT / "raw"

WS_COMBINED = "wss://fstream.binance.com/stream?streams="  # URL-subscribed combined streams
FAPI = "https://fapi.binance.com"
STREAMS_PER_CONN = 30      # small per-conn: a drop loses few symbols, light recv queue
RECONNECT_24H = 23 * 3600  # cycle before Binance's 24h connection cap
FLUSH_SECONDS = 5
WATCHDOG_SECS = 45         # no message in 45s → force reconnect (silent-death guard)
HEARTBEAT_SECS = 60

# Spread-diverse pilot: majors (control — expected uneconomic at the touch) through
# wide-spread mid-caps (the candidate niche). E1's universe scan should revise this.
DEFAULT_SYMS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
    "ADAUSDT", "AVAXUSDT", "LTCUSDT", "APTUSDT", "ARBUSDT", "FILUSDT",
    "ATOMUSDT", "AAVEUSDT", "GMXUSDT", "ICPUSDT",
]


def _snapshot_exchange_info(syms: list[str]) -> None:
    """Save tick/step sizes once per day — needed to convert spreads to ticks/bps."""
    out = ROOT / f"exchange_info_{time.strftime('%Y%m%d', time.gmtime())}.json"
    if out.exists():
        return
    try:
        info = requests.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=20).json()
        keep = {}
        for s in info.get("symbols", []):
            if s["symbol"] in syms:
                filt = {f["filterType"]: f for f in s.get("filters", [])}
                keep[s["symbol"]] = {
                    "tickSize": filt.get("PRICE_FILTER", {}).get("tickSize"),
                    "stepSize": filt.get("LOT_SIZE", {}).get("stepSize"),
                    "pricePrecision": s.get("pricePrecision"),
                }
        out.write_text(json.dumps(keep, indent=1))
        print(f"[hf] exchangeInfo snapshot → {out.name} ({len(keep)} syms)", flush=True)
    except Exception as ex:  # non-fatal — retry next startup
        print(f"[hf] exchangeInfo snapshot failed: {ex}", flush=True)


class HFCollector:
    def __init__(self, syms: list[str]):
        self.syms = [s.upper() for s in syms]
        self.stop = False
        self.buf: dict[tuple[str, str], list[str]] = defaultdict(list)  # (stream,sym) → lines
        self.open_hour: dict[tuple[str, str], str] = {}                 # (stream,sym) → "YYYYMMDD_HH"
        self.counts = defaultdict(int)
        self.lat_ms: list[float] = []          # recv-vs-event latency samples (1/100 sampled)
        self.last_msg = time.time()

    # ---- message handlers -----------------------------------------------------
    def _on_msg(self, m: dict, recv_ns: int) -> None:
        e = m.get("e")
        sym = m.get("s", "")
        E, T = m.get("E", 0), m.get("T", 0)
        if e == "bookTicker":
            line = f"{recv_ns},{E},{T},{m['u']},{m['b']},{m['B']},{m['a']},{m['A']}"
            key = ("bookTicker", sym)
        elif e == "depthUpdate":  # partial-book 20-level snapshot (we subscribe no diff streams)
            bids = "|".join(f"{p}@{q}" for p, q in m["b"])
            asks = "|".join(f"{p}@{q}" for p, q in m["a"])
            line = f"{recv_ns},{E},{T},{m['u']},{bids},{asks}"
            key = ("depth20", sym)
        elif e == "trade":
            line = f"{recv_ns},{E},{T},{m['t']},{m['p']},{m['q']},{int(m['m'])}"
            key = ("trade", sym)
        else:
            return
        self.buf[key].append(line)
        self.counts[key[0]] += 1
        if E and self.counts[key[0]] % 100 == 0:
            self.lat_ms.append(recv_ns / 1e6 - E)
        self.last_msg = time.time()

    # ---- storage --------------------------------------------------------------
    def _flush(self) -> int:
        """Append buffered lines to per-hour CSVs; gzip closed hours. Runs in executor."""
        now_hour = time.strftime("%Y%m%d_%H", time.gmtime())
        n = 0
        for key, lines in list(self.buf.items()):
            if not lines:
                continue
            self.buf[key] = []
            stream, sym = key
            d = RAW / stream / sym
            d.mkdir(parents=True, exist_ok=True)
            prev = self.open_hour.get(key)
            if prev and prev != now_hour:
                self._gzip_closed(d / f"{prev}.csv")
            self.open_hour[key] = now_hour
            with open(d / f"{now_hour}.csv", "a") as f:
                f.write("\n".join(lines) + "\n")
            n += len(lines)
        return n

    @staticmethod
    def _gzip_closed(p: Path) -> None:
        if not p.exists():
            return
        try:
            with open(p, "rb") as src, gzip.open(str(p) + ".gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            p.unlink()
        except Exception as ex:
            print(f"[hf] gzip {p.name} failed: {ex} (left uncompressed)", flush=True)

    # ---- connection -----------------------------------------------------------
    async def _conn(self, streams: list[str], idx: int) -> None:
        while not self.stop:
            try:
                url = WS_COMBINED + "/".join(streams)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              max_queue=2 ** 16, open_timeout=20) as ws:
                    print(f"[hf] conn#{idx} connected, {len(streams)} URL streams", flush=True)
                    t_start = time.time()
                    while not self.stop:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=WATCHDOG_SECS)
                        except asyncio.TimeoutError:
                            print(f"[hf] conn#{idx} silent {WATCHDOG_SECS}s — reconnecting", flush=True)
                            break
                        m = json.loads(raw)
                        m = m.get("data", m)          # combined-stream wrapper
                        if "e" in m:
                            self._on_msg(m, time.time_ns())
                        if time.time() - t_start > RECONNECT_24H:
                            print(f"[hf] conn#{idx} 23h cycle", flush=True)
                            break
            except Exception as ex:
                print(f"[hf] conn#{idx} dropped: {type(ex).__name__} {str(ex)[:80]} — reconnecting",
                      flush=True)
            if not self.stop:
                await asyncio.sleep(2)

    async def _sleep(self, secs: float) -> None:
        """Stop-aware sleep so shutdown isn't delayed by long intervals."""
        t0 = time.time()
        while not self.stop and time.time() - t0 < secs:
            await asyncio.sleep(1)

    async def _flusher(self) -> None:
        loop = asyncio.get_event_loop()
        while not self.stop:
            await self._sleep(FLUSH_SECONDS)
            await loop.run_in_executor(None, self._flush)

    async def _heartbeat(self) -> None:
        while not self.stop:
            await self._sleep(HEARTBEAT_SECS)
            if self.stop:
                break
            lat = sorted(self.lat_ms)
            med = lat[len(lat) // 2] if lat else float("nan")
            self.lat_ms = []
            print(f"[hf] {time.strftime('%H:%M:%S', time.gmtime())}Z "
                  + " ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
                  + f" | recv-lat~{med:.0f}ms (incl clock offset)", flush=True)

    async def run(self) -> None:
        RAW.mkdir(parents=True, exist_ok=True)
        _snapshot_exchange_info(self.syms)
        streams = [f"{s.lower()}@{ch}" for s in self.syms
                   for ch in ("bookTicker", "depth20@100ms", "trade")]
        chunks = [streams[i:i + STREAMS_PER_CONN] for i in range(0, len(streams), STREAMS_PER_CONN)]
        print(f"[hf] {len(self.syms)} syms, {len(streams)} streams over {len(chunks)} conn(s)", flush=True)
        tasks = [asyncio.create_task(self._conn(c, i)) for i, c in enumerate(chunks)]
        tasks += [asyncio.create_task(self._flusher()), asyncio.create_task(self._heartbeat())]
        await asyncio.gather(*tasks)
        self._flush()
        print("[hf] stopped, final flush done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", nargs="*", default=None, help="override the default 16-symbol pilot")
    a = ap.parse_args()
    c = HFCollector(a.syms or DEFAULT_SYMS)

    def _sig(*_):
        print("[hf] shutdown signal — final flush", flush=True)
        c.stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
