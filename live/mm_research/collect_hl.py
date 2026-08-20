"""
Hyperliquid forward data collector (Variant-B screen prerequisite, P-2026-002).

Streams per coin over the single HL websocket (wss://api.hyperliquid.xyz/ws):
  bbo      best bid/ask, pushed on BBO change
  l2Book   book snapshots (HL pushes full snapshots — no diff reconstruction),
           stored top BOOK_LEVELS per side
  trades   prints; side field recorded raw ("B"/"A") — side semantics are
           VERIFIED EMPIRICALLY in the screen (px vs prevailing bbo), not assumed

Design notes:
- HL requires APPLICATION-level ping ({"method":"ping"} → {"channel":"pong"});
  websocket protocol pings alone may not keep the session alive. Sent every 30 s.
- Universe is fetched at startup from POST /info {"type":"meta"} and intersected
  with the requested list; missing coins are warned, not fatal. If REST /info is
  unreachable (possible US geo-restrictions), the requested list is used as-is —
  subscriptions to unlisted coins simply never push, visible in the heartbeat.
- Storage mirrors collect_hf.py: per-hour CSV under data/mm_hf/hl_raw/, gzipped
  on rotation. data/ is gitignored.

Rows (recv_ns = local wall clock at parse; t_ms = exchange event time):
  bbo:    recv_ns,t_ms,bid_px,bid_sz,ask_px,ask_sz
  l2Book: recv_ns,t_ms,bids,asks       (levels "px@sz@n|..." top BOOK_LEVELS)
  trades: recv_ns,t_ms,tid,px,sz,side  (side raw: B/A)

Run:  nohup python3 live/mm_research/collect_hl.py > data/mm_hf/hl_collector.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import shutil
import signal
import time
from collections import defaultdict
from pathlib import Path

import requests
import websockets

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/mm_hf"
RAW = ROOT / "hl_raw"

WS_URL = "wss://api.hyperliquid.xyz/ws"
INFO_URL = "https://api.hyperliquid.xyz/info"
BOOK_LEVELS = 10
FLUSH_SECONDS = 5
APP_PING_SECONDS = 30
WATCHDOG_SECS = 60
HEARTBEAT_SECS = 60

# Binance-pilot equivalents (cross-venue comparison). HL lists majors under plain
# coin names. The screen may add HL-native mid/tails once first spreads are seen.
DEFAULT_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX",
                 "LTC", "APT", "ARB", "FIL", "ATOM", "AAVE", "GMX", "ICP"]


def fetch_universe(coins: list[str]) -> list[str]:
    """Intersect requested coins with the live HL perp universe (best effort)."""
    try:
        meta = requests.post(INFO_URL, json={"type": "meta"}, timeout=20).json()
        listed = {a["name"] for a in meta.get("universe", [])}
        missing = [c for c in coins if c not in listed]
        if missing:
            print(f"[hl] not listed on HL (skipped): {missing}", flush=True)
        kept = [c for c in coins if c in listed]
        (ROOT / f"hl_meta_{time.strftime('%Y%m%d', time.gmtime())}.json").write_text(
            json.dumps(meta, indent=1))
        return kept or coins
    except Exception as ex:
        print(f"[hl] /info meta fetch failed ({type(ex).__name__}: {str(ex)[:80]}) — "
              f"using requested list unverified", flush=True)
        return coins


class HLCollector:
    def __init__(self, coins: list[str]):
        self.coins = coins
        self.stop = False
        self.buf: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.open_hour: dict[tuple[str, str], str] = {}
        self.counts = defaultdict(int)
        self.last_msg = time.time()

    # ---- handlers -------------------------------------------------------------
    def _on_msg(self, m: dict, recv_ns: int) -> None:
        ch = m.get("channel")
        d = m.get("data")
        if ch == "pong" or d is None:
            return
        if ch == "bbo":
            coin = d.get("coin", "")
            bbo = d.get("bbo", [None, None])
            b, a = bbo[0] or {}, bbo[1] or {}
            line = (f"{recv_ns},{d.get('time', 0)},{b.get('px', '')},{b.get('sz', '')},"
                    f"{a.get('px', '')},{a.get('sz', '')}")
            self._push(("bbo", coin), line)
        elif ch == "l2Book":
            coin = d.get("coin", "")
            lv = d.get("levels", [[], []])
            bids = "|".join(f"{x['px']}@{x['sz']}@{x.get('n', '')}" for x in lv[0][:BOOK_LEVELS])
            asks = "|".join(f"{x['px']}@{x['sz']}@{x.get('n', '')}" for x in lv[1][:BOOK_LEVELS])
            self._push(("l2Book", coin), f"{recv_ns},{d.get('time', 0)},{bids},{asks}")
        elif ch == "trades":
            for tr in (d if isinstance(d, list) else [d]):
                coin = tr.get("coin", "")
                self._push(("trades", coin),
                           f"{recv_ns},{tr.get('time', 0)},{tr.get('tid', '')},"
                           f"{tr.get('px', '')},{tr.get('sz', '')},{tr.get('side', '')}")
        else:
            return
        self.last_msg = time.time()

    def _push(self, key: tuple[str, str], line: str) -> None:
        self.buf[key].append(line)
        self.counts[key[0]] += 1

    # ---- storage (same layout as collect_hf) ----------------------------------
    def _flush(self) -> int:
        now_hour = time.strftime("%Y%m%d_%H", time.gmtime())
        n = 0
        for key, lines in list(self.buf.items()):
            if not lines:
                continue
            self.buf[key] = []
            stream, coin = key
            d = RAW / stream / coin
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
            print(f"[hl] gzip {p.name} failed: {ex} (left uncompressed)", flush=True)

    # ---- connection -----------------------------------------------------------
    async def _conn(self) -> None:
        while not self.stop:
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20,
                                              max_queue=2 ** 16, open_timeout=20) as ws:
                    for coin in self.coins:
                        for typ in ("bbo", "l2Book", "trades"):
                            await ws.send(json.dumps({"method": "subscribe",
                                                      "subscription": {"type": typ, "coin": coin}}))
                        await asyncio.sleep(0.05)
                    print(f"[hl] subscribed {len(self.coins)} coins × 3 streams", flush=True)
                    last_ping = time.time()
                    while not self.stop:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=WATCHDOG_SECS)
                        except asyncio.TimeoutError:
                            print(f"[hl] silent {WATCHDOG_SECS}s — reconnecting", flush=True)
                            break
                        self._on_msg(json.loads(raw), time.time_ns())
                        if time.time() - last_ping > APP_PING_SECONDS:
                            await ws.send(json.dumps({"method": "ping"}))
                            last_ping = time.time()
            except Exception as ex:
                print(f"[hl] conn dropped: {type(ex).__name__} {str(ex)[:100]} — reconnecting",
                      flush=True)
            if not self.stop:
                await asyncio.sleep(2)

    async def _sleep(self, secs: float) -> None:
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
            print(f"[hl] {time.strftime('%H:%M:%S', time.gmtime())}Z "
                  + " ".join(f"{k}={v}" for k, v in sorted(self.counts.items())), flush=True)

    async def run(self) -> None:
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"[hl] {len(self.coins)} coins: {self.coins}", flush=True)
        await asyncio.gather(asyncio.create_task(self._conn()),
                             asyncio.create_task(self._flusher()),
                             asyncio.create_task(self._heartbeat()))
        self._flush()
        print("[hl] stopped, final flush done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="*", default=None)
    a = ap.parse_args()
    coins = fetch_universe(a.coins or DEFAULT_COINS)
    c = HLCollector(coins)

    def _sig(*_):
        print("[hl] shutdown signal — final flush", flush=True)
        c.stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
