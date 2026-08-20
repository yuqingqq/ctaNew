"""
Polymarket live-data price stream collector (P-2026-003).

wss://ws-live-data.polymarket.com, topic "crypto_prices" — the price feed
Polymarket's own UI uses for crypto markets (per-second, full_accuracy_value).
Probe-verified public 2026-08-19.

OPEN QUESTION (review-loop item): whether this equals the Chainlink 60s-TWAP
Data Stream that SETTLES the 5-min markets (resolutionSource:
data.chain.link/streams/{coin}-usd-twap-60s-streams) or is a spot mirror for
charts ("btcusdt" naming suggests possibly Binance spot). Until identified:
collect it AND compare offline vs our Binance bookTicker mid — identical ⇒
mirror; systematically distinct ⇒ candidate settlement input. Unknown topics
and non-crypto_prices payloads are stored raw for later schema discovery.

Rows: known topic → csv "recv_ns,t_ms,symbol,value_full"; else raw JSONL.
Storage: data/pm_5min/prices/<topic>/<YYYYMMDD_HH>.csv (gzip on rotation).

Run:  nohup python3 live/pm_research/collect_pm_prices.py > data/pm_5min/prices_collector.log 2>&1 &
"""
from __future__ import annotations

import asyncio
import gzip
import json
import shutil
import signal
import time
from collections import defaultdict
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/pm_5min/prices"
WS_URL = "wss://ws-live-data.polymarket.com"
SUB = {"action": "subscribe",
       "subscriptions": [
           {"topic": "crypto_prices", "type": "update"},          # Binance-spot mirror (iter-1 verdict)
           # THE SETTLEMENT STREAM (review iter 1): Chainlink RTDS TWAP relay,
           # 1s cadence, values 1e18-scaled, carries window_s. No replay exists —
           # unsubscribed time is truth lost. Stored raw JSONL per topic.
           {"topic": "crypto_prices_twap_sixty", "type": "update"},
           {"topic": "crypto_prices_twap_thirty", "type": "update"},
       ]}
FLUSH_S = 5
WATCHDOG_S = 30
HEARTBEAT_S = 60


class PriceCollector:
    def __init__(self):
        self.stop = False
        self.buf: dict[str, list[str]] = defaultdict(list)
        self.open_hour: dict[str, str] = {}
        self.counts = defaultdict(int)

    def _on_msg(self, raw: str, recv_ns: int) -> None:
        try:
            m = json.loads(raw)
        except Exception:
            return
        topic = m.get("topic") or "unknown"
        p = m.get("payload") or {}
        if topic == "crypto_prices" and "symbol" in p:
            line = f"{recv_ns},{p.get('timestamp','')},{p['symbol']},{p.get('full_accuracy_value', p.get('value',''))}"
        else:
            line = f"{recv_ns}\t{raw}"
        self.buf[topic].append(line)
        self.counts[topic] += 1

    def _flush(self) -> None:
        now_hour = time.strftime("%Y%m%d_%H", time.gmtime())
        for topic, lines in list(self.buf.items()):
            if not lines:
                continue
            self.buf[topic] = []
            d = ROOT / topic
            d.mkdir(parents=True, exist_ok=True)
            prev = self.open_hour.get(topic)
            if prev and prev != now_hour:
                p = d / f"{prev}.csv"
                if p.exists():
                    try:
                        with open(p, "rb") as src, gzip.open(str(p) + ".gz", "wb",
                                                             compresslevel=6) as dst:
                            shutil.copyfileobj(src, dst)
                        p.unlink()
                    except Exception as ex:
                        print(f"[pmp] gzip: {ex}", flush=True)
            self.open_hour[topic] = now_hour
            with open(d / f"{now_hour}.csv", "a") as f:
                f.write("\n".join(lines) + "\n")

    async def _conn(self) -> None:
        while not self.stop:
            try:
                async with websockets.connect(WS_URL, ping_interval=15, ping_timeout=15,
                                              open_timeout=15) as ws:
                    await ws.send(json.dumps(SUB))
                    print("[pmp] subscribed crypto_prices", flush=True)
                    while not self.stop:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=WATCHDOG_S)
                        except asyncio.TimeoutError:
                            print(f"[pmp] silent {WATCHDOG_S}s — reconnect", flush=True)
                            break
                        if raw:
                            self._on_msg(raw, time.time_ns())
            except Exception as ex:
                print(f"[pmp] conn: {type(ex).__name__} {str(ex)[:80]} — retry", flush=True)
            if not self.stop:
                await asyncio.sleep(2)

    async def _sleep(self, secs: float) -> None:
        t0 = time.time()
        while not self.stop and time.time() - t0 < secs:
            await asyncio.sleep(1)

    async def _housekeeping(self) -> None:
        n = 0
        while not self.stop:
            await self._sleep(FLUSH_S)
            self._flush()
            n += 1
            if n % (HEARTBEAT_S // FLUSH_S) == 0:
                print(f"[pmp] {time.strftime('%H:%M:%S', time.gmtime())}Z "
                      + " ".join(f"{k}={v}" for k, v in sorted(self.counts.items())), flush=True)

    async def run(self) -> None:
        ROOT.mkdir(parents=True, exist_ok=True)
        await asyncio.gather(self._conn(), self._housekeeping())
        self._flush()
        print("[pmp] stopped", flush=True)


def main() -> None:
    c = PriceCollector()

    def _sig(*_):
        print("[pmp] shutdown", flush=True)
        c.stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
