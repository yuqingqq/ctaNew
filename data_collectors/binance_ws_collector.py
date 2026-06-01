"""Real-time Binance USDⓈ-M market-data collector → Vision-format daily parquets.

Subscribes to <sym>@aggTrade (flow book) and <sym>@kline_5m (price-book live edge) for the
universe and writes the SAME daily-parquet layout the Vision archive produces, into the SAME
paths the convexity pipeline already reads:
    aggTrades → data/ml/test/parquet/aggTrades/<SYM>/<YYYY-MM-DD>.parquet
    klines    → data/ml/test/parquet/klines/<SYM>/5m/<YYYY-MM-DD>.parquet
So ingest_flow_daily.py and refresh_convexity_panel.py consume the live edge with NO code change
(binance_vision_loader._process_day short-circuits when a day's parquet already exists).

CRITICAL — routed endpoint. Binance Market streams ONLY push on the ROUTED path
`wss://fstream.binance.com/market/ws`. The legacy bare `/ws` connects and ACKs SUBSCRIBE but
sends zero market data (verified 2026-06-01). Mainnet-verified: WS aggId == REST aggId.

Robustness:
  - in-memory current-UTC-day buffers, dedup by agg_trade_id (aggTrade) / open_time (closed klines);
    flush-from-memory every FLUSH_SECONDS; evict the previous day after its final flush at rollover.
  - on startup: load any existing day-file back into the buffer, then REST gap-backfill today from the
    last buffered aggId so the current day has no hole before the socket connected.
  - on reconnect (drop or the 24h connection cap): REST gap-backfill per sym from last-seen aggId.
  - only CLOSED 5m klines are written (kline 'x' flag); never split a 5m bar across files.

Usage:
  PYTHONPATH=. .venv/bin/python data_collectors/binance_ws_collector.py --syms-file /tmp/universe174.txt
  PYTHONPATH=. .venv/bin/python data_collectors/binance_ws_collector.py --syms BTCUSDT ETHUSDT --no-backfill
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time, signal
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import websockets

REPO = Path("/home/yuqing/ctaNew")
KL_ROOT = REPO / "data/ml/test/parquet/klines"
AGG_ROOT = REPO / "data/ml/test/parquet/aggTrades"
WS_BASE = "wss://fstream.binance.com/market/ws"   # ROUTED — required for market data
FAPI = "https://fapi.binance.com"

STREAMS_PER_CONN = 150     # well under the 1024/conn cap; a few conns for resilience
FLUSH_SECONDS = 30
RECONNECT_24H = 23 * 3600  # proactively cycle before Binance's 24h connection limit

AGG_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
            "transact_time", "is_buyer_maker"]
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


class Collector:
    def __init__(self, syms: list[str], backfill: bool = True):
        self.syms = [s.upper() for s in syms]
        self.backfill = backfill
        # buffers[dataset][sym][day] = dict keyed by dedup-id → row-dict
        self.agg = defaultdict(lambda: defaultdict(dict))   # sym → day → {aggId: row}
        self.kl = defaultdict(lambda: defaultdict(dict))     # sym → day → {open_time_ms: row}
        self.dirty_agg = set()   # (sym, day)
        self.dirty_kl = set()
        self.last_aggid = {}     # sym → last aggId seen (for gap backfill)
        self.stop = False
        self.msg_count = 0
        self._sess = requests.Session()

    # ---- persistence ---------------------------------------------------------
    def _agg_path(self, sym, day): return AGG_ROOT / sym / f"{day}.parquet"
    def _kl_path(self, sym, day):  return KL_ROOT / sym / "5m" / f"{day}.parquet"

    def _load_existing_today(self):
        """Seed buffers from any existing current-day file so we continue, not clobber."""
        today = _day(int(time.time() * 1000))
        for sym in self.syms:
            p = self._agg_path(sym, today)
            if p.exists():
                df = pd.read_parquet(p)
                for r in df.to_dict("records"):
                    aid = int(r["agg_trade_id"])
                    r["transact_time"] = int(pd.Timestamp(r["transact_time"]).timestamp() * 1000)
                    self.agg[sym][today][aid] = r
                    self.last_aggid[sym] = max(self.last_aggid.get(sym, 0), aid)
            p = self._kl_path(sym, today)
            if p.exists():
                df = pd.read_parquet(p)
                for r in df.to_dict("records"):
                    t = int(pd.Timestamp(r["open_time"]).timestamp() * 1000)
                    r["open_time"] = t; r["close_time"] = int(pd.Timestamp(r["close_time"]).timestamp() * 1000)
                    self.kl[sym][today][t] = r

    def _flush(self):
        n = 0
        for (sym, day) in list(self.dirty_agg):
            rows = list(self.agg[sym][day].values())
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["transact_time"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
            df = df[AGG_COLS].sort_values("agg_trade_id").reset_index(drop=True)
            p = self._agg_path(sym, day); p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, compression="zstd", index=False)
            n += 1
        self.dirty_agg.clear()
        for (sym, day) in list(self.dirty_kl):
            rows = list(self.kl[sym][day].values())
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df[KL_COLS].sort_values("open_time").reset_index(drop=True)
            p = self._kl_path(sym, day); p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, compression="zstd", index=False)
            n += 1
        self.dirty_kl.clear()
        # evict any day older than today (already flushed) to bound memory
        today = _day(int(time.time() * 1000))
        for store in (self.agg, self.kl):
            for sym in store:
                for d in [d for d in store[sym] if d < today]:
                    del store[sym][d]
        return n

    # ---- message handling ----------------------------------------------------
    def _on_aggtrade(self, d):
        sym = d["s"]; aid = int(d["a"]); T = int(d["T"]); day = _day(T)
        self.agg[sym][day][aid] = {
            "agg_trade_id": aid, "price": float(d["p"]), "quantity": float(d["q"]),
            "first_trade_id": int(d["f"]), "last_trade_id": int(d["l"]),
            "transact_time": T, "is_buyer_maker": bool(d["m"])}
        self.dirty_agg.add((sym, day))
        self.last_aggid[sym] = max(self.last_aggid.get(sym, 0), aid)

    def _on_kline(self, d):
        k = d["k"]
        if not k["x"]:    # only persist CLOSED bars (never split a 5m bar across files)
            return
        sym = d["s"]; t = int(k["t"]); day = _day(t)
        self.kl[sym][day][t] = {
            "open_time": t, "open": float(k["o"]), "high": float(k["h"]), "low": float(k["l"]),
            "close": float(k["c"]), "volume": float(k["v"]), "close_time": int(k["T"]),
            "quote_volume": float(k["q"]), "count": int(k["n"]),
            "taker_buy_volume": float(k["V"]), "taker_buy_quote_volume": float(k["Q"])}
        self.dirty_kl.add((sym, day))

    # ---- REST gap backfill ---------------------------------------------------
    def _backfill_agg(self, sym, from_id):
        """Pull aggTrades fromId=from_id forward to fill a gap. Bounded, throttled."""
        url = f"{FAPI}/fapi/v1/aggTrades"
        cur = from_id; pulled = 0
        for _ in range(200):   # safety cap (~200k trades)
            try:
                r = self._sess.get(url, params={"symbol": sym, "fromId": cur, "limit": 1000}, timeout=15)
                if r.status_code in (418, 429):
                    time.sleep(5); continue
                r.raise_for_status(); chunk = r.json()
            except Exception:
                break
            if not chunk:
                break
            for t in chunk:
                self._on_aggtrade({"s": sym, "a": t["a"], "p": t["p"], "q": t["q"],
                                   "f": t["f"], "l": t["l"], "T": t["T"], "m": t["m"]})
            pulled += len(chunk); cur = chunk[-1]["a"] + 1
            if len(chunk) < 1000:
                break
            time.sleep(0.25)
        return pulled

    async def _backfill_all(self, reason):
        loop = asyncio.get_event_loop()
        total = 0
        for sym in self.syms:
            frm = self.last_aggid.get(sym)
            if not frm:
                continue
            total += await loop.run_in_executor(None, self._backfill_agg, sym, frm + 1)
        print(f"[ws] backfill ({reason}): +{total} aggTrades across gaps", flush=True)

    # ---- connection ----------------------------------------------------------
    async def _conn(self, streams, idx):
        url = f"{WS_BASE}"
        while not self.stop:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              max_queue=2**16, open_timeout=20) as ws:
                    # subscribe in sub-batches (respect 10 incoming msg/sec control limit)
                    for i in range(0, len(streams), 100):
                        await ws.send(json.dumps({"method": "SUBSCRIBE",
                                                  "params": streams[i:i + 100], "id": idx * 1000 + i}))
                        await asyncio.sleep(0.3)
                    print(f"[ws] conn#{idx} subscribed {len(streams)} streams", flush=True)
                    t_start = time.time()
                    async for raw in ws:
                        m = json.loads(raw)
                        if "e" not in m:      # subscribe/ack frames
                            continue
                        e = m["e"]
                        if e == "aggTrade":
                            self._on_aggtrade(m); self.msg_count += 1
                        elif e == "kline":
                            self._on_kline(m); self.msg_count += 1
                        if time.time() - t_start > RECONNECT_24H:
                            break   # cycle before the 24h cap
            except Exception as ex:
                print(f"[ws] conn#{idx} dropped: {type(ex).__name__} {str(ex)[:80]} — reconnecting", flush=True)
            if self.stop:
                break
            await asyncio.sleep(2)
            if self.backfill:
                await self._backfill_all(f"conn#{idx} reconnect")

    async def _flusher(self):
        while not self.stop:
            await asyncio.sleep(FLUSH_SECONDS)
            loop = asyncio.get_event_loop()
            n = await loop.run_in_executor(None, self._flush)
            print(f"[ws] flush: {n} day-files written | {self.msg_count} msgs total", flush=True)

    async def run(self):
        AGG_ROOT.mkdir(parents=True, exist_ok=True); KL_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[ws] loading existing today-files for {len(self.syms)} syms...", flush=True)
        self._load_existing_today()
        if self.backfill:
            await self._backfill_all("startup")
        streams = [f"{s.lower()}@aggTrade" for s in self.syms] + [f"{s.lower()}@kline_5m" for s in self.syms]
        chunks = [streams[i:i + STREAMS_PER_CONN] for i in range(0, len(streams), STREAMS_PER_CONN)]
        print(f"[ws] {len(streams)} streams over {len(chunks)} connection(s)", flush=True)
        tasks = [asyncio.create_task(self._conn(c, i)) for i, c in enumerate(chunks)]
        tasks.append(asyncio.create_task(self._flusher()))
        await asyncio.gather(*tasks)
        self._flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", nargs="*", default=None)
    ap.add_argument("--syms-file", default=None, help="file with whitespace-separated symbols")
    ap.add_argument("--no-backfill", action="store_true", help="skip REST gap-backfill (WS-only)")
    a = ap.parse_args()
    if a.syms_file:
        syms = Path(a.syms_file).read_text().split()
    elif a.syms:
        syms = a.syms
    else:
        print("provide --syms or --syms-file"); sys.exit(2)
    c = Collector(syms, backfill=not a.no_backfill)

    def _sig(*_):
        print("[ws] shutdown signal — final flush", flush=True); c.stop = True
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
