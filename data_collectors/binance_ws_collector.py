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
import argparse, asyncio, json, sys, time, signal, subprocess, threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import websockets

REPO = Path("/home/yuqing/ctaNew")
KL_ROOT = REPO / "data/ml/test/parquet/klines"
AGG_ROOT = REPO / "data/ml/test/parquet/aggTrades"
FUND_ROOT = REPO / "data/ml/cache"                # funding_{sym}.parquet — fed by @markPrice subscription
WS_BASE = "wss://fstream.binance.com/market/ws"   # ROUTED — required for market data
FAPI = "https://fapi.binance.com"

STREAMS_PER_CONN = 60      # smaller per-conn so a drop loses fewer syms; lighter queue (was 150 → kline drops)
FLUSH_SECONDS = 5          # write closed bars to disk fast so the cycle backfill doesn't race a slow flush
RECONNECT_24H = 23 * 3600  # proactively cycle before Binance's 24h connection limit
FOUR_H_MS = 4 * 3600 * 1000
# FIXED GRACE after the 4h boundary bar closes, before flush+fire. The PRIMARY fix is the _flush orphan-race
# (difference_update not .clear) so late-arriving boundary klines are no longer silently dropped; this grace is
# the belt-and-suspenders margin so the bars are ON DISK before the cycle reads — receipt (~4s spread) + at least
# one 5s drain flush. 24s spans ~4 flush cycles. The earlier "adaptive" grace fired on RECEIPT at ~+4s, before
# the tail was persisted → cohorts 130/126/161 at 00/04/12:00; the only full cohort (173 @ 08:00) happened to
# fire late at +22s. (NOT a 15-20s disk-write latency — a flush is sub-second; it's the drain-flush count.)
GRACE_SECONDS = 24.0       # s fixed wait — receipt spread + ≥1 drain flush, with margin (was the +4s regression)
CYCLE_SCRIPT = REPO / "live/convexity_v1_cycle_once.sh"   # decision pipeline, push-triggered on boundary

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
        self.fund = defaultdict(dict)   # sym → {calc_time_ms: row}  (settled funding via @markPrice)
        self.dirty_fund = set()
        self.last_fund = {}      # sym → {"r": float, "T": int}  (track settlement transitions)
        self.stop = False
        self.msg_count = 0
        self._sess = requests.Session()
        self.triggered_B = 0                  # last 4h-boundary (ms) we push-triggered (dedupe across syms)
        self._flush_lock = threading.Lock()   # serialize _flush (30s flusher vs boundary-trigger flush)

    # ---- persistence ---------------------------------------------------------
    def _agg_path(self, sym, day): return AGG_ROOT / sym / f"{day}.parquet"
    def _kl_path(self, sym, day):  return KL_ROOT / sym / "5m" / f"{day}.parquet"

    def _load_existing_today(self):
        """Seed buffers from any existing current-day file so we continue, not clobber."""
        today = _day(int(time.time() * 1000))
        for sym in self.syms:
            # NOTE: aggTrade not loaded — v1 collector streams only klines + markPrice (no flow).
            try:                                          # robust: a corrupt day-file must not crash startup
                p = self._kl_path(sym, today)
                if p.exists():
                    df = pd.read_parquet(p)
                    for r in df.to_dict("records"):
                        t = int(pd.Timestamp(r["open_time"]).timestamp() * 1000)
                        r["open_time"] = t; r["close_time"] = int(pd.Timestamp(r["close_time"]).timestamp() * 1000)
                        self.kl[sym][today][t] = r
            except Exception as e:
                print(f"[ws] skip corrupt kl {sym} {today}: {type(e).__name__}", flush=True)

    def _flush(self):
      with self._flush_lock:                  # serializes flush-vs-flush. NOTE: does NOT serialize against
        n = 0                                 # _on_kline on the loop thread — to_parquet() below releases the GIL,
        # ORPHAN-RACE FIX: snapshot the dirty set, write it, then remove ONLY the snapshot (difference_update),
        # NOT a blanket .clear(). A boundary bar that _on_kline adds DURING the write window would have its
        # (sym,day) flag wiped by .clear() and never get written until the symbol's next bar re-dirties it ~5min
        # later — that was the 13-symbol boundary-cohort drop (06-07 00/04/12:00). difference_update keeps those
        # late flags so the NEXT 5s flush writes them. (kl is what the decide cohort reads; agg/fund same hazard.)
        snap_agg = list(self.dirty_agg)
        for (sym, day) in snap_agg:
            rows = list(self.agg[sym][day].values())
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["transact_time"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
            df = df[AGG_COLS].sort_values("agg_trade_id").reset_index(drop=True)
            p = self._agg_path(sym, day); p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp"); df.to_parquet(tmp, compression="zstd", index=False); tmp.replace(p)
            n += 1
        self.dirty_agg.difference_update(snap_agg)
        snap_kl = list(self.dirty_kl)
        for (sym, day) in snap_kl:
            rows = list(self.kl[sym][day].values())
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df[KL_COLS].sort_values("open_time").reset_index(drop=True)
            p = self._kl_path(sym, day); p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp"); df.to_parquet(tmp, compression="zstd", index=False); tmp.replace(p)
            n += 1
        self.dirty_kl.difference_update(snap_kl)
        snap_fund = list(self.dirty_fund)
        for sym in snap_fund:
            rows = list(self.fund[sym].values())
            if not rows:
                continue
            new = pd.DataFrame(rows); new["calc_time"] = pd.to_datetime(new["calc_time"], unit="ms", utc=True)
            new = new[["calc_time", "interval_hours", "funding_rate"]]
            p = FUND_ROOT / f"funding_{sym}.parquet"
            if p.exists():
                old = pd.read_parquet(p); old["calc_time"] = pd.to_datetime(old["calc_time"], utc=True)
                comb = pd.concat([old, new], ignore_index=True)
            else:
                comb = new
            comb = comb.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)
            tmp = p.with_name(p.name + ".tmp"); comb.to_parquet(tmp, index=False); tmp.replace(p); n += 1
        self.dirty_fund.difference_update(snap_fund)
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
        if t % 300_000 != 0 or t > int(time.time() * 1000) + 60_000:   # reject mis-aligned / future open_time so a
            return                                                     # replay/garbage frame can't poison the buffer
        self.kl[sym][day][t] = {
            "open_time": t, "open": float(k["o"]), "high": float(k["h"]), "low": float(k["l"]),
            "close": float(k["c"]), "volume": float(k["v"]), "close_time": int(k["T"]),
            "quote_volume": float(k["q"]), "count": int(k["n"]),
            "taker_buy_volume": float(k["V"]), "taker_buy_quote_volume": float(k["Q"])}
        self.dirty_kl.add((sym, day))
        # INSTRUMENTATION: log how many boundary klines have been RECEIVED, and when (vs the flush logs which
        # show when they're WRITTEN). At a 4h funding-settlement boundary this separates "Binance pushes the
        # boundary bar slowly" (received late) from "collector stalls writing it" (received on time, written late).
        if t % FOUR_H_MS == 0:
            # MEASURE the cohort fill: DISTINCT symbols (set) vs RAW messages (counter). If msgs > distinct at
            # fire time, the closed bar is arriving more than once (duplicates — source TBD); if msgs == distinct
            # but distinct < ~171, the bars are genuinely arriving slowly. The grace waits on DISTINCT either way.
            self._brx = getattr(self, "_brx", {}); self._bmsg = getattr(self, "_bmsg", {})
            seen = self._brx.setdefault(t, set()); fresh = sym not in seen; seen.add(sym)
            self._bmsg[t] = self._bmsg.get(t, 0) + 1
            n = len(seen)
            if fresh and n in (1, 40, 80, 120, 150, 170, 174):
                print(f"[ws-instr] boundary {datetime.fromtimestamp(t/1000, tz=timezone.utc):%H:%M}: "
                      f"{n} distinct / {self._bmsg[t]} msgs at +{time.time()-(t/1000+300):.0f}s after close", flush=True)
        # PUSH TRIGGER: fire when the FIRST 5m bar of a new 4h period closes — i.e. this bar's OPEN is the
        # boundary (t % 4h == 0). That is exactly when the 4h decide-bar `t` first becomes buildable: its
        # features come from this 5m row (_build_sym_window anchors the 4h bar to the 5m row at that
        # open_time). Firing on the bar that closes AT the boundary is ~5min too early — the 4h row doesn't
        # exist yet → build_bar fails → stale preds. Deduped via triggered_B; guarded so any error stays out
        # of the WS feed; the fallback watchdog covers a miss.
        if t % FOUR_H_MS == 0 and t > self.triggered_B:
            self.triggered_B = t
            try:
                asyncio.create_task(self._fire_boundary(t))
            except RuntimeError:
                pass   # no running loop (e.g. during backfill) — fallback covers it

    async def _fire_boundary(self, B: int):
        """Push the decision the instant a 4h bar closes: brief grace for stragglers, flush so the reader
        sees the bar on disk, then spawn cycle_once detached. Never raises into the WS loop."""
        try:
            # FIXED GRACE: wait the disk-spread window so the boundary klines are WRITTEN (not just received)
            # before the cycle reads them. Receipt-based early-fire was the 12:00 regression (see GRACE_SECONDS).
            t_close = B / 1000 + 300                                  # the boundary 5m bar closes at B+5min
            await asyncio.sleep(GRACE_SECONDS)
            await asyncio.get_event_loop().run_in_executor(None, self._flush)
            subprocess.Popen(["bash", str(CYCLE_SCRIPT), "collector"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            for k in [k for k in getattr(self, "_brx", {}) if k < B]:           # bound memory: drop old boundaries
                self._brx.pop(k, None); getattr(self, "_bmsg", {}).pop(k, None)
            ts = datetime.fromtimestamp(B / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            seen = getattr(self, "_brx", {}).get(B, set()); rx = len(seen); waited = time.time() - t_close
            msgs = getattr(self, "_bmsg", {}).get(B, 0)
            missing = [s for s in self.syms if s not in seen][:10]              # WHO is absent at fire time
            print(f"[ws] boundary {ts} → {rx} distinct / {msgs} msgs in {waited:.0f}s | missing[{len(self.syms)-rx}]: "
                  f"{missing} → flushed + push-triggered cycle_once", flush=True)
            print(f"[ws] boundary {ts} → {rx}/175 received in {waited:.0f}s → flushed + push-triggered cycle_once", flush=True)
        except Exception as e:
            print(f"[ws] boundary trigger err: {type(e).__name__} {e}", flush=True)

    def _on_markprice(self, d):
        """@markPrice: capture SETTLED funding via subscription (replaces the FAPI pull). When the
        nextFundingTime T advances, the funding for the prior T settled at the rate active just before
        it (≈ the predicted rate `r`, which has converged to the realized rate by settlement)."""
        sym = d["s"]; r = float(d["r"]); T = int(d["T"])
        prev = self.last_fund.get(sym)
        if prev is not None and prev["T"] != T and prev["T"] > 0:
            iv = round((T - prev["T"]) / 3_600_000)
            self.fund[sym][prev["T"]] = {"calc_time": prev["T"], "funding_rate": prev["r"],
                                         "interval_hours": iv if iv in (4, 8) else 8}
            self.dirty_fund.add(sym)
        self.last_fund[sym] = {"r": r, "T": T}

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
                        elif e == "markPriceUpdate":
                            self._on_markprice(m)
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
        # v1 needs only klines (V0 features) + markPrice (funding). NO aggTrade — no flow features.
        # markPrice@1s is high-rate (1 msg/sym/sec); subscribing it for all 175 floods the WS queue and
        # drops kline bars. The 80 high-vol PEERS are klines-only (xs-rank cohort, never scored/traded), so
        # they need NO funding — stream markPrice only for the traded low-vol book.
        try:
            _excl = set(json.load(open(REPO / "live/models/convexity_v1_universe.json"))["exclude_high_vol"])
        except Exception:
            _excl = set()
        mp_syms = [s for s in self.syms if s not in _excl]
        streams = ([f"{s.lower()}@kline_5m" for s in self.syms]
                   + [f"{s.lower()}@markPrice@1s" for s in mp_syms])
        print(f"[ws] streams: {len(self.syms)} kline_5m + {len(mp_syms)} markPrice (skipped {len(self.syms)-len(mp_syms)} peer funding)", flush=True)
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
