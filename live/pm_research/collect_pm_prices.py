"""
Polymarket live-data price stream collector (P-2026-003).

wss://ws-live-data.polymarket.com, topic "crypto_prices" — the price feed
Polymarket's own UI uses for crypto markets (per-second, full_accuracy_value).
Probe-verified public 2026-08-19.

The `crypto_prices` topic is a Binance-spot mirror. E-M6 verified that
`crypto_prices_twap_sixty` is the settlement stream: S60(T) versus S60(t0)
reproduces every admissible winner. The thirty-second stream is collected as a
published predictor. Unknown topics and non-crypto payloads remain stored raw.

Rows: known topic → csv "recv_ns,t_ms,symbol,value_full"; else raw JSONL.
Storage: data/pm_5min/prices/<topic>/<YYYYMMDD_HH>.csv (gzip on rotation).

Run:  nohup python3 live/pm_research/collect_pm_prices.py > data/pm_5min/prices_collector.log 2>&1 &
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import shutil
import signal
import tempfile
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
GLOBAL_WATCHDOG_S = 8
TOPIC_WATCHDOG_S = 8
HEARTBEAT_S = 60
REQUIRED_TOPICS = ("crypto_prices_twap_thirty", "crypto_prices_twap_sixty")
GAP_LEDGER = ROOT / "collector_gaps.jsonl"
COLLECTOR_VERSION = "prices_v2"


def jl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


def gzip_atomic(path: Path, dest: Path) -> None:
    """Publish a complete archive atomically and preserve CSV on failure."""
    if dest.exists():
        raise FileExistsError(f"refusing to replace existing {dest}")
    tmp = Path(f"{dest}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with open(path, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(dest)
        path.unlink()
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class PriceCollector:
    def __init__(self):
        self.stop = False
        self.buf: dict[str, list[str]] = defaultdict(list)
        self.open_hour: dict[str, str] = {}
        self.counts = defaultdict(int)
        self.last_topic_recv_ns: dict[str, int] = {}
        self.open_gaps: dict[str, dict] = {}
        self.connection_id = 0

    def _audit(self, obj: dict) -> None:
        try:
            jl_append(GAP_LEDGER, {"recv_ns": time.time_ns(),
                                   "collector_version": COLLECTOR_VERSION, **obj})
        except Exception as ex:
            self.counts["audit_errors"] += 1
            print(f"[pmp] audit append: {type(ex).__name__} {str(ex)[:60]}", flush=True)

    def _mark_gap(self, cause: str, topics: list[str], detected_ns: int) -> None:
        for topic in topics:
            if topic in self.open_gaps:
                continue
            start_ns = self.last_topic_recv_ns.get(topic, detected_ns)
            self.open_gaps[topic] = {"cause": cause, "gap_start_ns": start_ns,
                                     "connection_id": self.connection_id}
            self._audit({
                "event": "gap_open", "topic": topic, "cause": cause,
                "gap_start_ns": start_ns, "detected_ns": detected_ns,
                "connection_id": self.connection_id,
            })

    def _on_msg(self, raw: str, recv_ns: int) -> str:
        try:
            m = json.loads(raw)
        except Exception:
            topic = "malformed"
            self.buf[topic].append(f"{recv_ns}\t{raw}")
            self.counts["malformed_json"] += 1
            return topic
        if not isinstance(m, dict):
            topic = "malformed_schema"
            self.buf[topic].append(f"{recv_ns}\t{raw}")
            self.counts["malformed_schema"] += 1
            return topic
        raw_topic = m.get("topic")
        topic = raw_topic if isinstance(raw_topic, str) and raw_topic else "unknown"
        p = m.get("payload") or {}
        if not isinstance(p, dict):
            p = {}
        if topic in REQUIRED_TOPICS and not (
                p.get("symbol") and p.get("timestamp") is not None
                and (p.get("full_accuracy_value") is not None
                     or p.get("value") is not None)):
            self.buf[topic].append(f"{recv_ns}\t{raw}")
            self.counts[f"{topic}_invalid_payload"] += 1
            return f"invalid:{topic}"
        if topic == "crypto_prices" and "symbol" in p:
            line = f"{recv_ns},{p.get('timestamp','')},{p['symbol']},{p.get('full_accuracy_value', p.get('value',''))}"
        else:
            line = f"{recv_ns}\t{raw}"
        self.buf[topic].append(line)
        self.counts[topic] += 1
        self.last_topic_recv_ns[topic] = recv_ns
        gap = self.open_gaps.pop(topic, None)
        if gap is not None:
            self._audit({
                "event": "gap_closed", "topic": topic, "cause": gap["cause"],
                "gap_start_ns": gap["gap_start_ns"], "gap_end_ns": recv_ns,
                "duration_ms": (recv_ns - gap["gap_start_ns"]) / 1e6,
                "opened_connection_id": gap["connection_id"],
                "closed_connection_id": self.connection_id,
            })
        return topic

    def _gzip(self, path: Path) -> None:
        try:
            gzip_atomic(path, Path(str(path) + ".gz"))
        except Exception as ex:
            self.counts["gzip_errors"] += 1
            print(f"[pmp] gzip: {ex}", flush=True)

    def _rotate_stale_csv(self) -> None:
        """Recover closed-hour CSVs left behind by a prior abrupt stop."""
        now_hour = time.strftime("%Y%m%d_%H", time.gmtime())
        for d in sorted(x for x in ROOT.iterdir() if x.is_dir()):
            for path in sorted(d.glob("*.csv")):
                if path.stem != now_hour:
                    self._gzip(path)

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
                    self._gzip(p)
            self.open_hour[topic] = now_hour
            with open(d / f"{now_hour}.csv", "a") as f:
                f.write("\n".join(lines) + "\n")

    async def _conn(self) -> None:
        while not self.stop:
            try:
                async with websockets.connect(WS_URL, ping_interval=15, ping_timeout=15,
                                              open_timeout=15) as ws:
                    self.connection_id += 1
                    connected_ns = time.time_ns()
                    seen_this_connection: set[str] = set()
                    await ws.send(json.dumps(SUB))
                    print(f"[pmp] subscribed topics connection={self.connection_id}", flush=True)
                    while not self.stop:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=GLOBAL_WATCHDOG_S)
                        except asyncio.TimeoutError:
                            now_ns = time.time_ns()
                            self.counts["global_silence_reconnects"] += 1
                            self.counts["reconnects"] += 1
                            self._mark_gap("GLOBAL_SOCKET_SILENCE", list(REQUIRED_TOPICS), now_ns)
                            print(f"[pmp] silent {GLOBAL_WATCHDOG_S}s — reconnect", flush=True)
                            break
                        if raw:
                            now_ns = time.time_ns()
                            topic = self._on_msg(raw, now_ns)
                            if topic in REQUIRED_TOPICS:
                                seen_this_connection.add(topic)
                            stale = []
                            for required in REQUIRED_TOPICS:
                                base = (self.last_topic_recv_ns.get(required, connected_ns)
                                        if required in seen_this_connection else connected_ns)
                                if now_ns - base > TOPIC_WATCHDOG_S * 10**9:
                                    stale.append(required)
                            if stale:
                                self.counts["topic_stale_reconnects"] += 1
                                self.counts["reconnects"] += 1
                                self._mark_gap("TOPIC_STALE", stale, now_ns)
                                peers = [x for x in REQUIRED_TOPICS if x not in stale]
                                self._mark_gap("PEER_TOPIC_RECONNECT", peers, now_ns)
                                print(f"[pmp] stale topics={stale} — reconnect", flush=True)
                                break
            except Exception as ex:
                now_ns = time.time_ns()
                self.counts["connection_errors"] += 1
                self.counts["reconnects"] += 1
                self._mark_gap(type(ex).__name__.upper(), list(REQUIRED_TOPICS), now_ns)
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
                now_ns = time.time_ns()
                ages = {k: round((now_ns - self.last_topic_recv_ns[k]) / 1e9, 2)
                        for k in REQUIRED_TOPICS if k in self.last_topic_recv_ns}
                print(f"[pmp] {time.strftime('%H:%M:%S', time.gmtime())}Z "
                      + " ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
                      + f" ages_s={ages} open_gaps={list(self.open_gaps)}", flush=True)

    async def run(self) -> None:
        ROOT.mkdir(parents=True, exist_ok=True)
        self._rotate_stale_csv()
        self._audit({"event": "collector_start", "pid": os.getpid()})
        print(f"[pmp] version={COLLECTOR_VERSION} watchdog_global="
              f"{GLOBAL_WATCHDOG_S}s watchdog_topic={TOPIC_WATCHDOG_S}s", flush=True)
        await asyncio.gather(self._conn(), self._housekeeping())
        self._flush()
        self._audit({"event": "collector_stop", "pid": os.getpid()})
        print("[pmp] stopped", flush=True)


def selftest() -> int:
    c = PriceCollector()
    valid = json.dumps({"topic": "crypto_prices_twap_sixty",
                        "payload": {"symbol": "btc/usd", "timestamp": 1,
                                    "full_accuracy_value": "2"}})
    invalid = json.dumps({"topic": "crypto_prices_twap_sixty", "payload": {}})
    invalid_result = c._on_msg(invalid, 13)
    odd_topic = c._on_msg(json.dumps({"topic": [], "payload": {}}), 14)
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "prices.csv"
        dest = Path(d) / "prices.csv.gz"
        raw.write_bytes(b"a,b\n1,2\n")
        gzip_atomic(raw, dest)
        atomic_ok = (not raw.exists() and gzip.open(dest, "rb").read() == b"a,b\n1,2\n"
                     and not list(Path(d).glob("*.tmp-*")))
    checks = [
        ("valid topic parsed", c._on_msg(valid, 10) == "crypto_prices_twap_sixty"),
        ("knowledge time retained", c.last_topic_recv_ns.get(
            "crypto_prices_twap_sixty") == 10),
        ("malformed payload preserved", c._on_msg("{bad", 11) == "malformed"
         and c.counts["malformed_json"] == 1 and len(c.buf["malformed"]) == 1),
        ("non-object JSON preserved", c._on_msg("[]", 12) == "malformed_schema"
         and c.counts["malformed_schema"] == 1),
        ("invalid required payload is not fresh", invalid_result.startswith("invalid:")
         and c.last_topic_recv_ns["crypto_prices_twap_sixty"] == 10),
        ("non-string topic is totalized", odd_topic == "unknown"
         and len(c.buf["unknown"]) == 1),
        ("watchdogs are below old 30s loss", GLOBAL_WATCHDOG_S < 30
         and TOPIC_WATCHDOG_S < 30),
        ("gzip publish is atomic", atomic_ok),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    c = PriceCollector()

    def _sig(*_):
        print("[pmp] shutdown", flush=True)
        c.stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
