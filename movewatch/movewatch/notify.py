"""Alert delivery: Telegram (optional), stdout log, and an append-only JSONL record.

Credentials (never stored in this repo): MOVEWATCH_TELEGRAM_BOT_TOKEN /
MOVEWATCH_TELEGRAM_CHAT_ID if set, else the shared TELEGRAM_BOT_TOKEN /
TELEGRAM_CHAT_ID (the systemd unit loads ctaNew/.env). Missing credentials ->
Telegram is a no-op and every alert still lands in the JSONL record.

Rate control: messages are drained every `batch_s`; when more than
`max_per_batch` are waiting they go out as ONE combined message, and sends are
spaced >= `min_interval_s` (Telegram: ~20 messages/min per group).
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

log = logging.getLogger("movewatch.notify")


class Notifier:
    def __init__(self, cfg: dict, record_dir: Path):
        self.cfg = cfg or {}
        self.token = os.environ.get("MOVEWATCH_TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat = os.environ.get("MOVEWATCH_TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
        self.telegram = bool(self.cfg.get("telegram", True) and self.token and self.chat)
        self.dry_run = bool(self.cfg.get("dry_run", False))
        self.batch_s = float(self.cfg.get("batch_s", 2.0))
        self.max_per_batch = int(self.cfg.get("max_per_batch", 4))
        self.min_interval_s = float(self.cfg.get("min_interval_s", 3.0))
        self.record_dir = record_dir
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.q: asyncio.Queue = asyncio.Queue()
        self.sent = 0
        self.failed = 0
        self._last_send = 0.0

    def status(self) -> str:
        if self.dry_run:
            return "dry-run (log + JSONL only)"
        return "telegram" if self.telegram else "no telegram credentials (log + JSONL only)"

    def record_only(self, text: str, record: dict | None = None) -> None:
        """Record + log an alert that is NOT sent on its own (e.g. held for a burst digest)."""
        self._record(text, record)

    def emit(self, text: str, record: dict | None = None) -> None:
        """Queue a message; record it immediately (the record never depends on delivery)."""
        self._record(text, record)
        self.q.put_nowait(text)

    def _record(self, text: str, record: dict | None) -> None:
        rec = dict(record or {})
        rec["text"] = text
        rec["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = self.record_dir / f"alerts-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
        log.info("%s %s", "HELD" if rec.get("held_for_digest") else "ALERT", text.replace("\n", " | "))

    async def run(self, session: aiohttp.ClientSession) -> None:
        while True:
            first = await self.q.get()
            await asyncio.sleep(self.batch_s)
            batch = [first]
            while not self.q.empty():
                batch.append(self.q.get_nowait())
            if len(batch) > self.max_per_batch:
                chunks = ["\n\n".join(batch[k: k + 8]) for k in range(0, len(batch), 8)]
            else:
                chunks = batch
            for text in chunks:
                wait = self.min_interval_s - (time.time() - self._last_send)
                if wait > 0:
                    await asyncio.sleep(wait)
                await self._send(session, text)
                self._last_send = time.time()

    async def _send(self, session: aiohttp.ClientSession, text: str) -> bool:
        if self.dry_run or not self.telegram:
            return False
        if len(text) > 4000:
            text = text[:3950] + "\n…(truncated)"
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        body = {"chat_id": self.chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        for k in range(3):
            try:
                async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=15)) as r:
                    if r.status == 429:
                        ra = (await r.json(content_type=None)).get("parameters", {}).get("retry_after", 5)
                        await asyncio.sleep(float(ra) + 0.5)
                        continue
                    if r.status >= 400:
                        log.warning("telegram HTTP %s: %s", r.status, (await r.text())[:200])
                        self.failed += 1
                        return False
                    self.sent += 1
                    return True
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("telegram send failed (%s), retry %d", e, k + 1)
                await asyncio.sleep(2 * (k + 1))
        self.failed += 1
        return False


def _json_default(o):
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# --------------------------------------------------------------------------- formatting
def esc(s) -> str:
    return html.escape(str(s))


def fmt_px(p: float) -> str:
    if p is None or p != p:
        return "?"
    if p >= 1000:
        return f"${p:,.0f}"
    if p >= 1:
        return f"${p:,.2f}"
    if p >= 0.01:
        return f"${p:.4f}"
    return f"${p:.3g}"


def fmt_pct(x: float, nd: int = 1) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:+.{nd}f}%"


def fmt_usd(x: float) -> str:
    if x is None or x != x:
        return "?"
    if x >= 1e9:
        return f"${x/1e9:.1f}B"
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    if x >= 1e3:
        return f"${x/1e3:.0f}k"
    return f"${x:.0f}"


def hhmm(ts: float, ref: float | None = None) -> str:
    d = datetime.fromtimestamp(ts, timezone.utc)
    r = datetime.fromtimestamp(ref or time.time(), timezone.utc)
    return d.strftime("%H:%MZ") if d.date() == r.date() else d.strftime("%m-%d %H:%MZ")
