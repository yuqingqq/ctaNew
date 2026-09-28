"""CLI.

  python -m movewatch run [--config config/movewatch.yaml] [--dry-run] [--venues a,b] [--duration S]
  python -m movewatch test-telegram        # one labelled test message through the real path
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="movewatch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default=str(ROOT / "config" / "movewatch.yaml"))
    r.add_argument("--dry-run", action="store_true", help="log + JSONL only, no Telegram")
    r.add_argument("--venues", default=None, help="comma list to restrict venues")
    r.add_argument("--duration", type=float, default=None, help="stop after S seconds (smoke tests)")
    r.add_argument("--no-watchers", action="store_true")
    r.add_argument("--data-dir", default=None, help="override data_dir (smoke tests)")
    r.add_argument("--warmup", type=int, default=None)
    stt = sub.add_parser("status", help="feed health + the latest alerts")
    stt.add_argument("--n", type=int, default=10)
    t = sub.add_parser("test-telegram")
    t.add_argument("--config", default=str(ROOT / "config" / "movewatch.yaml"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    if args.cmd == "status":                 # needs no config
        return _status(args.n)
    cfg = load_cfg(args.config)

    if args.cmd == "test-telegram":
        return asyncio.run(_test_telegram(cfg))

    if args.dry_run:
        cfg.setdefault("notify", {})["dry_run"] = True
    if args.venues:
        cfg["only_venues"] = [v.strip() for v in args.venues.split(",") if v.strip()]
    if args.no_watchers:
        cfg.setdefault("watchers", {})["enabled"] = False
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.warmup is not None:
        cfg["warmup_s"] = args.warmup
    from .engine import Engine
    try:
        asyncio.run(Engine(cfg, ROOT).run(args.duration))
    except KeyboardInterrupt:
        pass
    return 0


def _status(n: int) -> int:
    import json
    import time
    hp = ROOT / "data" / "state" / "health.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        age = time.time() - h["t"]
        print(f"health {int(age)}s old · {h['instruments']} instruments · {h['assets']} assets · "
              f"{h['active_episodes']} active episodes · sent {h['sent']} · failed {h['send_failed']}")
        for v, s in h["feeds"].items():
            print(f"  {v:14s} up {s['up']}/{s['shards']}  msgs {s['msgs']:>9}  reconnects {s['reconnects']}"
                  f"  max_silence {s['max_silence_s']}s")
    else:
        print("no health.json yet (service not running?)")
    files = sorted((ROOT / "data" / "alerts").glob("alerts-*.jsonl"))
    rows = []
    for f in files[-2:]:
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    print(f"\nlast {min(n, len(rows))} alerts:")
    for r in rows[-n:]:
        print(" ", r.get("logged_at", ""), r.get("kind", ""), "|", r.get("text", "").replace("\n", " | ")[:160])
    return 0


async def _test_telegram(cfg: dict) -> int:
    import aiohttp

    from .notify import Notifier
    n = Notifier(cfg.get("notify", {}), ROOT / "data" / "alerts")
    if not n.telegram:
        print("telegram: no credentials in env (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID or MOVEWATCH_*)")
        return 2
    async with aiohttp.ClientSession() as s:
        ok = await n._send(s, "🟦 movewatch test message — delivery path OK (P-2026-004 setup)")
    print("telegram:", "sent" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
