"""Hourly snapshot of open paper-trade portfolio + Telegram notification.

Runs once per invocation — designed for cron at minute :05 of every hour:
  5 * * * * cd /path/to/ctaNew && python -m live.hourly_monitor

Each invocation:
  1. Loads live/state/positions.json (current open positions)
  2. Fetches HL mids for held symbols + funding history since last hourly tick
  3. Marks each position to current mid → computes hourly MtM PnL delta
  4. Accrues hourly funding per position
  5. Appends a row to live/state/hourly_pnl.csv (running history)
  6. Sends a Telegram snapshot with positions + per-leg PnL + portfolio totals
  7. Updates each position's `last_marked_mid` and `funding_paid_usd`

If TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars are not set, the Telegram
push is skipped silently — CSV log is still written.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from live.paper_bot import (
    LegPosition, INITIAL_EQUITY_USD,
    _binance_to_hl_coin, fetch_hl_mids, fetch_hl_funding_history,
    POSITIONS_PATH, STATE_DIR,
)
from live.telegram import notify_telegram

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hourly")

HOURLY_PNL_PATH = STATE_DIR / "hourly_pnl.csv"
HOURLY_LAST_TICK_PATH = STATE_DIR / "hourly_last_tick.json"


def _load_positions() -> list[LegPosition]:
    if not POSITIONS_PATH.exists():
        return []
    with POSITIONS_PATH.open() as f:
        data = json.load(f)
    return [LegPosition(**d) for d in data] if data else []


def _save_positions(positions: list[LegPosition]) -> None:
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSITIONS_PATH.open("w") as f:
        json.dump([p.to_dict() for p in positions], f, indent=2)


def _load_last_tick() -> dict:
    if HOURLY_LAST_TICK_PATH.exists():
        with HOURLY_LAST_TICK_PATH.open() as f:
            return json.load(f)
    return {}


def _save_last_tick(d: dict) -> None:
    with HOURLY_LAST_TICK_PATH.open("w") as f:
        json.dump(d, f, indent=2)


def _fmt_bps(x: float) -> str:
    return f"{x:+.2f}"


def _fmt_usd(x: float) -> str:
    return f"${x:+.2f}" if abs(x) < 1000 else f"${x:+,.0f}"


def main():
    now = datetime.now(timezone.utc)
    positions = _load_positions()
    if not positions:
        log.info("No open positions — nothing to monitor.")
        notify_telegram("⚠️ <b>v6_clean hourly</b>: no open positions. "
                         "Run paper_bot to open the next cycle.")
        return 0

    # Determine "last tick" timestamp for funding window
    last_tick = _load_last_tick()
    prev_tick_iso = last_tick.get("ts_utc")
    if prev_tick_iso:
        try:
            prev_tick_dt = pd.Timestamp(prev_tick_iso)
        except Exception:
            prev_tick_dt = pd.Timestamp(now) - pd.Timedelta(hours=1)
    else:
        prev_tick_dt = pd.Timestamp(now) - pd.Timedelta(hours=1)

    log.info("Hourly tick: %s, prev tick: %s", now.isoformat(), prev_tick_dt)

    # 1. Fetch all HL mids in one call
    try:
        all_mids = fetch_hl_mids()
    except Exception as e:
        log.error("HL allMids fetch failed: %s", e)
        return 1

    # 2. Per-position MtM + funding accrual since last tick
    per_leg = []
    total_unrealized_usd = 0.0
    total_hourly_pnl_usd = 0.0
    total_hourly_funding_usd = 0.0
    total_notional_usd = 0.0
    funding_cache: dict[str, list[dict]] = {}
    start_ms = int(prev_tick_dt.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    for p in positions:
        coin = _binance_to_hl_coin(p.symbol)
        mid_now = float(all_mids.get(coin, np.nan))
        if not np.isfinite(mid_now):
            continue

        # Hourly MtM (signed by side)
        if p.last_marked_mid and np.isfinite(p.last_marked_mid):
            if p.side == "L":
                hourly_pnl_frac = (mid_now / p.last_marked_mid - 1.0)
            else:
                hourly_pnl_frac = (p.last_marked_mid / mid_now - 1.0)
        else:
            hourly_pnl_frac = 0.0
        hourly_pnl_usd = hourly_pnl_frac * p.entry_notional_usd
        total_hourly_pnl_usd += hourly_pnl_usd

        # Cumulative MtM since entry
        if p.entry_price_hl and np.isfinite(p.entry_price_hl):
            if p.side == "L":
                cum_pnl_frac = (mid_now / p.entry_price_hl - 1.0)
            else:
                cum_pnl_frac = (p.entry_price_hl / mid_now - 1.0)
        else:
            cum_pnl_frac = 0.0
        cum_pnl_usd = cum_pnl_frac * p.entry_notional_usd

        # Funding since last tick
        if coin not in funding_cache:
            try:
                funding_cache[coin] = fetch_hl_funding_history(coin, start_ms, end_ms)
            except Exception as e:
                log.warning("[%s] funding fetch failed: %s", coin, e)
                funding_cache[coin] = []
        side_sign = 1.0 if p.side == "L" else -1.0
        hourly_funding_usd = 0.0
        for entry in funding_cache[coin]:
            try:
                rate = float(entry["fundingRate"])
            except (KeyError, TypeError, ValueError):
                continue
            hourly_funding_usd += side_sign * rate * p.entry_notional_usd
        total_hourly_funding_usd += hourly_funding_usd

        per_leg.append({
            "symbol": p.symbol, "side": p.side,
            "weight": p.weight, "notional_usd": p.entry_notional_usd,
            "entry_price": p.entry_price_hl, "mid_now": mid_now,
            "cum_pnl_pct": cum_pnl_frac * 100, "cum_pnl_usd": cum_pnl_usd,
            "hourly_pnl_pct": hourly_pnl_frac * 100, "hourly_pnl_usd": hourly_pnl_usd,
            "hourly_funding_usd": hourly_funding_usd,
            "cum_funding_usd": p.funding_paid_usd + hourly_funding_usd,
        })
        total_unrealized_usd += cum_pnl_usd
        total_notional_usd += p.entry_notional_usd

        # Update position state
        p.last_marked_mid = mid_now
        p.funding_paid_usd += hourly_funding_usd

    # Save updated positions
    _save_positions(positions)
    _save_last_tick({"ts_utc": now.isoformat()})

    # 3. Append to hourly_pnl.csv
    bps_per_dollar = 1e4 / INITIAL_EQUITY_USD
    row = {
        "ts_utc": now.isoformat(),
        "n_positions": len(positions),
        "total_notional_usd": total_notional_usd,
        "hourly_pnl_usd": total_hourly_pnl_usd,
        "hourly_pnl_bps": total_hourly_pnl_usd * bps_per_dollar,
        "hourly_funding_usd": total_hourly_funding_usd,
        "hourly_funding_bps": total_hourly_funding_usd * bps_per_dollar,
        "hourly_net_usd": total_hourly_pnl_usd - total_hourly_funding_usd,
        "hourly_net_bps": (total_hourly_pnl_usd - total_hourly_funding_usd) * bps_per_dollar,
        "cumulative_unrealized_usd": total_unrealized_usd,
        "cumulative_unrealized_bps": total_unrealized_usd * bps_per_dollar,
    }
    df_new = pd.DataFrame([row])
    if HOURLY_PNL_PATH.exists():
        df = pd.concat([pd.read_csv(HOURLY_PNL_PATH), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(HOURLY_PNL_PATH, index=False)

    # 4. Build Telegram message
    bps_h_pnl = row["hourly_pnl_bps"]
    bps_h_fund = row["hourly_funding_bps"]
    bps_h_net = row["hourly_net_bps"]
    bps_cum = row["cumulative_unrealized_bps"]

    # Sort legs by hourly PnL (best first)
    per_leg_sorted = sorted(per_leg, key=lambda d: -d["hourly_pnl_usd"])
    legs_long = [d for d in per_leg_sorted if d["side"] == "L"]
    legs_short = [d for d in per_leg_sorted if d["side"] == "S"]

    msg_lines = [
        f"📊 <b>v6_clean hourly</b>  ({now:%Y-%m-%d %H:%M} UTC)",
        f"",
        f"Equity: ${INITIAL_EQUITY_USD:,.0f}  •  Open positions: {len(positions)}  •  Notional: ${total_notional_usd:,.0f}",
        f"",
        f"<b>Hourly</b>:    PnL {_fmt_bps(bps_h_pnl)} bps  •  funding {_fmt_bps(bps_h_fund)} bps  •  net {_fmt_bps(bps_h_net)} bps  ({_fmt_usd(row['hourly_net_usd'])})",
        f"<b>Since open</b>: cumulative MtM {_fmt_bps(bps_cum)} bps  ({_fmt_usd(total_unrealized_usd)})",
        f"",
    ]
    msg_lines.append("<b>Longs</b>:")
    for d in legs_long:
        msg_lines.append(
            f"  {d['symbol']:<10} ${d['notional_usd']:>5,.0f}  "
            f"cum {d['cum_pnl_pct']:+.2f}%  •  hr {d['hourly_pnl_pct']:+.3f}%  "
            f"fund {_fmt_usd(d['hourly_funding_usd'])}"
        )
    msg_lines.append("<b>Shorts</b>:")
    for d in legs_short:
        msg_lines.append(
            f"  {d['symbol']:<10} ${d['notional_usd']:>5,.0f}  "
            f"cum {d['cum_pnl_pct']:+.2f}%  •  hr {d['hourly_pnl_pct']:+.3f}%  "
            f"fund {_fmt_usd(d['hourly_funding_usd'])}"
        )
    text = "\n".join(msg_lines)
    print(text)
    sent = notify_telegram(text)
    if sent:
        log.info("telegram sent: %d chars", len(text))
    else:
        log.info("telegram skipped (no env vars or send failed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
