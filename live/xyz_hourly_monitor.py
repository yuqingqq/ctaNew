"""Hourly snapshot of open xyz paper-trade portfolio + Telegram notification.

Runs once per invocation. Cron suggestion: every hour at minute :05:
    5 * * * * cd /path/to/ctaNew && python3 -m live.xyz_hourly_monitor

Each invocation:
  1. Loads live/state/xyz/positions.json (current open positions)
  2. Fetches xyz mids + funding history since last hourly tick
  3. Marks each position to current mid → computes hourly MtM PnL delta
  4. Accrues hourly funding per position
  5. Appends a row to live/state/xyz/hourly_pnl.csv
  6. Sends a Telegram snapshot with positions + per-leg PnL + portfolio totals

Reads cycles.csv to compute strategy PnL since deploy:
    cumulative_strategy_pnl_bps = realized_net_bps_since_deploy + mtm_since_last_cycle_bps
where mtm_since_last_cycle_bps tracks each position's drift from prev rebalance mid.

Telegram is opt-in via TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.telegram import notify_telegram
from live.xyz_paper_bot import (
    HL_INFO_URL, POSITIONS_PATH, CYCLES_PATH, STATE_DIR,
    fetch_xyz_mids, load_state, save_state, _flush_pending_cycle_row,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xyz_hourly")

HOURLY_PNL_PATH = STATE_DIR / "hourly_pnl.csv"
HOURLY_LAST_TICK_PATH = STATE_DIR / "hourly_last_tick.json"


def fetch_xyz_funding_history(symbol: str, start_ms: int, end_ms: int) -> list[dict] | None:
    """xyz funding settles hourly. Coin form is 'xyz:SYMBOL'."""
    payload = {"type": "fundingHistory", "coin": f"xyz:{symbol}",
               "startTime": int(start_ms), "endTime": int(end_ms)}
    try:
        r = requests.post(HL_INFO_URL, json=payload, timeout=10)
        r.raise_for_status()
        return r.json() or []
    except Exception as e:
        log.warning("[%s] funding fetch failed: %s", symbol, e)
        return None


def _load_last_tick() -> dict:
    if HOURLY_LAST_TICK_PATH.exists():
        with HOURLY_LAST_TICK_PATH.open() as fh:
            return json.load(fh)
    return {}


def _save_last_tick(d: dict):
    with HOURLY_LAST_TICK_PATH.open("w") as fh:
        json.dump(d, fh, indent=2)


def _append_hourly_row(row: dict):
    ts = str(row.get("ts_utc"))
    df_new = pd.DataFrame([row])
    if HOURLY_PNL_PATH.exists():
        try:
            existing = pd.read_csv(HOURLY_PNL_PATH)
            if "ts_utc" in existing.columns and (existing["ts_utc"].astype(str) == ts).any():
                log.info("hourly row for %s already logged — skip duplicate append", ts)
                return
            df = pd.concat([existing, df_new], ignore_index=True)
        except Exception as exc:
            log.warning("could not read hourly_pnl.csv for dedup check: %s", exc)
            df_new.to_csv(HOURLY_PNL_PATH, mode="a", header=False, index=False)
            return
    else:
        df = df_new
    df.to_csv(HOURLY_PNL_PATH, index=False)


def _flush_pending_hourly_update(state: dict) -> dict:
    pending = state.get("pending_hourly_update")
    if pending is None:
        return state
    row = pending.get("row")
    if row:
        _append_hourly_row(row)
    last_tick = pending.get("last_tick")
    if last_tick:
        _save_last_tick(last_tick)
    if "last_marked_mids" in pending:
        state["last_marked_mids"] = pending["last_marked_mids"]
    state = {k: v for k, v in state.items() if k != "pending_hourly_update"}
    save_state(state)
    return state


def _fmt_bps(x: float) -> str:
    if x is None or not np.isfinite(x): return "n/a"
    return f"{x:+.2f}"


def _fmt_usd(x: float) -> str:
    if x is None or not np.isfinite(x): return "n/a"
    return f"${x:+.2f}" if abs(x) < 1000 else f"${x:+,.0f}"


def main() -> int:
    now = datetime.now(timezone.utc)
    state = load_state()
    if state is None:
        log.info("No open positions — nothing to monitor.")
        notify_telegram("⚠️ <b>v7 xyz hourly</b>: no open positions. "
                          "Run xyz_paper_bot to open the next cycle.")
        return 0
    state = _flush_pending_cycle_row(state)
    state = _flush_pending_hourly_update(state)
    if not state.get("long") and not state.get("short"):
        log.info("No open positions — nothing to monitor.")
        notify_telegram("⚠️ <b>v7 xyz hourly</b>: no open positions. "
                          "Run xyz_paper_bot to open the next cycle.")
        return 0

    long_set = state.get("long", [])
    short_set = state.get("short", [])
    entry_fills = state.get("entry_fills", {})
    last_marked = state.get("last_marked_mids", {})  # set by us each hour
    notional_usd = state.get("notional_usd", 10000.0)
    K = state.get("top_k", len(long_set) or 1)
    per_name_notional = notional_usd / max(K, 1)

    # Determine last tick for funding window
    last_tick = _load_last_tick()
    prev_tick_iso = last_tick.get("ts_utc")
    if prev_tick_iso:
        try:
            prev_tick_dt = pd.Timestamp(prev_tick_iso)
        except Exception:
            prev_tick_dt = pd.Timestamp(now) - pd.Timedelta(hours=1)
    else:
        prev_tick_dt = pd.Timestamp(now) - pd.Timedelta(hours=1)
    log.info("hourly tick=%s prev=%s", now.isoformat(), prev_tick_dt)

    universe = sorted(set(long_set) | set(short_set))
    try:
        mids = fetch_xyz_mids(universe)
    except Exception as e:
        log.error("xyz mid fetch failed: %s", e)
        return 1
    log.info("fetched %d/%d mids", len(mids), len(universe))

    start_ms = int(prev_tick_dt.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    per_leg = []
    total_hourly_pnl_usd = 0.0
    total_since_cycle_usd = 0.0
    total_since_entry_usd = 0.0
    total_funding_usd = 0.0
    funding_fetch_failed = False
    mark_fetch_failed = False

    def _process(sym, side):
        nonlocal total_hourly_pnl_usd, total_since_cycle_usd, total_since_entry_usd
        nonlocal total_funding_usd, funding_fetch_failed, mark_fetch_failed
        if sym not in mids:
            # Missing mid → can't mark this position AND we skip its funding
            # fetch. Treat the whole tick as partial so the funding checkpoint
            # doesn't advance — next tick re-tries the same window.
            mark_fetch_failed = True
            funding_fetch_failed = True
            log.warning("[%s] mid missing — treating tick as partial", sym)
            return None
        mid_now = mids[sym]
        side_sign = 1.0 if side == "L" else -1.0

        # Hourly drift: vs last_marked mid
        prev_mark = last_marked.get(sym, mid_now)
        h_pnl_frac = side_sign * np.log(mid_now / prev_mark) if prev_mark > 0 else 0.0
        h_pnl_usd = h_pnl_frac * per_name_notional
        total_hourly_pnl_usd += h_pnl_usd

        # Since-cycle (vs entry_mids — last rebalance mid for held, vwap for newly opened)
        entry_mids = state.get("entry_mids", {})
        prev_cycle_ref = entry_mids.get(sym, mid_now)
        c_pnl_frac = side_sign * np.log(mid_now / prev_cycle_ref) if prev_cycle_ref > 0 else 0.0
        c_pnl_usd = c_pnl_frac * per_name_notional
        total_since_cycle_usd += c_pnl_usd

        # Since-entry (vs original entry vwap, captures slippage embedded)
        entry_vwap = entry_fills.get(sym, {}).get("vwap")
        if entry_vwap and entry_vwap > 0:
            e_pnl_frac = side_sign * np.log(mid_now / entry_vwap)
            e_pnl_usd = e_pnl_frac * per_name_notional
        else:
            e_pnl_frac = c_pnl_frac
            e_pnl_usd = c_pnl_usd
        total_since_entry_usd += e_pnl_usd

        # Funding accrual since last tick
        h_fund = 0.0
        try:
            history = fetch_xyz_funding_history(sym, start_ms, end_ms)
            if history is None:
                funding_fetch_failed = True
                history = []
            for ev in history:
                try:
                    rate = float(ev.get("fundingRate", 0))
                except (TypeError, ValueError):
                    continue
                # long pays positive funding, short receives
                h_fund += -side_sign * rate * per_name_notional
        except Exception as exc:
            funding_fetch_failed = True
            log.warning("[%s] funding processing failed: %s", sym, exc)
        total_funding_usd += h_fund

        return {
            "symbol": sym, "side": side,
            "notional_usd": per_name_notional,
            "entry_vwap": entry_vwap, "mid_now": mid_now,
            "hourly_pnl_pct": h_pnl_frac * 100, "hourly_pnl_usd": h_pnl_usd,
            "since_cycle_pct": c_pnl_frac * 100, "since_cycle_usd": c_pnl_usd,
            "since_entry_pct": e_pnl_frac * 100, "since_entry_usd": e_pnl_usd,
            "hourly_funding_usd": h_fund,
        }

    for s in long_set:
        d = _process(s, "L")
        if d: per_leg.append(d)
    for s in short_set:
        d = _process(s, "S")
        if d: per_leg.append(d)

    if funding_fetch_failed:
        # Do not book partial funding. The funding checkpoint is intentionally
        # left unchanged below, so successful symbols will be refetched for the
        # same window on the next tick.
        total_funding_usd = 0.0
        for d in per_leg:
            d["hourly_funding_usd"] = float("nan")

    # Update last_marked_mids in state for next tick's hourly drift
    new_marks = {**last_marked}
    for d in per_leg:
        new_marks[d["symbol"]] = d["mid_now"]
    if funding_fetch_failed:
        log.warning("funding fetch failed for at least one symbol; "
                    "not advancing hourly funding checkpoint")

    # Compute strategy P&L since deploy from cycles.csv
    realized_net_bps = 0.0
    realized_usd = 0.0
    n_cycles = 0
    if CYCLES_PATH.exists():
        try:
            d = pd.read_csv(CYCLES_PATH)
            realized_net_bps = float(d["net_bps"].sum())
            n_cycles = len(d)
            if "notional_usd" in d.columns:
                notionals = pd.to_numeric(d["notional_usd"], errors="coerce").fillna(notional_usd)
                realized_usd = float(((d["net_bps"] / 1e4) * notionals).sum())
            else:
                # Legacy cycles lacked notional_usd. Fall back to current state
                # notional, matching the old behavior.
                realized_usd = float((d["net_bps"] / 1e4).sum() * notional_usd)
        except Exception as exc:
            log.warning("could not read cycles.csv: %s", exc)

    leg_notional_usd = notional_usd  # per leg (long or short separately)
    total_notional_usd = leg_notional_usd * 2
    bps_per_dollar_leg = 1e4 / leg_notional_usd if leg_notional_usd > 0 else 0.0

    # Convert per-name $ totals to per-leg bps using leg notional
    h_bps = total_hourly_pnl_usd * bps_per_dollar_leg
    cycle_bps = total_since_cycle_usd * bps_per_dollar_leg
    entry_bps = total_since_entry_usd * bps_per_dollar_leg
    fund_bps = total_funding_usd * bps_per_dollar_leg
    cumulative_strategy_bps = realized_net_bps + cycle_bps

    row = {
        "ts_utc": now.isoformat(),
        "n_positions": len(per_leg),
        "long_set": ",".join(sorted(long_set)),
        "short_set": ",".join(sorted(short_set)),
        "leg_notional_usd": leg_notional_usd,
        "hourly_pnl_usd": total_hourly_pnl_usd,
        "hourly_pnl_bps": h_bps,
        "hourly_funding_usd": total_funding_usd,
        "hourly_funding_bps": fund_bps,
        "funding_fetch_failed": funding_fetch_failed,
        "mark_fetch_failed": mark_fetch_failed,
        "pnl_incomplete": mark_fetch_failed,
        "since_cycle_pnl_usd": total_since_cycle_usd,
        "since_cycle_pnl_bps": cycle_bps,
        "since_entry_pnl_usd": total_since_entry_usd,
        "since_entry_pnl_bps": entry_bps,
        "realized_net_bps_since_deploy": realized_net_bps,
        "realized_usd_since_deploy": realized_usd,
        "cumulative_strategy_pnl_bps": cumulative_strategy_bps,
        "cumulative_strategy_pnl_usd": realized_usd + total_since_cycle_usd,
        "n_cycles_realized": n_cycles,
    }
    pending_hourly = {
        "row": row,
        "last_marked_mids": new_marks,
        "last_tick": None if funding_fetch_failed else {"ts_utc": now.isoformat()},
    }
    state["pending_hourly_update"] = pending_hourly
    save_state(state)
    state = _flush_pending_hourly_update(state)

    # Telegram message — lead with the cumulative number, then break down
    longs = sorted([d for d in per_leg if d["side"] == "L"],
                    key=lambda d: -d["since_entry_usd"])
    shorts = sorted([d for d in per_leg if d["side"] == "S"],
                     key=lambda d: -d["since_entry_usd"])
    is_first_tick = not last_marked  # was last_marked dict empty before this run?

    # Headline: total $ P&L since deployment + bps view.
    # IMPORTANT: open-cycle component must be `since_cycle` (drift since last
    # rebalance), NOT `since_entry`. Held names' P&L from prior cycles is
    # already in cycles.csv (entry_mids gets updated to cycle's mid each
    # rebalance). Using since_entry here would double-count multi-cycle holds.
    total_strategy_usd = total_since_cycle_usd + realized_usd
    msg = [
        f"⏱ <b>v7 xyz hourly</b>  ({now:%Y-%m-%d %H:%M} UTC)",
        f"",
        f"<b>Strategy P&amp;L since deploy</b>",
        f"  <b>{_fmt_usd(total_strategy_usd)}</b>  =  realized {_fmt_usd(realized_usd)} "
        f"(N={n_cycles} closed) + open {_fmt_usd(total_since_cycle_usd)}",
        f"  <i>{_fmt_bps(cumulative_strategy_bps)} bps total / "
        f"{_fmt_bps(realized_net_bps)} realized + {_fmt_bps(cycle_bps)} open</i>",
        f"",
        f"Positions: L={len(longs)} S={len(shorts)}  •  leg notional: ${leg_notional_usd:,.0f}",
        f"",
    ]
    if funding_fetch_failed:
        msg.append(
            "<i>Funding incomplete this tick — not booked; checkpoint left unchanged for retry.</i>"
        )
    if mark_fetch_failed:
        msg.append(
            "<i>PnL incomplete this tick — at least one open position could not be marked.</i>"
        )
    if is_first_tick:
        msg.append(
            f"<i>First tick after rebalance — hourly drift initialized to 0; "
            f"next tick will show real hourly $ change.</i>"
        )
    else:
        msg.append(
            f"<b>Last hour</b>: pnl {_fmt_usd(total_hourly_pnl_usd)}  "
            f"fund {_fmt_usd(total_funding_usd)}  "
            f"<b>net {_fmt_usd(total_hourly_pnl_usd + total_funding_usd)}</b>  "
            f"({_fmt_bps(h_bps + fund_bps)} bps)"
        )
    msg.append(
        f"<b>Since rebalance</b>: pnl {_fmt_usd(total_since_cycle_usd)} "
        f"({_fmt_bps(cycle_bps)} bps)"
    )
    msg.append("")

    msg.append("<b>Longs</b>:")
    for d in longs:
        msg.append(
            f"  {d['symbol']:<6} {_fmt_usd(d['since_entry_usd']):>8}  "
            f"({d['since_entry_pct']:+.2f}%)  "
            f"fund {_fmt_usd(d['hourly_funding_usd'])}"
        )
    msg.append("<b>Shorts</b>:")
    for d in shorts:
        msg.append(
            f"  {d['symbol']:<6} {_fmt_usd(d['since_entry_usd']):>8}  "
            f"({d['since_entry_pct']:+.2f}%)  "
            f"fund {_fmt_usd(d['hourly_funding_usd'])}"
        )

    text = "\n".join(msg)
    print(text)
    sent = notify_telegram(text)
    log.info("telegram %s (%d chars)",
             "sent" if sent else "skipped (no env vars or send failed)", len(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
