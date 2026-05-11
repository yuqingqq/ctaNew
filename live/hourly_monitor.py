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
    """Load positions from positions.json. Compatible with both formats:
      - Legacy: bare list of position dicts
      - New (paper_bot post-2026-05-09): {"positions": [...], "pending_cycle_row": ...}
    """
    if not POSITIONS_PATH.exists():
        return []
    with POSITIONS_PATH.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        # Legacy format
        return [LegPosition(**d) for d in data] if data else []
    # New dict format
    raw_positions = data.get("positions", [])
    return [LegPosition(**d) for d in raw_positions] if raw_positions else []


def _save_positions(positions: list[LegPosition]) -> None:
    """Atomic save preserving paper_bot's full state schema (positions +
    pending_cycle_row + any future fields). Reads existing dict if present
    so we don't clobber a pending cycle row that paper_bot is mid-commit on.
    """
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Read current full state (so we preserve pending_cycle_row etc. that
    # paper_bot may have staged but not yet flushed).
    if POSITIONS_PATH.exists():
        try:
            with POSITIONS_PATH.open() as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                state = dict(existing)  # shallow copy
            else:
                # Legacy bare-list — upgrade to dict format
                state = {"positions": [], "pending_cycle_row": None}
        except Exception:
            state = {"positions": [], "pending_cycle_row": None}
    else:
        state = {"positions": [], "pending_cycle_row": None}
    state["positions"] = [p.to_dict() for p in positions]
    # Atomic write via tmp + rename (matches paper_bot._write_positions_atomic)
    tmp = POSITIONS_PATH.with_suffix(POSITIONS_PATH.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2, default=str)
        f.flush()
        try:
            import os as _os
            _os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    tmp.replace(POSITIONS_PATH)


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
    total_since_last_cycle_usd = 0.0
    total_hourly_pnl_usd = 0.0
    total_hourly_funding_usd = 0.0
    total_notional_usd = 0.0
    funding_cache: dict[str, list[dict]] = {}
    start_ms = int(prev_tick_dt.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    # Partial-tick tracking: if any required fetch fails (mid or funding),
    # don't advance hourly_last_tick so next tick re-tries the same window.
    # Otherwise we permanently lose funding/MtM for that gap. (Matches the
    # xyz_hourly_monitor.py pattern.)
    mark_fetch_failed = False
    funding_fetch_failed = False

    for p in positions:
        coin = _binance_to_hl_coin(p.symbol)
        mid_now = float(all_mids.get(coin, np.nan))
        if not np.isfinite(mid_now):
            mark_fetch_failed = True
            funding_fetch_failed = True   # can't fetch funding without per-symbol pass
            log.warning("[%s] mid missing — treating tick as partial", p.symbol)
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

        # Cumulative MtM since entry — snapshot only; meaning shifts at every
        # rebalance because closed legs leave and new legs enter at fresh bases.
        # NOT the same as strategy PnL since deploy. Use cumulative_strategy_pnl_bps
        # for that (computed below from cycles.csv + since-last-cycle MtM).
        if p.entry_price_hl and np.isfinite(p.entry_price_hl):
            if p.side == "L":
                cum_pnl_frac = (mid_now / p.entry_price_hl - 1.0)
            else:
                cum_pnl_frac = (p.entry_price_hl / mid_now - 1.0)
        else:
            cum_pnl_frac = 0.0
        cum_pnl_usd = cum_pnl_frac * p.entry_notional_usd

        # MtM since the last cycle decision — composes correctly with
        # cycles.csv net_bps to give true strategy PnL since deployment.
        # last_cycle_mid is the basis at which next cycle's gross_pnl_bps will be computed.
        if p.last_cycle_mid and np.isfinite(p.last_cycle_mid):
            if p.side == "L":
                since_cycle_frac = (mid_now / p.last_cycle_mid - 1.0)
            else:
                since_cycle_frac = (p.last_cycle_mid / mid_now - 1.0)
        else:
            since_cycle_frac = 0.0
        since_cycle_usd = since_cycle_frac * p.entry_notional_usd
        total_since_last_cycle_usd += since_cycle_usd

        # Funding since last tick
        if coin not in funding_cache:
            try:
                funding_cache[coin] = fetch_hl_funding_history(coin, start_ms, end_ms)
            except Exception as e:
                log.warning("[%s] funding fetch failed: %s", coin, e)
                funding_cache[coin] = []
                funding_fetch_failed = True   # don't advance checkpoint
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

        # DEFER mid-mark and funding accumulation to end-of-loop. If any
        # fetch failed (mark or funding for any symbol), we discard pending
        # updates to avoid checkpoint inconsistency on the retry tick.
        # See "Why both must be deferred" comment below.
        p._pending_last_marked_mid = float(mid_now)
        p._pending_hourly_funding = float(hourly_funding_usd)

    # Commit (or discard) deferred updates atomically. Why both must be
    # deferred: if we updated last_marked_mid eagerly but kept the funding
    # checkpoint frozen, the retried window would refetch funding for
    # [prev_tick → now] but compute hourly drift only over [partial_tick → now]
    # (mark moved, funding window didn't). The two checkpoints must advance
    # together to keep the windows aligned.
    tick_complete = not (mark_fetch_failed or funding_fetch_failed)
    if tick_complete:
        for p in positions:
            pending_mid = getattr(p, "_pending_last_marked_mid", None)
            if pending_mid is not None:
                p.last_marked_mid = pending_mid
            pending_fund = getattr(p, "_pending_hourly_funding", None)
            if pending_fund is not None:
                p.funding_paid_usd += pending_fund
    else:
        log.warning("Tick incomplete (mark_fail=%s funding_fail=%s) — "
                    "discarding pending mid-marks and funding; next tick "
                    "will retry the full window from prev_tick_iso",
                    mark_fetch_failed, funding_fetch_failed)
        total_hourly_funding_usd = 0.0
        for d in per_leg:
            d["hourly_funding_usd"] = float("nan")
    # Always clear pending fields so subsequent calls don't re-apply
    for p in positions:
        if hasattr(p, "_pending_last_marked_mid"):
            delattr(p, "_pending_last_marked_mid")
        if hasattr(p, "_pending_hourly_funding"):
            delattr(p, "_pending_hourly_funding")

    # Save updated positions. On complete tick: marks + funding committed.
    # On partial tick: only previously-committed state is preserved (we
    # discarded pending updates above). Either way the file reflects a
    # consistent point-in-time.
    _save_positions(positions)

    # Advance hourly_last_tick ONLY if tick was complete (matches the
    # mark/funding commit decision so both checkpoints stay aligned).
    # On partial tick, leaving prev tick means next run re-fetches the
    # full window — and since marks weren't advanced either, hourly drift
    # over that window will be computed consistently.
    if tick_complete:
        _save_last_tick({"ts_utc": now.isoformat()})
    else:
        log.warning("Tick incomplete — leaving hourly_last_tick at prev "
                    "value to retry on next run (next tick covers full window)")

    # Realized strategy PnL = sum of past cycles' net_bps (closed bookkeeping).
    # Adding the current cycle's MtM-since-last-cycle gives true total PnL.
    #
    # Phase 4 honest-equity update: paper_bot now records the cycle's actual
    # equity (real HL balance in live mode, INITIAL_EQUITY_USD in sim) as
    # cycle_row["equity_usd"]. Use the latest cycle's equity as the bps
    # denominator so hourly alerts reflect real account size, not the
    # legacy $10k sim assumption.
    #
    # Cumulative PnL is filtered to only sum cycles in the SAME execution
    # mode as the latest cycle (live or sim). Mixing sim cycles at $10k
    # equity with live cycles at $472 equity in a single bps total is
    # meaningless because the denominators differ — they'd need to be
    # converted to dollars first then back. Matching the mode keeps the
    # cumulative number on a consistent denominator.
    realized_net_bps_since_deploy = 0.0
    realized_gross_bps_since_deploy = 0.0
    current_equity_usd = INITIAL_EQUITY_USD
    current_mode_is_live = False
    cycles_path = STATE_DIR / "cycles.csv"
    if cycles_path.exists():
        try:
            cdf = pd.read_csv(cycles_path)
            # Latest cycle's equity (Phase 4 column; missing in pre-Phase-4
            # rows → defaults to INITIAL_EQUITY_USD as legacy fallback).
            if "equity_usd" in cdf.columns and not cdf.empty:
                last_eq = cdf["equity_usd"].iloc[-1]
                if pd.notna(last_eq) and float(last_eq) > 0:
                    current_equity_usd = float(last_eq)
            if "live_execute" in cdf.columns and not cdf.empty:
                current_mode_is_live = bool(cdf["live_execute"].iloc[-1] is True
                                             or cdf["live_execute"].iloc[-1] == "True")
            # Filter cumulative to only same-mode rows (matching latest).
            if "live_execute" in cdf.columns:
                live_mask = cdf["live_execute"].fillna(False).astype(bool)
                same_mode = cdf[live_mask == current_mode_is_live]
            else:
                # Pre-Phase-4 file: assume all rows are sim
                same_mode = cdf if not current_mode_is_live else cdf.iloc[0:0]
            realized_net_bps_since_deploy = float(same_mode["net_bps"].sum())
            realized_gross_bps_since_deploy = float(same_mode["gross_pnl_bps"].sum())
        except Exception as e:
            log.warning("could not read cycles.csv: %s", e)

    # 3. Append to hourly_pnl.csv — but ONLY for COMPLETE ticks. Partial
    # ticks have already discarded mark/funding updates and don't advance
    # hourly_last_tick, so writing a row here would overlap with the next
    # successful tick's window and produce duplicate/inconsistent MtM
    # entries in the running log. (Issue 3c fix, 2026-05-09)
    bps_per_dollar = 1e4 / current_equity_usd if current_equity_usd > 0 else 0.0
    mtm_since_last_cycle_bps = total_since_last_cycle_usd * bps_per_dollar
    cumulative_strategy_pnl_bps = (
        realized_net_bps_since_deploy + mtm_since_last_cycle_bps
    )
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
        # New: composes correctly across rebalances. Use this for PnL tracking.
        "realized_net_bps_since_deploy": realized_net_bps_since_deploy,
        "realized_gross_bps_since_deploy": realized_gross_bps_since_deploy,
        "mtm_since_last_cycle_bps": mtm_since_last_cycle_bps,
        "cumulative_strategy_pnl_bps": cumulative_strategy_pnl_bps,
    }
    if tick_complete:
        df_new = pd.DataFrame([row])
        if HOURLY_PNL_PATH.exists():
            df = pd.concat([pd.read_csv(HOURLY_PNL_PATH), df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(HOURLY_PNL_PATH, index=False)
    else:
        log.warning("Tick incomplete — NOT appending to hourly_pnl.csv "
                    "(would overlap with next successful tick's window)")

    # 4. Build Telegram message
    bps_h_pnl = row["hourly_pnl_bps"]
    bps_h_fund = row["hourly_funding_bps"]
    bps_h_net = row["hourly_net_bps"]
    bps_cum = row["cumulative_unrealized_bps"]
    bps_realized = row["realized_net_bps_since_deploy"]
    bps_open_cycle = row["mtm_since_last_cycle_bps"]
    bps_total = row["cumulative_strategy_pnl_bps"]

    # Sort legs by hourly PnL (best first)
    per_leg_sorted = sorted(per_leg, key=lambda d: -d["hourly_pnl_usd"])
    legs_long = [d for d in per_leg_sorted if d["side"] == "L"]
    legs_short = [d for d in per_leg_sorted if d["side"] == "S"]

    msg_lines = [
        f"📊 <b>v6_clean hourly</b>  ({now:%Y-%m-%d %H:%M} UTC)",
        f"",
        f"Equity: ${current_equity_usd:,.2f}{' (live)' if current_mode_is_live else ' (sim)'}  •  "
        f"Open positions: {len(positions)}  •  Notional: ${total_notional_usd:,.0f}",
        f"",
        f"<b>Hourly</b>:    PnL {_fmt_bps(bps_h_pnl)} bps  •  funding {_fmt_bps(bps_h_fund)} bps  •  net {_fmt_bps(bps_h_net)} bps  ({_fmt_usd(row['hourly_net_usd'])})",
        f"<b>Strategy PnL since deploy</b>: <b>{_fmt_bps(bps_total)} bps</b>  "
        f"= realized {_fmt_bps(bps_realized)} + open-cycle {_fmt_bps(bps_open_cycle)}",
        f"<i>Held-leg MtM-vs-entry: {_fmt_bps(bps_cum)} bps "
        f"(snapshot view; resets at rebalance — not strategy PnL)</i>",
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
