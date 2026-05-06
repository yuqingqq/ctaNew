"""Hyperliquid live-execution adapter for paper_bot.

Bridges paper_bot's per-symbol delta-trade calls into the executeEngine
(SignalLimitStrategy on HL mainnet). Returns a dict shaped like
`simulate_taker_fill` so the call site swap is one line.

Topology: this module sys.path-prepends the executeEngine repo so we can
import its modules without packaging changes. Both repos live under
/home/yuqing/.

Usage:
    executor = await HLExecutor.create()
    fills = await executor.batch_fill_deltas([
        {"coin": "BTC", "side": "buy", "target_notional_usd": 25.0, "signal_mid": 81450.0},
        ...
    ])
    await executor.close()

Failure policy (per integration plan): a failed delta returns a sentinel
fill dict with qty=0 and notes set, never raises. Caller decides whether
to skip the leg or abort the cycle.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# --- sys.path bootstrap to find executeEngine alongside ctaNew ---
_ENGINE_ROOT = Path(os.environ.get("EXECUTEENGINE_ROOT", "/home/yuqing/executeEngine"))
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

# Auto-load the executeEngine .env (HL_ACCOUNT_ADDRESS / HL_SECRET_KEY) so
# paper_bot doesn't need to know about it.
_ENV_FILE = _ENGINE_ROOT / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

from engine_bootstrap import bootstrap_engine  # type: ignore  # noqa: E402
from execution_engine_skeleton import ExecutionPlan, ExecutionStatus  # noqa: E402
from exchange_adapter import OrderSide  # noqa: E402

log = logging.getLogger("hl_executor")


# Engine's hard-coded conservative min_notional floor (execution_engine_skeleton.py:674).
# Anything below this is silently rejected by the engine. We pre-filter so
# callers see a clean "skipped" result rather than a generic risk reject.
ENGINE_MIN_NOTIONAL_USD = 20.0

# Margin above the engine floor — gives rounding headroom.
DELTA_MIN_NOTIONAL_USD = 25.0


# Spread-aware duration tiers. SignalLimitStrategy's maker phase needs time
# to be hit; a tight HL spread (≤1 bp) means there's no maker price between
# bid and ask, so any post-only order rests at the bid and waits passively.
# Longer durations help only when spread is wide enough that the strategy
# can post inside the spread and chase aggressively.
#
# Phase 0 evidence: BTC at 1.2 bp spread → 0% maker fill at 60s, ~10% at
# 300s. APT at 3 bp spread → 50% maker fill (Phase 1 fired). ARB/DOGE/OP
# typically 2–5 bp → wide enough for chase to matter.
DURATION_TIERS = [
    # (max_spread_bps, max_duration_s, chase_wait_s, initial_wait_s, max_chase_attempts)
    ( 1.0,  60.0, 2.0, 2.0,  3),   # tight (BTC/ETH/SOL/LTC) — maker rare, IOC quickly
    ( 3.0, 180.0, 3.0, 3.0,  6),   # mid — partial maker chance
    (1e9, 300.0, 5.0, 5.0, 10),   # wide — full maker patience
]


def _strategy_params_for_spread(spread_bps: Optional[float]) -> dict:
    """Pick max_duration / chase tunables based on observed spread.

    None or unknown spread defaults to the wide tier (most patient).
    """
    s = spread_bps if (spread_bps is not None and spread_bps >= 0) else 1e9
    for cap, dur, chase, init, attempts in DURATION_TIERS:
        if s <= cap:
            return {
                "execution_max_duration_seconds": dur,
                "execution_chase_wait_seconds": chase,
                "execution_initial_wait_seconds": init,
                "execution_max_chase_attempts": attempts,
                "execution_ioc_reserve_seconds": min(5.0, dur * 0.05),
                "_chosen_tier_max_spread": cap,
                "_chosen_max_duration": dur,
            }
    # Should be unreachable — last tier matches everything.
    return {
        "execution_max_duration_seconds": 300.0,
        "execution_chase_wait_seconds": 5.0,
        "execution_initial_wait_seconds": 5.0,
        "execution_max_chase_attempts": 10,
        "execution_ioc_reserve_seconds": 5.0,
    }


class HLExecutor:
    """Async live-execute façade. One instance == one connected engine."""

    def __init__(self, engine, exchange, *, config_path: str, mode: str = "signal_limit"):
        self.engine = engine
        self.exchange = exchange
        self.config_path = config_path
        self.mode = mode  # "signal_limit" | "ioc"
        self._closed = False

    @classmethod
    async def create(
        cls,
        config_path: str = "/home/yuqing/executeEngine/hyperliquid_mainnet.yaml",
        *,
        mode: str = "signal_limit",
    ) -> "HLExecutor":
        engine, exchange, _ = await bootstrap_engine(config_path)
        log.info("HLExecutor connected via %s (mode=%s)", config_path, mode)
        return cls(engine, exchange, config_path=config_path, mode=mode)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.engine.stop_services()
        except Exception as e:
            log.warning("HLExecutor.engine.stop_services failed: %s", e)
        try:
            await self.exchange.disconnect()
        except Exception as e:
            log.warning("HLExecutor.exchange.disconnect failed: %s", e)

    # ------------------------------------------------------------------ #
    # Single-leg execution                                                #
    # ------------------------------------------------------------------ #
    async def execute_delta(
        self,
        *,
        coin: str,
        side: str,                       # "buy" | "sell"
        target_notional_usd: float,
        signal_mid: float,
        observed_spread_bps: Optional[float] = None,
    ) -> dict:
        """Execute one delta trade. Returns a fill dict matching the
        `simulate_taker_fill` shape with two extra keys (notes, over_fill_qty).

        observed_spread_bps: tier the SignalLimit max_duration on observed
        spread. Tight spreads (≤1 bp) → 60s (maker rarely fills, IOC fast).
        Wide spreads (>3 bp) → 300s (real maker savings possible).
        """
        symbol = f"{coin}/USDC"

        # Pre-filter: below engine floor → skip, no engine call.
        if target_notional_usd < ENGINE_MIN_NOTIONAL_USD:
            return _skipped_fill(
                signal_mid,
                f"skipped: notional ${target_notional_usd:.2f} < engine min ${ENGINE_MIN_NOTIONAL_USD:.2f}",
            )

        # Translate to ExecutionPlan.
        try:
            qty = self.exchange.round_quantity_to_step(symbol, target_notional_usd / signal_mid)
        except Exception as e:
            return _skipped_fill(signal_mid, f"qty rounding failed: {e}")

        if qty <= 0:
            return _skipped_fill(signal_mid, "qty rounded to zero")

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # Clamp to per-symbol risk cap from hyperliquid_mainnet.yaml. The
        # engine's pre-trade gate rejects any plan exceeding it.
        target_slip_bps = 25.0
        try:
            limits = self.engine.risk_manager.get_limits(symbol)
            cap = limits.get("max_slippage_bps") or limits.get("max_allowed_slippage_bps")
            if cap is not None:
                target_slip_bps = min(target_slip_bps, float(cap))
        except Exception as e:
            log.debug("[%s] could not read risk cap, using default %.1fbps: %s",
                      symbol, target_slip_bps, e)

        # Tier the strategy params (max_duration, chase wait, etc.) by spread.
        spread_params = _strategy_params_for_spread(observed_spread_bps)
        chosen_dur = spread_params.pop("_chosen_max_duration", 300.0)
        chosen_tier = spread_params.pop("_chosen_tier_max_spread", None)
        if observed_spread_bps is not None:
            log.debug(
                "[%s] spread=%.2fbps → tier_cap=%s, max_duration=%.0fs",
                symbol, observed_spread_bps, chosen_tier, chosen_dur,
            )

        metadata: dict[str, Any] = {
            "execution_price_mode": self.mode,  # "signal_limit" | "ioc"
            "signal_price": signal_mid,
            "execution_max_slippage_bps": target_slip_bps,
            "execution_chase_fraction": 0.3,
            **spread_params,
        }
        plan = ExecutionPlan(
            symbol=symbol,
            side=order_side,
            total_quantity=qty,
            urgency="low",
            max_slippage_bps=target_slip_bps,
            time_horizon_seconds=int(chosen_dur),
            metadata=metadata,
        )

        try:
            result, _meta = await self.engine.execute(plan)
        except Exception as e:
            log.error("[%s] engine.execute raised: %s", symbol, e)
            return _failed_fill(signal_mid, f"engine.execute raised: {e}")

        return _result_to_fill(result, fallback_mid=signal_mid)

    # ------------------------------------------------------------------ #
    # Batch fill                                                          #
    # ------------------------------------------------------------------ #
    async def batch_fill_deltas(
        self,
        deltas: list[dict],
        *,
        max_concurrent: int = 5,
    ) -> dict[str, dict]:
        """Fan out delta trades in parallel (bounded concurrency). Returns
        {coin: fill_dict}.

        Each delta dict: {"coin", "side", "target_notional_usd", "signal_mid"}.
        Failures are caught per-leg; one bad symbol does not abort the batch.

        Concurrency is bounded by `max_concurrent` so HL rate limits aren't
        tripped (HL allows ~50 RPS info / 10 RPS exchange — 5 concurrent
        signal-limit plans is well under). The engine has per-symbol locks,
        so distinct symbols are safe to run in parallel.

        Wall time goes from N×~60s sequential to ~60s + (N/concurrency)×fan-in.
        """
        import asyncio as _asyncio
        sem = _asyncio.Semaphore(max(1, int(max_concurrent)))

        async def _one(d: dict) -> tuple[str, dict]:
            coin = d["coin"]
            async with sem:
                try:
                    fill = await self.execute_delta(
                        coin=coin,
                        side=d["side"],
                        target_notional_usd=float(d["target_notional_usd"]),
                        signal_mid=float(d["signal_mid"]),
                        observed_spread_bps=d.get("spread_bps"),
                    )
                except Exception as e:
                    log.exception("[%s] execute_delta unexpected failure", coin)
                    fill = _failed_fill(d.get("signal_mid", 0.0), f"unexpected: {e}")
            log.info(
                "[%s] %s ${%.2f} -> qty=%.6f vwap=%.6f slip=%.2fbps fully_filled=%s%s",
                coin, d["side"], d["target_notional_usd"],
                fill.get("qty", 0.0), fill.get("vwap", 0.0),
                fill.get("slippage_bps", 0.0), fill.get("fully_filled", False),
                f"  notes={fill['notes']}" if fill.get("notes") else "",
            )
            return coin, fill

        results = await _asyncio.gather(*(_one(d) for d in deltas))
        return dict(results)

    async def fetch_equity_usd(self) -> float:
        """Account-value (margin equity) in USDC. Used by paper_bot to size
        delta trades against actual exchange capital."""
        bal = await self.exchange.fetch_balance()
        # Hyperliquid surfaces both spot USDC and account_value (incl PnL).
        # account_value is the right ceiling for new entries.
        v = bal.get("account_value")
        if v is None or v <= 0:
            v = bal.get("USDC", 0.0)
        return float(v or 0.0)


# -------------------------------------------------------------------------- #
# Result translation                                                          #
# -------------------------------------------------------------------------- #
def _result_to_fill(result, *, fallback_mid: float) -> dict:
    """Map ExecutionResult → fill dict shaped like simulate_taker_fill."""
    qty = float(result.filled_quantity or 0.0)
    vwap = float(result.average_price or 0.0)
    arrival = float(result.arrival_price or fallback_mid or 0.0)
    slippage_bps = float(result.realized_slippage_bps or 0.0)

    fully_filled = (
        result.status == ExecutionStatus.COMPLETED
        and qty > 0
    )

    notes = result.notes or ""
    over_fill_qty = 0.0
    if "OVER_FILL=" in notes:
        # Parse "OVER_FILL=0.001234 (..." for the magnitude.
        try:
            tail = notes.split("OVER_FILL=", 1)[1]
            num_str = tail.split(" ", 1)[0]
            over_fill_qty = float(num_str)
        except (ValueError, IndexError):
            log.warning("Could not parse OVER_FILL magnitude from notes: %s", notes)

    return {
        "vwap": vwap if qty > 0 else float("nan"),
        "mid": arrival if arrival > 0 else fallback_mid,
        "slippage_bps": slippage_bps,
        "qty": qty,
        "levels_consumed": len(result.orders or []),
        "fully_filled": fully_filled,
        "notes": notes,
        "over_fill_qty": over_fill_qty,
        "status": result.status.value if result.status else "UNKNOWN",
    }


def _skipped_fill(mid: float, reason: str) -> dict:
    return {
        "vwap": float("nan"),
        "mid": float(mid) if mid else 0.0,
        "slippage_bps": 0.0,
        "qty": 0.0,
        "levels_consumed": 0,
        "fully_filled": False,
        "notes": reason,
        "over_fill_qty": 0.0,
        "status": "SKIPPED",
    }


def _failed_fill(mid: float, reason: str) -> dict:
    return {
        "vwap": float("nan"),
        "mid": float(mid) if mid else 0.0,
        "slippage_bps": float("nan"),
        "qty": 0.0,
        "levels_consumed": 0,
        "fully_filled": False,
        "notes": reason,
        "over_fill_qty": 0.0,
        "status": "FAILED",
    }
