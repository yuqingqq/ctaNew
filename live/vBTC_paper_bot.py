"""vBTC paper-trading bot — single-cycle orchestrator (4-hour rebalance cadence).

Implements the validated production stack:
  - Rolling-IC top-15 universe with PIT 60d eligibility
  - LGBM 5-seed ensemble (model artifact loaded from models/vBTC_production.pkl)
  - K=4 long / K=4 short with conv_gate (30th-pctile dispersion)
  - PM_M2_b1 entry persistence
  - flat_real skip mode (close on gate, re-open on clear with 2-leg cost)
  - dd_tier_aggressive overlay: dd>10%→0.6, dd>20%→0.3, dd>30%→0.1

One invocation = one rebalance cycle. State persists in live/state/vBTC/.

Run modes:
  python -m live.vBTC_paper_bot                 # one cycle now (default paper)
  python -m live.vBTC_paper_bot --check-state   # print current positions/state
  python -m live.vBTC_paper_bot --backtest-cycle <ISO>  # simulate at past time

State files (live/state/vBTC/):
  positions.json     Open positions
  cycles.csv         Append-only cycle log
  dispersion.json    deque of trailing dispersion (252 entries)
  pm_history.json    PM-persistence picks (last PM_M cycles)
  cum_pnl.json       Cumulative PnL + peak for DD overlay
  features_cache.parquet   Latest computed features (for monitoring)

Cost model: 4.5 bps per leg taker fee (HL VIP-0 baseline). Update if account
fee tier differs.

LIVE EXECUTION HOOKS:
  - place_order_live(): currently raises NotImplementedError. Wire to HL Info+Exchange API.
  - fetch_latest_klines(): currently reads cached panel. Wire to Binance REST for
    fresh 5-min bars within the last 14 days per symbol.
"""
from __future__ import annotations
import argparse, json, logging, pickle, sys
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STATE_DIR = REPO / "live/state/vBTC"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = REPO / "models/vBTC_production.pkl"
PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"

POSITIONS_PATH = STATE_DIR / "positions.json"
CYCLES_LOG = STATE_DIR / "cycles.csv"
DISP_PATH = STATE_DIR / "dispersion.json"
PM_HIST_PATH = STATE_DIR / "pm_history.json"
CUM_PNL_PATH = STATE_DIR / "cum_pnl.json"

HORIZON = 48
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
TOP_K = 4

# --- DD overlay tiers ---
DD_TIERS = [(0.30, 0.10), (0.20, 0.30), (0.10, 0.60)]  # (threshold, size); checked desc


# =========================================================================
# State management
# =========================================================================

@dataclass
class Positions:
    """Currently held long/short legs."""
    long: list  # list of symbol strings
    short: list
    is_flat: bool
    entry_cycle_ts: Optional[str] = None  # ISO timestamp of entry

    @classmethod
    def empty(cls): return cls([], [], False, None)


def load_positions() -> Positions:
    if not POSITIONS_PATH.exists():
        return Positions.empty()
    with open(POSITIONS_PATH) as f:
        d = json.load(f)
    return Positions(**d)


def save_positions(p: Positions):
    with open(POSITIONS_PATH, "w") as f:
        json.dump(asdict(p), f, indent=2)


def load_dispersion_history() -> deque:
    if not DISP_PATH.exists():
        return deque(maxlen=GATE_LOOKBACK)
    with open(DISP_PATH) as f:
        d = json.load(f)
    return deque(d.get("values", []), maxlen=GATE_LOOKBACK)


def save_dispersion_history(dh: deque):
    with open(DISP_PATH, "w") as f:
        json.dump({"values": list(dh)}, f)


def load_pm_history() -> list:
    if not PM_HIST_PATH.exists():
        return []
    with open(PM_HIST_PATH) as f:
        d = json.load(f)
    return [{"long": set(x["long"]), "short": set(x["short"])} for x in d.get("history", [])]


def save_pm_history(history: list):
    with open(PM_HIST_PATH, "w") as f:
        json.dump({"history": [{"long": sorted(h["long"]), "short": sorted(h["short"])}
                                for h in history]}, f, indent=2)


def load_cum_pnl() -> dict:
    if not CUM_PNL_PATH.exists():
        return {"cum_pnl_bps": 0.0, "peak_bps": 0.0, "n_cycles": 0}
    with open(CUM_PNL_PATH) as f:
        return json.load(f)


def save_cum_pnl(d: dict):
    with open(CUM_PNL_PATH, "w") as f:
        json.dump(d, f, indent=2)


# =========================================================================
# DD overlay sizing
# =========================================================================

def compute_dd_size(cum_pnl_bps: float, peak_bps: float) -> float:
    """Return position size multiplier based on current drawdown from peak.

    dd_tier_aggressive: dd>10%→0.6, dd>20%→0.3, dd>30%→0.1
    """
    if peak_bps <= 0:
        return 1.0
    dd_pct = (peak_bps - cum_pnl_bps) / peak_bps
    for threshold, size in DD_TIERS:  # iterate desc; first match wins
        if dd_pct > threshold:
            return size
    return 1.0


# =========================================================================
# Strategy engine — one cycle
# =========================================================================

def run_strategy_cycle(
    cycle_ts: pd.Timestamp,
    features_panel: pd.DataFrame,  # rows for cycle_ts, columns=features
    artifact: dict,
    state: dict,  # contains positions, dispersion_history, pm_history, cum_pnl
) -> dict:
    """Run one rebalance cycle. Returns dict with decisions + new state.

    features_panel must have rows for current cycle_ts, with columns:
      - 'symbol', 'open_time', feature columns, optional 'return_pct' for prior cycle PnL
    """
    positions: Positions = state["positions"]
    disp_history: deque = state["dispersion_history"]
    pm_history: list = state["pm_history"]
    cum_pnl_d: dict = state["cum_pnl"]

    feat_set = artifact["feature_set"]
    universe = set(artifact["deployment_universe"])
    listings = {s: pd.Timestamp(t) for s, t in artifact["listings"].items()}

    # PIT eligibility at current cycle
    cutoff = cycle_ts - pd.Timedelta(days=artifact["min_history_days"])
    eligible = {s for s, t in listings.items() if t <= cutoff}
    trade_universe = universe & eligible
    n_universe = len(trade_universe)

    # Filter to universe + cycle
    cycle_panel = features_panel[
        (features_panel["open_time"] == cycle_ts) &
        (features_panel["symbol"].isin(trade_universe))
    ].copy()

    if len(cycle_panel) < 2 * TOP_K + 1:
        return {
            "cycle_ts": cycle_ts.isoformat(),
            "decision": "skip_insufficient_universe",
            "n_universe": n_universe,
            "n_panel": len(cycle_panel),
            "new_positions": positions,
            "size_multiplier": compute_dd_size(cum_pnl_d["cum_pnl_bps"],
                                                 cum_pnl_d["peak_bps"]),
        }

    # Predict
    X = cycle_panel[feat_set].to_numpy(np.float32)
    models = artifact["ensemble_models"]
    iters = artifact["ensemble_best_iters"]
    pred_ensemble = np.mean(
        [m.predict(X, num_iteration=it) for m, it in zip(models, iters)],
        axis=0
    )
    cycle_panel["pred"] = pred_ensemble

    sym_arr = cycle_panel["symbol"].to_numpy()
    pred_arr = cycle_panel["pred"].to_numpy()

    # Top/bottom K
    idx_top = np.argpartition(-pred_arr, TOP_K - 1)[:TOP_K]
    idx_bot = np.argpartition(pred_arr, TOP_K - 1)[:TOP_K]
    cand_long = set(sym_arr[idx_top])
    cand_short = set(sym_arr[idx_bot])

    # Conv gate: dispersion percentile
    dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())
    skip = False
    if len(disp_history) >= 30:
        thr = float(np.quantile(list(disp_history), GATE_PCTILE))
        if dispersion < thr:
            skip = True
    disp_history.append(dispersion)

    # PM history update (always update regardless of skip)
    band_k = max(TOP_K, int(round(PM_BAND * TOP_K)))
    n_panel = len(cycle_panel)
    bk = min(band_k, n_panel)
    idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < n_panel else np.arange(n_panel)
    idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < n_panel else np.arange(n_panel)
    pm_history.append({
        "long": set(sym_arr[idx_top_band]),
        "short": set(sym_arr[idx_bot_band])
    })
    if len(pm_history) > PM_M:
        pm_history[:] = pm_history[-PM_M:]

    # DD-overlay size multiplier
    size_mult = compute_dd_size(cum_pnl_d["cum_pnl_bps"], cum_pnl_d["peak_bps"])

    # Skip handling (flat_real mode)
    if skip:
        cur_l = set(positions.long); cur_s = set(positions.short)
        if not positions.is_flat and (cur_l or cur_s):
            # Close out → flat
            new_positions = Positions(long=[], short=[], is_flat=True,
                                        entry_cycle_ts=None)
            decision = "close_to_flat_skip"
        else:
            new_positions = positions
            decision = "hold_flat_skip"
        return {
            "cycle_ts": cycle_ts.isoformat(),
            "decision": decision, "skip_reason": "conv_gate",
            "dispersion": dispersion,
            "dispersion_pctile_threshold": thr if len(disp_history) > 30 else None,
            "new_positions": new_positions,
            "size_multiplier": size_mult,
            "n_universe": n_universe,
        }

    # PM persistence filter on entries
    cur_l = set(positions.long); cur_s = set(positions.short)
    if len(pm_history) >= PM_M:
        past_long = [h["long"] for h in pm_history[-PM_M:][:PM_M - 1]]
        past_short = [h["short"] for h in pm_history[-PM_M:][:PM_M - 1]]
        new_long = cur_l & cand_long
        new_short = cur_s & cand_short
        for s in cand_long - cur_l:
            if all(s in p for p in past_long): new_long.add(s)
        for s in cand_short - cur_s:
            if all(s in p for p in past_short): new_short.add(s)
        # Cap at K by pred-strength
        if len(new_long) > TOP_K:
            ranked = sorted(new_long, key=lambda s: -pred_arr[sym_arr == s][0])[:TOP_K]
            new_long = set(ranked)
        if len(new_short) > TOP_K:
            ranked = sorted(new_short, key=lambda s: pred_arr[sym_arr == s][0])[:TOP_K]
            new_short = set(ranked)
    else:
        new_long, new_short = cand_long, cand_short

    if not new_long or not new_short:
        return {
            "cycle_ts": cycle_ts.isoformat(),
            "decision": "skip_empty_leg",
            "new_positions": positions, "size_multiplier": size_mult,
            "n_universe": n_universe,
        }

    new_positions = Positions(
        long=sorted(new_long), short=sorted(new_short),
        is_flat=False, entry_cycle_ts=cycle_ts.isoformat()
    )
    return {
        "cycle_ts": cycle_ts.isoformat(),
        "decision": "rebalance",
        "long_syms": sorted(new_long),
        "short_syms": sorted(new_short),
        "dispersion": dispersion,
        "size_multiplier": size_mult,
        "new_positions": new_positions,
        "n_universe": n_universe,
        "n_long": len(new_long), "n_short": len(new_short),
    }


# =========================================================================
# Cycle PnL bookkeeping (paper mode)
# =========================================================================

def compute_realized_pnl(
    positions: Positions,
    prior_cycle_ts: pd.Timestamp,
    current_cycle_ts: pd.Timestamp,
    features_panel: pd.DataFrame,
) -> float:
    """Compute realized spread PnL between two cycles (paper mode).

    Uses prior cycle's positions and the realized return_pct on the current
    cycle's panel (which represents the forward return from prior cycle).
    """
    if not positions.long or not positions.short or positions.is_flat:
        return 0.0
    # We look at return_pct at current_cycle_ts which is fwd-return from prior
    sub = features_panel[features_panel["open_time"] == current_cycle_ts]
    long_g = sub[sub["symbol"].isin(positions.long)]
    short_g = sub[sub["symbol"].isin(positions.short)]
    if long_g.empty or short_g.empty:
        return 0.0
    long_ret = long_g["return_pct"].mean()
    short_ret = short_g["return_pct"].mean()
    return float((long_ret - short_ret) * 1e4)


def compute_trade_cost(prev: Positions, new: Positions, was_flat_to_active: bool) -> float:
    """Compute rebalance cost in bps."""
    if was_flat_to_active:
        return 2 * COST_PER_LEG  # re-open from flat: full entry cost
    if new.is_flat and not prev.is_flat:
        return 2 * COST_PER_LEG  # close to flat: full exit cost
    if not prev.long and not prev.short:
        return 2 * COST_PER_LEG  # cold start entry
    prev_l = set(prev.long); prev_s = set(prev.short)
    new_l = set(new.long); new_s = set(new.short)
    churn_long = len(new_l.symmetric_difference(prev_l)) / max(len(new_l | prev_l), 1)
    churn_short = len(new_s.symmetric_difference(prev_s)) / max(len(new_s | prev_s), 1)
    return (churn_long + churn_short) * COST_PER_LEG


# =========================================================================
# Data fetching — STUB (wire to live data source)
# =========================================================================

def fetch_features_panel_for_cycle(cycle_ts: pd.Timestamp) -> pd.DataFrame:
    """Return feature panel rows for the given cycle timestamp.

    PRODUCTION HOOK: replace this with:
      - Fetch latest Binance 5-min klines for all panel symbols
      - Compute features via features_ml/ pipeline
      - Return DataFrame with same schema as panel_variants_with_funding.parquet

    For now: reads the cached research panel. This works for paper-trading
    against historical cycles (--backtest-cycle mode).
    """
    panel = pd.read_parquet(PANEL_PATH)
    sub = panel[panel["open_time"] == cycle_ts]
    if sub.empty:
        raise ValueError(f"No panel rows for cycle_ts={cycle_ts}")
    return sub


# =========================================================================
# Execution — STUB (wire to broker)
# =========================================================================

def place_orders_paper(decision: dict, size_multiplier: float, notional_per_leg: float):
    """Paper execution: log the orders that would be placed."""
    if decision["decision"] != "rebalance":
        logging.info(f"  no trades ({decision['decision']})")
        return
    sized_notional = notional_per_leg * size_multiplier
    logging.info(f"  PAPER ORDERS (size_mult={size_multiplier:.2f}, notional/leg=${sized_notional:,.0f}):")
    for sym in decision["long_syms"]:
        logging.info(f"    BUY {sym}: target ${sized_notional / TOP_K:,.0f}")
    for sym in decision["short_syms"]:
        logging.info(f"    SELL {sym}: target ${sized_notional / TOP_K:,.0f}")


def place_orders_live(decision: dict, size_multiplier: float, notional_per_leg: float):
    raise NotImplementedError(
        "Live order execution not implemented. Wire to Hyperliquid Exchange API:\n"
        "  - Convert long_syms/short_syms to HL coin codes\n"
        "  - Fetch current marks via Info API\n"
        "  - Place market/IOC orders at sized notional\n"
        "  - Confirm fills, store actual fill prices in state."
    )


# =========================================================================
# Main cycle orchestrator
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-state", action="store_true",
                          help="print current state and exit")
    parser.add_argument("--backtest-cycle", type=str, default=None,
                          help="simulate cycle at this ISO timestamp (UTC)")
    parser.add_argument("--live", action="store_true",
                          help="execute real orders (otherwise paper)")
    parser.add_argument("--notional", type=float, default=10000.0,
                          help="base notional per leg in USD (before size multiplier)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.check_state:
        positions = load_positions()
        cum_pnl = load_cum_pnl()
        disp = load_dispersion_history()
        size_mult = compute_dd_size(cum_pnl["cum_pnl_bps"], cum_pnl["peak_bps"])
        print(f"\n=== vBTC paper bot state ===")
        print(f"Positions:")
        print(f"  long ({len(positions.long)}): {positions.long}")
        print(f"  short ({len(positions.short)}): {positions.short}")
        print(f"  is_flat: {positions.is_flat}")
        print(f"  entry_cycle: {positions.entry_cycle_ts}")
        print(f"Cumulative PnL: {cum_pnl['cum_pnl_bps']:+.0f} bps")
        print(f"Peak PnL: {cum_pnl['peak_bps']:+.0f} bps")
        print(f"Current DD from peak: "
              f"{(cum_pnl['peak_bps'] - cum_pnl['cum_pnl_bps']):+.0f} bps "
              f"({(cum_pnl['peak_bps'] - cum_pnl['cum_pnl_bps']) / max(cum_pnl['peak_bps'], 1) * 100:.1f}%)")
        print(f"Current size multiplier: {size_mult:.2f}")
        print(f"Dispersion history: {len(disp)} entries (max {GATE_LOOKBACK})")
        print(f"Cycles logged: {cum_pnl['n_cycles']}")
        return

    # Determine cycle timestamp
    if args.backtest_cycle:
        cycle_ts = pd.Timestamp(args.backtest_cycle, tz="UTC")
    else:
        # Production: floor to most recent HORIZON-aligned 5-min boundary
        now = pd.Timestamp.utcnow()
        # 5-min bars, sample every HORIZON
        epoch_min = int(now.timestamp() // 60)
        # Round down to nearest 5-min × HORIZON aligned to UTC midnight
        # HORIZON=48 × 5min = 240 min = 4h. Align to 00:00, 04:00, 08:00, ...
        cycle_min = (epoch_min // (HORIZON * 5)) * (HORIZON * 5)
        cycle_ts = pd.Timestamp(cycle_min * 60, unit="s", tz="UTC")

    logging.info(f"vBTC paper bot: cycle_ts = {cycle_ts}")

    # Load artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. "
                                  "Run: python -m live.train_vBTC_artifact")
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    logging.info(f"Loaded artifact v{artifact['version']}, "
                 f"trained {artifact['trained_at'][:19]}")
    logging.info(f"Deployment universe: {artifact['deployment_universe']}")

    # Load state
    positions = load_positions()
    disp_history = load_dispersion_history()
    pm_history = load_pm_history()
    cum_pnl_d = load_cum_pnl()
    logging.info(f"State: positions={len(positions.long)}L/{len(positions.short)}S, "
                 f"is_flat={positions.is_flat}, "
                 f"cum_pnl={cum_pnl_d['cum_pnl_bps']:+.0f}bps, "
                 f"peak={cum_pnl_d['peak_bps']:+.0f}bps")

    # Fetch features for this cycle
    features_panel = fetch_features_panel_for_cycle(cycle_ts)

    # Compute realized PnL from prior cycle (paper bookkeeping)
    realized_bps = 0.0
    if positions.entry_cycle_ts and (positions.long or positions.short):
        try:
            realized_bps = compute_realized_pnl(positions, None, cycle_ts, features_panel)
        except Exception as e:
            logging.warning(f"PnL bookkeeping skipped: {e}")
    if realized_bps != 0:
        logging.info(f"Realized PnL from prior cycle: {realized_bps:+.1f} bps (gross)")

    # Run strategy decision
    state = {
        "positions": positions,
        "dispersion_history": disp_history,
        "pm_history": pm_history,
        "cum_pnl": cum_pnl_d,
    }
    decision = run_strategy_cycle(cycle_ts, features_panel, artifact, state)
    logging.info(f"Decision: {decision['decision']}")
    logging.info(f"DD overlay size multiplier: {decision.get('size_multiplier', 1.0):.2f}")
    if decision["decision"] == "rebalance":
        logging.info(f"  longs:  {decision['long_syms']}")
        logging.info(f"  shorts: {decision['short_syms']}")
        logging.info(f"  dispersion: {decision.get('dispersion', 0):.4f}")

    # Apply realized PnL (with prior cycle's size multiplier — already-realized)
    new_positions: Positions = decision["new_positions"]
    new_size = decision["size_multiplier"]
    # Cost for THIS cycle's rebalance
    was_flat_to_active = positions.is_flat and not new_positions.is_flat
    if decision["decision"] == "rebalance":
        cost_bps = compute_trade_cost(positions, new_positions, was_flat_to_active)
    elif decision["decision"] == "close_to_flat_skip":
        cost_bps = 2 * COST_PER_LEG
    else:
        cost_bps = 0.0
    # Cycle net (size applied to gross spread; cost ALSO scaled by size since it's notional)
    net_cycle = realized_bps * new_size - cost_bps * new_size if not positions.is_flat else 0 - cost_bps * new_size
    # Actually, the standard convention: cost applies to the ENTRY being made at this cycle
    # So net = realized * prior_size - cost * new_size
    # But to keep it simple here, net = realized * size - cost * size
    logging.info(f"Cycle net PnL: realized={realized_bps:+.1f} * size={new_size:.2f} "
                 f"- cost={cost_bps:.1f} * size={new_size:.2f} = {net_cycle:+.1f} bps")

    # Update cum_pnl
    cum_pnl_d["cum_pnl_bps"] += net_cycle
    cum_pnl_d["peak_bps"] = max(cum_pnl_d["peak_bps"], cum_pnl_d["cum_pnl_bps"])
    cum_pnl_d["n_cycles"] += 1

    # Persist state
    save_positions(new_positions)
    save_dispersion_history(disp_history)
    save_pm_history(pm_history)
    save_cum_pnl(cum_pnl_d)

    # Append cycle log
    row = {
        "cycle_ts": cycle_ts.isoformat(),
        "decision": decision["decision"],
        "realized_bps": realized_bps,
        "cost_bps": cost_bps,
        "size_multiplier": new_size,
        "net_cycle_bps": net_cycle,
        "cum_pnl_bps": cum_pnl_d["cum_pnl_bps"],
        "peak_bps": cum_pnl_d["peak_bps"],
        "n_long": len(new_positions.long),
        "n_short": len(new_positions.short),
        "is_flat": new_positions.is_flat,
        "long_syms": ",".join(new_positions.long),
        "short_syms": ",".join(new_positions.short),
        "dispersion": decision.get("dispersion", None),
        "n_universe": decision.get("n_universe", None),
    }
    df_row = pd.DataFrame([row])
    if CYCLES_LOG.exists():
        df_row.to_csv(CYCLES_LOG, mode="a", header=False, index=False)
    else:
        df_row.to_csv(CYCLES_LOG, mode="w", header=True, index=False)
    logging.info(f"  appended cycle to {CYCLES_LOG}")

    # Place orders (paper or live)
    notional_per_leg = args.notional
    if args.live:
        place_orders_live(decision, new_size, notional_per_leg)
    else:
        place_orders_paper(decision, new_size, notional_per_leg)

    logging.info(f"  cum_pnl={cum_pnl_d['cum_pnl_bps']:+.0f}bps, "
                 f"peak={cum_pnl_d['peak_bps']:+.0f}bps, "
                 f"size_next_cycle={compute_dd_size(cum_pnl_d['cum_pnl_bps'], cum_pnl_d['peak_bps']):.2f}")


if __name__ == "__main__":
    main()
