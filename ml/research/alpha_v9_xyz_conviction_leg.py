"""Conviction-conditional single-leg + index hedge.

Each cycle:
  - Rank universe by pred. Top-K and bot-K candidates.
  - long_pred  = mean(pred over top-K candidates)
  - short_pred = mean(pred over bot-K candidates)   # typically negative
  - Decision rule (parametrized by `margin` in pred-space):
      if long_pred > |short_pred| + margin:  active = "long"
      elif |short_pred| > long_pred + margin: active = "short"
      else:                                   active = previous_leg (no flip)
  - Trade only the active leg with hysteresis. Hedge with the basket
    (idealized: P&L = mean of fwd_resid_1d on active leg).
  - On flip, fully liquidate and rebuild on the new leg; pay liquidation
    cost (full leg turnover) + index hedge flip (1 round-trip).

Compare to always-L/S, always-long-only, always-short-only.

Usage:
    python -m ml.research.alpha_v9_xyz_conviction_leg
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_tier_a import TIER_AB
from ml.research.alpha_v9_xyz_pm import load_or_compute_regime
from ml.research.alpha_v9_xyz_legs import (
    daily_portfolio_three_legs, metrics_on, boot_ci_on,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE = CACHE / "v7_tier_a_walkfwd_preds.parquet"

GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def daily_conviction_leg(test_pred: pd.DataFrame, signal: str, pnl_label: str,
                          allowed: set, top_k: int, exit_buffer: int,
                          cost_bps_side: float, margin_bps: float = 0.0,
                          stickiness_cycles: int = 0) -> pd.DataFrame:
    """Conviction-conditional single-leg trader.

    margin_bps: in bps of pred. Only flip leg if conviction asymmetry
        exceeds this margin (otherwise carry previous active leg).
    stickiness_cycles: minimum cycles before another flip is allowed.
        Prevents rapid leg-flipping when conviction hovers near zero.
    """
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    rows = []
    cur_holdings: set = set()
    cur_active: str | None = None
    cycles_since_flip = 0
    margin = margin_bps / 1e4

    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k + exit_buffer:
            continue
        bar = bar.sort_values(signal).reset_index(drop=True)
        n = len(bar)
        bar["rank_top"] = n - 1 - bar.index
        bar["rank_bot"] = bar.index

        # Conviction signal: mean pred of top-K vs bot-K
        top_K_set = set(bar[bar["rank_top"] < top_k]["symbol"])
        bot_K_set = set(bar[bar["rank_bot"] < top_k]["symbol"])
        long_pred = bar[bar["symbol"].isin(top_K_set)][signal].mean()
        short_pred = bar[bar["symbol"].isin(bot_K_set)][signal].mean()
        # |long| > |short| → favor long (long_pred is positive, short_pred is negative)
        proposed = "long" if long_pred > -short_pred else "short"
        diff = long_pred - (-short_pred)  # signed conviction asymmetry

        # Apply margin + stickiness
        if cur_active is None:
            new_active = proposed
        elif cycles_since_flip < stickiness_cycles:
            new_active = cur_active  # locked in
        elif abs(diff) <= margin:
            new_active = cur_active  # not enough asymmetry; stay
        else:
            new_active = proposed

        # Build new holdings on the active side w/ hysteresis
        if new_active == cur_active:
            # Continue: apply hysteresis exit + entry on the same side
            if cur_active == "long":
                rank_col = "rank_top"
            else:
                rank_col = "rank_bot"
            new_holdings = set(cur_holdings)
            for s in list(new_holdings):
                r = bar[bar["symbol"] == s]
                if r.empty:
                    new_holdings.discard(s); continue
                if r[rank_col].iloc[0] > top_k + exit_buffer - 1:
                    new_holdings.discard(s)
            cands = bar[bar[rank_col] < top_k]["symbol"].tolist()
            for s in cands:
                if len(new_holdings) >= top_k: break
                if s in new_holdings: continue
                new_holdings.add(s)
            if len(new_holdings) > top_k:
                ranked = bar[bar["symbol"].isin(new_holdings)].sort_values(rank_col)
                new_holdings = set(ranked.head(top_k)["symbol"])
            leg_chg = len(new_holdings.symmetric_difference(cur_holdings))
            switched = False
        else:
            # Switch: liquidate old leg fully, open new at top-K
            new_holdings = top_K_set if new_active == "long" else bot_K_set
            leg_chg = len(cur_holdings) + len(new_holdings)  # close + open
            switched = True

        # Cost accounting:
        # - Per-name notional 1/K. cost = (transactions / K) * cost_bps_side bps.
        # - Index hedge: assume 1 round-trip per leg flip = 2 transactions
        #   on the index, total cost = 2 * cost_bps_side bps on hedge notional 1.
        cost_names = (leg_chg / top_k) * cost_bps_side / 1e4
        cost_hedge_flip = (2 * cost_bps_side / 1e4) if switched else 0.0
        cost = cost_names + cost_hedge_flip

        # Realize P&L
        if not new_holdings:
            cur_holdings = new_holdings
            cur_active = new_active
            cycles_since_flip = 0 if switched else cycles_since_flip + 1
            continue
        leg_alpha_raw = bar[bar["symbol"].isin(new_holdings)][pnl_label].mean()
        leg_alpha = leg_alpha_raw if new_active == "long" else -leg_alpha_raw

        rows.append({
            "ts": ts, "active": new_active, "switched": switched,
            "leg_alpha": leg_alpha, "cost": cost,
            "net_alpha": leg_alpha - cost,
            "leg_turnover": leg_chg / top_k,
            "n_holdings": len(new_holdings),
            "long_pred": long_pred, "short_pred": short_pred, "diff": diff,
        })

        cur_holdings = new_holdings
        cur_active = new_active
        cycles_since_flip = 0 if switched else cycles_since_flip + 1
    return pd.DataFrame(rows)


def main() -> None:
    log.info("loading cached predictions: %s", PRED_CACHE)
    preds = pd.read_parquet(PRED_CACHE)
    regime = load_or_compute_regime()

    allowed = set(TIER_AB)
    K = 4; M_exit = 1

    log.info("running conviction-conditional variants (Tier A+B, K=%d, cost=%.1f bps/side)...",
             K, COST_BPS_SIDE)

    # Variants: (margin_bps, stickiness)
    variants = [
        ("conv-leg margin=0  stick=0", 0.0, 0),
        ("conv-leg margin=10 stick=0", 10.0, 0),
        ("conv-leg margin=25 stick=0", 25.0, 0),
        ("conv-leg margin=50 stick=0", 50.0, 0),
        ("conv-leg margin=10 stick=5", 10.0, 5),
        ("conv-leg margin=25 stick=5", 25.0, 5),
    ]
    out = []
    for name, m, s in variants:
        pnl_pre = daily_conviction_leg(preds, "pred", "fwd_resid_1d",
                                          allowed, K, M_exit, COST_BPS_SIDE,
                                          margin_bps=m, stickiness_cycles=s)
        pnl_g = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE,
                                window_days=GATE_WINDOW)
        if pnl_g.empty:
            log.warning("  %s empty", name); continue
        mt = metrics_on(pnl_g, "net_alpha")
        lo, hi = boot_ci_on(pnl_g, "net_alpha")
        n_switch = pnl_g["switched"].sum()
        long_share = (pnl_g["active"] == "long").mean()
        log.info("  %-32s n=%d Sh=%+.2f [%+.2f,%+.2f] net=%+.2f bps  "
                 "long-frac=%.0f%% switches=%d (%.1f%%)",
                 name, mt["n_rebal"], mt["active_sharpe"], lo, hi,
                 mt["net_bps_per_rebal"], long_share * 100,
                 n_switch, n_switch / mt["n_rebal"] * 100)
        out.append({"name": name, "metrics": mt, "ci": (lo, hi),
                    "pnl": pnl_g, "n_switch": int(n_switch),
                    "long_share": long_share})

    # Reference: always-L/S, always-long-only, always-short-only (from legs probe)
    log.info("\n=== REFERENCE (from alpha_v9_xyz_legs) ===")
    pnl_ref_pre = daily_portfolio_three_legs(preds, "pred", "fwd_resid_1d",
                                                allowed, K, M_exit, COST_BPS_SIDE)
    pnl_ref = gate_rolling(pnl_ref_pre, regime, pctile=GATE_PCTILE,
                              window_days=GATE_WINDOW)
    refs = []
    for label, col in [("L/S baseline (C5L)", "net_alpha"),
                        ("Always long-only", "net_long_only"),
                        ("Always short-only", "net_short_only")]:
        mt = metrics_on(pnl_ref, col)
        lo, hi = boot_ci_on(pnl_ref, col)
        log.info("  %-32s n=%d Sh=%+.2f [%+.2f,%+.2f] net=%+.2f bps",
                 label, mt["n_rebal"], mt["active_sharpe"], lo, hi,
                 mt["net_bps_per_rebal"])
        refs.append({"name": label, "metrics": mt, "ci": (lo, hi),
                     "pnl": pnl_ref, "col": col})

    # Conviction signal diagnostic
    log.info("\n=== Conviction asymmetry (long_pred − |short_pred|) ===")
    pre = pnl_ref_pre.copy()
    # Re-derive long_pred / short_pred from cached preds for diagnostic
    diag = daily_conviction_leg(preds, "pred", "fwd_resid_1d", allowed, K, M_exit,
                                  COST_BPS_SIDE, margin_bps=0.0, stickiness_cycles=0)
    diag_g = gate_rolling(diag, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    qs = diag_g["diff"].abs().quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    log.info("  |diff| pctile (in bps): 10%%=%.1f  25%%=%.1f  50%%=%.1f  75%%=%.1f  90%%=%.1f",
             qs.iloc[0]*1e4, qs.iloc[1]*1e4, qs.iloc[2]*1e4, qs.iloc[3]*1e4, qs.iloc[4]*1e4)
    log.info("  long-favored (diff>0) frac: %.1f%%  short-favored frac: %.1f%%",
             (diag_g["diff"] > 0).mean() * 100, (diag_g["diff"] <= 0).mean() * 100)

    # Conviction → realized alpha alignment: does the chosen leg actually have
    # the higher realized alpha?
    log.info("\n=== Does conviction predict the better leg? ===")
    realized_long_alpha = pnl_ref[["ts", "long_alpha", "short_alpha"]].copy()
    merged = diag_g[["ts", "diff", "active"]].merge(realized_long_alpha, on="ts")
    merged["realized_better_leg"] = np.where(
        merged["long_alpha"] > -merged["short_alpha"], "long", "short")
    hit = (merged["active"] == merged["realized_better_leg"]).mean()
    log.info("  conviction matches realized-better leg: %.1f%% of cycles", hit * 100)
    # When conviction is strong, is hit rate higher?
    for q_lo, q_hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        thresh_lo = merged["diff"].abs().quantile(q_lo)
        thresh_hi = merged["diff"].abs().quantile(q_hi)
        sub = merged[(merged["diff"].abs() >= thresh_lo) & (merged["diff"].abs() <= thresh_hi)]
        if len(sub) < 5: continue
        h = (sub["active"] == sub["realized_better_leg"]).mean()
        log.info("  |diff| in [%.0f%%, %.0f%%]  (n=%d):  hit=%.1f%%",
                 q_lo * 100, q_hi * 100, len(sub), h * 100)


if __name__ == "__main__":
    main()
