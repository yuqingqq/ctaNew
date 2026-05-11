"""Port crypto v6_clean PM_M2_b1 entry gate to xyz v7 strategy.

Tests whether requiring 2-cycle prediction-rank persistence at entry (a name
not currently held may only enter top-K if it was also in top-K at the prior
bar) improves the v7 deployable spec C5L (Tier A+B, K=4, M=1) on cached
walk-forward predictions.

Discipline gates (per crypto playbook):
  1. ΔSh > +0.20
  2. Bootstrap CI on Δnet > 0 (paired on overlapping ts)
  3. ≥ 60% folds (years) Δsh-positive
  4. K_avg ≥ 2.5 (concentration sanity)

The cached preds at data/ml/cache/v7_tier_a_walkfwd_preds.parquet were
computed once across full SP100 (so cross-sectional ranking is intact).
This script only changes the portfolio-construction step; no retraining.

Usage:
    python -m ml.research.alpha_v9_xyz_pm
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_multi import add_returns_and_basket, load_anchors
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, metrics_for, boot_ci,
)
from ml.research.alpha_v7_tier_a import TIER_A, TIER_B, TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE = CACHE / "v7_tier_a_walkfwd_preds.parquet"
REGIME_CACHE = CACHE / "v7_regime.parquet"

GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8  # current xyz growth-mode taker (deployable per memory)


# ---------------------------------------------------------------------------
# Portfolio construction with PM entry filter
# ---------------------------------------------------------------------------

def daily_portfolio_pm_hysteresis(test_pred: pd.DataFrame, signal: str,
                                    pnl_label: str, allowed: set,
                                    top_k: int, exit_buffer: int,
                                    cost_bps_side: float, M: int) -> pd.DataFrame:
    """PM_M{M} + hysteresis. Mirrors daily_portfolio_hysteresis except:

    Entry rule: a non-held candidate must have been in top-K at each of the
    M-1 prior bars (within `allowed` universe) to enter. M=1 disables the PM
    filter and is identical to baseline.

    Exit rule: held names auto-keep until rank > top_k + exit_buffer (same
    as baseline hysteresis).

    Per-name weight = 1 / n_active (matches baseline `.mean()` convention).
    """
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    rows = []
    cur_long, cur_short = set(), set()
    long_topk_history: list[set] = []
    short_topk_history: list[set] = []
    enter_long_rank = top_k
    exit_long_rank = top_k + exit_buffer

    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k + exit_buffer:
            continue
        bar = bar.sort_values(signal).reset_index(drop=True)
        n = len(bar)
        bar["rank_top"] = n - 1 - bar.index
        bar["rank_bot"] = bar.index

        cur_long_topk = set(bar[bar["rank_top"] < enter_long_rank]["symbol"])
        cur_short_topk = set(bar[bar["rank_bot"] < enter_long_rank]["symbol"])

        # PM filter: name must have been top-K in each of the past (M-1) bars.
        if M >= 2 and len(long_topk_history) >= M - 1:
            persistent_long = cur_long_topk.copy()
            for past in long_topk_history[-(M - 1):]:
                persistent_long &= past
        elif M == 1:
            persistent_long = cur_long_topk
        else:
            persistent_long = set()  # not enough history; no entries allowed

        if M >= 2 and len(short_topk_history) >= M - 1:
            persistent_short = cur_short_topk.copy()
            for past in short_topk_history[-(M - 1):]:
                persistent_short &= past
        elif M == 1:
            persistent_short = cur_short_topk
        else:
            persistent_short = set()

        # Long leg: hysteresis exit + PM-gated entry
        new_long = set(cur_long)
        for s in list(new_long):
            r = bar[bar["symbol"] == s]
            if r.empty:
                new_long.discard(s); continue
            if r["rank_top"].iloc[0] > exit_long_rank - 1:
                new_long.discard(s)
        candidates = bar[(bar["rank_top"] < enter_long_rank) &
                          (bar["symbol"].isin(persistent_long))]["symbol"].tolist()
        for s in candidates:
            if len(new_long) >= top_k:
                break
            if s in new_long:
                continue
            new_long.add(s)
        if len(new_long) > top_k:
            ranked = bar[bar["symbol"].isin(new_long)].sort_values("rank_top")
            new_long = set(ranked.head(top_k)["symbol"])

        # Short leg
        new_short = set(cur_short)
        for s in list(new_short):
            r = bar[bar["symbol"] == s]
            if r.empty:
                new_short.discard(s); continue
            if r["rank_bot"].iloc[0] > exit_long_rank - 1:
                new_short.discard(s)
        cands_s = bar[(bar["rank_bot"] < enter_long_rank) &
                       (bar["symbol"].isin(persistent_short))]["symbol"].tolist()
        for s in cands_s:
            if len(new_short) >= top_k:
                break
            if s in new_short:
                continue
            new_short.add(s)
        if len(new_short) > top_k:
            ranked = bar[bar["symbol"].isin(new_short)].sort_values("rank_bot")
            new_short = set(ranked.head(top_k)["symbol"])

        # Maintain history (last M-1 bars).
        long_topk_history.append(cur_long_topk)
        short_topk_history.append(cur_short_topk)
        if len(long_topk_history) > max(M - 1, 1):
            long_topk_history.pop(0)
            short_topk_history.pop(0)

        long_chg = len(new_long.symmetric_difference(cur_long))
        short_chg = len(new_short.symmetric_difference(cur_short))
        turnover = (long_chg + short_chg) / (2 * top_k)
        cost = turnover * cost_bps_side * 2 / 1e4

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            continue
        long_a = bar[bar["symbol"].isin(new_long)][pnl_label].mean()
        short_a = bar[bar["symbol"].isin(new_short)][pnl_label].mean()
        spread = long_a - short_a
        rows.append({"ts": ts, "spread_alpha": spread, "long_alpha": long_a,
                     "short_alpha": short_a, "turnover": turnover, "cost": cost,
                     "net_alpha": spread - cost, "n_universe": n,
                     "n_long": len(new_long), "n_short": len(new_short)})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def paired_block_bootstrap(pnl_a: pd.DataFrame, pnl_b: pd.DataFrame,
                             block_size: int = 10, n_boot: int = 2000,
                             seed: int = 42) -> tuple[float, float, float]:
    """Block-bootstrap CI on (mean(b) - mean(a)) net_alpha, paired on ts.

    Returns: (mean Δnet bps, lo, hi) at 95%.
    """
    merged = pnl_a[["ts", "net_alpha"]].rename(columns={"net_alpha": "a"}).merge(
        pnl_b[["ts", "net_alpha"]].rename(columns={"net_alpha": "b"}), on="ts")
    if len(merged) < 60:
        return np.nan, np.nan, np.nan
    diff = (merged["b"] - merged["a"]).to_numpy() * 1e4  # bps/cycle
    n = len(diff)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        means.append(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_year_table(pnl: pd.DataFrame) -> list[tuple]:
    if pnl.empty:
        return []
    out = []
    p = pnl.copy()
    p["year"] = pd.to_datetime(p["ts"]).dt.year
    for y, g in p.groupby("year"):
        if len(g) < 5: continue
        m = metrics_for(g, 1)
        out.append((y, m["n_rebal"], m["net_bps_per_rebal"], m["active_sharpe"]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_or_compute_regime() -> pd.DataFrame:
    if REGIME_CACHE.exists():
        log.info("loading cached regime: %s", REGIME_CACHE)
        return pd.read_parquet(REGIME_CACHE)
    log.info("computing regime indicators (cache miss; building panel)...")
    panel, _, _ = load_universe()
    if panel.empty:
        raise RuntimeError("empty panel from load_universe")
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    regime = compute_regime_indicators(panel, anchors)
    REGIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    regime.to_parquet(REGIME_CACHE)
    log.info("  cached regime: %d rows", len(regime))
    return regime


def evaluate_variant(preds: pd.DataFrame, regime: pd.DataFrame, *,
                      allowed: set, K: int, M_exit: int, M_pm: int,
                      use_gate: bool, name: str) -> dict | None:
    log.info(">>> %s  (univ=%d, K=%d, M_exit=%d, M_pm=%d, gate=%s)",
             name, len(allowed), K, M_exit, M_pm, use_gate)
    pnl_pre = daily_portfolio_pm_hysteresis(
        preds, "pred", "fwd_resid_1d", allowed, K, M_exit, COST_BPS_SIDE, M=M_pm)
    if pnl_pre.empty:
        log.warning("  empty pnl"); return None
    pnl = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW) \
        if use_gate else pnl_pre.copy()
    if pnl.empty:
        log.warning("  empty after gate"); return None
    m = metrics_for(pnl, 1)
    lo, hi = boot_ci(pnl, 1)
    k_long = pnl["n_long"].mean()
    k_short = pnl["n_short"].mean()
    log.info("  n=%d active_Sh=%+.2f [%+.2f,%+.2f] uncond=%+.2f net=%+.2f bps/cyc"
             "  K_avg L=%.2f S=%.2f turn=%.1f%%",
             m["n_rebal"], m["active_sharpe"], lo, hi, m["uncond_sharpe"],
             m["net_bps_per_rebal"], k_long, k_short, m["avg_turnover_pct"])
    return {"name": name, "metrics": m, "ci": (lo, hi), "pnl": pnl,
            "k_long": k_long, "k_short": k_short,
            "per_year": per_year_table(pnl)}


def main() -> None:
    log.info("loading cached predictions: %s", PRED_CACHE)
    preds = pd.read_parquet(PRED_CACHE)
    log.info("  preds: %d rows, %d symbols, ts %s..%s",
             len(preds), preds["symbol"].nunique(),
             preds["ts"].min().date(), preds["ts"].max().date())
    regime = load_or_compute_regime()

    allowed = set(TIER_AB)  # 11-name C5L deployable spec
    K = 4
    M_exit = 1

    out = []
    # Reference: baseline hysteresis no PM (exact reproduction of C5L).
    out.append(evaluate_variant(preds, regime, allowed=allowed, K=K, M_exit=M_exit,
                                  M_pm=1, use_gate=True,
                                  name="C5L baseline (gate, no PM)"))
    # PM_M2 + gate (analog of crypto conv+PM stacked)
    out.append(evaluate_variant(preds, regime, allowed=allowed, K=K, M_exit=M_exit,
                                  M_pm=2, use_gate=True,
                                  name="C5L + PM_M2 (gate)"))
    # PM_M3 + gate (stricter, 2 prior bars)
    out.append(evaluate_variant(preds, regime, allowed=allowed, K=K, M_exit=M_exit,
                                  M_pm=3, use_gate=True,
                                  name="C5L + PM_M3 (gate)"))
    # PM_M2 only (no gate)
    out.append(evaluate_variant(preds, regime, allowed=allowed, K=K, M_exit=M_exit,
                                  M_pm=2, use_gate=False,
                                  name="C5L + PM_M2 (no gate)"))
    # Baseline no gate (control)
    out.append(evaluate_variant(preds, regime, allowed=allowed, K=K, M_exit=M_exit,
                                  M_pm=1, use_gate=False,
                                  name="C5L baseline (no gate, no PM)"))
    # Sanity check: original C5L spec via daily_portfolio_hysteresis.
    log.info(">>> sanity: original daily_portfolio_hysteresis (gate, K=4, M=1)")
    sanity_pre = daily_portfolio_hysteresis(
        preds, "pred", "fwd_resid_1d", allowed, K, M_exit, COST_BPS_SIDE)
    sanity = gate_rolling(sanity_pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    sm = metrics_for(sanity, 1)
    slo, shi = boot_ci(sanity, 1)
    log.info("  sanity: n=%d active_Sh=%+.2f [%+.2f,%+.2f] net=%+.2f bps  (should match baseline above)",
             sm["n_rebal"], sm["active_sharpe"], slo, shi, sm["net_bps_per_rebal"])

    # ---- summary ----
    log.info("\n=== SUMMARY (Tier A+B, K=4, M_exit=1, cost=%.1f bps/side) ===", COST_BPS_SIDE)
    log.info("  %-32s %5s %10s %18s %10s %8s %8s %8s",
             "config", "n", "active_Sh", "95% CI", "net/cyc", "K_L", "K_S", "turn%")
    for r in out:
        if r is None: continue
        m = r["metrics"]; lo, hi = r["ci"]
        log.info("  %-32s %5d %+8.2f [%+5.2f,%+5.2f] %+8.2fbps %5.2f %5.2f %5.1f",
                 r["name"], m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["net_bps_per_rebal"], r["k_long"], r["k_short"],
                 m["avg_turnover_pct"])

    # ---- paired comparisons against C5L baseline (gate, no PM) ----
    base = next((r for r in out if r and r["name"] == "C5L baseline (gate, no PM)"), None)
    if base is None:
        log.error("no baseline; skipping paired"); return
    log.info("\n=== PAIRED Δnet vs C5L baseline (gate, no PM) ===")
    log.info("  %-32s %12s %22s %10s",
             "variant", "n_paired", "Δnet 95% block-CI", "Δnet bps")
    for r in out:
        if r is None or r["name"] == base["name"]: continue
        d, lo_d, hi_d = paired_block_bootstrap(base["pnl"], r["pnl"])
        log.info("  %-32s %12d  [%+5.2f,%+5.2f] bps  %+8.2f",
                 r["name"], len(base["pnl"][["ts"]].merge(r["pnl"][["ts"]], on="ts")),
                 lo_d, hi_d, d)

    # ---- per-year deltas ----
    log.info("\n=== PER-YEAR active_Sharpe ===")
    yrs = sorted({y for r in out if r for (y, *_) in r["per_year"]})
    header = "year  " + "  ".join(f"{r['name'][:18]:>18}" for r in out if r)
    log.info("  %s", header)
    for y in yrs:
        cells = []
        for r in out:
            if r is None: cells.append("    nan"); continue
            row = next((x for x in r["per_year"] if x[0] == y), None)
            cells.append(f"{row[3]:+8.2f}" if row else "     —  ")
        log.info("  %4d  " + "  ".join(f"{c:>18}" for c in cells), y)


if __name__ == "__main__":
    main()
