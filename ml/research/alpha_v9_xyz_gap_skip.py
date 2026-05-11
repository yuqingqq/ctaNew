"""Rigorous validation of the gap-skip overlay on v7 Tier A+B.

The overlay: skip (or size-down) the first cycle after a long gap between
traded cycles. Long gaps arise when the dispersion gate has been filtering
days; the first cycle back is hypothesized to be unconditionally noisier.

Phase 1 finding: with fixed threshold gap≥30d on full 10y, ΔSh +0.39 with
8/11 years positive but Δnet CI [−2.52, +5.39] crosses zero.

This script tests honestly:
  1. Walk-forward threshold tuning (per-fold; no full-sample optimum used)
  2. Aggregate test-panel Sharpe + paired Δnet CI
  3. Per-fold ΔSh consistency
  4. Single-event drop test (remove 2021-22 drawdown)
  5. Sensitivity to threshold choice (does ±10d matter?)

Folds: 7 expanding-train, 1 test year each (2019-2025).
For each fold: optimize threshold over grid {14, 21, 30, 45, 60} days using
training-period ΔSh (skip-action) as criterion; apply to test.

Discipline gates:
  G1: aggregated ΔSh > +0.20
  G2: paired Δnet block-bootstrap CI > 0
  G3: ≥4/7 test folds with positive ΔSh
  G4: drop 2021-22 — lift retains ≥50% of full
  G5: drawdown reduction (max-DD on weighted < baseline)

Plus G6 (new): threshold stability across folds (no wild swings)

Usage:
    python -m ml.research.alpha_v9_xyz_gap_skip
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import daily_portfolio_hysteresis
from ml.research.alpha_v9_xyz_pm import load_or_compute_regime
from ml.research.alpha_v7_tier_a import TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE_V7 = CACHE / "v7_tier_a_walkfwd_preds.parquet"
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8
GAP_GRID = [14, 21, 30, 45, 60]


def sharpe(s: pd.Series, rpy: float = 252.0) -> float:
    if len(s) < 2 or s.std() == 0: return 0.0
    return s.mean() / s.std() * np.sqrt(rpy)


def max_dd(s: pd.Series) -> float:
    cum = s.cumsum(); peak = cum.cummax()
    return float((cum - peak).min())


def paired_block_bs(diff: np.ndarray, block: int = 5,
                       n_boot: int = 2000) -> tuple[float, float, float]:
    n = len(diff)
    if n < 30: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    nb = int(np.ceil(n / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means.append(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def apply_skip(df: pd.DataFrame, threshold_days: int,
                 action: str = "skip") -> pd.DataFrame:
    out = df.copy()
    weights = np.where(out["gap_days"] >= threshold_days,
                         0.0 if action == "skip" else 0.5, 1.0)
    out["weight"] = weights
    out["weighted_net"] = weights * out["v7_net"]
    out["fired"] = weights < 1.0
    return out


def find_best_threshold(train: pd.DataFrame, action: str = "skip"
                         ) -> tuple[int, float]:
    base_sh = sharpe(train["v7_net"])
    best_d = -np.inf; best_t = None
    for t in GAP_GRID:
        w = apply_skip(train, t, action)
        sh = sharpe(w["weighted_net"])
        d = sh - base_sh
        if d > best_d:
            best_d = d; best_t = t
    return best_t, best_d


def main() -> None:
    log.info("loading v7 cached preds and computing v7 P&L ...")
    preds = pd.read_parquet(PRED_CACHE_V7)
    preds = preds[preds["symbol"].isin(TIER_AB)].copy()
    regime = load_or_compute_regime()
    pre = daily_portfolio_hysteresis(preds, "pred", "fwd_resid_1d",
                                       set(TIER_AB), 4, 1, COST_BPS_SIDE)
    pnl = gate_rolling(pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    pnl["ts"] = pd.to_datetime(pnl["ts"], utc=True).dt.tz_localize(None)
    pnl = pnl.sort_values("ts").reset_index(drop=True)
    pnl["prev_ts"] = pnl["ts"].shift(1)
    pnl["gap_days"] = (pnl["ts"] - pnl["prev_ts"]).dt.days
    pnl["v7_net"] = pnl["net_alpha"]
    pnl["year"] = pnl["ts"].dt.year
    log.info("  panel: %d cycles  %s..%s",
             len(pnl), pnl["ts"].min().date(), pnl["ts"].max().date())

    # 7-fold expanding-train walk-forward
    folds = []
    for test_year in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        train_end = pd.Timestamp(f"{test_year}-01-01")
        if test_year == 2025:
            test_end = pd.Timestamp("2026-12-31")
        else:
            test_end = pd.Timestamp(f"{test_year + 1}-01-01")
        folds.append({"name": f"fold_{test_year}",
                       "train_end": train_end, "test_start": train_end,
                       "test_end": test_end})

    log.info("\n=== 7-fold walk-forward (action=skip) ===")
    log.info("  %-10s %5s %10s %5s %10s %12s %10s",
             "fold", "n_test", "best_thr", "fired", "base_Sh", "weighted_Sh", "ΔSh")
    test_pnls = []
    for fold in folds:
        train = pnl[pnl["ts"] < fold["train_end"]]
        test = pnl[(pnl["ts"] >= fold["test_start"]) & (pnl["ts"] < fold["test_end"])]
        if len(train) < 50 or len(test) < 5:
            log.warning("  %s: skip (train=%d test=%d)", fold["name"], len(train), len(test))
            continue
        thr, _ = find_best_threshold(train, "skip")
        test_w = apply_skip(test, thr, "skip")
        base_sh = sharpe(test["v7_net"])
        w_sh = sharpe(test_w["weighted_net"])
        n_fired = test_w["fired"].sum()
        log.info("  %-10s %5d %8dd %5d %+10.2f %+12.2f %+10.2f",
                 fold["name"], len(test), thr, n_fired, base_sh, w_sh, w_sh - base_sh)
        test_w["fold"] = fold["name"]
        test_w["thr"] = thr
        test_pnls.append(test_w)

    all_test = pd.concat(test_pnls, ignore_index=True)
    log.info("\n  combined OOS test panel: %d cycles", len(all_test))

    # ---- aggregate ΔSh + Δnet CI ----
    log.info("\n=== Aggregated 7-fold OOS ===")
    base_sh = sharpe(all_test["v7_net"])
    w_sh = sharpe(all_test["weighted_net"])
    dsh = w_sh - base_sh
    diff = (all_test["weighted_net"] - all_test["v7_net"]).to_numpy() * 1e4
    d, lo, hi = paired_block_bs(diff)
    base_dd = max_dd(all_test["v7_net"]) * 1e4
    w_dd = max_dd(all_test["weighted_net"]) * 1e4
    log.info("  baseline Sharpe:  %+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             base_sh, all_test["v7_net"].mean() * 1e4, base_dd)
    log.info("  weighted Sharpe:  %+.2f  net=%+.2f bps  max-DD=%+.0f bps",
             w_sh, all_test["weighted_net"].mean() * 1e4, w_dd)
    log.info("  ΔSh = %+.2f   Δnet = %+.2f bps   95%% CI = [%+.2f, %+.2f] bps",
             dsh, d, lo, hi)

    # ---- per-fold consistency ----
    n_pos = sum(1 for p in test_pnls
                if sharpe(p["weighted_net"]) - sharpe(p["v7_net"]) > 0)
    n_total = len(test_pnls)
    log.info("\n  Folds with ΔSh > 0: %d/%d", n_pos, n_total)

    # ---- threshold stability ----
    log.info("\n=== Threshold stability across folds ===")
    thrs = [p["thr"].iloc[0] for p in test_pnls]
    log.info("  selected thresholds: %s", thrs)
    log.info("  median: %d  range: %d-%d  std: %.1f",
             int(np.median(thrs)), min(thrs), max(thrs), np.std(thrs))

    # ---- single-event drop ----
    log.info("\n=== Gate 4: drop 2021-2022 drawdown ===")
    no_2122 = all_test[~all_test["year"].isin([2021, 2022])]
    sh_b = sharpe(no_2122["v7_net"])
    sh_w = sharpe(no_2122["weighted_net"])
    dsh_drop = sh_w - sh_b
    retained = (dsh_drop / dsh) if abs(dsh) > 0.01 else 0
    log.info("  ΔSh full: %+.2f   ΔSh w/o 2021-22: %+.2f   retained: %.2f",
             dsh, dsh_drop, retained)

    # ---- threshold sensitivity (±10d from median) ----
    log.info("\n=== Threshold sensitivity (full panel, fixed-threshold) ===")
    log.info("  %-12s %10s %12s %18s",
             "threshold", "ΔSh", "Δnet bps", "95% CI")
    for t in [14, 21, 30, 45, 60]:
        w = apply_skip(pnl, t, "skip")
        sh_b_t = sharpe(w["v7_net"])
        sh_w_t = sharpe(w["weighted_net"])
        dsh_t = sh_w_t - sh_b_t
        diff_t = (w["weighted_net"] - w["v7_net"]).to_numpy() * 1e4
        d_t, lo_t, hi_t = paired_block_bs(diff_t)
        log.info("  gap≥%dd       %+10.2f %+12.2f  [%+5.2f, %+5.2f]",
                 t, dsh_t, d_t, lo_t, hi_t)

    # ---- discipline-gate verdict ----
    log.info("\n=== Discipline-gate verdict (7-fold honest OOS) ===")
    g1 = dsh > 0.20
    g2 = lo > 0
    g3 = n_pos >= 4
    g4 = retained >= 0.50
    g5 = w_dd > base_dd  # max-DD reduced (less negative)
    log.info("  G1: ΔSh > +0.20             %+.2f          %s", dsh, "✓" if g1 else "✗")
    log.info("  G2: Δnet 95%% CI > 0         [%+.2f, %+.2f]  %s", lo, hi, "✓" if g2 else "✗")
    log.info("  G3: ≥4/7 folds positive      %d/7         %s", n_pos, "✓" if g3 else "✗")
    log.info("  G4: 2021-22 drop survival    %.2f retained   %s", retained, "✓" if g4 else "✗")
    log.info("  G5: max-DD reduced           %+.0f vs %+.0f   %s", w_dd, base_dd, "✓" if g5 else "✗")
    if all([g1, g2, g3, g4, g5]):
        verdict = "PASS ALL GATES ✓"
    elif sum([g1, g2, g3, g4, g5]) >= 4:
        verdict = f"BORDERLINE PASS ({sum([g1, g2, g3, g4, g5])}/5)"
    else:
        verdict = f"FAIL ({sum([g1, g2, g3, g4, g5])}/5 gates)"
    log.info("\n  VERDICT: %s", verdict)

    # ---- per-year ΔSh on OOS test panel ----
    log.info("\n=== Per-year ΔSh (OOS test panel) ===")
    log.info("  %-6s %5s %12s %12s %12s",
             "year", "n", "base Sh", "weighted Sh", "ΔSh")
    for y, g in all_test.groupby("year"):
        if len(g) < 5: continue
        sh_b_y = sharpe(g["v7_net"])
        sh_w_y = sharpe(g["weighted_net"])
        log.info("  %-6d %5d %+12.2f %+12.2f %+12.2f",
                 y, len(g), sh_b_y, sh_w_y, sh_w_y - sh_b_y)


if __name__ == "__main__":
    main()
