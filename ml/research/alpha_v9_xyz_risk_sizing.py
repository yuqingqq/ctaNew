"""Phase 2: backtest pred_disp_TAB as a risk-sizing indicator on v7's full
2016-2026 history. 7-fold expanding walk-forward.

Per fold:
  - Train: tune threshold on all v7 cycles before fold's test year
  - Test:  apply the weight schedule to the test year's cycles
  - Save weighted P&L

Three weight schedules tested:
  binary_50: weight = 1.0 if indicator >= threshold, else 0.5
  binary_30: weight = 1.0 if indicator >= threshold, else 0.3
  kill:      weight = 1.0 if indicator >= threshold, else 0.0  (skip cycle)

Threshold optimization: search over training-set quantiles
{10, 20, 30, 40, 50} of pred_disp_TAB_sm. Pick the threshold that
maximizes training-period Sharpe of the weighted stream.

Discipline gates:
  G1: aggregated ΔSh > +0.20
  G2: paired Δnet block-bootstrap CI > 0
  G3: ≥ 4/7 test folds with positive ΔSh
  G4: single-event drop — remove 2021-2022 drawdown, re-aggregate;
      lift must retain ≥ 50%
  G5: drawdown reduction — max-DD on weighted < max-DD on baseline

Usage:
    python -m ml.research.alpha_v9_xyz_risk_sizing
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
TOP_K = 4
M_EXIT = 1


def build_panel() -> pd.DataFrame:
    log.info("loading v7 cached preds and computing v7 P&L + indicator ...")
    preds = pd.read_parquet(PRED_CACHE_V7)
    preds["ts"] = pd.to_datetime(preds["ts"], utc=True)
    regime = load_or_compute_regime()
    pre = daily_portfolio_hysteresis(preds, "pred", "fwd_resid_1d",
                                       set(TIER_AB), TOP_K, M_EXIT, COST_BPS_SIDE)
    v7_pnl = gate_rolling(pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)
    v7_pnl["ts"] = pd.to_datetime(v7_pnl["ts"], utc=True)

    # Compute pred_disp_TAB per ts
    preds_TAB = preds[preds["symbol"].isin(TIER_AB)]
    ind = (preds_TAB.groupby("ts")["pred"].std()
              .rename("pred_disp_TAB").reset_index())
    ind["pred_disp_TAB_sm"] = (ind["pred_disp_TAB"]
                                  .rolling(22, min_periods=10).mean().shift(1))

    panel = v7_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "v7_net"}).merge(
        ind[["ts", "pred_disp_TAB_sm"]], on="ts", how="left")
    panel = panel.sort_values("ts").reset_index(drop=True)
    panel["year"] = panel["ts"].dt.year
    panel = panel.dropna(subset=["pred_disp_TAB_sm"]).reset_index(drop=True)
    log.info("  panel: %d cycles  %s..%s",
             len(panel), panel["ts"].min().date(), panel["ts"].max().date())
    return panel


def apply_weights(panel: pd.DataFrame, mode: str, threshold: float
                    ) -> pd.DataFrame:
    df = panel.copy()
    fired = df["pred_disp_TAB_sm"] < threshold
    if mode == "binary_50":
        df["weight"] = np.where(fired, 0.5, 1.0)
    elif mode == "binary_30":
        df["weight"] = np.where(fired, 0.3, 1.0)
    elif mode == "kill":
        df["weight"] = np.where(fired, 0.0, 1.0)
    else:
        raise ValueError(mode)
    df["weighted_net"] = df["weight"] * df["v7_net"]
    df["fired"] = fired
    return df


def sharpe(s: pd.Series, rpy: float = 252.0) -> float:
    if len(s) < 2 or s.std() == 0: return 0.0
    return s.mean() / s.std() * np.sqrt(rpy)


def find_best_threshold(train: pd.DataFrame, mode: str) -> tuple[float, float]:
    qs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    best_sh = -np.inf; best_thr = None
    base_sh = sharpe(train["v7_net"])
    for q in qs:
        thr = train["pred_disp_TAB_sm"].quantile(q)
        w = apply_weights(train, mode, thr)
        sh = sharpe(w["weighted_net"])
        if sh > best_sh:
            best_sh = sh; best_thr = thr
    return best_thr, best_sh - base_sh


def max_drawdown(s: pd.Series) -> float:
    cum = s.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())


def paired_block_bootstrap(diff: np.ndarray, block_size: int = 5,
                              n_boot: int = 2000) -> tuple[float, float, float]:
    n = len(diff)
    if n < 30: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(n / block_size))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        means.append(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    panel = build_panel()

    # 7-fold expanding-train walk-forward
    folds = []
    for test_year in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        train_end = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
        test_start = train_end
        # Test runs through end of test year (2025 includes 2026 partial)
        if test_year == 2025:
            test_end = pd.Timestamp("2026-12-31", tz="UTC")
        else:
            test_end = pd.Timestamp(f"{test_year + 1}-01-01", tz="UTC")
        folds.append({
            "name": f"fold_{test_year}",
            "train_start": panel["ts"].min(),
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

    modes = ["binary_50", "binary_30", "kill"]
    fold_results = {m: [] for m in modes}

    log.info("\n=== 7-fold expanding-train OOS ===")
    log.info("  %-10s %-22s %5s %9s %5s",
             "fold", "mode", "thr×1e3", "n_test", "fire%")
    for fold in folds:
        train = panel[(panel["ts"] >= fold["train_start"]) &
                       (panel["ts"] < fold["train_end"])]
        test = panel[(panel["ts"] >= fold["test_start"]) &
                      (panel["ts"] < fold["test_end"])]
        if len(train) < 50 or len(test) < 5:
            log.warning("  %s: train=%d test=%d skip", fold["name"], len(train), len(test))
            continue
        for mode in modes:
            thr, train_dsh = find_best_threshold(train, mode)
            test_w = apply_weights(test, mode, thr)
            base_sh = sharpe(test["v7_net"])
            test_sh = sharpe(test_w["weighted_net"])
            fire_pct = test_w["fired"].mean() * 100
            log.info("  %-10s %-22s %+9.4f %5d %5.1f%%",
                     fold["name"], mode, thr * 1e3, len(test), fire_pct)
            fold_results[mode].append({
                "fold": fold["name"],
                "thr": thr,
                "n_test": len(test),
                "fire_pct": fire_pct,
                "base_sh": base_sh,
                "weighted_sh": test_sh,
                "delta_sh": test_sh - base_sh,
                "test_pnl": test_w[["ts", "v7_net", "weighted_net", "fired"]].copy(),
            })

    # ---- aggregate per-fold results ----
    log.info("\n=== Per-fold ΔSharpe ===")
    log.info("  %-10s %14s %14s %14s",
             "fold", "binary_50 ΔSh", "binary_30 ΔSh", "kill ΔSh")
    fold_names = [r["fold"] for r in fold_results["binary_50"]]
    for fname in fold_names:
        line = f"  {fname:<10}"
        for mode in modes:
            r = next((x for x in fold_results[mode] if x["fold"] == fname), None)
            if r is None: line += f"  {'---':>14}"
            else: line += f"  {r['delta_sh']:+14.2f}"
        log.info(line)

    # ---- aggregated test panel + paired CI ----
    log.info("\n=== Aggregated test panel: weighted vs baseline ===")
    log.info("  %-22s %5s %10s %10s %12s %18s %10s",
             "mode", "n", "base Sh", "weighted Sh", "ΔSh", "Δnet 95% CI", "max-DD")
    for mode in modes:
        rs = fold_results[mode]
        if not rs: continue
        all_pnl = pd.concat([r["test_pnl"] for r in rs], ignore_index=True)
        n = len(all_pnl)
        base_sh = sharpe(all_pnl["v7_net"])
        w_sh = sharpe(all_pnl["weighted_net"])
        dsh = w_sh - base_sh
        diff = (all_pnl["weighted_net"] - all_pnl["v7_net"]).to_numpy() * 1e4
        d_mean, d_lo, d_hi = paired_block_bootstrap(diff)
        max_dd_base = max_drawdown(all_pnl["v7_net"]) * 1e4
        max_dd_w = max_drawdown(all_pnl["weighted_net"]) * 1e4
        log.info("  %-22s %5d %+10.2f %+10.2f %+12.2f  [%+5.2f, %+5.2f] bps  %+8.0f→%+.0f",
                 mode, n, base_sh, w_sh, dsh, d_lo, d_hi, max_dd_base, max_dd_w)

    # ---- Gate 4: drop 2021-2022 drawdown ----
    log.info("\n=== Gate 4: drop 2021-2022 drawdown (single-event sensitivity) ===")
    log.info("  %-22s %14s %14s %10s",
             "mode", "ΔSh full", "ΔSh w/o 2021-22", "retained")
    for mode in modes:
        rs = fold_results[mode]
        if not rs: continue
        all_pnl = pd.concat([r["test_pnl"] for r in rs], ignore_index=True)
        # Drop 2021 and 2022 entirely
        no_2122 = all_pnl[~all_pnl["ts"].dt.year.isin([2021, 2022])]
        dsh_full = sharpe(all_pnl["weighted_net"]) - sharpe(all_pnl["v7_net"])
        dsh_drop = sharpe(no_2122["weighted_net"]) - sharpe(no_2122["v7_net"])
        retained = (dsh_drop / dsh_full) if abs(dsh_full) > 0.01 else 0
        log.info("  %-22s %+14.2f %+14.2f %10.2f", mode, dsh_full, dsh_drop, retained)

    # ---- per-year breakdown ----
    log.info("\n=== Per-year ΔSh (baseline vs binary_50) ===")
    rs = fold_results["binary_50"]
    if rs:
        all_pnl = pd.concat([r["test_pnl"] for r in rs], ignore_index=True)
        all_pnl["year"] = all_pnl["ts"].dt.year
        log.info("  %-6s %5s %12s %12s %12s",
                 "year", "n", "base Sh", "weighted Sh", "ΔSh")
        for y, g in all_pnl.groupby("year"):
            if len(g) < 5: continue
            log.info("  %-6d %5d %+12.2f %+12.2f %+12.2f",
                     y, len(g), sharpe(g["v7_net"]), sharpe(g["weighted_net"]),
                     sharpe(g["weighted_net"]) - sharpe(g["v7_net"]))

    # ---- discipline-gate verdict ----
    log.info("\n=== Discipline-gate verdict ===")
    log.info("  %-22s %10s %10s %10s %10s %18s %s",
             "mode", "ΔSh agg", "folds +", "G2 CI", "G4 retain", "G5 max-DD", "verdict")
    for mode in modes:
        rs = fold_results[mode]
        if not rs: continue
        all_pnl = pd.concat([r["test_pnl"] for r in rs], ignore_index=True)
        no_2122 = all_pnl[~all_pnl["ts"].dt.year.isin([2021, 2022])]
        dsh_full = sharpe(all_pnl["weighted_net"]) - sharpe(all_pnl["v7_net"])
        dsh_drop = sharpe(no_2122["weighted_net"]) - sharpe(no_2122["v7_net"])
        retained = (dsh_drop / dsh_full) if abs(dsh_full) > 0.01 else 0
        n_pos = sum(1 for r in rs if r["delta_sh"] > 0)
        diff = (all_pnl["weighted_net"] - all_pnl["v7_net"]).to_numpy() * 1e4
        _, lo, _ = paired_block_bootstrap(diff)
        max_dd_base = abs(max_drawdown(all_pnl["v7_net"]))
        max_dd_w = abs(max_drawdown(all_pnl["weighted_net"]))
        dd_reduced = max_dd_w < max_dd_base
        passes = (dsh_full > 0.20 and n_pos >= 4 and lo > 0
                   and retained >= 0.50 and dd_reduced)
        verdict = "PASS ✓" if passes else "fail"
        log.info("  %-22s %+10.2f %4d/%d %+10.2f %10.2f %10s     %s",
                 mode, dsh_full, n_pos, len(rs), lo, retained,
                 "yes" if dd_reduced else "no", verdict)


if __name__ == "__main__":
    main()
