"""Leg-asymmetry probe for v7 xyz: long-only and short-only basket-hedged P&L.

The model is trained on basket-residualized targets (`fwd_resid_1d`), so
"hedge with the model's training basket" is mechanically equivalent to taking
P&L on `fwd_resid_1d` directly (no basis vs the training target). This probe
evaluates:
  - L/S  (current C5L baseline)
  - long-only basket-hedged   = mean(fwd_resid_1d for K longs) − long-leg cost
  - short-only basket-hedged  = −mean(fwd_resid_1d for K shorts) − short-leg cost

Hysteresis construction is identical to baseline (K=4, M_exit=1, dispersion
gate 60th-pctile). Per-leg turnover is tracked so each variant pays only its
own leg's cost (the index hedge has near-zero turnover and is treated as
free; this is mildly optimistic for long/short-only).

If long-only Sharpe ≳ L/S, asymmetry is real and an index-hedged single-leg
book is competitive. If long-only ≈ L/S/√2, the strategy is symmetric and
swapping is a Sharpe downgrade.

Usage:
    python -m ml.research.alpha_v9_xyz_legs
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import metrics_for, boot_ci
from ml.research.alpha_v7_tier_a import TIER_AB
from ml.research.alpha_v9_xyz_pm import load_or_compute_regime, paired_block_bootstrap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE = CACHE / "v7_tier_a_walkfwd_preds.parquet"

GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def daily_portfolio_three_legs(test_pred: pd.DataFrame, signal: str,
                                pnl_label: str, allowed: set, top_k: int,
                                exit_buffer: int, cost_bps_side: float
                                ) -> pd.DataFrame:
    """Same hysteresis as baseline but exports per-leg turnover and three
    net-P&L variants (LS, long-only basket-hedged, short-only basket-hedged).

    Per-name notional = 1/K. cost_bps_side is per transaction (one side of a
    round trip). Long-only / short-only treat the index hedge as free (its
    turnover ≪ name leg turnover).
    """
    sub = test_pred[test_pred["symbol"].isin(allowed)].dropna(
        subset=[signal, pnl_label]).copy()
    rows = []
    cur_long, cur_short = set(), set()
    enter_rank = top_k
    exit_rank = top_k + exit_buffer

    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k + exit_buffer:
            continue
        bar = bar.sort_values(signal).reset_index(drop=True)
        n = len(bar)
        bar["rank_top"] = n - 1 - bar.index
        bar["rank_bot"] = bar.index

        new_long = set(cur_long)
        for s in list(new_long):
            r = bar[bar["symbol"] == s]
            if r.empty:
                new_long.discard(s); continue
            if r["rank_top"].iloc[0] > exit_rank - 1:
                new_long.discard(s)
        cands = bar[bar["rank_top"] < enter_rank]["symbol"].tolist()
        for s in cands:
            if len(new_long) >= top_k: break
            if s in new_long: continue
            new_long.add(s)
        if len(new_long) > top_k:
            ranked = bar[bar["symbol"].isin(new_long)].sort_values("rank_top")
            new_long = set(ranked.head(top_k)["symbol"])

        new_short = set(cur_short)
        for s in list(new_short):
            r = bar[bar["symbol"] == s]
            if r.empty:
                new_short.discard(s); continue
            if r["rank_bot"].iloc[0] > exit_rank - 1:
                new_short.discard(s)
        cs = bar[bar["rank_bot"] < enter_rank]["symbol"].tolist()
        for s in cs:
            if len(new_short) >= top_k: break
            if s in new_short: continue
            new_short.add(s)
        if len(new_short) > top_k:
            ranked = bar[bar["symbol"].isin(new_short)].sort_values("rank_bot")
            new_short = set(ranked.head(top_k)["symbol"])

        long_chg = len(new_long.symmetric_difference(cur_long))
        short_chg = len(new_short.symmetric_difference(cur_short))
        long_turn = long_chg / top_k
        short_turn = short_chg / top_k
        # Per-leg cost as fraction of leg notional (1.0 per leg).
        cost_long = long_turn * cost_bps_side / 1e4
        cost_short = short_turn * cost_bps_side / 1e4
        # Existing baseline cost (sum of both leg per-side fees).
        cost_ls = (long_turn + short_turn) * cost_bps_side / 1e4

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            continue
        long_a = bar[bar["symbol"].isin(new_long)][pnl_label].mean()
        short_a = bar[bar["symbol"].isin(new_short)][pnl_label].mean()
        spread = long_a - short_a
        rows.append({
            "ts": ts,
            "long_alpha": long_a, "short_alpha": short_a, "spread_alpha": spread,
            "long_turn": long_turn, "short_turn": short_turn,
            "cost": cost_ls, "cost_long": cost_long, "cost_short": cost_short,
            "net_alpha": spread - cost_ls,                # L/S
            "net_long_only": long_a - cost_long,          # long basket-hedged
            "net_short_only": -short_a - cost_short,      # short basket-hedged
            "turnover": (long_turn + short_turn) / 2,     # avg fraction per leg (matches existing)
            "n_long": len(new_long), "n_short": len(new_short),
        })
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(rows)


def metrics_on(pnl: pd.DataFrame, net_col: str) -> dict:
    if pnl.empty: return {"n": 0}
    n = len(pnl)
    rebals_per_year = 252.0
    p = pnl.copy()
    p["year"] = pd.to_datetime(p["ts"]).dt.year
    rebals_actual = n / max(p["year"].nunique(), 1)
    s = p[net_col]
    sh = s.mean() / s.std() * np.sqrt(rebals_per_year) if s.std() > 0 else 0
    annual_mean = s.mean() * rebals_actual
    annual_std = s.std() * np.sqrt(rebals_actual)
    uncond = annual_mean / annual_std if annual_std > 0 else 0
    return {"n_rebal": n, "active_sharpe": sh, "uncond_sharpe": uncond,
            "annual_return_pct": annual_mean * 100,
            "net_bps_per_rebal": s.mean() * 1e4}


def boot_ci_on(pnl: pd.DataFrame, net_col: str, n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < 30: return np.nan, np.nan
    n = len(pnl); rpy = 252.0
    rng = np.random.default_rng(42)
    sh = []
    arr = pnl[net_col].to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = arr[idx]
        if s.std() > 0:
            sh.append(s.mean() / s.std() * np.sqrt(rpy))
    if not sh: return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


def per_year(pnl: pd.DataFrame, net_col: str) -> list[tuple]:
    out = []
    p = pnl.copy()
    p["year"] = pd.to_datetime(p["ts"]).dt.year
    for y, g in p.groupby("year"):
        if len(g) < 5: continue
        m = metrics_on(g, net_col)
        out.append((y, m["n_rebal"], m["net_bps_per_rebal"], m["active_sharpe"]))
    return out


def main() -> None:
    log.info("loading cached predictions: %s", PRED_CACHE)
    preds = pd.read_parquet(PRED_CACHE)
    regime = load_or_compute_regime()

    allowed = set(TIER_AB)
    K = 4; M_exit = 1
    log.info("running daily_portfolio_three_legs (Tier A+B, K=%d, M=%d, cost=%.1f bps/side)...",
             K, M_exit, COST_BPS_SIDE)
    pnl_pre = daily_portfolio_three_legs(preds, "pred", "fwd_resid_1d",
                                            allowed, K, M_exit, COST_BPS_SIDE)
    pnl_gated = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE,
                                window_days=GATE_WINDOW)
    log.info("  pre-gate n=%d  post-gate n=%d", len(pnl_pre), len(pnl_gated))

    # ---- summary across three legs (gated) ----
    log.info("\n=== Gated (60th-pctile dispersion), Tier A+B K=4 M=1 ===")
    log.info("  %-26s %5s %10s %18s %10s %12s",
             "variant", "n", "active_Sh", "95% CI", "uncond", "net bps/cyc")
    for label, col in [("L/S baseline (C5L)", "net_alpha"),
                        ("Long-only basket-hedged", "net_long_only"),
                        ("Short-only basket-hedged", "net_short_only")]:
        m = metrics_on(pnl_gated, col)
        lo, hi = boot_ci_on(pnl_gated, col)
        log.info("  %-26s %5d %+8.2f [%+5.2f,%+5.2f] %+8.2f  %+10.2f",
                 label, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["net_bps_per_rebal"])

    # ---- summary no gate (full panel) ----
    log.info("\n=== Ungated ===")
    log.info("  %-26s %5s %10s %18s %10s %12s",
             "variant", "n", "active_Sh", "95% CI", "uncond", "net bps/cyc")
    for label, col in [("L/S baseline", "net_alpha"),
                        ("Long-only basket-hedged", "net_long_only"),
                        ("Short-only basket-hedged", "net_short_only")]:
        m = metrics_on(pnl_pre, col)
        lo, hi = boot_ci_on(pnl_pre, col)
        log.info("  %-26s %5d %+8.2f [%+5.2f,%+5.2f] %+8.2f  %+10.2f",
                 label, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["uncond_sharpe"], m["net_bps_per_rebal"])

    # ---- paired Δnet vs L/S baseline (gated) ----
    log.info("\n=== Paired Δnet bps vs L/S (gated, block-bootstrap) ===")
    base = pnl_gated[["ts", "net_alpha"]].rename(columns={"net_alpha": "x"})
    for label, col in [("Long-only basket-hedged", "net_long_only"),
                        ("Short-only basket-hedged", "net_short_only")]:
        cmp = pnl_gated[["ts", col]].rename(columns={col: "y"})
        merged = base.merge(cmp, on="ts")
        diff = (merged["y"] - merged["x"]).to_numpy() * 1e4
        n = len(diff); rng = np.random.default_rng(42); blocks = int(np.ceil(n / 10))
        means = []
        for _ in range(2000):
            starts = rng.integers(0, n - 10 + 1, size=blocks)
            idx = np.concatenate([np.arange(s, s + 10) for s in starts])[:n]
            means.append(diff[idx].mean())
        log.info("  %-26s  Δnet=%+7.2f bps  CI [%+5.2f, %+5.2f]",
                 label, diff.mean(),
                 np.percentile(means, 2.5), np.percentile(means, 97.5))

    # ---- per-year ----
    log.info("\n=== Per-year active Sharpe ===")
    log.info("  %-6s %10s %22s %22s",
             "year", "L/S", "Long-only basket-hedged", "Short-only basket-hedged")
    yrs = sorted(per_year(pnl_gated, "net_alpha"))
    p = pnl_gated.copy()
    p["year"] = pd.to_datetime(p["ts"]).dt.year
    for y in sorted(p["year"].unique()):
        g = p[p["year"] == y]
        if len(g) < 5: continue
        sh_ls = metrics_on(g, "net_alpha")["active_sharpe"]
        sh_lo = metrics_on(g, "net_long_only")["active_sharpe"]
        sh_so = metrics_on(g, "net_short_only")["active_sharpe"]
        log.info("  %-6d %+10.2f %+22.2f %+22.2f", y, sh_ls, sh_lo, sh_so)

    # ---- correlation between long & short alphas ----
    rho = pnl_gated[["long_alpha", "short_alpha"]].corr().iloc[0, 1]
    log.info("\n  corr(long_alpha, short_alpha) on gated panel = %+.3f", rho)
    log.info("  → if rho << 0, legs are anti-correlated and L/S diversifies maximally")
    log.info("  → if rho ~ 0, legs uncorrelated; L/S Sharpe ≈ √2 × single-leg")
    log.info("  → if rho > 0, legs co-move; single-leg can match or beat L/S")


if __name__ == "__main__":
    main()
