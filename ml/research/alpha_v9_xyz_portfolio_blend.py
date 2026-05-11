"""Portfolio-level blend: run v7 and intraday as two independent strategies,
combine P&Ls additively (no signal-level interference).

Both strategies share:
  - Universe: Tier A+B (11 names)
  - Position rule: top-K=4 long / bot-K=4 short, hysteresis M=1
  - Dispersion gate: 60th-pctile of trailing 252d xs dispersion
  - Cost: 0.8 bps/side
  - P&L target: fwd_resid_1d (close[T] → close[T+1] residual)

Differ only by signal:
  - v7        → lgbm_pred from cached walk-forward
  - intraday  → first_vs_last_xs (cross-sectional residual)

Combined P&L: blend = w_v7 * v7_net_alpha + w_id * intraday_net_alpha.
Each strategy maintains independent positions and pays its own cost.
This is the conservative (no order-netting) accounting.

Tests:
  1. Weight sweep on full 2024-2026 OOS panel (descriptive)
  2. 2-fold OOS: optimize weight on first half, evaluate on second half
  3. Paired block-bootstrap CI on Δnet vs v7-only
  4. Discipline gates: ΔSh > +0.20, Δnet CI > 0, both halves positive

Usage:
    python -m ml.research.alpha_v9_xyz_portfolio_blend
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, metrics_for, boot_ci,
)
from ml.research.alpha_v9_xyz_pm import load_or_compute_regime
from ml.research.alpha_v9_xyz_intraday_ic import session_features
from ml.research.alpha_v7_tier_a import TIER_AB

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE = CACHE / "v7_tier_a_walkfwd_preds.parquet"
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def build_panel() -> pd.DataFrame:
    log.info("loading cached v7 preds ...")
    preds = pd.read_parquet(PRED_CACHE)
    preds["date"] = pd.to_datetime(preds["ts"]).dt.tz_convert(None).dt.normalize()
    preds = preds[preds["symbol"].isin(TIER_AB)].copy()

    rows = []
    for sym in TIER_AB:
        poly_path = CACHE / f"poly_{sym}_5m.parquet"
        if not poly_path.exists(): continue
        poly = pd.read_parquet(poly_path)
        feats = session_features(poly)
        feats["symbol"] = sym
        sub = preds[preds["symbol"] == sym][["date", "ts", "pred", "fwd_resid_1d"]]
        merged = feats.merge(sub, on="date", how="inner")
        rows.append(merged)
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["first_vs_last", "pred", "fwd_resid_1d"]).reset_index(drop=True)
    df["first_vs_last_xs"] = (df["first_vs_last"]
                                 - df.groupby("date")["first_vs_last"].transform("median"))
    df["lgbm_pred"] = df["pred"]
    log.info("  panel: %d rows, %d names, %s..%s",
             len(df), df["symbol"].nunique(),
             df["date"].min().date(), df["date"].max().date())
    return df


def run_strategy(panel: pd.DataFrame, signal: str, regime: pd.DataFrame
                  ) -> pd.DataFrame:
    pre = daily_portfolio_hysteresis(
        panel, signal, "fwd_resid_1d", set(TIER_AB), 4, 1, COST_BPS_SIDE)
    return gate_rolling(pre, regime, pctile=GATE_PCTILE, window_days=GATE_WINDOW)


def metrics_on_pnl(pnl: pd.DataFrame, col: str = "net_alpha") -> dict:
    if pnl.empty: return {"n": 0}
    n = len(pnl); rpy = 252.0
    s = pnl[col]
    sh = s.mean() / s.std() * np.sqrt(rpy) if s.std() > 0 else 0
    p = pnl.copy(); p["year"] = pd.to_datetime(p["ts"]).dt.year
    rebals_actual = n / max(p["year"].nunique(), 1)
    annual_mean = s.mean() * rebals_actual
    annual_std = s.std() * np.sqrt(rebals_actual) if s.std() > 0 else 1
    return {"n_rebal": n, "active_sharpe": sh,
            "uncond_sharpe": annual_mean / annual_std if annual_std > 0 else 0,
            "annual_return_pct": annual_mean * 100,
            "net_bps_per_rebal": s.mean() * 1e4}


def boot_ci_on(pnl: pd.DataFrame, col: str = "net_alpha",
                 n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < 30: return np.nan, np.nan
    n = len(pnl); rpy = 252.0
    rng = np.random.default_rng(42)
    arr = pnl[col].to_numpy()
    sh = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = arr[idx]
        if s.std() > 0: sh.append(s.mean() / s.std() * np.sqrt(rpy))
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


def paired_bootstrap_diff(pnl_a: pd.DataFrame, pnl_b: pd.DataFrame,
                            col_a: str = "net_alpha", col_b: str = "net_alpha",
                            block_size: int = 5, n_boot: int = 2000
                            ) -> tuple[float, float, float]:
    merged = pnl_a[["ts", col_a]].rename(columns={col_a: "a"}).merge(
        pnl_b[["ts", col_b]].rename(columns={col_b: "b"}), on="ts")
    if len(merged) < 30: return np.nan, np.nan, np.nan
    diff = (merged["b"] - merged["a"]).to_numpy() * 1e4
    n = len(diff); rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(n / block_size))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        means.append(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    panel = build_panel()
    regime = load_or_compute_regime()

    log.info("running v7 strategy on Tier A+B (signal=lgbm_pred) ...")
    v7_pnl = run_strategy(panel, "lgbm_pred", regime)
    log.info("running intraday strategy (signal=first_vs_last_xs) ...")
    id_pnl = run_strategy(panel, "first_vs_last_xs", regime)
    log.info("  v7  n=%d  intraday n=%d", len(v7_pnl), len(id_pnl))

    # Both strategies' P&Ls on shared timestamps
    merged = (v7_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "v7_net"})
              .merge(id_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "id_net"}),
                      on="ts", how="inner"))
    log.info("  shared ts (both strategies traded): %d", len(merged))

    # Standalone Sharpe on shared panel
    log.info("\n=== Standalone Sharpe on shared (gated) panel ===")
    for label, col in [("v7 alone", "v7_net"), ("intraday alone", "id_net")]:
        m = metrics_on_pnl(merged, col)
        lo, hi = boot_ci_on(merged, col)
        log.info("  %-20s n=%d Sh=%+.2f [%+.2f,%+.2f] net=%+.2f bps",
                 label, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["net_bps_per_rebal"])

    # Correlation
    rho = merged[["v7_net", "id_net"]].corr().iloc[0, 1]
    log.info("\n  corr(v7_net, intraday_net) on shared ts = %+.3f", rho)

    # ---- weight sweep ----
    log.info("\n=== Weight sweep: blend = w_v7 × v7_net + (1−w_v7) × intraday_net ===")
    log.info("  %-12s %5s %10s %18s %12s %14s %14s",
             "weight", "n", "Sh", "95% CI", "net bps/cyc", "Δnet vs v7",
             "Δnet 95% CI")
    sweep_results = {}
    for w_v7 in [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]:
        merged["blend"] = w_v7 * merged["v7_net"] + (1 - w_v7) * merged["id_net"]
        m = metrics_on_pnl(merged, "blend")
        lo, hi = boot_ci_on(merged, "blend")
        # Paired diff vs v7-only (w_v7=1.0 is baseline)
        if w_v7 == 1.00:
            d_str = "      —      "; ci_str = "        —        "
        else:
            d, lo_d, hi_d = paired_bootstrap_diff(
                merged.assign(net_alpha=merged["v7_net"]),
                merged.assign(net_alpha=merged["blend"]),
                col_a="net_alpha", col_b="net_alpha")
            d_str = f"{d:+10.2f}"
            ci_str = f"[{lo_d:+5.2f},{hi_d:+5.2f}]"
        log.info("  w_v7=%.2f   %5d %+8.2f [%+5.2f,%+5.2f] %+10.2f  %s  %s",
                 w_v7, m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["net_bps_per_rebal"], d_str, ci_str)
        sweep_results[w_v7] = {"sh": m["active_sharpe"], "net": m["net_bps_per_rebal"]}

    # ---- 2-fold honest OOS ----
    log.info("\n=== 2-fold honest OOS (train weight on H1, evaluate on H2) ===")
    merged_sorted = merged.sort_values("ts").reset_index(drop=True)
    half = len(merged_sorted) // 2
    h1 = merged_sorted.iloc[:half]
    h2 = merged_sorted.iloc[half:]
    log.info("  H1: n=%d  ts %s..%s", len(h1),
             h1["ts"].min().date(), h1["ts"].max().date())
    log.info("  H2: n=%d  ts %s..%s", len(h2),
             h2["ts"].min().date(), h2["ts"].max().date())
    # Find argmax weight on H1
    best_w = None; best_sh = -np.inf
    for w_v7 in np.arange(0.0, 1.01, 0.05):
        b = w_v7 * h1["v7_net"] + (1 - w_v7) * h1["id_net"]
        sh = b.mean() / b.std() * np.sqrt(252) if b.std() > 0 else -np.inf
        if sh > best_sh: best_sh = sh; best_w = w_v7
    log.info("  H1 argmax: w_v7=%.2f → H1 Sh=%+.2f", best_w, best_sh)

    # Evaluate that weight on H2
    h2_blend = best_w * h2["v7_net"] + (1 - best_w) * h2["id_net"]
    h2_v7 = h2["v7_net"]
    sh_h2_blend = h2_blend.mean() / h2_blend.std() * np.sqrt(252) if h2_blend.std() > 0 else 0
    sh_h2_v7 = h2_v7.mean() / h2_v7.std() * np.sqrt(252) if h2_v7.std() > 0 else 0
    diff_h2 = h2_blend.mean() - h2_v7.mean()
    log.info("  H2 evaluation:")
    log.info("    H2 v7 alone:        Sh=%+.2f  net=%+.2f bps", sh_h2_v7, h2_v7.mean() * 1e4)
    log.info("    H2 blend (w_v7=%.2f): Sh=%+.2f  net=%+.2f bps  ΔSh=%+.2f  Δnet=%+.2f bps",
             best_w, sh_h2_blend, h2_blend.mean() * 1e4,
             sh_h2_blend - sh_h2_v7, diff_h2 * 1e4)

    # Symmetric: train on H2, evaluate on H1
    best_w2 = None; best_sh2 = -np.inf
    for w_v7 in np.arange(0.0, 1.01, 0.05):
        b = w_v7 * h2["v7_net"] + (1 - w_v7) * h2["id_net"]
        sh = b.mean() / b.std() * np.sqrt(252) if b.std() > 0 else -np.inf
        if sh > best_sh2: best_sh2 = sh; best_w2 = w_v7
    log.info("  H2 argmax: w_v7=%.2f → H2 Sh=%+.2f", best_w2, best_sh2)
    h1_blend = best_w2 * h1["v7_net"] + (1 - best_w2) * h1["id_net"]
    h1_v7 = h1["v7_net"]
    sh_h1_blend = h1_blend.mean() / h1_blend.std() * np.sqrt(252) if h1_blend.std() > 0 else 0
    sh_h1_v7 = h1_v7.mean() / h1_v7.std() * np.sqrt(252) if h1_v7.std() > 0 else 0
    log.info("  H1 evaluation:")
    log.info("    H1 v7 alone:        Sh=%+.2f", sh_h1_v7)
    log.info("    H1 blend (w_v7=%.2f): Sh=%+.2f  ΔSh=%+.2f",
             best_w2, sh_h1_blend, sh_h1_blend - sh_h1_v7)

    # ---- per-half breakdown for fixed canonical weights ----
    log.info("\n=== Per-half ΔSh vs v7-only at fixed weights ===")
    log.info("  %-12s %12s %12s",
             "weight", "H1 ΔSh", "H2 ΔSh")
    for w_v7 in [0.80, 0.70, 0.60, 0.50, 0.40, 0.30]:
        h1_blend = w_v7 * h1["v7_net"] + (1 - w_v7) * h1["id_net"]
        h2_blend = w_v7 * h2["v7_net"] + (1 - w_v7) * h2["id_net"]
        sh_h1_b = h1_blend.mean() / h1_blend.std() * np.sqrt(252) if h1_blend.std() > 0 else 0
        sh_h2_b = h2_blend.mean() / h2_blend.std() * np.sqrt(252) if h2_blend.std() > 0 else 0
        sh_h1_v7 = h1["v7_net"].mean() / h1["v7_net"].std() * np.sqrt(252)
        sh_h2_v7 = h2["v7_net"].mean() / h2["v7_net"].std() * np.sqrt(252)
        log.info("  w_v7=%.2f   %+10.2f   %+10.2f",
                 w_v7, sh_h1_b - sh_h1_v7, sh_h2_b - sh_h2_v7)


if __name__ == "__main__":
    main()
