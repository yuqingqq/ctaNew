"""Regime-conditional v7 + rolling blend probe.

Hypothesis: blending the rolling-24mo model into v7 helps during v7's
regime transitions (e.g., 2024Q2-Q3 PEAD-flip) and hurts during stable
strong-signal periods. A regime indicator that fires precisely during
transitions could capture the lift without paying the static-blend drag.

Five indicators × 1 hard-switch rule × honest 2-fold OOS:
  R1: v7 trailing-30d Sharpe                  (reactive)
  R2: v7 pred dispersion (xs std, 22d smooth) (forward-looking)
  R3: v7–rolling pred Spearman corr (22d smooth)
  R4: realized xs dispersion (inverted)
  R5: v7 running drawdown from peak           (reactive)

Hard-switch rule: when indicator < threshold, blend at w_v7=0.50; else 1.00.

Discipline gates:
  G1: aggregated ΔSh > +0.20 (over both H1-trained-eval-H2 and vice versa)
  G2: paired Δnet block-bootstrap CI > 0
  G3: per-half consistency (both halves' OOS evaluations positive)
  G4: single-event drop (remove 2024Q3) — lift must survive
  G5: multi-testing penalty — with 5 indicators, require ΔSh > +0.30
       (Bonferroni-ish guard against picking the lucky one)

Usage:
    python -m ml.research.alpha_v9_xyz_regime_blend
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
PRED_CACHE_R = CACHE / "v9_rolling_24mo_preds.parquet"
GATE_PCTILE = 0.6
GATE_WINDOW = 252
COST_BPS_SIDE = 0.8


def sharpe_annual(s: pd.Series, rpy: float = 252.0) -> float:
    if len(s) < 2 or s.std() == 0: return 0.0
    return s.mean() / s.std() * np.sqrt(rpy)


def build_pnls() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return v7_pnl, rolling_pnl, regime (all aligned)."""
    v7_preds = pd.read_parquet(PRED_CACHE_V7)
    r_preds = pd.read_parquet(PRED_CACHE_R)
    v7_preds["ts"] = pd.to_datetime(v7_preds["ts"], utc=True)
    r_preds["ts"] = pd.to_datetime(r_preds["ts"], utc=True)
    common_ts = sorted(set(v7_preds["ts"]) & set(r_preds["ts"]))
    log.info("  common preds ts: %d", len(common_ts))
    v7_sub = v7_preds[v7_preds["ts"].isin(common_ts)]
    r_sub = r_preds[r_preds["ts"].isin(common_ts)]

    regime = load_or_compute_regime()

    v7_pre = daily_portfolio_hysteresis(v7_sub, "pred", "fwd_resid_1d",
                                          set(TIER_AB), 4, 1, COST_BPS_SIDE)
    v7_pnl = gate_rolling(v7_pre, regime, pctile=GATE_PCTILE,
                             window_days=GATE_WINDOW)
    r_pre = daily_portfolio_hysteresis(r_sub, "pred", "fwd_resid_1d",
                                          set(TIER_AB), 4, 1, COST_BPS_SIDE)
    r_pnl = gate_rolling(r_pre, regime, pctile=GATE_PCTILE,
                             window_days=GATE_WINDOW)
    log.info("  v7_pnl: %d cycles  rolling_pnl: %d cycles", len(v7_pnl), len(r_pnl))

    # also return v7 + rolling preds to use for indicators R2/R3
    return v7_pnl, r_pnl, regime, v7_sub, r_sub


def build_indicators(v7_pnl: pd.DataFrame, r_pnl: pd.DataFrame,
                       v7_preds: pd.DataFrame, r_preds: pd.DataFrame,
                       regime: pd.DataFrame) -> pd.DataFrame:
    """Build all 5 regime indicators on the common ts."""
    merged = v7_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "v7"}).merge(
        r_pnl[["ts", "net_alpha"]].rename(columns={"net_alpha": "r"}), on="ts")
    merged = merged.sort_values("ts").reset_index(drop=True)

    # R1: v7 trailing 30-day Sharpe (computed from v7 net_alpha)
    merged["R1_v7_trail_sh_30d"] = (
        merged["v7"].rolling(30, min_periods=10).mean()
        / merged["v7"].rolling(30, min_periods=10).std()
        * np.sqrt(252)
    ).shift(1)  # use only past data

    # R2: v7 pred dispersion (per-day xs std), 22d smoothed
    v7p = v7_preds.copy()
    v7p = v7p[v7p["symbol"].isin(TIER_AB)]
    v7_disp = (v7p.groupby("ts")["pred"].std()
                 .rolling(22, min_periods=10).mean()
                 .shift(1)).reset_index()
    v7_disp.columns = ["ts", "R2_v7_pred_disp"]
    merged = merged.merge(v7_disp, on="ts", how="left")

    # R3: v7-rolling pred Spearman correlation per day, 22d smoothed
    v7p_sub = v7p[v7p["symbol"].isin(TIER_AB)][["ts", "symbol", "pred"]]
    rp_sub = r_preds[r_preds["symbol"].isin(TIER_AB)][["ts", "symbol", "pred"]].rename(
        columns={"pred": "pred_r"})
    pred_merged = v7p_sub.merge(rp_sub, on=["ts", "symbol"])
    daily_corrs = []
    for ts, g in pred_merged.groupby("ts"):
        if len(g) < 4:
            daily_corrs.append({"ts": ts, "R3_corr_v7_r": np.nan}); continue
        rho = g["pred"].rank().corr(g["pred_r"].rank())
        daily_corrs.append({"ts": ts, "R3_corr_v7_r": rho})
    rho_df = pd.DataFrame(daily_corrs)
    rho_df["R3_corr_v7_r"] = (rho_df["R3_corr_v7_r"]
                                  .rolling(22, min_periods=10).mean().shift(1))
    merged = merged.merge(rho_df, on="ts", how="left")

    # R4: realized xs dispersion (from regime), inverted (lower disp → higher indicator → fire)
    reg = regime[["ts", "disp_22d"]].copy()
    reg["ts"] = pd.to_datetime(reg["ts"], utc=True)
    merged = merged.merge(reg, on="ts", how="left")
    merged["R4_xs_disp_inverted"] = -merged["disp_22d"]

    # R5: v7 cumulative drawdown from peak
    cum = merged["v7"].cumsum()
    peak = cum.cummax()
    merged["R5_v7_drawdown"] = (cum - peak).shift(1)

    return merged


def evaluate_blend_at_threshold(merged: pd.DataFrame, indicator_col: str,
                                 threshold: float, w_low: float = 0.50
                                 ) -> dict:
    """Hard switch: when indicator < threshold, w_v7 = w_low; else 1.00."""
    df = merged.dropna(subset=[indicator_col, "v7", "r"]).copy()
    df["w_v7"] = np.where(df[indicator_col] < threshold, w_low, 1.00)
    df["blend"] = df["w_v7"] * df["v7"] + (1 - df["w_v7"]) * df["r"]
    sh_v7 = sharpe_annual(df["v7"])
    sh_blend = sharpe_annual(df["blend"])
    n_fired = (df["w_v7"] < 1.0).sum()
    return {"n": len(df), "sh_v7": sh_v7, "sh_blend": sh_blend,
            "delta_sh": sh_blend - sh_v7, "n_fired": int(n_fired),
            "fire_rate": n_fired / len(df) if len(df) else 0,
            "delta_net_bps": (df["blend"].mean() - df["v7"].mean()) * 1e4,
            "blend_pnl": df[["ts", "blend", "v7"]]}


def optimize_threshold(merged_subset: pd.DataFrame, indicator_col: str,
                         n_grid: int = 21) -> tuple[float, dict]:
    """Find threshold that maximizes ΔSh on the given subset."""
    df = merged_subset.dropna(subset=[indicator_col]).copy()
    if df.empty:
        return np.nan, {"delta_sh": -np.inf}
    qs = np.linspace(0.05, 0.95, n_grid)
    candidates = df[indicator_col].quantile(qs).unique()
    best_thresh = None; best_dsh = -np.inf; best_info = None
    for t in candidates:
        info = evaluate_blend_at_threshold(merged_subset, indicator_col, t)
        if info["delta_sh"] > best_dsh:
            best_dsh = info["delta_sh"]; best_thresh = t; best_info = info
    return best_thresh, best_info


def paired_block_bootstrap(merged: pd.DataFrame, blend_col: str = "blend",
                             v7_col: str = "v7", block_size: int = 5,
                             n_boot: int = 2000) -> tuple[float, float, float]:
    diff = (merged[blend_col] - merged[v7_col]).to_numpy() * 1e4
    n = len(diff); rng = np.random.default_rng(42)
    if n < 30: return np.nan, np.nan, np.nan
    n_blocks = int(np.ceil(n / block_size))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        means.append(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    log.info("loading preds and building pnls ...")
    v7_pnl, r_pnl, regime, v7_preds, r_preds = build_pnls()
    log.info("building regime indicators ...")
    merged = build_indicators(v7_pnl, r_pnl, v7_preds, r_preds, regime)
    log.info("  merged: %d rows  cols: %s", len(merged), [c for c in merged.columns if c.startswith("R")])

    indicators = ["R1_v7_trail_sh_30d", "R2_v7_pred_disp", "R3_corr_v7_r",
                   "R4_xs_disp_inverted", "R5_v7_drawdown"]

    # ---- in-sample full-panel best-threshold per indicator ----
    log.info("\n=== In-sample full-panel best threshold per indicator (illustrative only) ===")
    log.info("  %-26s %12s %12s %10s %12s %14s",
             "indicator", "best_thr", "ΔSh in-sample", "fire %", "n", "Δnet bps")
    full_results = {}
    for ind in indicators:
        thr, info = optimize_threshold(merged, ind)
        full_results[ind] = (thr, info)
        log.info("  %-26s %+12.4f %+12.2f %9.0f%% %12d %+12.2f",
                 ind, thr if not np.isnan(thr) else np.nan,
                 info["delta_sh"], info["fire_rate"] * 100, info["n"],
                 info["delta_net_bps"])

    # ---- 2-fold honest OOS ----
    log.info("\n=== 2-fold honest OOS ===")
    merged = merged.sort_values("ts").reset_index(drop=True)
    half = len(merged) // 2
    h1 = merged.iloc[:half]
    h2 = merged.iloc[half:]
    log.info("  H1: n=%d  ts %s..%s", len(h1),
             h1["ts"].min().date(), h1["ts"].max().date())
    log.info("  H2: n=%d  ts %s..%s", len(h2),
             h2["ts"].min().date(), h2["ts"].max().date())

    log.info("\n  %-26s %12s %12s %12s %12s %18s",
             "indicator", "thr (H1)", "ΔSh on H2", "thr (H2)", "ΔSh on H1",
             "Avg ΔSh OOS")
    oos_results = {}
    for ind in indicators:
        # Train on H1, eval on H2
        thr_h1, _ = optimize_threshold(h1, ind)
        if np.isnan(thr_h1):
            d_sh_h2 = 0.0
        else:
            info_h2 = evaluate_blend_at_threshold(h2, ind, thr_h1)
            d_sh_h2 = info_h2["delta_sh"]
        # Train on H2, eval on H1
        thr_h2, _ = optimize_threshold(h2, ind)
        if np.isnan(thr_h2):
            d_sh_h1 = 0.0
        else:
            info_h1 = evaluate_blend_at_threshold(h1, ind, thr_h2)
            d_sh_h1 = info_h1["delta_sh"]
        avg_oos = (d_sh_h1 + d_sh_h2) / 2
        log.info("  %-26s %+12.4f %+12.2f %+12.4f %+12.2f %+18.2f",
                 ind, thr_h1, d_sh_h2, thr_h2, d_sh_h1, avg_oos)
        oos_results[ind] = {"thr_h1": thr_h1, "d_sh_h2": d_sh_h2,
                              "thr_h2": thr_h2, "d_sh_h1": d_sh_h1,
                              "avg_oos": avg_oos}

    # ---- paired Δnet bootstrap (full panel, using H1-trained threshold for fairness) ----
    log.info("\n=== Paired Δnet block-bootstrap (using H1-trained threshold, applied to full panel) ===")
    log.info("  %-26s %12s %18s",
             "indicator", "Δnet bps", "95% block CI")
    for ind in indicators:
        thr_h1 = oos_results[ind]["thr_h1"]
        if np.isnan(thr_h1): continue
        info = evaluate_blend_at_threshold(merged, ind, thr_h1)
        d, lo, hi = paired_block_bootstrap(info["blend_pnl"])
        log.info("  %-26s %+12.2f  [%+5.2f, %+5.2f]",
                 ind, d, lo, hi)

    # ---- single-event sensitivity (drop 2024Q3) ----
    log.info("\n=== Gate 4: single-event sensitivity (drop 2024Q3) ===")
    merged_no_q3 = merged[~((merged["ts"].dt.year == 2024) & (merged["ts"].dt.quarter == 3))]
    log.info("  full panel n=%d  without 2024Q3 n=%d", len(merged), len(merged_no_q3))
    log.info("  %-26s %14s %14s %14s",
             "indicator", "ΔSh full", "ΔSh w/o Q3", "lift retained")
    for ind in indicators:
        thr_h1 = oos_results[ind]["thr_h1"]
        if np.isnan(thr_h1): continue
        info_full = evaluate_blend_at_threshold(merged, ind, thr_h1)
        info_no_q3 = evaluate_blend_at_threshold(merged_no_q3, ind, thr_h1)
        retained = (info_no_q3["delta_sh"] / info_full["delta_sh"]
                     if abs(info_full["delta_sh"]) > 0.01 else 0)
        log.info("  %-26s %+14.2f %+14.2f %14.2f",
                 ind, info_full["delta_sh"], info_no_q3["delta_sh"], retained)

    # ---- discipline-gate verdict ----
    log.info("\n=== Discipline-gate verdict ===")
    log.info("  %-26s %12s %14s %14s %s",
             "indicator", "avg OOS ΔSh", "both halves +", "Q3-drop ratio", "verdict")
    for ind in indicators:
        r = oos_results[ind]
        avg_oos = r["avg_oos"]
        both_pos = (r["d_sh_h1"] > 0) and (r["d_sh_h2"] > 0)
        thr_h1 = r["thr_h1"]
        if np.isnan(thr_h1):
            q3_ratio = np.nan; ci_pos = False
        else:
            info_full = evaluate_blend_at_threshold(merged, ind, thr_h1)
            info_no_q3 = evaluate_blend_at_threshold(merged_no_q3, ind, thr_h1)
            q3_ratio = (info_no_q3["delta_sh"] / info_full["delta_sh"]
                         if abs(info_full["delta_sh"]) > 0.01 else 0)
            d, lo, hi = paired_block_bootstrap(info_full["blend_pnl"])
            ci_pos = lo > 0
        # Discipline: avg_oos > +0.20 (G1+G3 implicit), CI > 0 (G2),
        # both halves positive (G3), Q3-drop retains ≥ 50% (G4)
        passes = (avg_oos > 0.20 and both_pos
                   and not np.isnan(q3_ratio) and q3_ratio >= 0.50
                   and ci_pos)
        # Also G5 (multi-test): require avg > 0.30
        passes_g5 = passes and avg_oos > 0.30
        verdict = "PASS ✓ (incl G5)" if passes_g5 else (
            "passes G1-G4 only" if passes else "fail")
        log.info("  %-26s %+12.2f %14s %14.2f  %s",
                 ind, avg_oos, "yes" if both_pos else "no",
                 q3_ratio if not np.isnan(q3_ratio) else 0, verdict)


if __name__ == "__main__":
    main()
