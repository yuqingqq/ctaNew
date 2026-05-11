"""Walk-forward IC stability for intraday features (v7 xyz universe).

Reuses features from `alpha_v9_xyz_intraday_ic.py`. Adds:
  1. Per-quarter IC: time-series of cross-sectional IC across ~8 quarters
  2. First-half / second-half OOS split: train-window IC vs test-window IC
  3. Block-bootstrap CI on full-sample pooled IC
  4. Sign-stability check: fraction of quarters with consistent IC sign

Discipline: a feature passes IF
  (a) full-sample |IC| ≥ 0.02
  (b) bootstrap 95% CI on IC excludes 0
  (c) ≥ 60% of quarters have IC of the consistent sign
  (d) second-half (OOS) |IC| ≥ 50% of first-half (IS)

Usage:
    python -m ml.research.alpha_v9_xyz_intraday_walkfwd
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v9_xyz_intraday_ic import (
    session_features, XYZ_NAMES, CACHE, PRED_CACHE,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def pooled_ic(df: pd.DataFrame, feat: str, label: str,
                method: str = "pearson") -> tuple[float, int]:
    x = df[feat].to_numpy(); y = df[label].to_numpy()
    msk = np.isfinite(x) & np.isfinite(y)
    if msk.sum() < 30: return np.nan, int(msk.sum())
    if method == "pearson":
        return float(np.corrcoef(x[msk], y[msk])[0, 1]), int(msk.sum())
    rx = pd.Series(x[msk]).rank().to_numpy()
    ry = pd.Series(y[msk]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), int(msk.sum())


def bootstrap_ic_ci(df: pd.DataFrame, feat: str, label: str,
                      n_boot: int = 2000, block_days: int = 5) -> tuple[float, float, float]:
    """Block-bootstrap by date to preserve cross-sectional structure within
    each date and serial correlation across dates."""
    df = df.dropna(subset=[feat, label]).copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(dates)
    if n_dates < block_days * 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(n_dates / block_days))
    ics = []
    for _ in range(n_boot):
        starts = rng.integers(0, n_dates - block_days + 1, size=n_blocks)
        idx_dates = []
        for s in starts:
            idx_dates.extend(dates.iloc[s:s + block_days].tolist())
        idx_dates = idx_dates[:n_dates]
        sub = df[df["date"].isin(idx_dates)]
        x = sub[feat].to_numpy(); y = sub[label].to_numpy()
        if len(x) < 30 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        ics.append(np.corrcoef(x, y)[0, 1])
    if not ics: return np.nan, np.nan, np.nan
    return float(np.mean(ics)), float(np.percentile(ics, 2.5)), float(np.percentile(ics, 97.5))


def main() -> None:
    log.info("loading cached preds and intraday features ...")
    preds = pd.read_parquet(PRED_CACHE)
    preds["date"] = pd.to_datetime(preds["ts"]).dt.tz_convert(None).dt.normalize()
    preds = preds[preds["symbol"].isin(XYZ_NAMES)].copy()

    feature_cols = ["opening_gap", "first_30_ret", "last_30_ret",
                     "first_vs_last", "intraday_range", "vwap_dev",
                     "close_pos_in_range", "day0_intra"]
    rows = []
    for sym in XYZ_NAMES:
        poly_path = CACHE / f"poly_{sym}_5m.parquet"
        if not poly_path.exists(): continue
        poly = pd.read_parquet(poly_path)
        feats = session_features(poly)
        feats["symbol"] = sym
        sub = preds[preds["symbol"] == sym][["date", "fwd_resid_1d", "fwd_resid_5d"]]
        merged = feats.merge(sub, on="date", how="inner")
        rows.append(merged)
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=feature_cols + ["fwd_resid_1d"]).reset_index(drop=True)
    # Cross-sectional residualize features
    for c in feature_cols:
        df[c + "_xs"] = df[c] - df.groupby("date")[c].transform("median")
    df["date"] = pd.to_datetime(df["date"])
    log.info("  pooled rows: %d  date range %s..%s  n_dates=%d  n_syms=%d",
             len(df), df["date"].min().date(), df["date"].max().date(),
             df["date"].nunique(), df["symbol"].nunique())

    # ---- per-quarter IC ----
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    qs = sorted(df["quarter"].unique())
    log.info("\n=== Per-quarter Pearson IC vs fwd_resid_1d ===")
    log.info("  %-22s " + "  ".join(f"{q:>9}" for q in qs) + "  full   pos_qts/total",
             "feature")
    rank_summary = []
    for c in feature_cols:
        ics_q = []
        for q in qs:
            g = df[df["quarter"] == q]
            ic, _ = pooled_ic(g, c + "_xs", "fwd_resid_1d")
            ics_q.append(ic)
        ic_full, _ = pooled_ic(df, c + "_xs", "fwd_resid_1d")
        ics_arr = np.array(ics_q)
        # Sign stability: fraction of quarters matching the full-sample sign
        full_sign = np.sign(ic_full)
        stable_fraction = ((np.sign(ics_arr) == full_sign).sum()
                           / len(ics_arr)) if full_sign != 0 else 0
        log.info("  %-22s " + "  ".join(f"{ic:+9.4f}" for ic in ics_q)
                 + f"  {ic_full:+.4f}  {int(stable_fraction*len(ics_arr))}/{len(ics_arr)}",
                 c)
        rank_summary.append({"feature": c, "ic_full": ic_full,
                              "stable_qts": int(stable_fraction * len(ics_arr)),
                              "total_qts": len(ics_arr),
                              "ics_q": ics_arr})

    # ---- IS (first-half) vs OOS (second-half) ----
    median_date = df["date"].quantile(0.5)
    df_is = df[df["date"] <= median_date]
    df_oos = df[df["date"] > median_date]
    log.info("\n=== First-half (IS) vs Second-half (OOS) split ===")
    log.info("  IS:  n=%d  dates %s..%s",
             len(df_is), df_is["date"].min().date(), df_is["date"].max().date())
    log.info("  OOS: n=%d  dates %s..%s",
             len(df_oos), df_oos["date"].min().date(), df_oos["date"].max().date())
    log.info("  %-22s %10s %10s %10s",
             "feature", "IS IC", "OOS IC", "OOS/IS")
    for c in feature_cols:
        is_ic, _ = pooled_ic(df_is, c + "_xs", "fwd_resid_1d")
        oos_ic, _ = pooled_ic(df_oos, c + "_xs", "fwd_resid_1d")
        ratio = oos_ic / is_ic if abs(is_ic) > 1e-6 else np.nan
        log.info("  %-22s %+10.4f %+10.4f %10.2f", c, is_ic, oos_ic, ratio)

    # ---- bootstrap CI on full-sample IC ----
    log.info("\n=== Block-bootstrap (5-day blocks) 95%% CI on pooled IC ===")
    log.info("  %-22s %10s %20s",
             "feature", "IC", "95% CI")
    for c in feature_cols:
        mean_ic, lo, hi = bootstrap_ic_ci(df, c + "_xs", "fwd_resid_1d",
                                              n_boot=1000, block_days=5)
        marker = "  pass" if (lo > 0 or hi < 0) and abs(mean_ic) >= 0.02 else ""
        log.info("  %-22s %+10.4f  [%+6.4f, %+6.4f]%s", c, mean_ic, lo, hi, marker)

    # ---- summary verdict ----
    log.info("\n=== Verdict (pass = full|IC|≥0.02 AND bootstrap CI excludes 0 AND ≥60%% sign-stable AND OOS keeps ≥50%% of IS magnitude) ===")
    log.info("  %-22s %8s %8s %8s %14s %s",
             "feature", "|IC|", "stable_q", "OOS/IS", "boot CI excl 0", "verdict")
    for c in feature_cols:
        ic_full, _ = pooled_ic(df, c + "_xs", "fwd_resid_1d")
        is_ic, _ = pooled_ic(df_is, c + "_xs", "fwd_resid_1d")
        oos_ic, _ = pooled_ic(df_oos, c + "_xs", "fwd_resid_1d")
        oos_ratio = abs(oos_ic / is_ic) if abs(is_ic) > 1e-6 else 0
        # stable quarters
        ics_q = np.array([pooled_ic(df[df["quarter"] == q],
                                     c + "_xs", "fwd_resid_1d")[0] for q in qs])
        stable_pct = (np.sign(ics_q) == np.sign(ic_full)).sum() / len(ics_q)
        # bootstrap
        _, lo, hi = bootstrap_ic_ci(df, c + "_xs", "fwd_resid_1d",
                                       n_boot=1000, block_days=5)
        ci_excl_zero = (lo > 0) or (hi < 0)
        passes = (abs(ic_full) >= 0.02 and ci_excl_zero
                   and stable_pct >= 0.60 and oos_ratio >= 0.50)
        log.info("  %-22s %8.4f %8.0f%% %8.2f  %14s  %s",
                 c, abs(ic_full), stable_pct * 100, oos_ratio,
                 "yes" if ci_excl_zero else "no",
                 "PASS ✓" if passes else "fail")


if __name__ == "__main__":
    main()
