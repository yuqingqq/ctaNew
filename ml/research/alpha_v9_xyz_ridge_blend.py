"""Ridge-head intraday-feature blend on top of v7 LGBM (Tier A+B, 2024-2026).

Architecture (mirrors crypto v6_clean hybrid LGBM + Ridge_pos):
  v7 LGBM  → lgbm_pred  (cached walk-forward, 18 features, 2013-2026)
  Ridge    → ridge_pred (intraday features, walk-forward over 2024-2026)
  Blend:   final_pred = (1-w) × z(lgbm_pred) + w × z(ridge_pred)
           where z() standardizes per training fold

Walk-forward folds (3, expanding train, fixed-length test):
  Fold 0: train 2024-06 .. 2024-12,  test 2025-01 .. 2025-05
  Fold 1: train 2024-06 .. 2025-05,  test 2025-06 .. 2025-10
  Fold 2: train 2024-06 .. 2025-10,  test 2025-11 .. 2026-04

Discipline gates:
  1. ΔSh > +0.20 vs v7-only on aggregated test panel
  2. Block-bootstrap Δnet CI > 0
  3. ≥ 2/3 folds with positive ΔSh (consistency)
  4. Hard-split: skipped — Polygon coverage insufficient

Usage:
    python -m ml.research.alpha_v9_xyz_ridge_blend
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
RIDGE_FEATURES = ["first_vs_last_xs", "first_30_ret_xs"]  # passed walk-forward gates
RIDGE_ALPHA = 1.0


def build_panel() -> pd.DataFrame:
    """Build merged panel: (date, ts, symbol, lgbm_pred, intraday features, fwd_resid_1d)."""
    log.info("loading cached v7 preds ...")
    preds = pd.read_parquet(PRED_CACHE)
    preds["date"] = pd.to_datetime(preds["ts"]).dt.tz_convert(None).dt.normalize()
    preds = preds[preds["symbol"].isin(TIER_AB)].copy()
    preds = preds.rename(columns={"pred": "lgbm_pred"})

    rows = []
    for sym in TIER_AB:
        poly_path = CACHE / f"poly_{sym}_5m.parquet"
        if not poly_path.exists(): continue
        poly = pd.read_parquet(poly_path)
        feats = session_features(poly)
        feats["symbol"] = sym
        sub = preds[preds["symbol"] == sym][
            ["date", "ts", "lgbm_pred", "fwd_resid_1d"]]
        merged = feats.merge(sub, on="date", how="inner")
        rows.append(merged)
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["first_vs_last", "first_30_ret",
                            "lgbm_pred", "fwd_resid_1d"]).reset_index(drop=True)
    # Cross-sectional residualize within Tier A+B per date
    for c in ["first_vs_last", "first_30_ret"]:
        df[c + "_xs"] = df[c] - df.groupby("date")[c].transform("median")
    log.info("  panel: %d rows, %d names, %s..%s",
             len(df), df["symbol"].nunique(),
             df["date"].min().date(), df["date"].max().date())
    return df


def make_folds(panel: pd.DataFrame) -> list[dict]:
    """Three expanding-train folds in 2025-2026."""
    return [
        {"name": "fold0",
         "train_end": pd.Timestamp("2025-01-01"),
         "test_start": pd.Timestamp("2025-01-01"),
         "test_end": pd.Timestamp("2025-06-01")},
        {"name": "fold1",
         "train_end": pd.Timestamp("2025-06-01"),
         "test_start": pd.Timestamp("2025-06-01"),
         "test_end": pd.Timestamp("2025-11-01")},
        {"name": "fold2",
         "train_end": pd.Timestamp("2025-11-01"),
         "test_start": pd.Timestamp("2025-11-01"),
         "test_end": pd.Timestamp("2026-05-01")},
    ]


def fit_ridge_and_predict(train: pd.DataFrame, test: pd.DataFrame,
                            features: list[str], target: str, alpha: float
                            ) -> tuple[np.ndarray, dict]:
    """Train Ridge on features → target. Return (predictions on test, info dict)."""
    Xtr = train[features].to_numpy(dtype=float)
    ytr = train[target].to_numpy(dtype=float)
    msk = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    Xtr, ytr = Xtr[msk], ytr[msk]
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    m = Ridge(alpha=alpha).fit(Xs, ytr)
    Xte = test[features].to_numpy(dtype=float)
    Xte_msk = np.isfinite(Xte).all(axis=1)
    pred = np.full(len(test), np.nan)
    pred[Xte_msk] = m.predict(sc.transform(Xte[Xte_msk]))
    info = {"coef": m.coef_.tolist(), "intercept": float(m.intercept_),
            "n_train": int(msk.sum())}
    return pred, info


def zscore_inplace(df: pd.DataFrame, col: str, ref_mean: float, ref_std: float
                    ) -> None:
    df[col + "_z"] = (df[col] - ref_mean) / ref_std if ref_std > 0 else 0.0


def evaluate_blend(panel: pd.DataFrame, w: float, regime: pd.DataFrame,
                    K: int = 4, M: int = 1, *, label: str) -> dict:
    """Construct portfolio on `final_pred = (1-w)*z_lgbm + w*z_ridge`."""
    sub = panel.copy()
    sub["final_pred"] = (1 - w) * sub["lgbm_pred_z"] + w * sub["ridge_pred_z"]
    pnl_pre = daily_portfolio_hysteresis(
        sub, "final_pred", "fwd_resid_1d",
        set(TIER_AB), K, M, COST_BPS_SIDE)
    if pnl_pre.empty:
        return {"label": label, "metrics": None, "pnl": pd.DataFrame()}
    pnl = gate_rolling(pnl_pre, regime, pctile=GATE_PCTILE,
                          window_days=GATE_WINDOW)
    if pnl.empty:
        return {"label": label, "metrics": None, "pnl": pd.DataFrame()}
    m = metrics_for(pnl, 1)
    lo, hi = boot_ci(pnl, 1)
    return {"label": label, "metrics": m, "ci": (lo, hi), "pnl": pnl}


def paired_block_bootstrap(pnl_a: pd.DataFrame, pnl_b: pd.DataFrame,
                             block_size: int = 5, n_boot: int = 2000) -> tuple[float, float, float]:
    merged = pnl_a[["ts", "net_alpha"]].rename(columns={"net_alpha": "a"}).merge(
        pnl_b[["ts", "net_alpha"]].rename(columns={"net_alpha": "b"}), on="ts")
    if len(merged) < 30: return np.nan, np.nan, np.nan
    diff = (merged["b"] - merged["a"]).to_numpy() * 1e4
    n = len(diff)
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
    regime = load_or_compute_regime()
    folds = make_folds(panel)

    # Walk-forward: train Ridge per fold, predict on test, compute z-stats per fold
    fold_outputs = []
    for fold in folds:
        train = panel[panel["date"] < fold["train_end"]].copy()
        test = panel[(panel["date"] >= fold["test_start"]) &
                       (panel["date"] < fold["test_end"])].copy()
        if len(train) < 200 or len(test) < 50:
            log.warning("  %s: train=%d test=%d skip", fold["name"], len(train), len(test))
            continue
        ridge_pred, info = fit_ridge_and_predict(
            train, test, RIDGE_FEATURES, "fwd_resid_1d", RIDGE_ALPHA)
        test["ridge_pred"] = ridge_pred
        # Standardize lgbm_pred and ridge_pred using TRAIN distributions
        lgbm_mean, lgbm_std = train["lgbm_pred"].mean(), train["lgbm_pred"].std()
        # For ridge_pred, predict on train to get its train distribution
        train_ridge_pred, _ = fit_ridge_and_predict(
            train, train, RIDGE_FEATURES, "fwd_resid_1d", RIDGE_ALPHA)
        ridge_mean = np.nanmean(train_ridge_pred)
        ridge_std = np.nanstd(train_ridge_pred)
        test["lgbm_pred_z"] = (test["lgbm_pred"] - lgbm_mean) / max(lgbm_std, 1e-9)
        test["ridge_pred_z"] = (test["ridge_pred"] - ridge_mean) / max(ridge_std, 1e-9)
        log.info("  %s: train_n=%d test_n=%d  Ridge coef=%s  intercept=%+.5f",
                 fold["name"], info["n_train"], len(test),
                 [f"{c:+.4f}" for c in info["coef"]], info["intercept"])
        fold_outputs.append(test)

    full_test = pd.concat(fold_outputs, ignore_index=True)
    log.info("\n  combined OOS test panel: %d rows  %s..%s",
             len(full_test),
             full_test["date"].min().date(), full_test["date"].max().date())

    # ---- weight sweep ----
    log.info("\n=== Weight sweep on aggregated 3-fold OOS panel ===")
    log.info("  %-22s %5s %10s %18s %12s",
             "config", "n", "active_Sh", "95% CI", "net bps/cyc")
    out = {}
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        r = evaluate_blend(full_test, w, regime, K=4, M=1,
                            label=f"w={w:.2f}")
        out[w] = r
        if r["metrics"] is None:
            log.info("  %-22s empty", r["label"]); continue
        m = r["metrics"]; lo, hi = r["ci"]
        log.info("  %-22s %5d %+8.2f [%+5.2f,%+5.2f] %+10.2f",
                 r["label"], m["n_rebal"], m["active_sharpe"], lo, hi,
                 m["net_bps_per_rebal"])

    base = out[0.0]
    if base["metrics"] is None:
        log.error("baseline empty; abort"); return

    # ---- paired Δnet CI ----
    log.info("\n=== Paired Δnet bps vs w=0 (block-bootstrap, 5-day) ===")
    log.info("  %-12s %7s %18s %s",
             "weight", "Δnet", "95% CI", "(point Sh diff)")
    for w in [0.05, 0.10, 0.15, 0.20, 0.30]:
        r = out[w]
        if r["metrics"] is None: continue
        d, lo_d, hi_d = paired_block_bootstrap(base["pnl"], r["pnl"])
        d_sh = r["metrics"]["active_sharpe"] - base["metrics"]["active_sharpe"]
        log.info("  w=%.2f   %+7.2f bps  [%+5.2f, %+5.2f]   ΔSh = %+.2f",
                 w, d, lo_d, hi_d, d_sh)

    # ---- per-fold ΔSh ----
    log.info("\n=== Per-fold active_Sharpe ===")
    log.info("  %-8s " + "  ".join(f"{f['name']:>10}" for f in folds), "weight")
    full_test["fold"] = ""
    for fold in folds:
        msk = ((full_test["date"] >= fold["test_start"]) &
                (full_test["date"] < fold["test_end"]))
        full_test.loc[msk, "fold"] = fold["name"]
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        r = out[w]
        if r["metrics"] is None: continue
        line = f"  w={w:.2f}  "
        for fold in folds:
            sub = r["pnl"].copy()
            sub["fold"] = sub["ts"].apply(
                lambda t: next((f["name"] for f in folds
                                  if f["test_start"] <= t.tz_convert(None) < f["test_end"]),
                                  None))
            g = sub[sub["fold"] == fold["name"]]
            if len(g) < 5:
                line += f"  {'---':>10}"; continue
            sh = metrics_for(g, 1)["active_sharpe"]
            line += f"  {sh:+10.2f}"
        log.info(line)

    # ---- discipline gate verdict ----
    log.info("\n=== Discipline gate verdict ===")
    log.info("  %-8s %8s %18s %12s %s",
             "weight", "ΔSh", "Δnet 95% CI", "folds_pos", "verdict")
    for w in [0.05, 0.10, 0.15, 0.20, 0.30]:
        r = out[w]
        if r["metrics"] is None: continue
        d_sh = r["metrics"]["active_sharpe"] - base["metrics"]["active_sharpe"]
        d, lo_d, hi_d = paired_block_bootstrap(base["pnl"], r["pnl"])
        # per-fold
        n_pos = 0; n_total = 0
        for fold in folds:
            sub = r["pnl"].copy()
            sub["fold"] = sub["ts"].apply(
                lambda t: next((f["name"] for f in folds
                                  if f["test_start"] <= t.tz_convert(None) < f["test_end"]),
                                  None))
            g = sub[sub["fold"] == fold["name"]]
            base_sub = base["pnl"].copy()
            base_sub["fold"] = base_sub["ts"].apply(
                lambda t: next((f["name"] for f in folds
                                  if f["test_start"] <= t.tz_convert(None) < f["test_end"]),
                                  None))
            base_g = base_sub[base_sub["fold"] == fold["name"]]
            if len(g) < 5 or len(base_g) < 5: continue
            d_fold = (metrics_for(g, 1)["active_sharpe"]
                       - metrics_for(base_g, 1)["active_sharpe"])
            n_total += 1
            if d_fold > 0: n_pos += 1
        passes = (d_sh > 0.20 and lo_d > 0 and n_pos >= 2)
        log.info("  w=%.2f  %+8.2f  [%+6.2f, %+6.2f]  %d/%d         %s",
                 w, d_sh, lo_d, hi_d, n_pos, n_total,
                 "PASS ✓" if passes else "fail")


if __name__ == "__main__":
    main()
