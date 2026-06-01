"""X79 — Beta-neutralization construction lever on 3-year panel.

Motivation (X74-B): cross-sectional ranking edge (IC/spread) is regime-robust —
positive even in strong bull — but the PORTFOLIO fails in bull. Hypothesis: the
failure is residual BTC-beta leakage in the L/S basket during trends. Neutralizing
beta each cycle could harvest the bull-regime signal that's currently lost, lifting
the ~+1.0 regime-aware ceiling WITHOUT a hard regime gate.

Tests (on 3yr V0 preds; rerun on V5 when panel_3yr_v5 ready):
  A. Baseline V0 (no neutralization) = +0.12
  B. Beta-neutral selection: residualize predictions on per-sym BTC beta before ranking
  C. Beta-weighted sizing: scale leg sizes to make net basket beta ≈ 0
  D. Combine beta-neutral + HMM/gate (if beta-neutral alone helps)

We approximate via the prediction/selection layer since the sleeve is fixed:
  - Each sym has corr_to_btc_1d / beta features in panel; use trailing beta to
    orthogonalize the cross-sectional pred against beta each cycle.
"""
from __future__ import annotations
import sys, importlib.util, time, warnings
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    t0 = time.time()
    print("=== X79 beta-neutralization construction lever (3yr) ===\n", flush=True)
    v5p = REPO/"outputs/vBTC_features/panel_3yr_v5.parquet"
    panel_path = v5p if v5p.exists() else REPO/"outputs/vBTC_features/panel_3yr_v0.parquet"
    print(f"Panel: {panel_path.name}")
    panel = pd.read_parquet(panel_path)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    if "target_z" not in panel.columns: panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    folds = x6.get_folds(panel)

    # Baseline V0 preds
    apd = x6.train_per_sym_ridge(panel, folds, feats, label="x79_base")
    apd.to_parquet(RCACHE/"x79_base_preds.parquet", index=False)
    m0 = x6.run_sleeve_on_preds(RCACHE/"x79_base_preds.parquet", "x79_base")
    print(f"A. Baseline V0: Sharpe={m0.get('sharpe',0):+.2f} folds={m0.get('folds_pos','?')} conc={m0.get('concentration','?')}\n")

    # Need a beta proxy per (sym, time). Use trailing corr_to_btc_1d as beta proxy if
    # beta col absent; better: idio_vol ratio. We'll use corr_to_btc_1d (PIT, shifted).
    beta_col = None
    for c in ["beta_to_btc", "corr_to_btc_1d"]:
        if c in panel.columns: beta_col = c; break
    if beta_col is None:
        print("No beta proxy column; cannot neutralize"); return
    print(f"Using beta proxy: {beta_col}\n")
    bmap = panel[["symbol","open_time",beta_col]].drop_duplicates(["symbol","open_time"])
    apd = apd.merge(bmap, on=["symbol","open_time"], how="left")

    # B. Beta-neutral selection: per cycle, regress pred on beta_proxy, use residual as new pred
    apd = apd.reset_index(drop=True)
    pred_bn = apd["pred"].to_numpy(dtype=float).copy()
    med = apd[beta_col].median()
    for ot, idx in apd.groupby("open_time").indices.items():
        b = np.nan_to_num(apd[beta_col].to_numpy()[idx], nan=med)
        p = apd["pred"].to_numpy()[idx]
        if np.std(b) < 1e-9 or len(idx) < 5:
            pred_bn[idx] = p - p.mean()
        else:
            A = np.vstack([np.ones_like(b), b]).T
            coef, *_ = np.linalg.lstsq(A, p, rcond=None)
            pred_bn[idx] = p - A @ coef
    apd_bn = apd.copy(); apd_bn["pred_bn"] = pred_bn
    a = apd_bn[["symbol","open_time","alpha_A","return_pct","exit_time","fold"]].copy()
    a["pred"] = pred_bn
    a.to_parquet(RCACHE/"x79_betaneutral_preds.parquet", index=False)
    mB = x6.run_sleeve_on_preds(RCACHE/"x79_betaneutral_preds.parquet", "x79_bn")
    print(f"B. Beta-neutral selection: Sharpe={mB.get('sharpe',0):+.2f} folds={mB.get('folds_pos','?')} conc={mB.get('concentration','?')}", flush=True)

    # C. Per-regime: beta-neutral only in bull (PIT btc_30d>0.10), normal else
    if "btc_ret_30d" in panel.columns:
        r30 = panel[["symbol","open_time","btc_ret_30d"]].drop_duplicates(["symbol","open_time"])
    else:
        r30 = None
    if r30 is not None:
        ab = apd_bn.merge(r30, on=["symbol","open_time"], how="left")
        ab["pred_mix"] = np.where(ab["btc_ret_30d"]>0.10, ab["pred_bn"], ab["pred"])
        a2 = ab[["symbol","open_time","alpha_A","return_pct","exit_time","fold"]].copy()
        a2["pred"] = ab["pred_mix"].values
        a2.to_parquet(RCACHE/"x79_bn_in_bull_preds.parquet", index=False)
        mC = x6.run_sleeve_on_preds(RCACHE/"x79_bn_in_bull_preds.parquet", "x79_bn_bull")
        print(f"C. Beta-neutral in bull only: Sharpe={mC.get('sharpe',0):+.2f} folds={mC.get('folds_pos','?')}", flush=True)

    print(f"\nReference: V0 uncond +0.12, KMeans-routed +1.07, hard gate +1.13")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
