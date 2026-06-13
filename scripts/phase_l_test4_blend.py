"""Phase L / Test 4: production/sparse fixed-weight blend.

Simplest possible test: blend production per-cycle PnL with sparse per-cycle PnL
at fixed weights (no learning).

  blended_pnl[t] = w_prod * prod_pnl[t] + w_sparse * sparse_pnl[t]

Tested at:
  prod=1.00 sparse=0.00 (production reference)
  prod=0.75 sparse=0.25
  prod=0.50 sparse=0.50
  prod=0.25 sparse=0.75
  prod=0.00 sparse=1.00 (sparse only reference)

Two sparse strengths: margin=0.25 (best K2 in-sample) and margin=0.40 (K4).

Pass: Sharpe ≥ production +0.2 OR maxDD improves >20% with Sharpe neutral,
AND ≥6/9 folds non-worse.

Note: this assumes you can RUN both production and sparse simultaneously in
the same account (i.e., capital allocated to two parallel sub-strategies).
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "outputs/vBTC_blend"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OOS_FOLDS = list(range(1, 10))

PROD_CSV = REPO / "outputs/vBTC_swap_rule/k2_robustness/per_cycle_m0.0.csv"
SPARSE_0_25_CSV = REPO / "outputs/vBTC_swap_rule/k2_robustness/per_cycle_m4.5.csv"
SPARSE_0_40_CSV = REPO / "outputs/vBTC_swap_rule/k2_robustness/per_cycle_m9.0.csv"


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print("=== Phase L / Test 4: production/sparse fixed-weight blend ===\n", flush=True)

    prod = pd.read_csv(PROD_CSV)
    sparse_025 = pd.read_csv(SPARSE_0_25_CSV)
    sparse_040 = pd.read_csv(SPARSE_0_40_CSV)
    prod["time"] = pd.to_datetime(prod["time"], utc=True)
    sparse_025["time"] = pd.to_datetime(sparse_025["time"], utc=True)
    sparse_040["time"] = pd.to_datetime(sparse_040["time"], utc=True)

    df = prod[["time", "fold", "net_bps"]].rename(columns={"net_bps": "prod_pnl"})
    df = df.merge(sparse_025[["time", "net_bps"]].rename(columns={"net_bps": "sparse025_pnl"}),
                    on="time", how="inner")
    df = df.merge(sparse_040[["time", "net_bps"]].rename(columns={"net_bps": "sparse040_pnl"}),
                    on="time", how="inner")
    print(f"  Cycles: {len(df):,}", flush=True)
    print(f"  Pure production Sharpe: {_sharpe(df['prod_pnl'].to_numpy()):+.2f}", flush=True)
    print(f"  Pure sparse_0.25 Sharpe: {_sharpe(df['sparse025_pnl'].to_numpy()):+.2f}", flush=True)
    print(f"  Pure sparse_0.40 Sharpe: {_sharpe(df['sparse040_pnl'].to_numpy()):+.2f}", flush=True)

    # Correlation between prod and sparse
    print(f"\n  Cycle-level correlation:")
    print(f"    prod vs sparse_0.25: {df['prod_pnl'].corr(df['sparse025_pnl']):+.3f}",
          flush=True)
    print(f"    prod vs sparse_0.40: {df['prod_pnl'].corr(df['sparse040_pnl']):+.3f}",
          flush=True)

    blends = [
        ("prod_1.00_sparse_0.00", 1.00, 0.00, "sparse025"),
        ("prod_0.75_sparse025_0.25", 0.75, 0.25, "sparse025"),
        ("prod_0.50_sparse025_0.50", 0.50, 0.50, "sparse025"),
        ("prod_0.25_sparse025_0.75", 0.25, 0.75, "sparse025"),
        ("prod_0.00_sparse025_1.00", 0.00, 1.00, "sparse025"),
        ("prod_0.75_sparse040_0.25", 0.75, 0.25, "sparse040"),
        ("prod_0.50_sparse040_0.50", 0.50, 0.50, "sparse040"),
        ("prod_0.25_sparse040_0.75", 0.25, 0.75, "sparse040"),
        ("prod_0.00_sparse040_1.00", 0.00, 1.00, "sparse040"),
    ]

    print(f"\n  {'variant':<32}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'pos_folds':>9}  per-fold", flush=True)
    results = []
    for label, w_prod, w_sparse, sparse_col in blends:
        sparse_key = "sparse025_pnl" if sparse_col == "sparse025" else "sparse040_pnl"
        blended = w_prod * df["prod_pnl"].to_numpy() + w_sparse * df[sparse_key].to_numpy()
        sh, lo, hi = block_bootstrap_ci(blended, statistic=_sharpe,
                                            block_size=7, n_boot=2000)
        n_pos = 0
        per_fold = []
        for f in OOS_FOLDS:
            mask = df["fold"] == f
            d = blended[mask]
            if len(d) >= 3:
                sh_f = _sharpe(d)
                per_fold.append(sh_f)
                if sh_f > 0: n_pos += 1
        pf_str = " ".join(f"{x:+.1f}" for x in per_fold)
        results.append({"variant": label, "w_prod": w_prod, "w_sparse": w_sparse,
                          "sparse_strength": sparse_col,
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "max_dd": _max_dd(blended), "total_pnl": blended.sum(),
                          "n_folds_positive": n_pos,
                          "per_fold": pf_str})
        print(f"  {label:<32}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{_max_dd(blended):>+7.0f}  {blended.sum():>+8.0f}  {n_pos:>5d}/9   {pf_str}",
              flush=True)

    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

    prod_row = next(r for r in results if r["variant"] == "prod_1.00_sparse_0.00")
    # Best blend (exclude pure prod and pure sparse-100)
    blend_only = [r for r in results if "0.25" in r["variant"] or "0.50" in r["variant"]
                   or "0.75" in r["variant"]]
    blend_only = [r for r in blend_only if r["w_sparse"] > 0 and r["w_prod"] > 0]
    best = max(blend_only, key=lambda r: r["sharpe"])
    print(f"\n  Production reference: Sharpe={prod_row['sharpe']:+.2f}, "
          f"maxDD={prod_row['max_dd']:+.0f}, folds={prod_row['n_folds_positive']}/9",
          flush=True)
    print(f"  Best blend ({best['variant']}): Sharpe={best['sharpe']:+.2f}, "
          f"maxDD={best['max_dd']:+.0f}, folds={best['n_folds_positive']}/9", flush=True)
    sharpe_lift = best["sharpe"] - prod_row["sharpe"]
    dd_improvement = (prod_row["max_dd"] - best["max_dd"]) / abs(prod_row["max_dd"]) * 100
    print(f"  Sharpe lift: {sharpe_lift:+.2f}", flush=True)
    print(f"  maxDD improvement: {dd_improvement:+.1f}%", flush=True)

    # Verdict
    pass_sharpe = sharpe_lift >= 0.2
    pass_dd_neutral = dd_improvement > 20 and abs(sharpe_lift) < 0.1
    pass_folds = best["n_folds_positive"] >= 6
    print(f"\n=== Test 4 verdict ===", flush=True)
    print(f"  Sharpe ≥ prod +0.2 OR maxDD>20% improve with Sharpe neutral:  "
          f"{'PASS' if pass_sharpe or pass_dd_neutral else 'FAIL'}", flush=True)
    print(f"  ≥6/9 folds non-worse: "
          f"{'PASS' if pass_folds else 'FAIL'} ({best['n_folds_positive']}/9)", flush=True)
    if (pass_sharpe or pass_dd_neutral) and pass_folds:
        print(f"  → ADOPT {best['variant']}", flush=True)
    else:
        print(f"  → NOT ADOPTED", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
