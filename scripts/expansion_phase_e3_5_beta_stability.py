"""Phase E3.5 (parallel diagnostic): does β-to-basket stability separate
the loser cohort (ICP, ORDI, HBAR, TAO, AAVE) from the winner cohort (VVV,
WIF, AVAX, WLD, AXS)?

Tests the hypothesis: unstable β = noisy hedge = bad picks.

Uses existing panel data (already has rolling beta_short_vs_bk per cycle).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs/vBTC_beta_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Loser / winner cohorts from per-symbol attribution earlier
LOSERS = ["ICPUSDT", "ORDIUSDT", "HBARUSDT", "TAOUSDT", "AAVEUSDT",
          "TIAUSDT", "ENAUSDT", "RUNEUSDT"]
WINNERS = ["VVVUSDT", "WIFUSDT", "AVAXUSDT", "WLDUSDT", "AXSUSDT",
           "LTCUSDT", "NEARUSDT", "LINKUSDT"]


def main():
    print(f"=== Phase E3.5: β-stability diagnostic ===\n", flush=True)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                              columns=["symbol", "open_time", "beta_short_vs_bk"])
    panel = panel.dropna(subset=["beta_short_vs_bk"])
    print(f"  Loaded {len(panel):,} rows, {panel['symbol'].nunique()} symbols", flush=True)

    # Per-symbol β-stability metrics
    print(f"\n  Computing per-symbol β-statistics over full panel window...", flush=True)
    rows = []
    for sym, g in panel.groupby("symbol"):
        b = g["beta_short_vs_bk"].to_numpy()
        if len(b) < 100:
            continue
        rows.append({
            "symbol": sym,
            "n_obs": len(b),
            "mean_beta": float(b.mean()),
            "std_beta": float(b.std()),
            "p5_beta": float(np.percentile(b, 5)),
            "p95_beta": float(np.percentile(b, 95)),
            "range_beta": float(np.percentile(b, 95) - np.percentile(b, 5)),
            "abs_beta_mean": float(np.abs(b).mean()),
            "min_beta": float(b.min()),
            "max_beta": float(b.max()),
        })
    stats = pd.DataFrame(rows).sort_values("std_beta")
    print(f"  Computed stats for {len(stats)} symbols", flush=True)

    # Label cohorts
    stats["cohort"] = stats["symbol"].apply(
        lambda s: "loser" if s in LOSERS else ("winner" if s in WINNERS else "other")
    )

    # ============ Cohort comparison ============
    print(f"\n=== Cohort comparison ===\n", flush=True)
    print(f"  {'cohort':<7}  {'n':>3}  {'mean_β':>7}  {'std_β':>7}  {'range_β':>8}  "
          f"{'min_β':>7}  {'max_β':>7}", flush=True)
    for c in ["loser", "winner", "other"]:
        sub = stats[stats["cohort"] == c]
        if len(sub) == 0: continue
        print(f"  {c:<7}  {len(sub):>3}  "
              f"{sub['mean_beta'].mean():>+7.3f}  "
              f"{sub['std_beta'].mean():>7.3f}  "
              f"{sub['range_beta'].mean():>8.3f}  "
              f"{sub['min_beta'].min():>+7.3f}  "
              f"{sub['max_beta'].max():>+7.3f}", flush=True)

    # ============ Detailed per-symbol view ============
    print(f"\n=== Per-symbol detail ===\n", flush=True)
    print(f"  {'symbol':<14} {'cohort':<7}  {'mean_β':>7}  {'std_β':>7}  {'range_β':>8}",
          flush=True)
    show = stats[stats["cohort"] != "other"].sort_values("std_beta")
    for _, r in show.iterrows():
        print(f"  {r['symbol']:<14} {r['cohort']:<7}  "
              f"{r['mean_beta']:>+7.3f}  {r['std_beta']:>7.3f}  {r['range_beta']:>8.3f}",
              flush=True)

    # ============ Hypothesis tests ============
    print(f"\n=== Hypothesis tests ===\n", flush=True)
    losers_stats = stats[stats["cohort"] == "loser"]
    winners_stats = stats[stats["cohort"] == "winner"]

    from scipy import stats as scipy_stats
    for metric in ["std_beta", "range_beta", "mean_beta"]:
        l_vals = losers_stats[metric].values
        w_vals = winners_stats[metric].values
        if len(l_vals) > 1 and len(w_vals) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(l_vals, w_vals)
            print(f"  {metric:<15}  losers={l_vals.mean():+.3f}  winners={w_vals.mean():+.3f}  "
                  f"t={t_stat:+.2f}  p={p_val:.3f}", flush=True)

    # ============ Distribution of std_β across full universe ============
    print(f"\n=== β-stability distribution across all 51 symbols ===\n", flush=True)
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    print(f"  std_β quantiles:", flush=True)
    for q in quantiles:
        print(f"    p{int(q*100):>2}: {stats['std_beta'].quantile(q):.3f}", flush=True)

    # ============ Cross-correlation: does std_β correlate with our IC universe ============
    # Symbols with low β-stability — are they already filtered by IC universe?
    print(f"\n=== Top-10 most β-stable and least β-stable symbols ===\n", flush=True)
    print(f"  Most stable (low std_β):", flush=True)
    for _, r in stats.head(10).iterrows():
        cohort_tag = f" [{r['cohort']}]" if r['cohort'] != 'other' else ""
        print(f"    {r['symbol']:<14}  std_β={r['std_beta']:.3f}  mean_β={r['mean_beta']:+.3f}"
              f"{cohort_tag}", flush=True)
    print(f"\n  Least stable (high std_β):", flush=True)
    for _, r in stats.tail(10).iterrows():
        cohort_tag = f" [{r['cohort']}]" if r['cohort'] != 'other' else ""
        print(f"    {r['symbol']:<14}  std_β={r['std_beta']:.3f}  mean_β={r['mean_beta']:+.3f}"
              f"{cohort_tag}", flush=True)

    stats.to_csv(OUT_DIR / "per_symbol_beta_stats.csv", index=False)
    print(f"\n  saved → {OUT_DIR}/per_symbol_beta_stats.csv", flush=True)


if __name__ == "__main__":
    main()
