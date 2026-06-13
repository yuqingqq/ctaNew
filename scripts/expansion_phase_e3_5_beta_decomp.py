"""Phase E3.5 v2: per-cycle β-leakage decomposition (correct test).

For each non-skipped trade cycle, decompose the realized spread:
  realized_spread = (mean_alpha_long - mean_alpha_short)           [ALPHA TERM]
                  + (mean_beta_long - mean_beta_short) * basket_fwd [β-LEAKAGE TERM]
                  + noise

If β-leakage is a meaningful share of spread VARIANCE, then β-matching per cycle
would reduce noise → Sharpe improves.

Uses audit panel + basket forward return.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main():
    print(f"=== Phase E3.5 v2: β-leakage decomposition ===\n", flush=True)

    # Load audit panel — has picked flags + alpha_A + return per pick
    audit = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/audit_panel.parquet")
    audit["time"] = pd.to_datetime(audit["time"])
    print(f"  audit: {len(audit):,} rows, {audit['time'].nunique()} cycles", flush=True)

    # Need beta + basket_fwd per (cycle, symbol) — get from feature panel
    feats = pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
        columns=["symbol", "open_time", "beta_short_vs_bk", "basket_fwd",
                  "alpha_A", "return_pct"]
    )
    feats["open_time"] = pd.to_datetime(feats["open_time"])
    feats = feats.rename(columns={"open_time": "time"})

    # Merge: audit rows already restricted to rebalance times; bring beta+basket_fwd
    merged = audit.merge(
        feats[["symbol", "time", "beta_short_vs_bk", "basket_fwd"]],
        on=["symbol", "time"], how="left"
    )

    # For each cycle, compute decomposition
    print(f"\n  Building per-cycle β-leakage decomposition...", flush=True)
    rows = []
    for t, g in merged.groupby("time"):
        long_g = g[g["picked_long"] == 1]
        short_g = g[g["picked_short"] == 1]
        if len(long_g) == 0 or len(short_g) == 0: continue

        # β-mismatch term
        beta_long = float(long_g["beta_short_vs_bk"].mean())
        beta_short = float(short_g["beta_short_vs_bk"].mean())
        basket_fwd = float(long_g["basket_fwd"].iloc[0])  # same per cycle
        beta_leakage_bps = (beta_long - beta_short) * basket_fwd * 1e4

        # Alpha term (using PIT alpha_A column, which is realized residual)
        alpha_long = float(long_g["alpha_A"].mean())
        alpha_short = float(short_g["alpha_A"].mean())
        alpha_term_bps = (alpha_long - alpha_short) * 1e4

        # Realized spread (raw returns)
        ret_long = float(long_g["return_pct"].mean())
        ret_short = float(short_g["return_pct"].mean())
        spread_bps = (ret_long - ret_short) * 1e4

        rows.append({
            "time": t, "spread_bps": spread_bps,
            "alpha_term_bps": alpha_term_bps,
            "beta_leakage_bps": beta_leakage_bps,
            "residual_bps": spread_bps - alpha_term_bps - beta_leakage_bps,
            "beta_long": beta_long, "beta_short": beta_short,
            "beta_diff": beta_long - beta_short,
            "basket_fwd": basket_fwd,
            "n_long": len(long_g), "n_short": len(short_g),
        })

    df = pd.DataFrame(rows)
    print(f"  Decomposed {len(df)} non-skipped cycles", flush=True)

    # Variance decomposition
    print(f"\n=== Variance contributions ===\n", flush=True)
    var_spread = df["spread_bps"].var()
    var_alpha = df["alpha_term_bps"].var()
    var_beta = df["beta_leakage_bps"].var()
    var_resid = df["residual_bps"].var()

    print(f"  Total spread variance:     {var_spread:>12,.0f} bps²", flush=True)
    print(f"    Alpha term variance:     {var_alpha:>12,.0f} bps² "
          f"({var_alpha/var_spread*100:>5.1f}%)", flush=True)
    print(f"    β-leakage variance:      {var_beta:>12,.0f} bps² "
          f"({var_beta/var_spread*100:>5.1f}%)", flush=True)
    print(f"    Residual variance:       {var_resid:>12,.0f} bps² "
          f"({var_resid/var_spread*100:>5.1f}%)", flush=True)

    # Correlation between components
    print(f"\n  Correlation alpha_term vs spread:    "
          f"{df['alpha_term_bps'].corr(df['spread_bps']):+.3f}", flush=True)
    print(f"  Correlation β-leakage vs spread:     "
          f"{df['beta_leakage_bps'].corr(df['spread_bps']):+.3f}", flush=True)

    # Means
    print(f"\n=== Means (per cycle) ===\n", flush=True)
    print(f"  Mean spread:              {df['spread_bps'].mean():+8.2f} bps", flush=True)
    print(f"  Mean alpha_term:          {df['alpha_term_bps'].mean():+8.2f} bps", flush=True)
    print(f"  Mean β-leakage:           {df['beta_leakage_bps'].mean():+8.2f} bps", flush=True)
    print(f"  Mean residual:            {df['residual_bps'].mean():+8.2f} bps", flush=True)

    # β-diff distribution
    print(f"\n=== β-diff distribution (β_long − β_short per cycle) ===\n", flush=True)
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
        print(f"  p{int(q*100):>2}: {df['beta_diff'].quantile(q):+.3f}", flush=True)
    print(f"  mean: {df['beta_diff'].mean():+.3f}", flush=True)
    print(f"  std:  {df['beta_diff'].std():.3f}", flush=True)

    # If we constrained |β_diff| < 0.2, how many cycles survive?
    for thresh in [0.1, 0.2, 0.3, 0.5]:
        kept = df[df["beta_diff"].abs() < thresh]
        kept_sharpe_proxy = kept["spread_bps"].mean() / kept["spread_bps"].std() if len(kept) > 1 else 0
        full_sharpe = df["spread_bps"].mean() / df["spread_bps"].std()
        print(f"\n  If we kept only |β_diff| < {thresh}: {len(kept)}/{len(df)} cycles "
              f"({len(kept)/len(df)*100:.0f}%)", flush=True)
        print(f"    kept Sharpe-proxy (mean/std on spread): {kept_sharpe_proxy:+.3f}  "
              f"(vs all-cycle {full_sharpe:+.3f})", flush=True)
        print(f"    kept mean spread: {kept['spread_bps'].mean():+.2f} bps  "
              f"(vs {df['spread_bps'].mean():+.2f})", flush=True)

    # Save
    df.to_csv(REPO / "outputs/vBTC_beta_stability/per_cycle_beta_decomp.csv", index=False)
    print(f"\n  saved → outputs/vBTC_beta_stability/per_cycle_beta_decomp.csv", flush=True)


if __name__ == "__main__":
    main()
