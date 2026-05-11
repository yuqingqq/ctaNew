"""Signal-width diagnostic: characterize basket vs BTC residuals from
panel data alone (no retraining).

Per cycle, compares:
  - Distribution of realized basket-residual (alpha_realized) vs BTC-residual
  - Oracle top-K minus bot-K spread (max alpha available with perfect rank)
  - Cross-sectional standard deviation per cycle

Answer: how much alpha is THEORETICALLY available in each target signal?
If basket >> BTC by oracle spread, even a perfect predictor on BTC target
captures less alpha than a same-quality predictor on basket target.

Combined with the prior IC test from `alpha_v9_btc_beta_target.py`'s actual
Sharpe results, this characterizes the gap.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_btc_beta_target import add_btc_beta_target

HORIZON = 48
TOP_K = 7
OUT_DIR = REPO / "outputs/btc_target_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building panel + adding β-adjusted BTC target...")
    panel = build_wide_panel()
    panel = add_btc_beta_target(panel)
    panel = panel.sort_values(["open_time", "symbol"]).reset_index(drop=True)

    # Cycle-level analysis (sample at HORIZON cadence to match strategy)
    times_sorted = sorted(panel["open_time"].unique())
    cycle_times = times_sorted[::HORIZON]
    print(f"  {len(cycle_times):,} cycles (HORIZON={HORIZON})")
    cycle_df = panel[panel["open_time"].isin(set(cycle_times))]

    # Per-cycle metrics
    rows = []
    for t, g in cycle_df.groupby("open_time"):
        g_clean = g.dropna(subset=["alpha_realized", "alpha_vs_btc_realized"])
        if len(g_clean) < 2 * TOP_K + 1: continue
        # Oracle spread: top-K minus bot-K when sorted by realized alpha
        sorted_b = g_clean.sort_values("alpha_realized")
        oracle_spread_basket = float(sorted_b.tail(TOP_K)["alpha_realized"].mean()
                                       - sorted_b.head(TOP_K)["alpha_realized"].mean())
        sorted_btc = g_clean.sort_values("alpha_vs_btc_realized")
        oracle_spread_btc = float(sorted_btc.tail(TOP_K)["alpha_vs_btc_realized"].mean()
                                     - sorted_btc.head(TOP_K)["alpha_vs_btc_realized"].mean())
        # Per-cycle dispersion
        std_basket = float(g_clean["alpha_realized"].std())
        std_btc = float(g_clean["alpha_vs_btc_realized"].std())
        # Mean abs residual
        mean_abs_basket = float(g_clean["alpha_realized"].abs().mean())
        mean_abs_btc = float(g_clean["alpha_vs_btc_realized"].abs().mean())
        # Correlation between the two residuals across symbols (cycle-level)
        corr = float(g_clean["alpha_realized"].corr(g_clean["alpha_vs_btc_realized"]))
        rows.append({
            "time": t, "n_syms": len(g_clean),
            "oracle_spread_basket_bps": oracle_spread_basket * 1e4,
            "oracle_spread_btc_bps": oracle_spread_btc * 1e4,
            "std_basket_bps": std_basket * 1e4,
            "std_btc_bps": std_btc * 1e4,
            "mean_abs_basket_bps": mean_abs_basket * 1e4,
            "mean_abs_btc_bps": mean_abs_btc * 1e4,
            "xsec_corr": corr,
        })
    df = pd.DataFrame(rows)
    print(f"  Analyzed {len(df):,} cycles\n")

    print("=" * 100)
    print("SIGNAL WIDTH: BASKET vs BTC-β-ADJUSTED RESIDUALS")
    print("=" * 100)
    print()
    print(f"  {'metric':<40} {'basket':>10}  {'BTC':>10}  {'BTC/basket ratio':>18}")
    metrics = [
        ("Oracle spread per cycle (bps)",       "oracle_spread_basket_bps", "oracle_spread_btc_bps"),
        ("Cross-sectional std per cycle (bps)", "std_basket_bps",           "std_btc_bps"),
        ("Mean |residual| per cycle (bps)",     "mean_abs_basket_bps",      "mean_abs_btc_bps"),
    ]
    for label, b_col, btc_col in metrics:
        b = df[b_col].mean(); btc = df[btc_col].mean()
        ratio = btc / b if b != 0 else float("nan")
        print(f"  {label:<40} {b:>+10.2f}  {btc:>+10.2f}  {ratio:>17.2f}×")

    print()
    print(f"  {'Cross-section corr (basket vs BTC α)':<40} {df['xsec_corr'].mean():>+10.3f}")
    print(f"  (How similar are the two residuals' rankings within each cycle?)")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    spread_b = df["oracle_spread_basket_bps"].mean()
    spread_btc = df["oracle_spread_btc_bps"].mean()
    std_b = df["std_basket_bps"].mean()
    std_btc = df["std_btc_bps"].mean()
    print(f"\n  Oracle spread:    {spread_b:+.1f} bps (basket) vs {spread_btc:+.1f} bps (BTC)")
    print(f"  Per-cycle std:    {std_b:+.1f} bps (basket) vs {std_btc:+.1f} bps (BTC)")
    print()
    if spread_btc > 1.05 * spread_b:
        print(f"  → BTC oracle spread is LARGER than basket. Theoretical alpha is greater")
        print(f"    in BTC residuals; underperformance must be from model not predicting them well.")
    elif spread_btc < 0.95 * spread_b:
        print(f"  → BTC oracle spread is SMALLER than basket. Even with perfect ranking,")
        print(f"    BTC residual gives less alpha to capture per cycle.")
    else:
        print(f"  → Oracle spreads are similar. Underperformance is mostly about")
        print(f"    prediction quality (IC), not signal availability.")

    if std_btc > 1.20 * std_b:
        print(f"  → BTC residuals have higher per-cycle std ({std_btc/std_b:.2f}× basket).")
        print(f"    This is the 'BTC contamination' effect: alts share BTC-direction noise")
        print(f"    that doesn't fully orthogonalize even with proper β-adjustment.")
    elif std_btc < 0.85 * std_b:
        print(f"  → BTC residuals have lower per-cycle std ({std_btc/std_b:.2f}× basket).")
        print(f"    BTC residuals are more compressed cross-sectionally.")
    else:
        print(f"  → Residual variances are comparable.")

    print()
    print(f"  Sharpe-per-spread efficiency:")
    print(f"    (Sharpe) ≈ (mean spread captured) / (std of cycle PnL)")
    print(f"    Per-cycle Sharpe is bounded by: spread × IC / (std × √breadth)")
    print(f"    If both targets had IC = 0.05 (typical), expected basket spread captured")
    print(f"      ≈ {spread_b * 0.05:.2f} bps and BTC ≈ {spread_btc * 0.05:.2f} bps.")

    df.to_csv(OUT_DIR / "signal_width.csv", index=False)
    print(f"\n  saved → {OUT_DIR / 'signal_width.csv'}")


if __name__ == "__main__":
    main()
