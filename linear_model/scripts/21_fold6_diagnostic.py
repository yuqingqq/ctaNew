"""Step 21: What's special about fold 6?

Investigate:
  1. Fold 6 time range
  2. Per-cycle PnL within fold 6 for R3_BTC vs R3
  3. Top contributing cycles within fold 6
  4. Symbols picked during fold 6
  5. BTC price + vol regime during fold 6
  6. α_β distribution during fold 6 vs other folds
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

OUT = REPO / "linear_model/results"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"

# Per-cycle CSVs
R3_BTC_B = OUT / "r3_btc_backtest_B_IC_signed.csv"
R3_B = OUT / "r3_backtest_B_IC_signed.csv"


def main():
    print("=== Step 21: Fold 6 diagnostic ===\n", flush=True)

    # ===== Per-cycle PnL by fold =====
    r3_btc = pd.read_csv(R3_BTC_B)
    r3_btc["time"] = pd.to_datetime(r3_btc["time"], utc=True)
    r3 = pd.read_csv(R3_B)
    r3["time"] = pd.to_datetime(r3["time"], utc=True)

    print("Per-fold breakdown:", flush=True)
    print(f"  {'fold':>4}  {'time range':>40}  {'R3 PnL':>10}  {'R3_BTC PnL':>12}  "
          f"{'Δ':>10}", flush=True)
    for fold in range(1, 10):
        g_btc = r3_btc[r3_btc["fold"]==fold]
        g_r3 = r3[r3["fold"]==fold]
        if len(g_btc) == 0: continue
        t_min, t_max = g_btc["time"].min(), g_btc["time"].max()
        pnl_btc = g_btc["net_pnl_bps"].sum()
        pnl_r3 = g_r3["net_pnl_bps"].sum() if len(g_r3) > 0 else np.nan
        d = pnl_btc - pnl_r3 if not pd.isna(pnl_r3) else np.nan
        marker = "  ← fold 6" if fold == 6 else ""
        print(f"  {fold:>4}  {str(t_min)[:10]} → {str(t_max)[:10]}  "
              f"{pnl_r3:>+10.0f}  {pnl_btc:>+12.0f}  {d:>+10.0f}{marker}", flush=True)

    # ===== Top cycles in fold 6 =====
    print(f"\n\nTop 10 cycles in fold 6 for R3_BTC (by net_pnl_bps):", flush=True)
    f6 = r3_btc[r3_btc["fold"]==6].sort_values("net_pnl_bps", ascending=False)
    print(f"  {'time':>20}  {'gross':>8}  {'cost':>6}  {'net':>8}", flush=True)
    for _, row in f6.head(10).iterrows():
        print(f"  {str(row['time'])[:19]}  {row['gross_pnl_bps']:>+8.1f}  "
              f"{row['cost_bps']:>+6.2f}  {row['net_pnl_bps']:>+8.1f}", flush=True)

    print(f"\n  Bottom 10 cycles in fold 6 (worst losses):", flush=True)
    for _, row in f6.tail(10).iterrows():
        print(f"  {str(row['time'])[:19]}  {row['gross_pnl_bps']:>+8.1f}  "
              f"{row['cost_bps']:>+6.2f}  {row['net_pnl_bps']:>+8.1f}", flush=True)

    # ===== BTC price + vol during fold 6 =====
    print(f"\n\nBTC market conditions in fold 6 vs other folds:", flush=True)
    fold_ranges = {}
    for fold in range(1, 10):
        g = r3_btc[r3_btc["fold"]==fold]
        if len(g) == 0: continue
        fold_ranges[fold] = (g["time"].min(), g["time"].max())

    # Load BTC 5m klines for time-range stats
    btc_files = sorted(KLINES.glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close","quote_volume"])
                      for f in btc_files], ignore_index=True)
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.sort_values("open_time")
    btc["ret"] = btc["close"].pct_change()
    btc["rvol_24h"] = btc["ret"].rolling(288).std() * np.sqrt(288)  # daily vol

    print(f"  {'fold':>4}  {'days':>5}  {'BTC start':>10}  {'BTC end':>10}  "
          f"{'BTC chg':>9}  {'avg vol':>10}  {'max draw':>10}", flush=True)
    for fold, (t0, t1) in fold_ranges.items():
        sub = btc[(btc["open_time"] >= t0) & (btc["open_time"] <= t1)]
        if len(sub) < 100: continue
        c_start = float(sub["close"].iloc[0]); c_end = float(sub["close"].iloc[-1])
        chg = (c_end / c_start - 1) * 100
        avg_vol = float(sub["rvol_24h"].mean()) * 100
        # Max drawdown from start
        peak = sub["close"].cummax()
        dd = ((sub["close"] / peak - 1) * 100).min()
        days = (t1 - t0).days
        marker = "  ← fold 6" if fold == 6 else ""
        print(f"  {fold:>4}  {days:>5}  ${c_start:>9.0f}  ${c_end:>9.0f}  "
              f"{chg:>+8.1f}%  {avg_vol:>+9.1f}%  {dd:>+9.1f}%{marker}", flush=True)

    # ===== R3_BTC predictions in fold 6 — does the model itself act differently? =====
    print(f"\n\nR3_BTC prediction distribution by fold:", flush=True)
    preds = pd.read_parquet(REPO / "linear_model/results/ridge_r3_btc_preds.parquet")
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds = preds[preds["fold"].isin(range(1, 10))]
    preds["alpha_bps"] = preds["alpha_beta"] * 1e4
    # Per-fold IC of pred_z vs alpha_beta
    print(f"  {'fold':>4}  {'n_pred':>8}  {'pred_std':>9}  {'alpha_std':>10}  "
          f"{'per-cycle IC':>14}", flush=True)
    for fold in range(1, 10):
        g = preds[preds["fold"]==fold]
        if len(g) == 0: continue
        cyc_ic = g.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda gg: gg["pred_z"].rank().corr(gg["alpha_beta"].rank())
            if len(gg) >= 5 else np.nan).dropna()
        marker = "  ← fold 6" if fold == 6 else ""
        print(f"  {fold:>4}  {len(g):>8,}  {g['pred_z'].std():>9.4f}  "
              f"{g['alpha_bps'].std():>10.1f}  {cyc_ic.mean():>+14.4f}{marker}",
              flush=True)


if __name__ == "__main__":
    main()
