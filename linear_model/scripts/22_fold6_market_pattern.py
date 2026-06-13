"""Step 22: What actually happened in fold 6's mid-January 2026 window?

The lift comes from 5 trading days: 2026-01-17 to 2026-01-22.
Look at:
  1. BTC price action during that window
  2. Per-symbol alt returns — was there rotation?
  3. Cross-sectional return dispersion
  4. Which symbols did R3_BTC long/short?
  5. Funding rates regime
  6. What story does this paint?
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
R3_BTC_B = REPO / "linear_model/results/r3_btc_backtest_B_IC_signed.csv"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

# The 5-day winning window in fold 6
WIN_START = pd.Timestamp("2026-01-15", tz="UTC")
WIN_END = pd.Timestamp("2026-01-25", tz="UTC")


def main():
    print("=== Step 22: What happened in fold 6's mid-Jan 2026 window? ===\n",
          flush=True)
    print(f"Window: {WIN_START.date()} to {WIN_END.date()}\n", flush=True)

    # ===== 1. BTC price action =====
    btc_files = sorted((KLINES_DIR / "BTCUSDT" / "5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close","volume"])
                      for f in btc_files], ignore_index=True)
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.sort_values("open_time")
    btc_win = btc[(btc.open_time >= WIN_START) & (btc.open_time <= WIN_END)]
    print(f"--- BTC price action ---", flush=True)
    print(f"  Start: ${btc_win.close.iloc[0]:.0f}", flush=True)
    print(f"  End:   ${btc_win.close.iloc[-1]:.0f}", flush=True)
    print(f"  Change: {(btc_win.close.iloc[-1]/btc_win.close.iloc[0]-1)*100:+.2f}%", flush=True)
    print(f"  Max:   ${btc_win.close.max():.0f}", flush=True)
    print(f"  Min:   ${btc_win.close.min():.0f}", flush=True)
    print(f"  Intraday range: {(btc_win.close.max()/btc_win.close.min()-1)*100:+.2f}%",
          flush=True)
    # Daily summary
    btc_win["date"] = btc_win["open_time"].dt.floor("1D")
    daily = btc_win.groupby("date")["close"].agg(["first","last","min","max"])
    daily["chg_pct"] = (daily["last"]/daily["first"]-1)*100
    daily["range_pct"] = (daily["max"]/daily["min"]-1)*100
    print(f"\n  Daily breakdown:", flush=True)
    for d, r in daily.iterrows():
        print(f"    {str(d)[:10]}  open ${r['first']:.0f}  close ${r['last']:.0f}  "
              f"chg {r['chg_pct']:+.2f}%  range {r['range_pct']:.2f}%", flush=True)

    # ===== 2. Alt-coin behavior during this window =====
    print(f"\n\n--- Alt-coin behavior (vs BTC) ---", flush=True)
    panel = pd.read_parquet(PANEL,
        columns=["symbol","open_time","return_pct","funding_rate"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)

    win_panel = panel[(panel.open_time >= WIN_START) & (panel.open_time <= WIN_END)]
    # Aggregate per-symbol stats
    sym_stats = []
    for sym, g in win_panel.groupby("symbol"):
        if sym == "BTCUSDT": continue
        if len(g) < 100: continue
        # cumulative return = (1 + r1)(1+r2)... - 1 over the window's first row to last
        # But return_pct is 4h forward return. Use bar-to-bar from kline data instead.
        sym_files = sorted((KLINES_DIR / sym / "5m").glob("*.parquet"))
        if not sym_files: continue
        sym_btc = pd.concat([pd.read_parquet(f, columns=["open_time","close","quote_volume"])
                              for f in sym_files], ignore_index=True)
        sym_btc["open_time"] = pd.to_datetime(sym_btc["open_time"], utc=True)
        sym_w = sym_btc[(sym_btc.open_time >= WIN_START) &
                         (sym_btc.open_time <= WIN_END)]
        if len(sym_w) < 100: continue
        chg = (sym_w.close.iloc[-1] / sym_w.close.iloc[0] - 1) * 100
        rng = (sym_w.close.max() / sym_w.close.min() - 1) * 100
        vol_5m = sym_w.close.pct_change().std() * np.sqrt(288) * 100  # daily vol
        avg_funding = g.funding_rate.mean() * 1e4 * 365  # annualized bps
        sym_stats.append({"symbol":sym, "chg_pct":chg, "range_pct":rng,
                          "daily_vol":vol_5m, "ann_funding_bps":avg_funding})
    df_sym = pd.DataFrame(sym_stats).sort_values("chg_pct", ascending=False)

    print(f"  {'symbol':<14} {'chg %':>8} {'range %':>9} {'vol %':>7} {'fund bps/yr':>12}",
          flush=True)
    print(f"  TOP 10 GAINERS:", flush=True)
    for _, r in df_sym.head(10).iterrows():
        print(f"  {r['symbol']:<14} {r['chg_pct']:>+8.2f} {r['range_pct']:>9.2f} "
              f"{r['daily_vol']:>7.2f} {r['ann_funding_bps']:>+12.0f}", flush=True)
    print(f"\n  BOTTOM 10 LOSERS:", flush=True)
    for _, r in df_sym.tail(10).iterrows():
        print(f"  {r['symbol']:<14} {r['chg_pct']:>+8.2f} {r['range_pct']:>9.2f} "
              f"{r['daily_vol']:>7.2f} {r['ann_funding_bps']:>+12.0f}", flush=True)

    print(f"\n  Overall stats during window:", flush=True)
    print(f"    BTC change: {(btc_win.close.iloc[-1]/btc_win.close.iloc[0]-1)*100:+.2f}%",
          flush=True)
    print(f"    Median alt change: {df_sym.chg_pct.median():+.2f}%", flush=True)
    print(f"    Mean alt change: {df_sym.chg_pct.mean():+.2f}%", flush=True)
    print(f"    Top-vs-bottom spread: {df_sym.chg_pct.iloc[0]-df_sym.chg_pct.iloc[-1]:+.2f}%",
          flush=True)
    print(f"    Std of alt changes: {df_sym.chg_pct.std():.2f}%", flush=True)

    # ===== 3. R3_BTC's picks during this window =====
    print(f"\n\n--- R3_BTC's actual picks during the winning cycles ---", flush=True)
    # Need pick-level info — load from r3_btc preds or backtest CSV
    r3_btc = pd.read_csv(R3_BTC_B)
    r3_btc["time"] = pd.to_datetime(r3_btc["time"], utc=True)
    win = r3_btc[(r3_btc.time >= WIN_START) & (r3_btc.time <= WIN_END)
                  & (r3_btc.fold == 6)]
    print(f"  Cycles in window: {len(win)}", flush=True)
    print(f"  Total PnL: {win.net_pnl_bps.sum():+.0f} bps", flush=True)
    print(f"  Avg gross/cycle: {win.gross_pnl_bps.mean():+.2f} bps", flush=True)
    print(f"  Avg cost/cycle:  {win.cost_bps.mean():+.2f} bps", flush=True)
    print(f"  Avg turnover:    {win.turnover.mean():.2f}", flush=True)

    # Per-day total
    win = win.copy()
    win["date"] = win["time"].dt.floor("1D")
    daily_pnl = win.groupby("date").agg(net=("net_pnl_bps","sum"),
                                          n_cycles=("net_pnl_bps","count")).sort_index()
    print(f"\n  Daily PnL during winning window:", flush=True)
    for d, r in daily_pnl.iterrows():
        print(f"    {str(d)[:10]}  PnL = {r['net']:+8.0f} bps  ({r['n_cycles']:.0f} cycles)",
              flush=True)


if __name__ == "__main__":
    main()
