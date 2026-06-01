"""LONG-PRED iter-014 — Clean leg-by-leg audit, no confusion

ONE simple question:
  Per cycle, in absolute realized returns, does the LONG leg make money?
  Per cycle, in absolute realized returns, does the SHORT leg make money?
  In each regime (H1 vs H2), with hl=14d preds (production model).

Definitions (no "vs median" tricks):
  Long leg per cycle:  mean of top-K returns. If positive → long made money.
  Short leg per cycle: −mean of bot-K returns. If positive → short made money.
  Long-Short basket:   top_mean − bot_mean (= spread). If positive → basket profitable.

Measure: gross expectation, std, t-stat. Then subtract realistic cost.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import ttest_1samp
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
PREDS_HL14 = S/"x132_p2_hl14_full_fullOOS_preds.parquet"
ALLOWLIST = S/"dyn_allow/allow_W180_t0.02.parquet"
H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
# Cost: ~3 bps per leg per cycle in 6-sleeve held book (approximately)
# Actually we'll subtract it explicitly per leg trade
COST_BPS_PER_LEG_TRADE = 4.5
TURNS_PER_CYCLE = 1/6   # rough: each leg turns over 1/HOLD per cycle in steady state

def main():
    print("=== LONG-PRED iter-014: Clean leg audit ===\n", flush=True)

    preds = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred","return_pct"])
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds = preds[(preds["open_time"].dt.hour%4==0) & (preds["open_time"].dt.minute==0)]
    allowlist = pd.read_parquet(ALLOWLIST) if ALLOWLIST.exists() else None
    if allowlist is not None:
        allowlist["open_time"] = pd.to_datetime(allowlist["open_time"], utc=True)
        allow_set = set((r.open_time, r.symbol) for _, r in allowlist.iterrows())
    else:
        allow_set = None

    def measure(df, K, use_filter):
        if use_filter:
            df = df[df.apply(lambda r: (r["open_time"], r["symbol"]) in allow_set, axis=1)]
        per_cycle = []
        for ot, g in df.groupby("open_time"):
            if len(g) < 2*K: continue
            g = g.sort_values("pred")
            top_mean = g.tail(K)["return_pct"].mean()
            bot_mean = g.head(K)["return_pct"].mean()
            per_cycle.append((ot, top_mean, bot_mean))
        return pd.DataFrame(per_cycle, columns=["open_time","top_mean","bot_mean"])

    def stats(arr, label):
        arr_bps = arr * 1e4
        mean = arr_bps.mean()
        se = arr_bps.std() / np.sqrt(len(arr_bps))
        t = mean / se if se>0 else float("nan")
        return dict(label=label, mean_bps=mean, se=se, t=t, n=len(arr_bps))

    # Run for full universe and filtered universe, K=5 and K=3, H1 and H2
    print(f"{'configuration':<40} {'mean_bps':>10} {'SE':>8} {'t-stat':>8} {'sig':>4}")
    print("-"*72)
    for use_filter, label_filt in [(False, "FULL"), (True, "FILT")]:
        for K in [5, 3]:
            res = measure(preds, K=K, use_filter=use_filter)
            res["open_time"] = pd.to_datetime(res["open_time"], utc=True)
            for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
                sub = res[(res["open_time"]>=s)&(res["open_time"]<e)]
                if len(sub)<10: continue
                # Long leg per cycle = top_mean (realized return of long position)
                long_pnl = sub["top_mean"].values
                # Short leg per cycle = -bot_mean (realized PnL of short position)
                short_pnl = -sub["bot_mean"].values
                # L-S basket per cycle = top_mean - bot_mean
                basket = sub["top_mean"].values - sub["bot_mean"].values

                for arr, leg_label in [(long_pnl, "Long-only"), (short_pnl, "Short-only"), (basket, "L-S basket")]:
                    s_ = stats(arr, "")
                    sig = "★★" if abs(s_["t"])>2.58 else "★" if abs(s_["t"])>1.96 else ""
                    print(f"  {label_filt:<6} K={K} {period_label} {leg_label:<14} ({s_['n']:>3} cyc)  "
                          f"{s_['mean_bps']:>+8.2f}  {s_['se']:>7.2f}  {s_['t']:>+7.2f}  {sig:>4}")
                print()

    # Cost-adjusted view
    print("\n" + "="*72)
    print("=== COST-ADJUSTED (subtract 1 bp per leg per cycle steady-state) ===\n")
    print(f"Approximation: 6-sleeve held book → ~1 bp/cycle cost per leg traded\n")
    # show production-relevant config: FILT, K=5, both halves
    print(f"{'leg':<15} {'H1 gross':>10} {'H1 net':>10}  {'H2 gross':>10} {'H2 net':>10}")
    print("-"*60)
    res5 = measure(preds, K=5, use_filter=True)
    res5["open_time"] = pd.to_datetime(res5["open_time"], utc=True)
    for period_label, (s, e), col in [("H1", (H1_START, H2_START), 0), ("H2", (H2_START, H2_END), 1)]:
        sub = res5[(res5["open_time"]>=s)&(res5["open_time"]<e)]
        long_pnl_h = sub["top_mean"].mean()*1e4
        short_pnl_h = -sub["bot_mean"].mean()*1e4
        basket_h = (sub["top_mean"]-sub["bot_mean"]).mean()*1e4
        # Net = gross - 1 bps per leg per cycle
        if col==0:
            row_long = [long_pnl_h, long_pnl_h-1]
            row_short = [short_pnl_h, short_pnl_h-1]
            row_basket = [basket_h, basket_h-2]
        else:
            row_long.extend([long_pnl_h, long_pnl_h-1])
            row_short.extend([short_pnl_h, short_pnl_h-1])
            row_basket.extend([basket_h, basket_h-2])

    print(f"{'Long-only':<15} {row_long[0]:>+8.2f}  {row_long[1]:>+8.2f}    {row_long[2]:>+8.2f}  {row_long[3]:>+8.2f}")
    print(f"{'Short-only':<15} {row_short[0]:>+8.2f}  {row_short[1]:>+8.2f}    {row_short[2]:>+8.2f}  {row_short[3]:>+8.2f}")
    print(f"{'L-S basket':<15} {row_basket[0]:>+8.2f}  {row_basket[1]:>+8.2f}    {row_basket[2]:>+8.2f}  {row_basket[3]:>+8.2f}")

    print(f"\n=== BOTTOM-LINE ===\n")
    print(f"In ABSOLUTE realized terms (no median tricks):")
    print(f"  Long leg gross PnL (top-K=5 mean realized return):")
    print(f"    H1: {row_long[0]:+.2f} bps/cycle")
    print(f"    H2: {row_long[2]:+.2f} bps/cycle")
    print(f"  Short leg gross PnL (-bot-K=5 mean realized return):")
    print(f"    H1: {row_short[0]:+.2f} bps/cycle")
    print(f"    H2: {row_short[2]:+.2f} bps/cycle")
    print(f"  Long-short basket (top - bot mean per cycle):")
    print(f"    H1: {row_basket[0]:+.2f} bps/cycle")
    print(f"    H2: {row_basket[2]:+.2f} bps/cycle")

if __name__=="__main__": main()
