"""LONG-PRED iter-017 — Why does V_FULL_filtered underperform V0 in specific months?

Two possible root causes to test:
  (A) Filter excludes the big-winner syms: per-sym IC filter removes syms that
      ended up making big PnL in those months
  (B) hl=14d preds make different selections: recency-weighted model picks
      different top-K/bot-K than the static V0 model

Diagnostic:
  1. Load V0 and V_FULL_filtered cycles, focus on bad months (2025-12, 2026-04)
  2. For each cycle, compare V0's selected positions vs V_FULL's
  3. Identify which symbols V0 traded that V_FULL missed (and vice versa)
  4. Check if missed symbols were excluded by filter or by prediction difference
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS_STATIC = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PREDS_HL14 = S/"x132_p2_hl14_full_fullOOS_preds.parquet"
ALLOWLIST = S/"dyn_allow/allow_W180_t0.02.parquet"
V0_CYCLES = S/"iter010_v0_cycles.csv"
VFULL_CYCLES = S/"v_full_filtered_cycles.csv"

def main():
    t0 = time.time()
    print("=== iter-017: Why does V_FULL underperform V0 in specific months? ===\n", flush=True)

    # Load both cycles
    v0 = pd.read_csv(V0_CYCLES, parse_dates=["open_time"])
    vf = pd.read_csv(VFULL_CYCLES, parse_dates=["open_time"])
    v0["open_time"] = pd.to_datetime(v0["open_time"], utc=True)
    vf["open_time"] = pd.to_datetime(vf["open_time"], utc=True)
    # Merge on open_time
    merged = v0[["open_time","pnl_bps","top_k_long","bot_k_short","regime","n_universe","stop_engaged"]].rename(
        columns={"pnl_bps":"v0_pnl","top_k_long":"v0_long","bot_k_short":"v0_short","n_universe":"v0_n","stop_engaged":"v0_stop"}
    ).merge(
        vf[["open_time","pnl_bps","top_k_long","bot_k_short","n_universe","stop_engaged"]].rename(
            columns={"pnl_bps":"vf_pnl","top_k_long":"vf_long","bot_k_short":"vf_short","n_universe":"vf_n","stop_engaged":"vf_stop"}
        ), on="open_time"
    )
    merged["pnl_diff"] = merged["vf_pnl"] - merged["v0_pnl"]
    merged["month"] = merged["open_time"].dt.to_period("M").astype(str)

    # Focus on bad months
    BAD_MONTHS = ["2025-12", "2026-04"]
    GOOD_MONTHS = ["2025-10", "2026-03", "2026-05"]
    for month in BAD_MONTHS + GOOD_MONTHS:
        m = merged[merged["month"]==month]
        if len(m)==0: continue
        print(f"\n=== {month} ({'BAD' if month in BAD_MONTHS else 'GOOD'}: V_FULL pnl={int(m['vf_pnl'].sum()):+d}, V0 pnl={int(m['v0_pnl'].sum()):+d}, diff={int(m['pnl_diff'].sum()):+d}) ===")
        # Top 5 worst cycles for V_FULL relative to V0
        worst = m.nsmallest(5, "pnl_diff")
        print(f"  Top 5 cycles where V_FULL underperformed V0 most:")
        for _, row in worst.iterrows():
            v0_long = str(row["v0_long"])[:60]
            vf_long = str(row["vf_long"])[:60]
            v0_short = str(row["v0_short"])[:60]
            vf_short = str(row["vf_short"])[:60]
            print(f"    {row['open_time']} regime={row['regime']} V0 pnl={row['v0_pnl']:+8.1f} VF pnl={row['vf_pnl']:+8.1f} diff={row['pnl_diff']:+8.1f}")
            print(f"      V0 longs:  {v0_long}")
            print(f"      VF longs:  {vf_long}")
            print(f"      V0 shorts: {v0_short}")
            print(f"      VF shorts: {vf_short}")
            print(f"      V0 universe size: {row['v0_n']:.0f}, VF universe size: {row['vf_n']:.0f}")
            print(f"      V0 stop: {row['v0_stop']}, VF stop: {row['vf_stop']}")
            print()
        # universe sizes
        print(f"\n  Average universe size: V0={m['v0_n'].mean():.0f}, V_FULL={m['vf_n'].mean():.0f} (filtered)")
        print(f"  Stop engagement: V0={100*m['v0_stop'].mean():.1f}%, V_FULL={100*m['vf_stop'].mean():.1f}%")
        # regime breakdown
        print(f"  Regime distribution:")
        for reg, count in m["regime"].value_counts().items():
            print(f"    {reg}: {count} cycles ({100*count/len(m):.0f}%)")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
