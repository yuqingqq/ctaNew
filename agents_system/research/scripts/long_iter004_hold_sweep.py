"""LONG-PRED iter-004 — HOLD sweep on H2

iter-003 revealed a ~7 bps/cycle construction loss vs the fresh signal.
Hypothesis: in H2's low-dispersion regime, signal half-life is much shorter
than 24h. The 6-sleeve held book aggregates 5 stale sleeves with the 1 fresh
sleeve, diluting the +5 bps fresh signal with noise from stale positions.

Test: sweep HOLD ∈ {1, 2, 3, 6} and measure both GROSS (cost=0) and NET (cost=4.5)
Sharpe on H2. If HOLD=1 gross Sharpe is positive and HOLD=6 gross is negative,
the construction-mismatch hypothesis is confirmed.

Also report turnover per HOLD (shorter hold → more turnover → more cost) so
the cost/signal trade-off is visible.
"""
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END = pd.Timestamp("2026-05-26",tz="UTC")

def sharpe(p_bps):
    p = p_bps/1e4
    return p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float("nan")

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-004: HOLD sweep on H2 ===\n", flush=True)
    results = []
    for HOLD in [1, 2, 3, 6]:
        for cost_bps in [0, 4.5]:
            tag = f"H{HOLD}_c{cost_bps}"
            print(f"-- HOLD={HOLD} COST={cost_bps} bps/leg --", flush=True)
            env = os.environ.copy()
            env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                        "SIDE_MODE":"default","STRAT_HOLD":str(HOLD),"COST_BPS_LEG":str(cost_bps)})
            res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                                  "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                                 capture_output=True, text=True, env=env)
            if res.returncode != 0:
                print(f"  FAILED: {res.stderr[-300:]}"); continue
            c = pd.read_csv(S/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
            h2 = c[(c["open_time"]>=H2_START)&(c["open_time"]<=H2_END)]
            h1 = c[c["open_time"]<H2_START]
            full_sh = sharpe(c["pnl_bps"]); h1_sh = sharpe(h1["pnl_bps"]); h2_sh = sharpe(h2["pnl_bps"])
            avg_turn = c["turnover"].mean()
            avg_cost = c["cost_bps"].mean()
            avg_gross = c["gross_pnl_bps"].mean()
            results.append(dict(HOLD=HOLD, cost=cost_bps, full_Sh=round(full_sh,3),
                                H1_Sh=round(h1_sh,3), H2_Sh=round(h2_sh,3),
                                turnover=round(avg_turn,3), avg_cost_bps=round(avg_cost,2),
                                avg_gross_bps=round(avg_gross,2),
                                full_totPnL=int(c['pnl_bps'].sum()),
                                H2_totPnL=int(h2['pnl_bps'].sum())))
    df = pd.DataFrame(results)
    print(f"\n=== HOLD × COST SWEEP RESULT (H2 focus) ===")
    print(df.to_string(index=False))

    # Verdict
    print(f"\n=== VERDICT ===")
    h2_h1_c0 = df[(df["HOLD"]==1)&(df["cost"]==0)]["H2_Sh"].iloc[0] if len(df[(df["HOLD"]==1)&(df["cost"]==0)]) else float("nan")
    h2_h6_c0 = df[(df["HOLD"]==6)&(df["cost"]==0)]["H2_Sh"].iloc[0] if len(df[(df["HOLD"]==6)&(df["cost"]==0)]) else float("nan")
    print(f"  H2 gross Sharpe: HOLD=1 → {h2_h1_c0:+.2f}  vs  HOLD=6 → {h2_h6_c0:+.2f}  Δ {h2_h1_c0 - h2_h6_c0:+.2f}")
    if h2_h1_c0 > 0 and h2_h6_c0 < -1:
        print(f"  ✓ CONSTRUCTION-MISMATCH CONFIRMED: shorter hold recovers fresh signal in H2")
        # find best net HOLD
        best_net = df[df["cost"]==4.5].sort_values("H2_Sh", ascending=False).iloc[0]
        print(f"  Best NET H2 Sh: HOLD={int(best_net['HOLD'])} Sh {best_net['H2_Sh']:+.2f} (turnover {best_net['turnover']:.2f}, cost {best_net['avg_cost_bps']:.1f} bps)")
    elif h2_h1_c0 < 0:
        print(f"  ✗ Even HOLD=1 doesn't help — signal genuinely broken at the fresh-prediction level too")
    else:
        print(f"  Intermediate — modest improvement; trade-off binds")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
