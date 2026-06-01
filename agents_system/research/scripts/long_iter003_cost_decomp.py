"""LONG-PRED iter-003 — Cost vs Alpha decomposition

Tests whether H2 underperformance is cost-binding or genuine alpha-loss.

(a) Decompose H2 GROSS vs NET PnL per cycle across Phase 2 variants (A/B/C/D)
(b) Cost-sweep replay at COST_BPS_LEG in {0, 1, 2, 4.5, 9} on variant A (static+default)
(c) Verdict:
    - cost≤2 → Sharpe>0 in H2  → cost is binding
    - even cost=0 → Sharpe ~0 → alpha truly gone in H2
    - intermediate → both contribute
"""
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END = pd.Timestamp("2026-05-26",tz="UTC")

def sharpe(p_bps):
    p = p_bps/1e4
    return p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float("nan")

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-003: Cost vs Alpha decomposition ===\n", flush=True)

    # (a) Gross vs Net decomposition for Phase 2 variants
    print("=== (a) Phase 2 variants: GROSS vs NET H2 Sharpe ===\n")
    for v in ["A_static_default","B_hl14_default","C_static_short_hedge","D_hl14_short_hedge"]:
        path = S/f"phase2_{v}.csv"
        if not path.exists():
            print(f"  {v}: missing {path.name}"); continue
        c = pd.read_csv(path); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
        h2 = c[(c["open_time"]>=H2_START)&(c["open_time"]<=H2_END)]
        if not len(h2): continue
        gross_sh = sharpe(h2["gross_pnl_bps"])
        net_sh = sharpe(h2["pnl_bps"])
        print(f"  {v:<32} n={len(h2):>4}  gross Sh {gross_sh:+.3f}  net Sh {net_sh:+.3f}  "
              f"avg cost {h2['cost_bps'].mean():.2f} bps/cyc  cost/gross={abs(h2['cost_bps'].sum()/max(1,abs(h2['gross_pnl_bps'].sum())))*100:.0f}%")

    # (b) Cost sweep on variant A (static + default)
    print("\n=== (b) COST SWEEP on static_default — what cost level flips H2 Sharpe positive? ===\n")
    results = []
    for cost_bps in [0, 1, 2, 4.5, 9]:
        print(f"  -- COST={cost_bps} bps/leg replay --", flush=True)
        env = os.environ.copy()
        env.update({"PYTHONPATH":str(REPO), "BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                    "SIDE_MODE":"default","COST_BPS_LEG":str(cost_bps)})
        res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                              "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                             capture_output=True, text=True, env=env)
        if res.returncode != 0:
            print(f"  FAILED: {res.stderr[-300:]}"); continue
        c = pd.read_csv(S/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
        h2 = c[(c["open_time"]>=H2_START)&(c["open_time"]<=H2_END)]
        h1 = c[c["open_time"]<H2_START]
        full_sh = sharpe(c["pnl_bps"]); h1_sh = sharpe(h1["pnl_bps"]); h2_sh = sharpe(h2["pnl_bps"])
        results.append(dict(cost_bps=cost_bps, full_Sh=round(full_sh,3),
                            H1_Sh=round(h1_sh,3), H2_Sh=round(h2_sh,3),
                            full_totPnL=int(c['pnl_bps'].sum()), H2_totPnL=int(h2['pnl_bps'].sum())))

    print("\n=== COST SWEEP RESULT ===")
    print(pd.DataFrame(results).to_string(index=False))

    # (c) Verdict
    print("\n=== VERDICT ===")
    df = pd.DataFrame(results)
    h2_at_0 = df[df["cost_bps"]==0]["H2_Sh"].iloc[0] if len(df) else float("nan")
    h2_at_2 = df[df["cost_bps"]==2]["H2_Sh"].iloc[0] if len(df) else float("nan")
    h2_at_45 = df[df["cost_bps"]==4.5]["H2_Sh"].iloc[0] if len(df) else float("nan")
    if h2_at_0 > 1.0:
        print(f"  cost=0 H2 Sharpe = {h2_at_0:+.2f} → ALPHA IS THERE, cost is the binding constraint")
        print(f"  cost=2 H2 Sharpe = {h2_at_2:+.2f} → {'flips positive' if h2_at_2>0 else 'still negative — partial'}")
        print(f"  cost=4.5 H2 Sharpe = {h2_at_45:+.2f} (current) — diagnostic confirms cost dominates")
    elif h2_at_0 < 0.3:
        print(f"  cost=0 H2 Sharpe = {h2_at_0:+.2f} → ALPHA IS TRULY GONE; even free trading doesn't save H2")
    else:
        print(f"  cost=0 H2 Sharpe = {h2_at_0:+.2f} → mixed; some alpha but small")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
