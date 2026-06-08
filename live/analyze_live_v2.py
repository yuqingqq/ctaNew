"""Analyze the LIVE convexity-v2 forward-test performance from the real-fill ledger + modeled cycles.
RUN ON THE EXEC SERVER (where the records live; they're gitignored, not on the research box):
    python3 live/analyze_live_v2.py            # auto-reads live/state/convexity_v2
    python3 live/analyze_live_v2.py <book_dir> # e.g. live/state/convexity_v1
Reports: real-fill equity/Sharpe/maxDD, EXECUTION-COST breakdown (slippage/latency/fee/unfilled), regime &
per-cycle records, real-vs-modeled drift, the long-winner gate's live effect, and optimization flags.
"""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

BOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("live/state/convexity_v2")
ANN = np.sqrt(6*365)
def sh(p): p=np.asarray(p,float); return p.mean()/p.std()*ANN if len(p)>2 and p.std()>0 else float("nan")
def dd(eq): eq=np.asarray(eq,float); return float((eq-np.maximum.accumulate(eq)).min())

led_p = BOOK/"realfill"/"ledger.json"
if not led_p.exists():
    print(f"NO real-fill ledger at {led_p} — run on the exec box, or pass the right book dir."); sys.exit(1)
led = json.load(open(led_p)); cyc = pd.DataFrame(led.get("cycles", []))
if cyc.empty: print("ledger has no cycles yet."); sys.exit(0)
cyc["open_time"] = pd.to_datetime(cyc["open_time"], utc=True); cyc = cyc.sort_values("open_time")
eq0 = float(led.get("equity0", 10000.0))

print(f"===== LIVE REAL-FILL PERFORMANCE ({BOOK.name}) =====")
print(f"cycles {len(cyc)}  |  {cyc.open_time.min()} -> {cyc.open_time.max()}  |  equity0 ${eq0:,.0f}")
realized = cyc["realized_pnl"].astype(float)
eqcurve = cyc["equity"].astype(float).to_numpy()
ret = np.diff(np.concatenate([[eq0], eqcurve]))            # per-cycle equity change ($)
print(f"  equity now ${eqcurve[-1]:,.0f}  | realized_cum ${float(led.get('realized_cum',0)):,.1f}  "
      f"unreal ${cyc['unrealized_pnl'].astype(float).iloc[-1]:+,.1f}")
print(f"  total return {(eqcurve[-1]/eq0-1)*100:+.2f}%  | per-cycle Sharpe {sh(ret):+.2f}  | maxDD ${dd(eqcurve):+,.0f} ({dd(eqcurve)/eq0*100:+.1f}%)")
print(f"  win rate {(realized>0).mean()*100:.0f}%  | best {realized.max():+.1f} worst {realized.min():+.1f}")

# Execution-cost breakdown — is slippage/latency eating the edge? (the live-only drag)
print("\n--- EXECUTION COST (the live-only drag; backtest assumed flat 4.5bps/leg) ---")
for c in ["exec_cost_bps","book_slip_bps","latency_drift_bps","fee_bps","basis_bps"]:
    if c in cyc: print(f"  {c:18s}: mean {cyc[c].astype(float).mean():+6.2f}  median {cyc[c].astype(float).median():+6.2f}  p90 {cyc[c].astype(float).quantile(.9):+6.2f}")
if "n_trades" in cyc and "n_unfilled" in cyc:
    nt=cyc["n_trades"].sum(); nu=cyc["n_unfilled"].sum()
    print(f"  fills: {nt-nu}/{nt} ({nu} unfilled, {nu/max(1,nt)*100:.0f}%) — unfilled = missed/partial orders")

# Regime breakdown
if "regime" in cyc:
    print("\n--- by regime ---")
    for r,g in cyc.groupby("regime"):
        rr=g["realized_pnl"].astype(float); print(f"  {r:5s} (n={len(g):3d}): realized ${rr.sum():+7.1f}  mean {rr.mean():+5.1f}  Sharpe {sh(np.diff(np.concatenate([[0],g['equity'].astype(float).to_numpy()]))):+.2f}")

# Worst cycles (optimization flags)
print("\n--- worst 5 real-fill cycles (where it bled — inspect for pattern) ---")
for _,r in cyc.nsmallest(5,"realized_pnl").iterrows():
    extra = f" exec_cost {r.get('exec_cost_bps','?')}bps" if "exec_cost_bps" in r else ""
    print(f"  {r['open_time']} [{r.get('regime','?')}]: realized {float(r['realized_pnl']):+.1f}  unfilled {r.get('n_unfilled','?')}{extra}")

# real vs modeled (does live track the backtest? gate live-vs-modeled consistency)
mc = BOOK/"state"/"cycles.csv"
if mc.exists():
    m=pd.read_csv(mc); m["open_time"]=pd.to_datetime(m["open_time"],utc=True)
    j=cyc[["open_time","realized_pnl"]].merge(m[["open_time","pnl_bps"]],on="open_time",how="inner")
    if len(j)>5:
        print(f"\n--- real-fill vs MODELED (n={len(j)} overlapping) ---")
        print(f"  corr(real $, modeled bps) = {j['realized_pnl'].corr(j['pnl_bps']):+.2f}  (high = live tracks model)")
        print(f"  modeled cycles Sharpe {sh(m['pnl_bps']):+.2f} vs backtest-monthly-PIT expectation +4.22 (long-winner gate)")

print("\nNOTE: forward expectation is wide (universe-overfit + new gate pending live confirmation). Watch: exec-cost")
print("drag vs the +0.33 gate edge, unfilled rate, and whether the long-winner gate is skipping rockets as designed.")
