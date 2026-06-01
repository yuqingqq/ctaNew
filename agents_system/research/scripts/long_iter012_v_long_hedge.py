"""LONG-PRED iter-012 — V_LONG_HEDGE: the architecture iter-011 actually supports.

iter-011's honest leg audit revealed: in H2 with hl=14d preds + filter at K=3:
  - Long top-K=3 vs median: +12.98 bps (t=+3.10, SIGNIFICANT)
  - Short bot-K=3 vs median:  -1.30 bps (NS, essentially zero)

V6 drops the WORKING leg (long) and keeps the NON-WORKING leg (short). The data
literally says we should do the OPPOSITE.

V_LONG_HEDGE = K=3 alt LONGS + BTC SHORT hedge (neutralizes basket beta).
This is the symmetric mirror of V6's short_btc_hedge.

Compare V_LONG_HEDGE vs V6 vs V0 (and the "natural" 5L/5S full-filter mix).
"""
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
HL14_PREDS = S/"x132_p2_hl14_full_fullOOS_preds.parquet"
ALLOWLIST = S/"dyn_allow/allow_W180_t0.02.parquet"
H1_START = pd.Timestamp("2025-10-04",tz="UTC"); H1_END = pd.Timestamp("2026-01-22",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC"); H2_END = pd.Timestamp("2026-05-26",tz="UTC")

def sharpe(p):
    p = np.array(p)/1e4
    return float(p.mean()/p.std()*np.sqrt(6*365)) if p.std()>0 else float("nan")

def replay(label, side_mode, use_filter, use_hl14):
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                "SIDE_MODE":side_mode,"STRAT_HOLD":"6","COST_BPS_LEG":"4.5"})
    if use_filter:
        env["CONVEXITY_DYNAMIC_ALLOWLIST_PATH"] = str(ALLOWLIST)
    else:
        env.pop("CONVEXITY_DYNAMIC_ALLOWLIST_PATH", None)
    if use_hl14:
        env["CONVEXITY_PREDS_PATH"] = str(HL14_PREDS)
    else:
        env.pop("CONVEXITY_PREDS_PATH", None)
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  REPLAY FAILED {label}: {res.stderr[-300:]}"); return None
    c = pd.read_csv(S/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    h1c = c[(c["open_time"]>=H1_START)&(c["open_time"]<H1_END)]
    h2c = c[(c["open_time"]>=H2_START)&(c["open_time"]<=H2_END)]
    cum = pd.Series(c["pnl_bps"]).cumsum(); dd = float((cum-cum.cummax()).min())
    return dict(label=label, n=len(c),
                full_Sh=round(sharpe(c["pnl_bps"]),3),
                H1_Sh=round(sharpe(h1c["pnl_bps"]),3),
                H2_Sh=round(sharpe(h2c["pnl_bps"]),3),
                totPnL=int(c["pnl_bps"].sum()),
                H1_totPnL=int(h1c["pnl_bps"].sum()),
                H2_totPnL=int(h2c["pnl_bps"].sum()),
                maxDD=int(dd),
                stop_pct=round(100*c["stop_engaged"].mean(),1))

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-012: V_LONG_HEDGE (what the data actually supports) ===\n", flush=True)
    variants = [
        ("V0_baseline",        "default",          False, False),  # full universe, static, 5L/5S
        ("V6_short_hedge",     "short_btc_hedge",  True,  True),   # V6 (the inverted one)
        ("V_LONG_HEDGE",       "long_btc_hedge",   True,  True),   # NEW: mirror of V6, longs+BTC short hedge
        ("V_LONG_HEDGE_static","long_btc_hedge",   True,  False),  # static preds variant
        ("V_FULL_filtered",    "default",          True,  True),   # full 5L/5S with filter (captures both legs)
    ]
    results = []
    for label, side_mode, use_filter, use_hl14 in variants:
        print(f"-- {label}: side={side_mode} filter={use_filter} hl14={use_hl14} --", flush=True)
        r = replay(label, side_mode, use_filter, use_hl14)
        if r:
            results.append(r)
            print(f"   → full {r['full_Sh']:+.3f}  H1 {r['H1_Sh']:+.3f}  H2 {r['H2_Sh']:+.3f}  "
                  f"totPnL {r['totPnL']:+d}  maxDD {r['maxDD']}  stop {r['stop_pct']}%", flush=True)
    df = pd.DataFrame(results)
    print(f"\n=== COMPARISON ===")
    print(df[["label","full_Sh","H1_Sh","H2_Sh","totPnL","H1_totPnL","H2_totPnL","maxDD","stop_pct"]].to_string(index=False))

    # Verdict
    print(f"\n=== VERDICT ===")
    v6 = df[df["label"]=="V6_short_hedge"].iloc[0]
    vlong = df[df["label"]=="V_LONG_HEDGE"].iloc[0]
    vfull = df[df["label"]=="V_FULL_filtered"].iloc[0]
    print(f"\nV6 (short+BTC hedge):    full {v6['full_Sh']:+.3f}  H1 {v6['H1_Sh']:+.3f}  H2 {v6['H2_Sh']:+.3f}")
    print(f"V_LONG_HEDGE (mirror):   full {vlong['full_Sh']:+.3f}  H1 {vlong['H1_Sh']:+.3f}  H2 {vlong['H2_Sh']:+.3f}")
    print(f"V_FULL filtered (both):  full {vfull['full_Sh']:+.3f}  H1 {vfull['H1_Sh']:+.3f}  H2 {vfull['H2_Sh']:+.3f}")
    if vlong["H2_Sh"] > v6["H2_Sh"] + 0.3:
        print(f"\n✓ V_LONG_HEDGE BEATS V6 by Δ {vlong['H2_Sh']-v6['H2_Sh']:+.2f} in H2 — confirms iter-011")
    elif vfull["full_Sh"] > max(vlong["full_Sh"], v6["full_Sh"]) + 0.15:
        print(f"\n✓ V_FULL (BOTH LEGS with filter) wins — capture both signals")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
