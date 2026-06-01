"""LONG-PRED iter-007 — Architecture stacking test

Tests whether the architectural fixes from earlier iterations COMPOUND or
SATURATE. Available levers:
  - iter-006 (best): per-sym IC filter W=180 τ=0.02 (allowlist parquet)
  - iter-004: HOLD={2, 6}
  - Phase 2: SIDE_MODE={default, short_btc_hedge}

6 variants covering the interactions:
  V0 baseline                 HOLD=6, default, no filter   (reference)
  V1 filter only              HOLD=6, default, +filter
  V2 filter + HOLD=2          HOLD=2, default, +filter
  V3 filter + short_hedge     HOLD=6, short_btc_hedge, +filter
  V4 ALL THREE stacked        HOLD=2, short_btc_hedge, +filter
  V5 HOLD=2 + short_hedge     HOLD=2, short_btc_hedge, no filter (isolate compound)
"""
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
ALLOWLIST = S/"dyn_allow/allow_W180_t0.02.parquet"
H1_START = pd.Timestamp("2025-10-04",tz="UTC"); H1_END = pd.Timestamp("2026-01-22",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC"); H2_END = pd.Timestamp("2026-05-26",tz="UTC")

def sharpe(p):
    p = np.array(p)/1e4
    return float(p.mean()/p.std()*np.sqrt(6*365)) if p.std()>0 else float("nan")

def replay(label, *, hold, side_mode, use_filter):
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                "SIDE_MODE":side_mode,"STRAT_HOLD":str(hold),"COST_BPS_LEG":"4.5"})
    if use_filter: env["CONVEXITY_DYNAMIC_ALLOWLIST_PATH"] = str(ALLOWLIST)
    else: env.pop("CONVEXITY_DYNAMIC_ALLOWLIST_PATH", None)
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  REPLAY FAILED {label}: {res.stderr[-300:]}"); return None
    c = pd.read_csv(S/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    h1c = c[(c["open_time"]>=H1_START)&(c["open_time"]<H1_END)]
    h2c = c[(c["open_time"]>=H2_START)&(c["open_time"]<=H2_END)]
    pb = c["pnl_bps"].values
    cum = pd.Series(pb).cumsum(); dd = float((cum-cum.cummax()).min())
    return dict(label=label, n=len(c),
                full_Sh=round(sharpe(pb),3),
                H1_Sh=round(sharpe(h1c["pnl_bps"]),3),
                H2_Sh=round(sharpe(h2c["pnl_bps"]),3),
                totPnL=int(pb.sum()),
                H1_totPnL=int(h1c["pnl_bps"].sum()),
                H2_totPnL=int(h2c["pnl_bps"].sum()),
                maxDD=int(dd),
                stop_pct=round(100*c["stop_engaged"].mean(),1),
                avg_n=round(c["n_universe"].mean(),0))

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-007: Architecture stacking ===\n", flush=True)
    if not ALLOWLIST.exists():
        print(f"  !!! ALLOWLIST {ALLOWLIST} missing; rerun iter-006 first"); sys.exit(2)
    variants = [
        ("V0_baseline",        dict(hold=6, side_mode="default",         use_filter=False)),
        ("V1_filter_only",     dict(hold=6, side_mode="default",         use_filter=True)),
        ("V2_filter+H2",       dict(hold=2, side_mode="default",         use_filter=True)),
        ("V3_filter+hedge",    dict(hold=6, side_mode="short_btc_hedge", use_filter=True)),
        ("V4_ALL_STACKED",     dict(hold=2, side_mode="short_btc_hedge", use_filter=True)),
        ("V5_H2+hedge_nofilt", dict(hold=2, side_mode="short_btc_hedge", use_filter=False)),
    ]
    results = []
    for label, kwargs in variants:
        print(f"-- {label}: {kwargs} --", flush=True)
        r = replay(label, **kwargs)
        if r:
            results.append(r)
            print(f"   → full Sh {r['full_Sh']:+.3f}  H1 {r['H1_Sh']:+.3f}  H2 {r['H2_Sh']:+.3f}  "
                  f"totPnL {r['totPnL']:+d}", flush=True)
    df = pd.DataFrame(results)
    print(f"\n=== STACKING SWEEP RESULTS ===")
    print(df[["label","full_Sh","H1_Sh","H2_Sh","totPnL","H2_totPnL","maxDD","stop_pct","avg_n"]].to_string(index=False))

    # Verdict
    print(f"\n=== VERDICT — compound vs saturate vs hurt ===")
    if len(df)<6:
        print(f"  partial run; can't fully analyze"); return
    by = df.set_index("label")
    v0, v1, v2, v3, v4, v5 = (by.loc[v] for v in ["V0_baseline","V1_filter_only","V2_filter+H2","V3_filter+hedge","V4_ALL_STACKED","V5_H2+hedge_nofilt"])
    print(f"  H2 Sharpe: V0 {v0['H2_Sh']:+.2f}  V1 {v1['H2_Sh']:+.2f}  V2 {v2['H2_Sh']:+.2f}  "
          f"V3 {v3['H2_Sh']:+.2f}  V4 {v4['H2_Sh']:+.2f}  V5 {v5['H2_Sh']:+.2f}")
    print(f"  Full Sh:   V0 {v0['full_Sh']:+.2f}  V1 {v1['full_Sh']:+.2f}  V2 {v2['full_Sh']:+.2f}  "
          f"V3 {v3['full_Sh']:+.2f}  V4 {v4['full_Sh']:+.2f}  V5 {v5['full_Sh']:+.2f}")
    best_h2 = df.sort_values("H2_Sh", ascending=False).iloc[0]
    best_full = df.sort_values("full_Sh", ascending=False).iloc[0]
    print(f"\n  Best H2: {best_h2['label']} H2 {best_h2['H2_Sh']:+.2f}, full {best_h2['full_Sh']:+.2f}")
    print(f"  Best Full: {best_full['label']} full {best_full['full_Sh']:+.2f}, H2 {best_full['H2_Sh']:+.2f}")
    # V4 vs individual best
    indiv_best_h2 = max(v1["H2_Sh"], v2["H2_Sh"], v3["H2_Sh"])
    if v4["H2_Sh"] > indiv_best_h2 + 0.15:
        print(f"  ✓ COMPOUND: V4 H2 {v4['H2_Sh']:+.2f} > best individual {indiv_best_h2:+.2f}")
    elif v4["H2_Sh"] < indiv_best_h2 - 0.15:
        print(f"  ✗ NEGATIVE INTERACTION: V4 H2 {v4['H2_Sh']:+.2f} < best individual {indiv_best_h2:+.2f}")
    else:
        print(f"  ≈ SATURATE: V4 H2 {v4['H2_Sh']:+.2f} ≈ best individual {indiv_best_h2:+.2f} — fixes address same component")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
