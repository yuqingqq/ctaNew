"""LONG-PRED iter-008 — Compound V3 with hl=14d retrained preds

V3 (current best from iter-007): static preds + per-sym IC filter W180 τ=0.02
                                 + SIDE_MODE=short_btc_hedge + HOLD=6
V6 (test): same architecture but using hl=14d retrained preds

iter-001/002 showed hl=14d preds have ~similar IC but better top-1 fresh
signal AND substantially better short-side edge (+19.9 bps vs +8 bps at K=3).
V3's architecture uses K=3 shorts via short_btc_hedge — so the question is:
does hl=14d's stronger short selection compound with V3's filter+hedge?

If yes: V6 is the final production spec (model + architecture both optimized).
If no: V3 is final, simpler operationally (uses standard static preds).
"""
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
STATIC_PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
HL14_PREDS = S/"x132_p2_hl14_full_fullOOS_preds.parquet"
ALLOWLIST = S/"dyn_allow/allow_W180_t0.02.parquet"
H1_START = pd.Timestamp("2025-10-04",tz="UTC"); H1_END = pd.Timestamp("2026-01-22",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC"); H2_END = pd.Timestamp("2026-05-26",tz="UTC")

def sharpe(p):
    p = np.array(p)/1e4
    return float(p.mean()/p.std()*np.sqrt(6*365)) if p.std()>0 else float("nan")

def replay(label, preds_path):
    env = os.environ.copy()
    env.update({
        "PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
        "SIDE_MODE":"short_btc_hedge","STRAT_HOLD":"6","COST_BPS_LEG":"4.5",
        "CONVEXITY_DYNAMIC_ALLOWLIST_PATH":str(ALLOWLIST),
        "CONVEXITY_PREDS_PATH":str(preds_path),
    })
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  REPLAY FAILED {label}: {res.stderr[-400:]}"); return None
    c = pd.read_csv(S/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    h1c = c[(c["open_time"]>=H1_START)&(c["open_time"]<H1_END)]
    h2c = c[(c["open_time"]>=H2_START)&(c["open_time"]<=H2_END)]
    cum = pd.Series(c["pnl_bps"]).cumsum(); dd = float((cum-cum.cummax()).min())
    return dict(label=label, n=len(c),
                full_Sh=round(sharpe(c["pnl_bps"]),3),
                H1_Sh=round(sharpe(h1c["pnl_bps"]),3),
                H2_Sh=round(sharpe(h2c["pnl_bps"]),3),
                totPnL=int(c["pnl_bps"].sum()),
                H2_totPnL=int(h2c["pnl_bps"].sum()),
                maxDD=int(dd),
                stop_pct=round(100*c["stop_engaged"].mean(),1))

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-008: Compound V3 + hl=14d ===\n", flush=True)
    for path in [STATIC_PREDS, HL14_PREDS, ALLOWLIST]:
        if not path.exists():
            print(f"  !!! MISSING {path}"); sys.exit(2)

    results = []
    for label, preds in [("V3_static_preds", STATIC_PREDS), ("V6_hl14_preds", HL14_PREDS)]:
        print(f"-- {label}: preds={preds.name} --", flush=True)
        r = replay(label, preds)
        if r:
            results.append(r)
            print(f"   → full {r['full_Sh']:+.3f}  H1 {r['H1_Sh']:+.3f}  H2 {r['H2_Sh']:+.3f}  "
                  f"totPnL {r['totPnL']:+d}  maxDD {r['maxDD']}  stop {r['stop_pct']}%", flush=True)
    df = pd.DataFrame(results)
    print(f"\n=== COMPARISON ===")
    print(df.to_string(index=False))

    if len(df)==2:
        v3 = df[df["label"]=="V3_static_preds"].iloc[0]
        v6 = df[df["label"]=="V6_hl14_preds"].iloc[0]
        delta_h2 = v6["H2_Sh"] - v3["H2_Sh"]
        delta_full = v6["full_Sh"] - v3["full_Sh"]
        print(f"\n=== VERDICT ===")
        print(f"  V3 → V6 H2 Sh:   {v3['H2_Sh']:+.3f} → {v6['H2_Sh']:+.3f}  Δ {delta_h2:+.3f}")
        print(f"  V3 → V6 full Sh: {v3['full_Sh']:+.3f} → {v6['full_Sh']:+.3f}  Δ {delta_full:+.3f}")
        if delta_h2 >= 0.3:
            print(f"  ✓ COMPOUND: V6 is the final deploy spec (hl=14d adds +0.3+ H2 Sh on V3)")
        elif delta_h2 <= -0.3:
            print(f"  ✗ HL14 HURTS: V3 is final (static preds simpler operationally)")
        else:
            print(f"  ≈ SATURATE: V3 is final (no meaningful compound; simpler is better)")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
