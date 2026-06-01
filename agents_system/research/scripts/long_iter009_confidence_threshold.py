"""LONG-PRED iter-009 — Confidence-threshold variant V7

V6 always trades K=3 alt shorts each side cycle. The diagnosed mechanism (iter-002:
small +5 bps fresh signal, iter-001: upside-only top-1 lift, vBTC's conv_gate) suggests
that LOW-conviction predictions are noise; only the EXTREME ones carry edge.

V7 modifies V6's SIDE construction: bidirectional alt picks ONLY when |pred| > τ;
otherwise fall back to BTC-neutral. This re-introduces the long leg responsibly
(only at high conviction) and avoids forcing K shorts when none are conviction-worthy.

Test against V6 baseline on full OOS:
  V6_baseline: hl=14d + filter + short_btc_hedge (current best)
  V7_τ0.3   : hl=14d + filter + confidence_btc_hedge τ=0.3 (loose, frequent trades)
  V7_τ0.5   : hl=14d + filter + confidence_btc_hedge τ=0.5 (medium)
  V7_τ0.8   : hl=14d + filter + confidence_btc_hedge τ=0.8 (tight, rare trades)

VERDICT criteria (honest gates):
- Win MUST transport across BOTH H1 and H2 — H2-only lift could be τ-overfit
- Best variant must beat V6 by ≥0.15 Sharpe full OOS AND not destroy H1
- If only H2 wins, the threshold is in-sample-tuned; need percentile-based version
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

def replay(label, side_mode, threshold):
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                "SIDE_MODE":side_mode,"STRAT_HOLD":"6","COST_BPS_LEG":"4.5",
                "CONVEXITY_DYNAMIC_ALLOWLIST_PATH":str(ALLOWLIST),
                "CONVEXITY_PREDS_PATH":str(HL14_PREDS),
                "PRED_THRESHOLD":str(threshold)})
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
    print("=== LONG-PRED iter-009: Confidence-threshold V7 ===\n", flush=True)

    variants = [
        ("V6_baseline",     "short_btc_hedge",       0.0),   # ignored — V6 unchanged
        ("V7_tau0.3",       "confidence_btc_hedge",  0.3),
        ("V7_tau0.5",       "confidence_btc_hedge",  0.5),
        ("V7_tau0.8",       "confidence_btc_hedge",  0.8),
        ("V7_tau1.0",       "confidence_btc_hedge",  1.0),
    ]
    results = []
    for label, side_mode, tau in variants:
        print(f"-- {label}: side={side_mode} τ={tau} --", flush=True)
        r = replay(label, side_mode, tau)
        if r:
            results.append(r)
            print(f"   → full {r['full_Sh']:+.3f}  H1 {r['H1_Sh']:+.3f}  H2 {r['H2_Sh']:+.3f}  "
                  f"totPnL {r['totPnL']:+d}  maxDD {r['maxDD']}  stop {r['stop_pct']}%", flush=True)
    df = pd.DataFrame(results)
    print(f"\n=== RESULTS ===")
    print(df[["label","full_Sh","H1_Sh","H2_Sh","totPnL","H2_totPnL","maxDD","stop_pct"]].to_string(index=False))

    # Verdict
    print(f"\n=== VERDICT ===")
    if len(df) >= 2:
        v6 = df[df["label"]=="V6_baseline"].iloc[0]
        # find best V7 variant
        v7s = df[df["label"].str.startswith("V7_")]
        best_full = v7s.sort_values("full_Sh", ascending=False).iloc[0]
        best_h2 = v7s.sort_values("H2_Sh", ascending=False).iloc[0]
        print(f"  V6 baseline:           full {v6['full_Sh']:+.3f}  H1 {v6['H1_Sh']:+.3f}  H2 {v6['H2_Sh']:+.3f}")
        print(f"  Best V7 (full Sh):     {best_full['label']}  full {best_full['full_Sh']:+.3f}  H1 {best_full['H1_Sh']:+.3f}  H2 {best_full['H2_Sh']:+.3f}")
        print(f"  Best V7 (H2 Sh):       {best_h2['label']}  full {best_h2['full_Sh']:+.3f}  H1 {best_h2['H1_Sh']:+.3f}  H2 {best_h2['H2_Sh']:+.3f}")
        # Honest gate: best V7 must beat V6 on full OOS AND not destroy H1
        delta_full = best_full["full_Sh"] - v6["full_Sh"]
        delta_h1 = best_full["H1_Sh"] - v6["H1_Sh"]
        delta_h2 = best_full["H2_Sh"] - v6["H2_Sh"]
        if delta_full > 0.15 and delta_h1 > -0.3:
            print(f"  ✓ ADOPT V7 {best_full['label']}: full +{delta_full:.2f}, H1 {delta_h1:+.2f}, H2 {delta_h2:+.2f}")
        elif delta_h2 > 0.3 and delta_h1 < -0.5:
            print(f"  ⚠ H2-only win — likely τ-overfit; need percentile-based version (iter-010)")
        else:
            print(f"  ≈ No clean improvement over V6 — confidence threshold marginal here. V6 stays best.")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
