"""X110 — What CAUSES the sign flip, and does any market var LEAD it? (basis for leading predictor)

Hypothesis: signal efficacy = sign of cross-sec IC(pred, alpha). The flip = a cross-sectional
momentum<->reversal regime switch. If true, a market state var (BTC vol/trend, funding,
dispersion, realized reversal-strength) should LEAD the efficacy regime, giving a LEADING
sign predictor (vs the lagging trailing-IC which only reacts after the turn).

Steps:
  1. efficacy series: per-cycle cross-sec IC(pred, alpha_A); smoothed eff90 = trailing-90 mean.
  2. market state vars (PIT at t): btc_ret_7d/30d, btc_rvol_7d/30d, xs_ret_disp (std return_1d),
     funding_mean, funding_absmean, funding_disp, realized reversal-strength rev_str
     (= -cross-sec corr(return_1d, return_pct), trailing-smoothed).
  3. lead-lag: corr(var_t, eff90_{t+k}) for k in {0,+45,+90,+180} cycles vs the
     persistence baseline corr(eff90_{t}, eff90_{t+k}). A var that BEATS persistence at k>0
     leads the flip.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
PANEL = REPO/"outputs/vBTC_features/panel_3yr_v5.parquet"
HOLD=6


def main():
    t0=time.time()
    print("=== X110 sign-flip cause + lead analysis ===\n", flush=True)
    p=pd.read_parquet(RC/"x70_v0_3yr_preds.parquet", columns=["symbol","open_time","pred","alpha_A","return_pct"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]
    feats=pd.read_parquet(PANEL, columns=["symbol","open_time","return_1d","funding_rate","btc_ret_7d","btc_ret_30d","btc_rvol_7d","btc_rvol_30d"])
    feats["open_time"]=pd.to_datetime(feats["open_time"],utc=True)
    d=p.merge(feats,on=["symbol","open_time"],how="left")

    # per-cycle market series
    rows=[]
    for ot,g in d.groupby("open_time"):
        gv=g.dropna(subset=["pred","alpha_A"])
        ic=spearmanr(gv["pred"],gv["alpha_A"]).correlation if len(gv)>=8 else np.nan
        gr=g.dropna(subset=["return_1d","return_pct"])
        rev=-spearmanr(gr["return_1d"],gr["return_pct"]).correlation if len(gr)>=8 else np.nan
        rows.append((ot, ic, rev,
                     g["btc_ret_7d"].iloc[0], g["btc_ret_30d"].iloc[0],
                     g["btc_rvol_7d"].iloc[0], g["btc_rvol_30d"].iloc[0],
                     g["return_1d"].std(), g["funding_rate"].mean(),
                     g["funding_rate"].abs().mean(), g["funding_rate"].std()))
    m=pd.DataFrame(rows,columns=["open_time","ic","rev_str","btc_ret7","btc_ret30","btc_rvol7","btc_rvol30",
                                 "xs_disp","fund_mean","fund_absmean","fund_disp"]).set_index("open_time").sort_index()
    m["eff90"]=m["ic"].rolling(90,min_periods=45).mean()
    m["rev_str_s"]=m["rev_str"].rolling(30,min_periods=15).mean()

    # show efficacy regime over time (quarterly)
    print("=== efficacy regime eff90 (smoothed cross-sec IC) by quarter ===")
    q=m["eff90"].resample("QE").mean()
    for ts,v in q.items():
        print(f"  {ts.date()}  eff90={v:+.4f}  {'POSITIVE' if v>0 else 'NEGATIVE/flat'}", flush=True)

    # lead-lag vs persistence baseline
    vars_=["rev_str_s","btc_ret7","btc_ret30","btc_rvol7","btc_rvol30","xs_disp","fund_mean","fund_absmean","fund_disp"]
    lags=[0,45,90,180]
    print(f"\n=== corr(var_t, eff90_(t+k)) — does any var LEAD efficacy? ===")
    print(f"  {'variable':<14}" + "".join(f"k=+{k:<6}" for k in lags))
    # persistence baseline
    base_row="".join(f"{m['eff90'].corr(m['eff90'].shift(-k)):>+8.2f}" for k in lags)
    print(f"  {'eff90(persist)':<14}{base_row}", flush=True)
    print("  " + "-"*46)
    scores={}
    for v in vars_:
        cells=[];
        for k in lags:
            c=m[v].corr(m["eff90"].shift(-k))
            cells.append(c)
        scores[v]=cells[2]  # k=90 lead
        print(f"  {v:<14}" + "".join(f"{c:>+8.2f}" for c in cells), flush=True)

    print(f"\n  (compare each var's k=+90 to persistence k=+90={m['eff90'].corr(m['eff90'].shift(-90)):+.2f};")
    print(f"   a |corr| as large as persistence but from a var KNOWN AT t = a leading predictor.)")
    best=sorted(scores.items(), key=lambda kv:-abs(kv[1]))[:3]
    print(f"  strongest k=+90 leaders: " + ", ".join(f"{v}({c:+.2f})" for v,c in best))
    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
