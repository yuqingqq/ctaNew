"""LONG-PRED iter-044 — Regime diagnosis: H1 vs H2, and which PIT signal flags 'model works'.

Verify the user's hypothesis: both H1/H2 are bear, difference is H1 has stronger vol/dispersion.
Then find the best PIT switch signal for the classifier = the observable most correlated with
the model's per-cycle long alpha.

Per-cycle metrics:
  btc_rvol_7d        (market vol; PIT feature)
  market_ret         (median alt return = market proxy)
  xs_ret_disp        (std of realized returns across syms)
  pred_disp          (std of model pred across syms; PIT, known at decision)
  long_IC            (spearman pred vs realized return)
  long_alpha         (top-K=5 mean - median; the thing we want to predict)

Switch-signal power: for each PIT candidate (btc_rvol_7d, pred_disp, trailing xs_disp,
trailing long_IC, trailing long_alpha), Spearman vs current long_alpha + quartile spread.
The signal where high-bucket=good-alpha / low-bucket=bad-alpha is the switch signal.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
K=5

def main():
    t0=time.time()
    print("=== iter-044: Regime diagnosis H1 vs H2 + switch-signal search ===\n", flush=True)
    p=pd.read_parquet(PREDS, columns=["symbol","open_time","return_pct","pred"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p.open_time>=H1S)&(p.open_time<H2E)&(p.open_time.dt.hour%4==0)&(p.open_time.dt.minute==0)]
    pf=pd.read_parquet(PANEL, columns=["symbol","open_time","btc_rvol_7d"])
    pf["open_time"]=pd.to_datetime(pf["open_time"],utc=True)
    d=p.merge(pf,on=["symbol","open_time"],how="left")

    # per-cycle metrics
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<2*K: continue
        med=g["return_pct"].median()
        ic=spearmanr(g["pred"],g["return_pct"])[0]
        la=(g.nlargest(K,"pred")["return_pct"].mean()-med)*1e4
        rows.append(dict(open_time=ot, btc_rvol_7d=g["btc_rvol_7d"].median(),
                         market_ret=med*1e4, xs_ret_disp=g["return_pct"].std()*1e4,
                         pred_disp=g["pred"].std(), long_IC=ic, long_alpha=la))
    c=pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)
    c["regime"]=np.where(c.open_time<H2S,"H1","H2")

    # ---- A: H1 vs H2 regime characterization ----
    print("=== A: H1 vs H2 means (verify 'both bear, H1 stronger vol/dispersion') ===\n")
    agg=c.groupby("regime").agg(btc_rvol=("btc_rvol_7d","mean"), mkt_ret=("market_ret","mean"),
        xs_disp=("xs_ret_disp","mean"), pred_disp=("pred_disp","mean"),
        long_IC=("long_IC","mean"), long_alpha=("long_alpha","mean"), n=("long_alpha","size"))
    print(agg.to_string(float_format=lambda x:f"{x:+.4f}"))
    print(f"\n  market_ret in bps/cycle (both negative = bear?); btc_rvol & xs_disp = vol/dispersion")
    h1,h2=agg.loc["H1"],agg.loc["H2"]
    print(f"\n  Verdict:")
    print(f"    both bear? H1 mkt_ret {h1.mkt_ret:+.1f}, H2 {h2.mkt_ret:+.1f} bps/cyc -> {'BOTH BEAR' if h1.mkt_ret<0 and h2.mkt_ret<0 else 'NOT both bear'}")
    print(f"    H1 vol > H2 vol? btc_rvol {h1.btc_rvol:.4f} vs {h2.btc_rvol:.4f} -> {'YES' if h1.btc_rvol>h2.btc_rvol else 'NO'}")
    print(f"    H1 dispersion > H2? xs_disp {h1.xs_disp:.1f} vs {h2.xs_disp:.1f} -> {'YES' if h1.xs_disp>h2.xs_disp else 'NO'}")
    print(f"    H1 pred_disp > H2? {h1.pred_disp:.4f} vs {h2.pred_disp:.4f} -> {'YES' if h1.pred_disp>h2.pred_disp else 'NO'}")

    # ---- monthly ----
    print(f"\n=== monthly breakdown ===\n")
    c["month"]=c.open_time.dt.to_period("M").astype(str)
    m=c.groupby("month").agg(btc_rvol=("btc_rvol_7d","mean"), mkt_ret=("market_ret","mean"),
        xs_disp=("xs_ret_disp","mean"), pred_disp=("pred_disp","mean"),
        long_IC=("long_IC","mean"), long_alpha=("long_alpha","mean"))
    print(m.to_string(float_format=lambda x:f"{x:+.3f}"))

    # ---- B: switch-signal power (PIT candidates vs current long_alpha) ----
    print(f"\n=== B: which PIT signal best flags 'model works' (vs current long_alpha)? ===\n")
    # trailing (PIT through t-1) versions
    c["tr_xs_disp"]=c["xs_ret_disp"].rolling(30,min_periods=10).mean().shift(1)
    c["tr_long_IC"]=c["long_IC"].rolling(30,min_periods=10).mean().shift(1)
    c["tr_long_alpha"]=c["long_alpha"].rolling(30,min_periods=10).mean().shift(1)
    # btc_rvol_7d and pred_disp are known AT t (PIT at decision time)
    cand=["btc_rvol_7d","pred_disp","tr_xs_disp","tr_long_IC","tr_long_alpha"]
    print(f"  {'signal':<16}{'Spearman vs LA':>15}{'Q1 LA':>9}{'Q4 LA':>9}{'Q4-Q1':>9}")
    print("  "+"-"*58)
    for s in cand:
        sub=c.dropna(subset=[s,"long_alpha"])
        rho=spearmanr(sub[s],sub["long_alpha"])[0]
        q=pd.qcut(sub[s],4,labels=False,duplicates="drop")
        q1=sub["long_alpha"][q==0].mean(); q4=sub["long_alpha"][q==q.max()].mean()
        print(f"  {s:<16}{rho:>+14.3f}{q1:>+8.1f}{q4:>+8.1f}{q4-q1:>+8.1f}")
    print(f"\n  (want: signal with high +Spearman & large Q4-Q1 spread = best regime/switch indicator)")
    print(f"  (PIT-at-t: btc_rvol_7d, pred_disp; trailing-PIT: tr_*)")
    c.to_parquet(REPO/"agents_system/research/outputs/iter044_cycle_metrics.parquet")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
