"""LONG-PRED iter-034 — Test design A: model-selected longs + PASSIVE hedge.

The short leg carries no model signal (iter-033 decisive). So the short leg
becomes a pure beta hedge. Test how to construct it.

Construction variants (all use the SAME model longs = top-K by raw pred):
  A0  baseline reference: model longs + model shorts (K=5/K=5)  [what we'd replace]
  A1  long-only (no hedge)                       -> full bear beta exposure
  A2  longs + equal-weight FULL-basket short (beta-1 dollar-neutral)
  A3  longs + beta-matched basket short (size hedge so net beta ~ 0)
  A4  longs + BTC-proxy short (short the cross-sym median return as a single hedge)

Metric per slice (VAL / INTERIM / FINAL), with turnover cost:
  - net PnL bps/cycle, t-stat, annualized Sharpe-like
  - long selection alpha (vs market median) -- should be the alpha source
  - net market exposure (beta) proxy

Long selection uses top-K=5 by raw pred (no per-cycle z; no gate -- keep it simple
and robust, since iter-033 showed gating longs doesn't reliably help OOS).

Beta proxy: regress per-cycle long-basket return on market-median return across
cycles in TRAIN-like fashion; but here we just report net exposure = mean(long_ret)
correlation to market and the realized net pnl's market sensitivity.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"

VAL_S,VAL_E = pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2025-12-01",tz="UTC")
INT_S,INT_E = pd.Timestamp("2025-12-01",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC")
FIN_S,FIN_E = pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC")
SLICES = [("VAL",VAL_S,VAL_E),("INTERIM",INT_S,INT_E),("FINAL",FIN_S,FIN_E)]

K = 5
COST_RT_BPS = 9.0
CYCLES_PER_YEAR = 6*365

def run_variant(sub, variant):
    """Return per-cycle dataframe with net pnl etc. Long book = top-K by pred (equal wt)."""
    rows=[]
    prev_long=set(); prev_short=set()
    for ot,gc in sub.groupby("open_time"):
        if len(gc) < 2*K: continue
        mkt = gc["return_pct"].median()
        longs = gc.nlargest(K,"pred")
        long_ret = longs["return_pct"].mean()
        long_set = set(longs["symbol"])
        # turnover long (equal wt, full $1 side)
        to_long = len(long_set - prev_long)/K if K else 0
        prev_long = long_set

        short_set=set(); to_short=0.0; short_ret=0.0; hedge_notional=0.0
        if variant=="A0_model_short":
            shorts = gc.nsmallest(K,"pred"); short_ret=shorts["return_pct"].mean()
            short_set=set(shorts["symbol"]); to_short=len(short_set-prev_short)/K
            gross = long_ret - short_ret
        elif variant=="A1_long_only":
            gross = long_ret            # no hedge, full beta
        elif variant=="A2_full_basket":
            gross = long_ret - mkt      # short equal-wt basket (== median proxy), $1 vs $1
            hedge_notional = 1.0        # basket re-struck each cycle (approx full turnover)
        elif variant=="A3_beta_half":
            gross = long_ret - 0.5*mkt  # half-sized hedge (if long book beta<1)
            hedge_notional = 0.5
        elif variant=="A4_btc_proxy":
            gross = long_ret - mkt      # single-instrument proxy = median; lower turnover
            hedge_notional = 0.15       # one liquid instrument, cheap to roll
        prev_short = short_set
        cost = COST_RT_BPS*(to_long+to_short+hedge_notional)/1e4
        rows.append({"ot":ot,"net":gross-cost,"gross":gross,
                     "long_alpha":long_ret-mkt,"mkt":mkt,"long_ret":long_ret})
    return pd.DataFrame(rows)

def stat(res):
    net=res["net"].values*1e4; n=len(res)
    # market beta of the net pnl (sensitivity to market median move)
    mkt=res["mkt"].values*1e4
    beta = np.polyfit(mkt, res["net"].values*1e4, 1)[0] if n>2 and mkt.std()>0 else np.nan
    return dict(n=n, net=net.mean(),
                t=net.mean()/(net.std()/np.sqrt(n)) if net.std()>0 else 0,
                sharpe=net.mean()/net.std()*np.sqrt(CYCLES_PER_YEAR) if net.std()>0 else 0,
                gross=res["gross"].mean()*1e4,
                long_alpha=res["long_alpha"].mean()*1e4,
                beta=beta)

def main():
    t0=time.time()
    print("=== iter-034: Design A — model longs + passive hedge ===\n",flush=True)
    print(f"  long book = top-K={K} by raw pred (no gate); cost={COST_RT_BPS}bps RT\n")
    preds=pd.read_parquet(PREDS); preds["open_time"]=pd.to_datetime(preds["open_time"],utc=True)

    variants=["A0_model_short","A1_long_only","A2_full_basket","A3_beta_half","A4_btc_proxy"]
    labels={"A0_model_short":"A0 model longs+shorts (ref)",
            "A1_long_only":"A1 long-only (no hedge)",
            "A2_full_basket":"A2 longs + full-basket short",
            "A3_beta_half":"A3 longs + half-basket short",
            "A4_btc_proxy":"A4 longs + single-instr hedge"}

    for slabel,s,e in SLICES:
        sub=preds[(preds.open_time>=s)&(preds.open_time<e)].copy()
        print(f"=== {slabel}  ({sub.open_time.nunique()} cycles) ===")
        print(f"  {'variant':<30}{'net bps':>9}{'t':>6}{'Sharpe':>8}{'gross':>8}{'L_alpha':>9}{'beta':>7}")
        print("  "+"-"*76)
        for v in variants:
            st=stat(run_variant(sub,v))
            tag="★" if abs(st["t"])>1.96 else " "
            print(f"  {labels[v]:<30}{st['net']:>+8.2f}{tag}{st['t']:>+5.1f}{st['sharpe']:>+7.2f}"
                  f"{st['gross']:>+7.1f}{st['long_alpha']:>+8.1f}{st['beta']:>+6.2f}")
        print()
    print(f"DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
