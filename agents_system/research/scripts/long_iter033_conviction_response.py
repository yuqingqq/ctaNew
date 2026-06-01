"""LONG-PRED iter-033 — Conviction-response curve (decides if an ABSOLUTE gate can work).

Uses RAW model output (NOT per-cycle z-scored) = the true absolute conviction signal.

Question: as conviction deepens, does selection alpha rise monotonically?
  - Longs: raw_pred high  -> long alpha should rise (this is why longs work)
  - Shorts: raw_pred low (very negative) -> short alpha should rise IF gate can help

Tests, per slice (VAL / INTERIM / FINAL):
  T1: Bin every observation by raw_pred decile -> mean forward return per decile
      (the pure response curve; monotone = conviction informative)
  T2: For shorts specifically: among names with pred < 0, bucket by conviction depth
      (most-negative quintile ... least) -> short alpha per bucket
      + cross-tab: do high-conviction shorts land in the extreme-pump ret_3d zone?
  T3: Absolute raw-pred threshold sweep on SHORT side:
      for T in grid, take shorts = {pred < -T}; report n, short alpha, per slice
      -> does a STRICTER absolute bar raise short alpha on the holdout?
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

VAL_S,VAL_E = pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2025-12-01",tz="UTC")
INT_S,INT_E = pd.Timestamp("2025-12-01",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC")
FIN_S,FIN_E = pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC")
SLICES = [("VAL",VAL_S,VAL_E),("INTERIM",INT_S,INT_E),("FINAL",FIN_S,FIN_E)]

def short_alpha_series(g):
    """per-cycle: market_med - picked short mean, given a sub-df of short picks merged w/ market_med."""
    pass

def main():
    t0=time.time()
    print("=== iter-033: Conviction-response (absolute-gate feasibility) ===\n", flush=True)
    preds = pd.read_parquet(PREDS)
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    panel = pd.read_parquet(PANEL, columns=["symbol","open_time","ret_3d"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0)&(panel["open_time"].dt.minute==0)]
    df = preds.merge(panel, on=["symbol","open_time"], how="left")
    # market median per cycle for alpha computation
    df["mkt_med"] = df.groupby("open_time")["return_pct"].transform("median")
    print(f"  raw pred: mean={df.pred.mean():.3f} std={df.pred.std():.3f} "
          f"p01={df.pred.quantile(.01):.3f} p50={df.pred.quantile(.5):.3f} p99={df.pred.quantile(.99):.3f}\n", flush=True)

    # ===== T1: raw-pred decile response curve (per slice) =====
    print("=== T1: Mean forward return by RAW-pred decile (monotone = conviction informative) ===\n")
    for slabel,s,e in SLICES:
        sub = df[(df.open_time>=s)&(df.open_time<e)].copy()
        sub["dec"] = pd.qcut(sub["pred"], 10, labels=False, duplicates="drop")
        g = sub.groupby("dec").agg(fwd=("return_pct","mean"), alpha=("return_pct", lambda x: 0)).reset_index()
        # alpha vs market median
        agg = sub.groupby("dec").apply(lambda x: pd.Series({
            "fwd_bps": x["return_pct"].mean()*1e4,
            "alpha_bps": (x["return_pct"]-x["mkt_med"]).mean()*1e4,
            "n": len(x)})).reset_index()
        print(f"  {slabel}:")
        print(f"    {'decile':<7}{'fwd bps':>10}{'alpha vs med':>14}{'n':>8}")
        for _,r in agg.iterrows():
            d=int(r['dec']); tag = "  <-TOP(long)" if d==9 else ("  <-BOT(short)" if d==0 else "")
            print(f"    {d:<7}{r['fwd_bps']:>+9.1f}{r['alpha_bps']:>+13.1f}{int(r['n']):>8}{tag}")
        # monotonicity check (Spearman of decile vs alpha)
        rho = agg["dec"].corr(agg["alpha_bps"], method="spearman")
        print(f"    monotonicity (Spearman dec vs alpha) = {rho:+.2f}\n")

    # ===== T2: short-side conviction buckets + ret_3d zone cross-tab =====
    print("=== T2: Short-side conviction depth -> short alpha + recent-pump zone ===\n")
    print("  (among pred<0 names, bucket by how negative; Q5=deepest short conviction)\n")
    for slabel,s,e in SLICES:
        sub = df[(df.open_time>=s)&(df.open_time<e)].copy()
        neg = sub[sub["pred"]<0].copy()
        if len(neg)<200:
            print(f"  {slabel}: too few neg-pred names\n"); continue
        neg["cb"] = pd.qcut(neg["pred"], 5, labels=["Q1_mild","Q2","Q3","Q4","Q5_deep"], duplicates="drop")
        agg = neg.groupby("cb").apply(lambda x: pd.Series({
            "short_alpha_bps": (x["mkt_med"]-x["return_pct"]).mean()*1e4,
            "mean_ret3d_pct": x["ret_3d"].mean()*100,
            "n": len(x)})).reset_index()
        print(f"  {slabel}:")
        print(f"    {'conviction':<10}{'short_alpha':>12}{'ret_3d %':>10}{'n':>8}")
        for _,r in agg.iterrows():
            print(f"    {r['cb']:<10}{r['short_alpha_bps']:>+11.1f}{r['mean_ret3d_pct']:>+9.2f}{int(r['n']):>8}")
        rho = pd.Series(range(len(agg))).corr(agg["short_alpha_bps"], method="spearman")
        print(f"    monotonicity (deeper conviction -> higher short alpha) = {rho:+.2f}\n")

    # ===== T3: absolute raw-pred SHORT threshold sweep =====
    print("=== T3: Absolute SHORT threshold sweep (shorts = {pred < -T}) ===\n")
    # choose T grid from the negative tail quantiles (global)
    qs = [df.pred.quantile(q) for q in [0.30,0.20,0.10,0.05,0.02]]
    Tgrid = sorted([-q for q in qs if q<0]) or [0.05,0.1,0.2]
    print(f"  T grid (abs): {[f'{t:.3f}' for t in Tgrid]}\n")
    print(f"  {'slice':<9}{'T':>7}{'avg #S/cyc':>11}{'%cyc w/short':>13}{'short_alpha bps':>16}{'t':>6}")
    print("  "+"-"*64)
    for slabel,s,e in SLICES:
        sub = df[(df.open_time>=s)&(df.open_time<e)].copy()
        ncyc = sub.open_time.nunique()
        for T in Tgrid:
            rows=[]
            for ot,gc in sub.groupby("open_time"):
                sh = gc[gc["pred"] < -T]
                if len(sh)==0:
                    rows.append({"a":np.nan,"has":0}); continue
                a = (gc["return_pct"].median()-sh["return_pct"].mean())
                rows.append({"a":a,"has":1})
            rr=pd.DataFrame(rows)
            arr=rr["a"].dropna().values*1e4
            if len(arr)==0:
                print(f"  {slabel:<9}{T:>7.3f}{'0.0':>11}{'0%':>13}{'--':>16}"); continue
            avg_n = sub.groupby("open_time").apply(lambda x:(x['pred']<-T).sum()).mean()
            tstat = arr.mean()/(arr.std()/np.sqrt(len(arr))) if arr.std()>0 else 0
            tag="★" if abs(tstat)>1.96 else " "
            print(f"  {slabel:<9}{T:>7.3f}{avg_n:>11.1f}{rr['has'].mean()*100:>12.0f}%{arr.mean():>+15.1f}{tag}{tstat:>+5.1f}")
        print()

    print(f"DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
