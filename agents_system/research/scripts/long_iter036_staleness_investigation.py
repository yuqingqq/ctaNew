"""LONG-PRED iter-036 — Staleness investigation (before the retrain test).

Question: is long-alpha decay driven by MODEL STALENESS (frozen at fold start,
predicting months later) or by REGIME (H2 just bad)?

The production preds are 8 walk-forward folds, each ~7 months. Within each fold the
model is trained at the fold boundary and FROZEN. So "model age" = open_time - fold_start.

KEY DISENTANGLING TEST:
  For EACH fold, measure long selection alpha (top-K=5 vs cycle median) bucketed by
  "months into fold" (= months since the model was trained). Average across folds.
  - If alpha consistently DECAYS month-0 -> month-7 ACROSS folds -> STALENESS is real,
    monthly retrain should help.
  - If no consistent within-fold decay (only H2/fold-8 is bad) -> REGIME, retrain won't fix.

Also:
  T2: per-fold first-2-months vs last-2-months long alpha (paired across folds)
  T3: IC (Spearman pred vs fwd) by months-into-fold
  T4: universe drift — # symbols active late-in-fold that were NOT active at fold start
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
K = 5

def main():
    t0 = time.time()
    print("=== iter-036: Staleness vs regime investigation ===\n", flush=True)
    d = pd.read_parquet(PREDS, columns=["symbol","open_time","return_pct","pred","fold"])
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
    # model age = months since fold start
    fold_start = d.groupby("fold")["open_time"].transform("min")
    d["months_in_fold"] = ((d["open_time"] - fold_start).dt.total_seconds()/(86400*30.4)).astype(int)
    d["mkt_med"] = d.groupby("open_time")["return_pct"].transform("median")
    print(f"  {len(d):,} rows, folds {sorted(d.fold.unique())}\n", flush=True)

    # ---- T1: long selection alpha by months-in-fold, per fold + averaged ----
    print("=== T1: top-K=5 long selection alpha (bps vs median) by MONTHS SINCE RETRAIN ===\n")
    def cyc_long_alpha(g):
        if len(g) < 2*K: return np.nan
        return (g.nlargest(K,"pred")["return_pct"].mean() - g["return_pct"].median())*1e4
    # per (fold, month) bucket
    rows=[]
    for (fold,mif), grp in d.groupby(["fold","months_in_fold"]):
        cyc = grp.groupby("open_time").apply(cyc_long_alpha).dropna()
        if len(cyc) < 20: continue
        rows.append({"fold":fold,"mif":mif,"long_alpha":cyc.mean(),"n":len(cyc)})
    r = pd.DataFrame(rows)
    # matrix: rows=fold, cols=months_in_fold
    piv = r.pivot_table(index="fold", columns="mif", values="long_alpha")
    print("  Long alpha (bps) by fold × months-into-fold:")
    print("  fold |" + "".join(f"{m:>7}" for m in piv.columns))
    print("  " + "-"*(6+7*len(piv.columns)))
    for fold in piv.index:
        print(f"   {int(fold):>3} |" + "".join(f"{piv.loc[fold,m]:>+7.0f}" if not np.isnan(piv.loc[fold,m]) else f"{'·':>7}" for m in piv.columns))
    # average across folds (the disentangling row)
    avg = piv.mean(axis=0)
    print("  " + "-"*(6+7*len(piv.columns)))
    print("  AVG |" + "".join(f"{avg[m]:>+7.0f}" if not np.isnan(avg[m]) else f"{'·':>7}" for m in piv.columns))
    # monotonicity of the AVG curve
    valid = avg.dropna()
    rho = spearmanr(valid.index, valid.values)[0] if len(valid)>2 else np.nan
    print(f"\n  Avg-curve monotonicity (months vs alpha) Spearman = {rho:+.2f}")
    print(f"  (strong NEGATIVE = alpha decays as model ages = STALENESS confirmed)")

    # ---- T2: first-2-months vs last-2-months, paired across folds ----
    print(f"\n=== T2: fresh (months 0-1) vs stale (months 5+) long alpha, paired per fold ===\n")
    fresh_l, stale_l = [], []
    print(f"  {'fold':<6}{'fresh(0-1mo)':>14}{'stale(5+mo)':>13}{'decay':>9}")
    for fold in sorted(d.fold.unique()):
        df=d[d.fold==fold]
        fresh=df[df.months_in_fold<=1]; stale=df[df.months_in_fold>=5]
        fa=fresh.groupby("open_time").apply(cyc_long_alpha).dropna()
        sa=stale.groupby("open_time").apply(cyc_long_alpha).dropna()
        if len(fa)<20 or len(sa)<20: continue
        fresh_l.append(fa.mean()); stale_l.append(sa.mean())
        print(f"   {int(fold):<5}{fa.mean():>+12.0f}  {sa.mean():>+11.0f}  {sa.mean()-fa.mean():>+8.0f}")
    fresh_l,stale_l=np.array(fresh_l),np.array(stale_l)
    print(f"\n  mean fresh={fresh_l.mean():+.0f}  mean stale={stale_l.mean():+.0f}  mean decay={stale_l.mean()-fresh_l.mean():+.0f} bps")
    n_decay=(stale_l<fresh_l).sum()
    print(f"  folds where stale < fresh: {n_decay}/{len(fresh_l)}")
    # paired t
    diff=stale_l-fresh_l
    t=diff.mean()/(diff.std()/np.sqrt(len(diff))) if diff.std()>0 else np.nan
    print(f"  paired decay t-stat = {t:+.2f} ({'consistent staleness' if abs(t)>1.5 and diff.mean()<0 else 'not consistent'})")

    # ---- T3: IC by months-in-fold (averaged across folds) ----
    print(f"\n=== T3: per-cycle IC (Spearman pred vs fwd) by months since retrain ===\n")
    rows=[]
    for (fold,mif),grp in d.groupby(["fold","months_in_fold"]):
        ics=[]
        for ot,g in grp.groupby("open_time"):
            if len(g)<20: continue
            ic=spearmanr(g["pred"],g["return_pct"])[0]
            if np.isfinite(ic): ics.append(ic)
        if len(ics)>=20: rows.append({"fold":fold,"mif":mif,"ic":np.mean(ics)})
    ric=pd.DataFrame(rows)
    avg_ic=ric.groupby("mif")["ic"].mean()
    print(f"  {'months':<8}{'avg IC':>9}")
    for m,ic in avg_ic.items(): print(f"   {m:<7}{ic:>+8.4f}")
    valid=avg_ic.dropna()
    rho_ic=spearmanr(valid.index,valid.values)[0] if len(valid)>2 else np.nan
    print(f"  IC monotonicity (months vs IC) Spearman = {rho_ic:+.2f}")

    # ---- T4: universe drift within fold ----
    print(f"\n=== T4: universe drift — new symbols appearing late in fold ===\n")
    print(f"  {'fold':<6}{'syms@start(mo0-1)':>18}{'new syms by mo5+':>18}{'%new':>7}")
    for fold in sorted(d.fold.unique()):
        df=d[d.fold==fold]
        start_syms=set(df[df.months_in_fold<=1]["symbol"].unique())
        late=df[df.months_in_fold>=5]
        if len(late)==0: continue
        late_syms=set(late["symbol"].unique())
        new=late_syms-start_syms
        pct=100*len(new)/max(1,len(late_syms))
        print(f"   {int(fold):<5}{len(start_syms):>16}  {len(new):>16}  {pct:>6.0f}%")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
