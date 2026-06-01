"""LOOP iter-02 (P2) — Does the raw_rank target win transfer to an EXTENSIBLE model?

iter-01 (pooled+sym_id, not adoptable): raw_rank target -> H2 IC +0.0608 vs resid_z +0.0006.
Now test extensible architectures:
  M_persym_resid : per-sym Ridge, resid_z target, recency60      [current production arch, control]
  M_persym_rank  : per-sym Ridge, raw_rank target, recency60     [EXTENSIBLE default — does rank help here?]
  M_pool_rank    : pooled NO-sym_id LGBM, raw_rank, recency60     [extensible alt — must prove]
  LSO            : pooled NO-sym_id, raw_rank, LEAVE-SYMBOLS-OUT (train 70% syms, test held-out 30%)
                   -> IC on UNSEEN symbols = extensibility proof.

Report H1/H2 long alpha + per-cycle IC. WIN = rank target lifts H2 on an extensible model AND
(pooled) generalizes to unseen symbols.
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
try: import lightgbm as lgb
except ImportError:
    import subprocess; subprocess.run([sys.executable,"-m","pip","install","lightgbm"],check=False); import lightgbm as lgb

REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location("x6",REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6=importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
PANEL=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
FIT=pd.Timestamp("2025-10-02",tz="UTC"); K=5; HL=60.0
FEAT=x6.BASE+x6.COHORT_EXTRAS   # 17

def long_alpha(df,col,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    cyc=sub.groupby("open_time").apply(lambda x:(x.nlargest(K,col)["return_pct"].mean()-x["return_pct"].median())*1e4 if len(x)>=2*K else np.nan).dropna()
    a=cyc.values; return (a.mean(),a.mean()/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan)
def cyc_ic(df,col,s,e):
    sub=df[(df.open_time>=s)&(df.open_time<e)]
    ics=[spearmanr(x[col],x["return_pct"])[0] for _,x in sub.groupby("open_time") if len(x)>=20]
    return np.nanmean(ics)

def main():
    t0=time.time(); print("=== LOOP iter-02: extensible rank-target transfer ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct","target_z","rmean","rstd"]+FEAT)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    pan["raw_rank"]=pan.groupby("open_time")["return_pct"].rank(pct=True)
    train=pan[pan.exit_time<FIT].copy(); test=pan[(pan.open_time>=H1S)&(pan.open_time<=H2E)].copy()
    print(f"  train {len(train):,}  test {len(test):,}  syms {pan.symbol.nunique()}\n",flush=True)

    def persym(tcol):
        t_end=train["open_time"].max(); models,ss,hh={},{},{}
        for sym,g in train.groupby("symbol"):
            if len(g)<300 or g[tcol].notna().sum()<300: continue
            s,h=x6.fit_preproc(g,FEAT); X=x6.apply_preproc(g,FEAT,s,h)
            w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            m=g[tcol].notna().values
            try: mdl=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X[m],g[tcol].to_numpy()[m],sample_weight=w[m])
            except: continue
            models[sym]=mdl; ss[sym]=s; hh[sym]=h
        out=[]
        for sym,g in test.groupby("symbol"):
            if sym not in models: continue
            X=x6.apply_preproc(g,FEAT,ss[sym],hh[sym]); o=g[["symbol","open_time","return_pct"]].copy(); o["pred"]=models[sym].predict(X); out.append(o)
        return pd.concat(out,ignore_index=True)

    def pooled(tcol, train_syms=None, test_syms=None):
        tr=train if train_syms is None else train[train.symbol.isin(train_syms)]
        te=test if test_syms is None else test[test.symbol.isin(test_syms)]
        t_end=tr["open_time"].max(); m=tr[tcol].notna().values
        w=np.exp(-((t_end-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)[m]
        Xtr=tr[FEAT].fillna(0).values[m]; Xte=te[FEAT].fillna(0).values
        p=dict(objective="regression",metric="rmse",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,
               feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,verbose=-1,num_threads=8)
        ds=lgb.Dataset(Xtr,label=tr[tcol].to_numpy()[m],weight=w)
        mdl=lgb.train(p,ds,num_boost_round=500,callbacks=[lgb.log_evaluation(0)])
        o=te[["symbol","open_time","return_pct"]].copy(); o["pred"]=mdl.predict(Xte); return o

    print(f"  {'config':<22}{'H1 long':>9}{'(t)':>7}{'H2 long':>9}{'(t)':>7}{'H1 IC':>9}{'H2 IC':>9}")
    print("  "+"-"*64)
    for label,fn in [("persym_resid_z",lambda:persym("target_z")),
                     ("persym_raw_rank",lambda:persym("raw_rank")),
                     ("pool_nosymid_rank",lambda:pooled("raw_rank"))]:
        d=fn(); h1=long_alpha(d,"pred",H1S,H2S); h2=long_alpha(d,"pred",H2S,H2E)
        i1=cyc_ic(d,"pred",H1S,H2S); i2=cyc_ic(d,"pred",H2S,H2E)
        print(f"  {label:<22}{h1[0]:>+8.1f}{h1[1]:>+7.1f}{h2[0]:>+8.1f}{h2[1]:>+7.1f}{i1:>+9.4f}{i2:>+9.4f}",flush=True)

    # leave-symbols-out extensibility proof
    print(f"\n=== leave-symbols-out (pooled no-sym_id, raw_rank): IC on UNSEEN symbols ===\n")
    syms=sorted(pan.symbol.unique())
    h=lambda s: int(__import__("hashlib").sha1(s.encode()).hexdigest(),16)
    test_syms=[s for s in syms if h(s)%10<3]; train_syms=[s for s in syms if h(s)%10>=3]
    d=pooled("raw_rank",train_syms=train_syms,test_syms=test_syms)
    h1=long_alpha(d,"pred",H1S,H2S); h2=long_alpha(d,"pred",H2S,H2E); i1=cyc_ic(d,"pred",H1S,H2S); i2=cyc_ic(d,"pred",H2S,H2E)
    print(f"  train {len(train_syms)} syms -> test {len(test_syms)} HELD-OUT syms")
    print(f"  held-out: H1 long {h1[0]:+.1f}(t{h1[1]:+.1f})  H2 long {h2[0]:+.1f}(t{h2[1]:+.1f})  H1 IC {i1:+.4f}  H2 IC {i2:+.4f}")
    print(f"  -> EXTENSIBLE if held-out H2 IC stays ~comparable to full-universe (>0.03)")
    print(f"\n  recency-A reference: H2 long ~+4, H2 IC ~0.")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
