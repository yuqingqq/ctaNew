"""LOOP iter-01 (P1) — Target redesign: can a pooled LGBM learn the defensive axis if the
target doesn't strip it?

Root cause: target_z = per-symbol-z of BTC-residualized return removes the defensive axis
(BTC-beta + per-symbol vol-scale). Test POOLED LGBM (sym_id, recency 60d, V0+defensive features)
on targets that RETAIN that axis while staying cross-symbol learnable:

  T_resid_z  : current baseline (per-sym z of alpha-vs-btc residual)         [control]
  T_raw_rank : cross-sectional rank of raw fwd return (scale-free; keeps beta+vol)
  T_raw_z    : per-symbol z of RAW fwd return (keeps vol-norm; retains BTC co-movement)

Eval: top-K=5 long selection alpha (vs cycle median) H1/H2 + per-cycle IC vs raw return.
Hypothesis: non-residualized targets let LGBM use corr/vol -> better H2.
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
try: import lightgbm as lgb
except ImportError:
    import subprocess; subprocess.run([sys.executable,"-m","pip","install","lightgbm"],check=False); import lightgbm as lgb

REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
PANEL=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT=REPO/"agents_system/research/outputs/loop"; OUT.mkdir(parents=True,exist_ok=True)
H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
FIT=pd.Timestamp("2025-10-02",tz="UTC"); K=5; CPD=6; HL=60.0
V0=["return_1d","atr_pct","obv_z_1d","vwap_slope_96","bars_since_high","autocorr_pctile_7d",
    "corr_to_btc_1d","beta_to_btc_change_5d","idio_vol_to_btc_1h","idio_vol_to_btc_1d",
    "funding_rate","funding_rate_z_7d","funding_rate_1d_change","rvol_7d","ret_3d","btc_rvol_7d"]

def main():
    t0=time.time(); print("=== LOOP iter-01: target redesign (pooled LGBM) ===\n",flush=True)
    cols=["symbol","open_time","exit_time","return_pct","target_z","alpha_vs_btc_realized","rmean","rstd"]+V0
    pan=pd.read_parquet(PANEL,columns=[c for c in cols if c])
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    # defensive feature
    g=pan.groupby("symbol",group_keys=False)
    pan["vol_ratio_recent"]=pan["idio_vol_to_btc_1d"]/g["idio_vol_to_btc_1d"].transform(lambda x:x.rolling(180,min_periods=90).mean().shift(1)).replace(0,np.nan)
    FEATS=V0+["vol_ratio_recent"]
    # alt targets
    pan["raw_z"]=((pan["return_pct"]-pan["rmean"])/pan["rstd"].replace(0,np.nan))   # per-sym z of RAW return
    pan["raw_rank"]=pan.groupby("open_time")["return_pct"].rank(pct=True)            # XS rank of raw return
    targets={"T_resid_z":"target_z","T_raw_rank":"raw_rank","T_raw_z":"raw_z"}

    syms=sorted(pan["symbol"].unique()); s2i={s:i for i,s in enumerate(syms)}
    pan["sym_id"]=pan["symbol"].map(s2i).astype("int32")
    train=pan[(pan.exit_time<FIT)].copy(); test=pan[(pan.open_time>=H1S)&(pan.open_time<=H2E)].copy()
    t_end=train["open_time"].max()
    w=np.exp(-((t_end-train["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
    Xtr=train[FEATS+["sym_id"]].fillna(0).values; Xte=test[FEATS+["sym_id"]].fillna(0).values
    p=dict(objective="regression",metric="rmse",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,
           feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,verbose=-1,num_threads=8)

    def long_alpha(df,col,s,e):
        sub=df[(df.open_time>=s)&(df.open_time<e)]
        cyc=sub.groupby("open_time").apply(lambda x:(x.nlargest(K,col)["return_pct"].mean()-x["return_pct"].median())*1e4 if len(x)>=2*K else np.nan).dropna()
        a=cyc.values; return a.mean(),(a.mean()/(a.std()/np.sqrt(len(a))) if a.std()>0 else np.nan)
    def cyc_ic(df,col,s,e):
        sub=df[(df.open_time>=s)&(df.open_time<e)]
        ics=[spearmanr(x[col],x["return_pct"])[0] for _,x in sub.groupby("open_time") if len(x)>=20]
        return np.nanmean(ics)

    print(f"  pooled LGBM, recency hl={HL}d, {len(FEATS)} feats + sym_id; train {len(train):,} rows\n",flush=True)
    print(f"  {'target':<12}{'H1 long':>9}{'(t)':>7}{'H2 long':>9}{'(t)':>7}{'H1 IC':>8}{'H2 IC':>8}")
    print("  "+"-"*60)
    res={}
    for label,tcol in targets.items():
        m=train[tcol].notna().values
        ds=lgb.Dataset(Xtr[m],label=train[tcol].values[m],categorical_feature=[len(FEATS)])
        model=lgb.train(p,ds,num_boost_round=500,callbacks=[lgb.log_evaluation(0)])
        test["pred"]=model.predict(Xte)
        h1m,h1t=long_alpha(test,"pred",H1S,H2S); h2m,h2t=long_alpha(test,"pred",H2S,H2E)
        ic1=cyc_ic(test,"pred",H1S,H2S); ic2=cyc_ic(test,"pred",H2S,H2E)
        print(f"  {label:<12}{h1m:>+8.1f}{h1t:>+7.1f}{h2m:>+8.1f}{h2t:>+7.1f}{ic1:>+8.4f}{ic2:>+8.4f}",flush=True)
        res[label]=dict(h1=h1m,h2=h2m,ic1=ic1,ic2=ic2)
        test[["symbol","open_time","return_pct","pred"]].to_parquet(OUT/f"iter01_{label}.parquet")
    print(f"\n  Reference recency-A long: H1 ~+34, H2 ~+4 (per-sym Ridge).")
    print(f"  WIN if a non-resid target gives H2 long > +4 AND H2 IC > baseline.")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
