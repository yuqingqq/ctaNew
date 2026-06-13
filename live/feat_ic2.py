"""feat-iter2: NEW signal axes (not panel recombinations). Multi-horizon TRAILING reversal, relative-strength-vs-BTC
reversal, funding-carry persistence, vol-of-vol — built from raw 4h closes (load_close_4h). Same nested-OOS marginal
IC test over V0+RR. Anti-overfit: walk-forward, marginal, per-fold reported.
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
from live.convexity_paper_bot import load_close_4h
x6 = tt.x6; V0 = list(tt.V0); RR = ["resid_rev_2","resid_rev_3"]; EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-06-30"]]

PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized","atr_pct","funding_rate","rvol_7d"]+[c for c in V0 if c not in ("atr_pct","funding_rate","rvol_7d")])
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)

# ---- raw-kline trailing features (PIT: trailing closes up to bar t) ----
print("loading 4h closes for trailing multi-horizon features...", flush=True)
btc = load_close_4h("BTCUSDT")
btc.index = pd.to_datetime(btc.index, utc=True)
def trail_ret(close, n): return (close/close.shift(n) - 1.0)
btc_r7, btc_r14 = trail_ret(btc,42), trail_ret(btc,84)
rows=[]; nbad=0
for sym, gg in PAN.groupby("symbol"):
    try:
        c = load_close_4h(sym); c.index = pd.to_datetime(c.index, utc=True)
        c = c[~c.index.duplicated(keep="last")].sort_index()
        r7, r14, r30 = trail_ret(c,42), trail_ret(c,84), trail_ret(c,180)
        volv = c.pct_change().rolling(42).std().rolling(42).std()         # vol-of-vol (instability)
        df = pd.DataFrame({"open_time": c.index, "ret_7d": r7.values, "ret_14d": r14.values,
                           "ret_30d": r30.values, "vol_of_vol": volv.values})
        df["rel_str_7d"]  = df["ret_7d"]  - btc_r7.reindex(c.index).values   # rel-strength vs BTC (reversal axis)
        df["rel_str_14d"] = df["ret_14d"] - btc_r14.reindex(c.index).values
        df["symbol"]=sym; rows.append(df)
    except Exception:
        nbad+=1
KF = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
KF["open_time"]=pd.to_datetime(KF["open_time"],utc=True)
PAN = PAN.merge(KF, on=["symbol","open_time"], how="left")
# panel-based funding carry (trailing) + atr-normalized reversal
PAN=PAN.sort_values(["symbol","open_time"])
PAN["funding_carry_7d"] = PAN.groupby("symbol")["funding_rate"].transform(lambda s:s.shift(1).rolling(42).mean())
PAN["ret_7d_per_atr"]   = PAN["ret_7d"]/(PAN["atr_pct"]+1e-9)
PAN["ret_30d_per_atr"]  = PAN["ret_30d"]/(PAN["atr_pct"]+1e-9)
CANDS=["ret_7d","ret_14d","ret_30d","rel_str_7d","rel_str_14d","funding_carry_7d","vol_of_vol","ret_7d_per_atr","ret_30d_per_atr"]
for c in CANDS: PAN[c]=PAN[c].replace([np.inf,-np.inf],np.nan).fillna(0.0)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"built {len(CANDS)} new features ({nbad} syms had no klines)", flush=True)

def gen_ic(feats):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; tr=PAN[(PAN.exit_time<c0-EMB)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"open_time":gte["open_time"].values,
                    "pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"xs_z":gte["xs_z"].values}))
            except Exception: pass
    W=pd.concat(rec,ignore_index=True); W["open_time"]=pd.to_datetime(W["open_time"],utc=True)
    ic=W.dropna().groupby("open_time").apply(lambda x:x["pred"].corr(x["xs_z"],method="spearman")).dropna()
    cp=pd.to_datetime(CUTS,utc=True)
    pf=[ic[(ic.index>=cp[i])&(ic.index<cp[i+1])].mean() for i in range(len(cp)-1)]
    return ic.mean(), pf

t0=time.time(); base_ic, base_pf = gen_ic(V0+RR)
print(f"\nBASELINE (V0+RR) IC {base_ic:+.4f}  ({time.time()-t0:.0f}s)")
print(f"MARGINAL IC LIFT (nested-OOS; bar lift>=+0.004 & >=6/9 folds):")
res=[]
for name in CANDS:
    t0=time.time(); ic,pf=gen_ic(V0+RR+[name]); fu=sum(1 for b,n in zip(base_pf,pf) if n>b)
    res.append((name, ic-base_ic, fu)); print(f"  {name:18s} lift {ic-base_ic:+.4f}  IC {ic:+.4f}  folds_up {fu}/9  ({time.time()-t0:.0f}s)", flush=True)
# also test the 3 rel-strength/reversal together (a 'reversal block')
t0=time.time(); ic,pf=gen_ic(V0+RR+["ret_7d","ret_30d","rel_str_14d"]); fu=sum(1 for b,n in zip(base_pf,pf) if n>b)
print(f"  [BLOCK ret7+ret30+relstr14] lift {ic-base_ic:+.4f}  IC {ic:+.4f}  folds_up {fu}/9  ({time.time()-t0:.0f}s)")
res.sort(key=lambda r:-r[1])
win=[r for r in res if r[1]>=0.004 and r[2]>=6]
print(f"\nTOP: {[(n,round(l,4),f) for n,l,f in res[:5]]}")
print(f"WINNERS: {[(n,round(l,4),f) for n,l,f in win] or 'NONE'}")
print("DONE feat_ic2")
