"""Clustering / nonlinear separability of winners vs losers — OOS-honest.
Fit everything on PRE-OOS data, score win-rate on OOS (no look-ahead). Three views:
  1. ceiling      : walk-forward GBM classifier win~features -> OOS AUC (vs logistic linear baseline)
  2. tree leaves  : depth-3 tree on win/loss; read leaves with extreme OOS win-rate + their feature path
  3. kmeans pocket: k clusters on features; per-cluster OOS win-rate + centroid; flag robustly-bad pockets
Traded population = production-equivalent (wfund per-symbol preds) top/bottom-K legs, labeled by leg PnL sign.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0=tt.V0; EMB=pd.Timedelta(days=1); K=3
OOS=pd.Timestamp("2025-10-04",tz="UTC")
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-12-01","2026-02-01","2026-04-01","2026-05-27"]]
F,flowcols=tt.build_flow()
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F,on=["symbol","open_time"],how="left")
FEATS=V0+flowcols
g=PAN.groupby("open_time"); PAN["fwd_demean"]=PAN["return_pct"]-g["return_pct"].transform("mean")

pr=pd.read_parquet(REPO/"live/state/convexity/hl_wfund175/fullflow_hl60.parquet")
pr["open_time"]=pd.to_datetime(pr["open_time"],utc=True)
D=pr[["symbol","open_time","pred"]].merge(PAN,on=["symbol","open_time"],how="inner")
D["rk"]=D.groupby("open_time")["pred"].rank(ascending=False,method="first")
D["rkb"]=D.groupby("open_time")["pred"].rank(ascending=True,method="first")
lo=D[D.rk<=K].copy(); lo["pnl"]=lo["fwd_demean"]; lo["side"]="L"
sh=D[D.rkb<=K].copy(); sh["pnl"]=-sh["fwd_demean"]; sh["side"]="S"
B=pd.concat([lo,sh],ignore_index=True)
B["win"]=(B["pnl"]>0).astype(int)
B=B.dropna(subset=FEATS+["win"]).reset_index(drop=True)
print(f"traded legs {len(B)} | overall OOS win-rate {B['win'].mean():.3f} (base rate to beat)\n")

# ---- 1. separability ceiling (walk-forward classifier OOS AUC) ----
gb_auc, lr_auc, ys, pgb, plr = [], [], [], [], []
for i in range(len(CUTS)-1):
    c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
    tr=B[B.exit_time<fc]; te=B[(B.open_time>=c0)&(B.open_time<c1)]
    if len(tr)<2000 or len(te)<100 or tr["win"].nunique()<2: continue
    Xtr,Xte=tr[FEATS].fillna(0).to_numpy(),te[FEATS].fillna(0).to_numpy()
    mu,sg=Xtr.mean(0),Xtr.std(0); sg[sg==0]=1
    gb=HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=0.05,min_samples_leaf=200).fit(Xtr,tr["win"])
    lr=LogisticRegression(C=0.1,max_iter=2000).fit((Xtr-mu)/sg,tr["win"])
    pg=gb.predict_proba(Xte)[:,1]; pl=lr.predict_proba((Xte-mu)/sg)[:,1]
    ys+=list(te["win"]); pgb+=list(pg); plr+=list(pl)
ys=np.array(ys)
print("=== 1. SEPARABILITY CEILING (pooled OOS) ===")
print(f"  GBM classifier  OOS AUC = {roc_auc_score(ys,pgb):.3f}")
print(f"  Logistic linear OOS AUC = {roc_auc_score(ys,plr):.3f}")
print(f"  (0.50 = coin-flip; <0.53 => winners/losers NOT separable by current features)\n")

# ---- interpretable models: within-OOS temporal split (traded legs exist only in OOS) ----
SPLIT=pd.Timestamp("2026-02-01",tz="UTC")   # fit on first ~4mo OOS, score on rest
pre=B[B.open_time<SPLIT]; oos=B[B.open_time>=SPLIT]
print(f"  (pocket split: fit {len(pre)} legs < {SPLIT.date()}, score {len(oos)} legs >=)\n")
Xpre,Xoos=pre[FEATS].fillna(0).to_numpy(),oos[FEATS].fillna(0).to_numpy()
mu,sg=Xpre.mean(0),Xpre.std(0); sg[sg==0]=1

# ---- 2. decision-tree leaves ----
print("=== 2. DECISION-TREE LEAVES (fit pre-OOS, win-rate scored OOS) ===")
dt=DecisionTreeClassifier(max_depth=3,min_samples_leaf=300,random_state=0).fit(Xpre,pre["win"])
oo=oos.copy(); oo["leaf"]=dt.apply(Xoos)
lt=oo.groupby("leaf").agg(n=("win","size"),oos_winrate=("win","mean"),mean_pnl=("pnl","mean")).sort_values("oos_winrate")
print(lt.round(3).to_string())
print("  tree rules (top splits):")
print("   "+export_text(dt,feature_names=FEATS,max_depth=2).replace("\n","\n   ")[:1200])

# ---- 3. kmeans pockets ----
print("\n=== 3. KMEANS POCKETS (fit pre-OOS on standardized feats, win-rate scored OOS) ===")
km=KMeans(n_clusters=8,n_init=10,random_state=0).fit((Xpre-mu)/sg)
oo["clu"]=km.predict((Xoos-mu)/sg)
ct=oo.groupby("clu").agg(n=("win","size"),oos_winrate=("win","mean"),mean_pnl=("pnl","mean")).sort_values("oos_winrate")
# centroid z (which features define each cluster)
cen=pd.DataFrame(km.cluster_centers_,columns=FEATS)
def topfeat(c):
    s=cen.loc[c].abs().sort_values(ascending=False).head(3)
    return ", ".join(f"{f}{'+' if cen.loc[c,f]>0 else '-'}{abs(cen.loc[c,f]):.1f}" for f in s.index)
ct["defining_feats(z)"]=[topfeat(c) for c in ct.index]
print(ct.round(3).to_string())
worst=ct.index[0]; base=B["win"].mean()
print(f"\n  worst pocket clu{worst}: OOS win-rate {ct.loc[worst,'oos_winrate']:.3f} vs base {base:.3f} "
      f"(n={int(ct.loc[worst,'n'])}) -> defined by {ct.loc[worst,'defining_feats(z)']}")
isvol = any(k in ct.loc[worst,'defining_feats(z)'] for k in ["idio_vol","atr_pct","rvol"])
print(f"  => worst pocket is {'the VOL-TAIL we already found (nothing new)' if isvol else 'a NEW region — investigate'}")
