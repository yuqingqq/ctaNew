"""Push HARD on: can we tell the model's future-losers from future-winners?
Part A  separability  : within the names the model LONGS (and SHORTS), do losers carry a feature
                        signature winners don't? single-feature Welch-t + rank-AUC; tail-loser signature.
Part B  IC ladder     : pooled-linear  ->  pooled-linear + hand-crafted nonlinear basis  ->  GBM.
                        If the basis closes the gap -> the edge is a KNOWN transform (hand-craftable).
                        If only the tree gets it    -> it's high-order interaction Ridge can't represent.
Read-only; uses production-equivalent wfund preds for the traded population + panel features.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0 = tt.V0; EMB = pd.Timedelta(days=1); HL = 60.0; K = 3
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-12-01","2026-02-01","2026-04-01","2026-05-27"]]

# ---- build feature panel ----
F, flowcols = tt.build_flow()
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F,on=["symbol","open_time"],how="left")
FEATS = V0 + flowcols
g = PAN.groupby("open_time")
PAN["fwd_demean"] = PAN["return_pct"] - g["return_pct"].transform("mean")     # long-PnL proxy (beta-neutral)
sd = g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

# ================= PART A: separability within traded longs/shorts =================
pr = pd.read_parquet(REPO/"live/state/convexity/hl_wfund175/fullflow_hl60.parquet")
pr["open_time"]=pd.to_datetime(pr["open_time"],utc=True)
D = pr[["symbol","open_time","pred"]].merge(PAN, on=["symbol","open_time"], how="inner")
D = D[D.open_time>=pd.Timestamp("2025-10-04",tz="UTC")]
# per-cycle pred rank -> top-K long, bottom-K short
D["rk"]  = D.groupby("open_time")["pred"].rank(ascending=False, method="first")
D["rkb"] = D.groupby("open_time")["pred"].rank(ascending=True,  method="first")
longs  = D[D.rk<=K].copy();  longs["pnl"]  =  longs["fwd_demean"]
shorts = D[D.rkb<=K].copy(); shorts["pnl"] = -shorts["fwd_demean"]
book = pd.concat([longs, shorts], ignore_index=True)
print(f"PART A  traded legs: {len(longs)} longs + {len(shorts)} shorts = {len(book)} | "
      f"win-rate L={ (longs.pnl>0).mean():.3f} S={(shorts.pnl>0).mean():.3f}")

def auc(x, y):  # rank-AUC of feature x predicting binary y (winner=1), nan-robust
    m = np.isfinite(x); x, y = x[m], y[m]
    if y.sum()==0 or y.sum()==len(y): return np.nan
    r = pd.Series(x).rank().to_numpy(); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum() - n1*(n1+1)/2)/(n1*n0)

def sep_table(df, title):
    win = (df.pnl>0).astype(int).to_numpy()
    rows=[]
    for f in FEATS:
        x = df[f].to_numpy(float); w=df[win==1][f]; l=df[win==0][f]
        w,l = w.dropna(), l.dropna()
        if len(w)<30 or len(l)<30: continue
        sp = np.sqrt(w.var()/len(w)+l.var()/len(l)); t = (w.mean()-l.mean())/sp if sp>0 else 0
        a = auc(x, win)
        rows.append((f, t, a, abs(a-0.5)))
    R = pd.DataFrame(rows, columns=["feat","welch_t","auc","auc_edge"]).sort_values("auc_edge",ascending=False)
    print(f"\n--- {title}: top separating features (|AUC-0.5| desc) ---")
    print(R.head(12).round(3).to_string(index=False))
    print(f"  best single-feature AUC={R.auc_edge.max()+0.5:.3f}  (0.5=coin-flip; >0.55 = real separation)")
    return R

sep_table(longs,  "LONGS winner-vs-loser")
sep_table(shorts, "SHORTS winner-vs-loser")

# tail-loser signature: worst-10% PnL legs vs the rest, standardized mean gap
tail = book[book.pnl <= book.pnl.quantile(0.10)]
rest = book[book.pnl >  book.pnl.quantile(0.10)]
print("\n--- TAIL-LOSER signature (worst-10% legs vs rest; z-gap of feature means) ---")
zr=[]
for f in FEATS:
    a,b = tail[f].dropna(), rest[f].dropna()
    if len(a)<30: continue
    sd_=PAN[f].std()
    if sd_>0: zr.append((f,(a.mean()-b.mean())/sd_))
ZR=pd.DataFrame(zr,columns=["feat","z_gap"]).reindex(pd.DataFrame(zr,columns=["feat","z_gap"]).z_gap.abs().sort_values(ascending=False).index)
print(ZR.head(12).round(3).to_string(index=False))

# nonlinear single-feature check: can a depth-2 tree on the SINGLE best linear-IC feature separate
# better than its linear AUC? (detects threshold/U-shaped separation Ridge misses)
print("\n--- linear vs nonlinear single-feature AUC on longs (gap = Ridge-missable shape) ---")
ywin = (longs.pnl>0).astype(int).to_numpy()
rows=[]
for f in ["ret_3d","return_1d","rvol_7d","funding_rate_z","dist_sma200"]:
    if f not in longs: continue
    x = longs[f].to_numpy(float); m=np.isfinite(x)
    lin = abs(auc(x,ywin)-0.5)+0.5
    t = HistGradientBoostingRegressor(max_depth=2,max_iter=60,min_samples_leaf=200).fit(x[m].reshape(-1,1), ywin[m])
    nl = abs(auc(t.predict(x[m].reshape(-1,1)), ywin[m])-0.5)+0.5
    rows.append((f, round(lin,3), round(nl,3), round(nl-lin,3)))
print(pd.DataFrame(rows,columns=["feat","linear_AUC","tree_AUC","nonlin_gain"]).to_string(index=False))

# ================= PART B: IC ladder (pooled, walk-forward) =================
P = PAN[PAN["xs_z"].notna()].copy()
# hand-crafted nonlinear basis (what a quant would add to a LINEAR model)
P["ret3d_pos"]=P["ret_3d"].clip(lower=0); P["ret3d_neg"]=P["ret_3d"].clip(upper=0)
P["r1d_pos"]=P["return_1d"].clip(lower=0); P["r1d_neg"]=P["return_1d"].clip(upper=0)
P["ret3d_x_rvol"]=P["ret_3d"]*P["rvol_7d"]; P["ret3d_sq"]=P["ret_3d"]**2; P["r1d_sq"]=P["return_1d"]**2
P["fund_x_ret3d"]=P.get("funding_rate_z",0)*P["ret_3d"]; P["rvol_x_dist"]=P["rvol_7d"]*P.get("dist_sma200",0)
BASIS = ["ret3d_pos","ret3d_neg","r1d_pos","r1d_neg","ret3d_x_rvol","ret3d_sq","r1d_sq","fund_x_ret3d","rvol_x_dist"]

def cyc_ic(df):
    return df.groupby("open_time").apply(lambda x: x["pred_"].corr(x["xs_z"],method="spearman")).mean()

def ladder(feats, kind):
    ics=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[(P.exit_time<fc)]; te=P[(P.open_time>=c0)&(P.open_time<c1)].copy()
        if len(tr)<5000 or len(te)==0: continue
        w=np.exp(-((tr["open_time"].max()-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
        Xtr=tr[feats].fillna(0).to_numpy(); Xte=te[feats].fillna(0).to_numpy()
        if kind=="lin":
            mu,sg=Xtr.mean(0),Xtr.std(0); sg[sg==0]=1
            m=RidgeCV(alphas=(0.1,1,10,100)).fit((Xtr-mu)/sg, tr["xs_z"], sample_weight=w)
            te["pred_"]=m.predict((Xte-mu)/sg)
        else:
            m=HistGradientBoostingRegressor(max_depth=4,max_iter=200,learning_rate=0.05,
                  min_samples_leaf=200,l2_regularization=1.0).fit(Xtr,tr["xs_z"],sample_weight=w)
            te["pred_"]=m.predict(Xte)
        ics.append(cyc_ic(te))
    return float(np.mean(ics))

L1 = ladder(FEATS, "lin")
L2 = ladder(FEATS+BASIS, "lin")
L3 = ladder(FEATS, "gbm")
print("\n================  IC LADDER (pooled, mean OOS per-cycle Spearman IC)  ================")
print(f"  L1  pooled linear (V0+flow)                  IC = {L1:+.4f}")
print(f"  L2  pooled linear + hand-crafted NL basis    IC = {L2:+.4f}   (basis recovers: {L2-L1:+.4f})")
print(f"  L3  pooled GBM (same V0+flow, native NL)      IC = {L3:+.4f}   (tree-only edge: {L3-L2:+.4f})")
gap = L3-L1
print(f"\n  total nonlinear gap L3-L1 = {gap:+.4f} | basis closes {100*(L2-L1)/gap if gap else 0:.0f}% | "
      f"{'HAND-CRAFTABLE (known transform)' if (L2-L1)>0.6*gap else 'TREE-ONLY (high-order interaction)' if gap>0.01 else 'NO real NL gap'}")
