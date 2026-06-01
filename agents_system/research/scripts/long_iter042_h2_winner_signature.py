"""LONG-PRED iter-042 — What's SPECIAL about H2-regime winners?

iter-038 found the stable defensive signature (low-vol/high-corr). Here we go broader and
contrast H1 vs H2 directly to find what is DIFFERENT/special about the new-regime winners.

Winner = top-K=5 by realized return per cycle; loser = bottom-K. All features PIT (entry-time).

A: per-feature rank-AUC (winner vs loser), H1 & H2 side-by-side, sorted by H2 |AUC-0.5|,
   flag features where H2 differs from H1 (|ΔAUC| large) = "special to new regime".
B: multivariate logistic (winner vs loser), 5-fold CV AUC for H1 vs H2 + top |coef| features
   = are H2 winners JOINTLY more/less separable than single features suggest?
C: symbol concentration of H2 winners + funding-sign profile (are they specific names / states?)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
H1S=pd.Timestamp("2025-10-04",tz="UTC"); H2S=pd.Timestamp("2026-01-22",tz="UTC"); H2E=pd.Timestamp("2026-05-26",tz="UTC")
K=5; CPD=6
V0=["return_1d","atr_pct","obv_z_1d","vwap_slope_96","bars_since_high","autocorr_pctile_7d",
    "corr_to_btc_1d","beta_to_btc_change_5d","idio_vol_to_btc_1h","idio_vol_to_btc_1d",
    "funding_rate","funding_rate_z_7d","funding_rate_1d_change","rvol_7d","ret_3d","btc_rvol_7d"]

def engineer(panel):
    g=panel.groupby("symbol",group_keys=False)
    for dd in [7,30,60]:
        b=dd*CPD; panel[f"ret_{dd}d"]=g["return_1d"].transform(lambda x:x.rolling(b,min_periods=b//2).sum().shift(1))
    panel["rvol_30d"]=g["rvol_7d"].transform(lambda x:x.rolling(180,min_periods=90).mean().shift(1))
    panel["i_cum_funding_7d"]=g["funding_rate"].transform(lambda x:x.rolling(42,min_periods=21).sum().shift(1))
    panel["i_cum_funding_30d"]=g["funding_rate"].transform(lambda x:x.rolling(180,min_periods=90).sum().shift(1))
    panel["i_pos_funding_share_7d"]=g["funding_rate"].transform(lambda x:(x>0).rolling(42,min_periods=21).mean().shift(1))
    panel["i_funding_vol_7d"]=g["funding_rate"].transform(lambda x:x.rolling(42,min_periods=21).std().shift(1))
    panel["k_rvol_std_30d"]=g["rvol_7d"].transform(lambda x:x.rolling(180,min_periods=90).std().shift(1))
    panel["k_ret_skew_30d"]=g["return_1d"].transform(lambda x:x.rolling(180,min_periods=90).skew().shift(1))
    panel["k_ret_kurt_30d"]=g["return_1d"].transform(lambda x:x.rolling(180,min_periods=90).kurt().shift(1))
    panel["k_vol_change_30d"]=panel["rvol_7d"]/panel["rvol_30d"].replace(0,np.nan)-1
    panel["vol_ratio_recent"]=panel["idio_vol_to_btc_1d"]/g["idio_vol_to_btc_1d"].transform(lambda x:x.rolling(180,min_periods=90).mean().shift(1)).replace(0,np.nan)
    extra=["ret_7d","ret_30d","ret_60d","i_cum_funding_7d","i_cum_funding_30d","i_pos_funding_share_7d",
           "i_funding_vol_7d","k_rvol_std_30d","k_ret_skew_30d","k_ret_kurt_30d","k_vol_change_30d","vol_ratio_recent"]
    return panel, V0+extra

def rank_auc(W,L,f):
    wv=W[f].dropna().values; lv=L[f].dropna().values
    if len(wv)<50 or len(lv)<50: return np.nan,np.nan,np.nan
    allv=np.concatenate([wv,lv]); r=pd.Series(allv).rank().values
    auc=(r[:len(wv)].sum()-len(wv)*(len(wv)+1)/2)/(len(wv)*len(lv))
    return wv.mean(),lv.mean(),auc

def win_lose(d,s,e):
    sub=d[(d.open_time>=s)&(d.open_time<e)]; W,L=[],[]
    for ot,g in sub.groupby("open_time"):
        if len(g)<2*K: continue
        W.append(g.nlargest(K,"return_pct")); L.append(g.nsmallest(K,"return_pct"))
    return pd.concat(W),pd.concat(L)

def main():
    t0=time.time()
    print("=== iter-042: What's special about H2 winners? ===\n", flush=True)
    p=pd.read_parquet(PREDS, columns=["symbol","open_time","return_pct"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p.open_time>=H1S)&(p.open_time<H2E)&(p.open_time.dt.hour%4==0)&(p.open_time.dt.minute==0)]
    pf=pd.read_parquet(PANEL, columns=["symbol","open_time"]+V0)
    pf["open_time"]=pd.to_datetime(pf["open_time"],utc=True)
    pf=pf.sort_values(["symbol","open_time"]).reset_index(drop=True)
    pf,FEATS=engineer(pf)
    d=p.merge(pf,on=["symbol","open_time"],how="left")
    print(f"  {len(d):,} rows, {len(FEATS)} features\n", flush=True)

    W1,L1=win_lose(d,H1S,H2S); W2,L2=win_lose(d,H2S,H2E)
    # ---- A: per-feature AUC H1 vs H2, sorted by H2 separation, flag regime-different ----
    print("=== A: rank-AUC winner-vs-loser (sorted by H2 |AUC-0.5|); ΔAUC flags new-regime-special ===\n")
    print(f"  {'feature':<22}{'H1 AUC':>8}{'H2 AUC':>8}{'ΔAUC':>8}  note")
    print("  "+"-"*60)
    rows=[]
    for f in FEATS:
        _,_,a1=rank_auc(W1,L1,f); _,_,a2=rank_auc(W2,L2,f)
        if np.isnan(a2): continue
        rows.append((f,a1,a2,a2-a1))
    for f,a1,a2,da in sorted(rows,key=lambda x:-abs(x[2]-0.5)):
        note=""
        if abs(a2-0.5)>0.05: note="H2-discriminative"
        if abs(da)>0.05: note+=(" / " if note else "")+("STRONGER in H2" if abs(a2-0.5)>abs(a1-0.5) else "weaker in H2")
        print(f"  {f:<22}{a1:>8.3f}{a2:>8.3f}{da:>+8.3f}  {note}")

    # ---- B: multivariate logistic separability H1 vs H2 ----
    print(f"\n=== B: multivariate logistic (winner vs loser), 5-fold CV AUC + top coefs ===\n")
    for label,W,L in [("H1",W1,L1),("H2",W2,L2)]:
        Xy=pd.concat([W.assign(_y=1),L.assign(_y=0)])[FEATS+["_y"]].dropna()
        X=(Xy[FEATS]-Xy[FEATS].mean())/Xy[FEATS].std().replace(0,1)
        y=Xy["_y"].values
        lr=LogisticRegression(max_iter=2000,C=1.0)
        auc=cross_val_score(lr,X,y,cv=5,scoring="roc_auc").mean()
        lr.fit(X,y); coef=pd.Series(lr.coef_[0],index=FEATS).sort_values(key=abs,ascending=False)
        print(f"  {label}: CV AUC={auc:.3f}  (n={len(y):,})  top coefs:")
        for f,c in coef.head(6).items(): print(f"      {f:<22}{c:>+7.2f}")
    print(f"  (higher AUC = winners more jointly separable from losers in that regime)")

    # ---- C: symbol concentration + funding state of H2 winners ----
    print(f"\n=== C: H2 winner concentration + funding state ===\n")
    topc=W2["symbol"].value_counts()
    print(f"  H2 distinct winner symbols: {W2['symbol'].nunique()} of {d[(d.open_time>=H2S)]['symbol'].nunique()} traded")
    print(f"  top-10 most-frequent H2 winners: {', '.join(f'{s}({n})' for s,n in topc.head(10).items())}")
    print(f"  H2 winners funding_rate: {100*(W2['funding_rate']<0).mean():.0f}% negative (vs losers {100*(L2['funding_rate']<0).mean():.0f}%)")
    print(f"  H2 winners ret_30d mean: {W2['ret_30d'].mean()*100:+.1f}% (losers {L2['ret_30d'].mean()*100:+.1f}%)")
    print(f"  H2 winners corr_to_btc mean: {W2['corr_to_btc_1d'].mean():.3f} (losers {L2['corr_to_btc_1d'].mean():.3f})")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
