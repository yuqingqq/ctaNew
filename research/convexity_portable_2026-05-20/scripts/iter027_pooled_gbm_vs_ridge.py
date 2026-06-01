"""
iter-027 DECISIVE Phase-2 model test: pooled LightGBM vs per-sym Ridge.

Question: does a pooled GBM (sym_id + BASE/cohort + new transport-stable features,
exploiting nonlinear interactions per Gu-Kelly-Xiu 2020) produce a pred whose
out-of-sample cross-sectional IC on the 4h alpha-residual BEATS the per-sym Ridge
baseline -- on BOTH HL70 and EXT (transport), walk-forward (no look-ahead)?

If the GBM pred OOS XS-IC does NOT exceed Ridge on both universes, a model rebuild
will not beat baseline -> NO-CANDIDATE on the model axis.

Walk-forward: expanding folds, 1-day embargo. Target = target_z (per-sym z of 4h
alpha-residual), same as V0. Evaluate XS-IC of pred vs raw alpha_vs_btc_realized.
"""
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
pd.options.mode.chained_assignment=None

GRID=48
BASE=['return_1d','atr_pct','vwap_slope_96','bars_since_high','autocorr_pctile_7d',
      'obv_z_1d','corr_to_btc_1d','beta_to_btc_change_5d','idio_vol_to_btc_1h',
      'idio_vol_to_btc_1d','funding_rate','funding_rate_z_7d','funding_rate_1d_change']
COHORT=['rvol_7d','ret_3d','btc_rvol_7d']

def load(path):
    import pyarrow.parquet as pq
    have=set(pq.ParquetFile(path).schema.names)
    cols=['symbol','open_time','alpha_vs_btc_realized','return_pct']+ \
         (['target_z'] if 'target_z' in have else [])+ \
         [c for c in BASE+COHORT if c in have]
    p=pd.read_parquet(path, columns=cols).sort_values(['symbol','open_time'])
    p['ot']=pd.to_datetime(p['open_time'])
    uts=np.sort(p['ot'].unique()); grid=set(uts[::GRID])
    p['trail4h']=p.groupby('symbol')['return_pct'].shift(GRID)
    g=p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
    grp=g.groupby('symbol')
    # NEW transport-stable features
    g['rev_6']=-grp['trail4h'].transform(lambda s:s.rolling(6,min_periods=4).sum())
    g['rev_2']=-grp['trail4h'].transform(lambda s:s.rolling(2,min_periods=2).sum())
    g['mom_42']=grp['trail4h'].transform(lambda s:s.rolling(42,min_periods=28).sum())
    g['rv_6']=grp['trail4h'].transform(lambda s:s.rolling(6,min_periods=4).std())
    g['vov']=grp['rv_6'].transform(lambda s:s.rolling(18,min_periods=12).std())
    g['sym_id']=g['symbol'].astype('category').cat.codes
    return g, [c for c in BASE+COHORT if c in g.columns]

def xs_ic_series(df, predcol, tgt='alpha_vs_btc_realized'):
    d=df.dropna(subset=[predcol,tgt])
    ics=d.groupby('ot').apply(lambda x:x[predcol].corr(x[tgt],method='spearman')
                              if x[predcol].nunique()>3 else np.nan).dropna()
    return ics

def walkforward(g, feats_base, NF=9):
    g=g.dropna(subset=['target_z']).copy()
    ots=np.sort(g['ot'].unique())
    bounds=np.linspace(0,len(ots),NF+1).astype(int)
    NEW=['rev_2','rev_6','mom_42','vov']
    feats_gbm=feats_base+NEW+['sym_id']
    feats_gbmnf=feats_base+['sym_id']  # GBM WITHOUT new feats (model-change only)
    pred_ridge=pd.Series(np.nan,index=g.index)
    pred_gbm=pd.Series(np.nan,index=g.index)
    pred_gbmnf=pd.Series(np.nan,index=g.index)
    for k in range(2,NF):  # expanding, start after 2 blocks
        tr_end=ots[bounds[k]]
        emb=tr_end - pd.Timedelta('1D')
        te_lo=ots[bounds[k]]; te_hi=ots[bounds[k+1]-1]
        tr=g[g['ot']<emb]; te=g[(g['ot']>=te_lo)&(g['ot']<=te_hi)]
        if len(tr)<5000 or len(te)<200: continue
        # ---- per-sym Ridge (V0-style) ----
        for s,trs in tr.groupby('symbol'):
            tes=te[te['symbol']==s]
            if len(tes)==0 or len(trs)<200: continue
            X=trs[feats_base].fillna(0.0).values; y=trs['target_z'].values
            r=Ridge(alpha=10.0).fit(X,y)
            pred_ridge.loc[tes.index]=r.predict(tes[feats_base].fillna(0.0).values)
        # ---- pooled GBM (with + without new feats) ----
        for feats,store in [(feats_gbm,pred_gbm),(feats_gbmnf,pred_gbmnf)]:
            Xt=tr[feats].fillna(0.0); yt=tr['target_z'].clip(-5,5)
            m=lgb.LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=0.03,
                                min_child_samples=200,subsample=0.8,colsample_bytree=0.8,
                                reg_lambda=5.0,n_jobs=4,verbose=-1)
            m.fit(Xt,yt,categorical_feature=['sym_id'])
            store.loc[te.index]=m.predict(te[feats].fillna(0.0))
    g['pred_ridge']=pred_ridge; g['pred_gbm']=pred_gbm; g['pred_gbmnf']=pred_gbmnf
    return g

print("="*78); print("iter-027 POOLED GBM vs per-sym RIDGE  (OOS walk-forward XS-IC)"); print("="*78)
for name,path in [('EXT','outputs/vBTC_features/panel_ext2021_v0.parquet'),
                  ('HL70','outputs/vBTC_features/panel_hl70.parquet')]:
    g,fb=load(path)
    print(f"\n### {name}  syms={g.symbol.nunique()} cycles={g.ot.nunique()} feats_base={len(fb)}")
    if 'target_z' not in g.columns or g['target_z'].notna().sum()<1000:
        # HL70 panel has no target_z; build PIT per-sym z of FORWARD alpha as target.
        # alpha_vs_btc_realized is the forward 4h alpha; z by trailing per-sym mean/std
        # (shifted to exclude the current label) -- PIT.
        grp=g.groupby('symbol')['alpha_vs_btc_realized']
        mu=grp.transform(lambda s:s.shift(1).rolling(2000,min_periods=200).mean())
        sd=grp.transform(lambda s:s.shift(1).rolling(2000,min_periods=200).std()).replace(0,np.nan)
        g['target_z']=(g['alpha_vs_btc_realized']-mu)/sd
        print(f"   (built PIT target_z for {name}; notna={g['target_z'].notna().sum()})")
    g=walkforward(g,fb)
    for col,lab in [('pred_ridge','Ridge(V0)'),('pred_gbmnf','GBM(base only)'),('pred_gbm','GBM(+new feats)')]:
        ics=xs_ic_series(g,col)
        if len(ics)<5: print(f"   {lab:<18} n/a"); continue
        t=ics.mean()/(ics.std()/np.sqrt(len(ics)))
        print(f"   {lab:<18} OOS XS-IC {ics.mean():+.4f}  t{t:+.1f}  cyc{len(ics)}")
