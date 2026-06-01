"""Decisive R-marginal: does MAX add info OVER rel_ret_1d (the iter-022 reject)?
If MAX is just a noisier rel_ret_1d, it inherits iter-022's PnL death. Residualize
fwd alpha-residual on BOTH pred AND rel_ret_1d per cycle, then IC of MAX vs that
double-residual. Also report XS corr(MAX, rel_ret_1d). HL70 + EXT."""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None
GRID=48; WINS=[3,6,12]
def build(path):
    p=pd.read_parquet(path,columns=['symbol','open_time','return_pct','return_1d','alpha_vs_btc_realized'])
    p=p.sort_values(['symbol','open_time']).copy()
    p['trail4h']=p.groupby('symbol')['return_pct'].shift(GRID)
    p['ot']=pd.to_datetime(p['open_time'])
    uts=np.sort(p['ot'].unique()); grid=set(uts[::GRID])
    g4=p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
    for w in WINS:
        g4[f'max_{w}']=g4.groupby('symbol')['trail4h'].transform(lambda s:s.rolling(w,min_periods=max(2,w//2)).max())
    gg=g4.groupby('ot'); g4['rel_ret_1d']=g4['return_1d']-gg['return_1d'].transform('mean')
    return g4
def xs_ic(df,sig,tgt):
    d=df.dropna(subset=[sig,tgt]); g=d.groupby('ot')
    ics=g.apply(lambda x:x[sig].corr(x[tgt],method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    if len(ics)<3: return np.nan,np.nan
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics)))
def resid_on(df,tgt,ctrls):
    """residualize tgt on a SET of controls per cycle (rank-space, sequential Gram-Schmidt)."""
    d=df.dropna(subset=[tgt]+ctrls).copy(); g=d.groupby('ot')
    d['r']=g[tgt].rank(); d['r']=d['r']-g['r'].transform('mean')
    for c in ctrls:
        d[c+'_r']=g[c].rank(); d[c+'_r']=d[c+'_r']-g[c+'_r'].transform('mean')
    for c in ctrls:
        cr=c+'_r'
        beta=((d['r']*d[cr]).groupby(d['ot']).transform('sum')/(d[cr]**2).groupby(d['ot']).transform('sum').replace(0,np.nan))
        d['r']=d['r']-beta*d[cr]
    return d
for name,path in [('HL70','outputs/vBTC_features/panel_hl70.parquet'),('EXT','outputs/vBTC_features/panel_ext2021_v0.parquet')]:
    g4=build(path)
    print(f"\n### {name}")
    for w in WINS:
        c=xs_ic(g4,f'max_{w}','rel_ret_1d')[0]
        # residualize fwd alpha-resid on rel_ret_1d only, then IC of max_w
        dr=resid_on(g4,'alpha_vs_btc_realized',['rel_ret_1d'])
        m,t=xs_ic(dr.assign(**{f'max_{w}':g4[f'max_{w}']}),f'max_{w}','r')
        print(f"  corr(max_{w},rel_1d)={c:+.3f} | IC(max_{w} -> alpha_resid RESID-on-rel_1d): {m:+.4f} t{t:+.2f}")
