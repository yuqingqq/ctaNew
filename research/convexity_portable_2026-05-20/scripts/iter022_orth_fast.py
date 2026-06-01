"""Fast vectorized residual-IC: is rel_ret_1d orthogonal to pred, and does its
information survive after removing pred (cross-sectionally)? Spearman via ranks."""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None

pan = pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet',
                      columns=['symbol','open_time','return_1d','alpha_vs_btc_realized'])
pred = pd.read_parquet('research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet',
                       columns=['symbol','open_time','pred','alpha_A'])
m = pred.merge(pan, on=['symbol','open_time'], how='left').dropna(subset=['return_1d','pred','alpha_A'])
m['ot']=m['open_time']
g=m.groupby('ot')
m['rel']=m['return_1d']-g['return_1d'].transform('mean')

# within-cycle ranks (Spearman = Pearson on ranks)
for c in ['rel','pred','alpha_A']:
    m[c+'_r']=g[c].rank()
g=m.groupby('ot')
# center ranks
for c in ['rel_r','pred_r','alpha_A_r']:
    m[c+'_c']=m[c]-g[c].transform('mean')

def pear(a,b):  # pooled per-cycle correlation averaged via per-cycle then mean
    gg=m.groupby('ot')
    num=(m[a]*m[b]).groupby(m['ot']).sum()
    da=np.sqrt((m[a]**2).groupby(m['ot']).sum()); db=np.sqrt((m[b]**2).groupby(m['ot']).sum())
    ic=(num/(da*db)).replace([np.inf,-np.inf],np.nan).dropna()
    return ic.mean(), ic.mean()/(ic.std()/np.sqrt(len(ic))), len(ic)

print("IC(rel -> alpha_A):", pear('rel_r_c','alpha_A_r_c'))
print("IC(pred -> alpha_A):", pear('pred_r_c','alpha_A_r_c'))
print("XS corr(rel,pred):", pear('rel_r_c','pred_r_c'))

# residualize rel-rank on pred-rank per cycle (vectorized OLS via groupby sums)
gg=m.groupby('ot')
beta=( (m['rel_r_c']*m['pred_r_c']).groupby(m['ot']).transform('sum') /
       (m['pred_r_c']**2).groupby(m['ot']).transform('sum') )
m['rel_resid']=m['rel_r_c']-beta*m['pred_r_c']
print("IC(rel AFTER removing pred -> alpha_A):", pear('rel_resid','alpha_A_r_c'))
