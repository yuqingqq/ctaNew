"""iter-022: era-stability of the reversal lead-lag IC. Split EXT 2021-26 by year and
check the cross-sectional IC sign is stable across regimes (the test that killed funding/mom180)."""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None

p = pd.read_parquet('outputs/vBTC_features/panel_ext2021_v0.parquet',
                    columns=['symbol','open_time','return_1d','alpha_vs_btc_realized'])
p['ot']=pd.to_datetime(p['open_time'])
uts=np.sort(p['ot'].unique()); grid=set(uts[::48]); p=p[p['ot'].isin(grid)].copy()
g=p.groupby('ot'); p['rel']=p['return_1d']-g['return_1d'].transform('mean')
p['yr']=p['ot'].dt.year

def ic(df):
    gg=df.dropna(subset=['rel','alpha_vs_btc_realized']).groupby('ot')
    ics=gg.apply(lambda x: x['rel'].corr(x['alpha_vs_btc_realized'],method='spearman') if x['rel'].nunique()>3 else np.nan).dropna()
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))), len(ics)

print("EXT reversal IC by YEAR (sign must stay negative for era-stability):")
for yr in sorted(p['yr'].unique()):
    m,t,n=ic(p[p['yr']==yr]); print(f"  {yr}: IC {m:+.4f}  t {t:+.2f}  n {n}")
