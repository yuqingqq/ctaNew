"""Presentation readings from frozen exhaustive tables, no new fitted rule."""
import json
import pandas as pd
import common as C
r=json.loads((C.H/'readings.json').read_text());p=pd.read_parquet(C.OUT/'pairs.parquet');out={}
def select(asset,axis,role,clock='pre',scope='all19'):
 return next(x for x in r['attribution'] if(x['scope'],x['day'],x['asset'],x['axis'],x['role'],x['clock'])==(scope,'all',asset,axis,role,clock))
out['passive_exit']={c:select('pf','exit','MM',c) for c in C.CLOCKS}
out['volatile_exit_majority_all_clocks']=all(next(x for x in v['rows'] if x['vol']=='volatile')['positive_share']>.5 for v in out['passive_exit'].values())
a=p[(p.clock=='pre')&(p.asset=='pf')&(p.axis=='exit')&(p.vol=='volatile')];g=a.groupby(['day','mid']).net_c.sum()/100;top=g.nlargest(2)
out['volatile_exit_by_day_usd']={day:float(g.loc[day].sum()) if day in g.index.get_level_values(0) else None for day in ['20261024','20261025','20261026','20261027']}
out['top_two_volatile_exit_usd']={str(mid):float(v) for (day,mid),v in top.items()};out['excluding_top_two_volatile_exit_usd']=float(g.sum()-top.sum());out['top_three_abs_contribution_fraction']=float(g.abs().nlargest(3).sum()/g.abs().sum())
out['top_two_day_sensitive']=out['excluding_top_two_volatile_exit_usd']<0
out['volatile_entry_marked_usd']={c:next(x['marked_usd'] for x in r['acquisition'] if(x['scope'],x['day'],x['clock'],x['asset'],x['vol'])==('all19','all',c,'pf','volatile')) for c in C.CLOCKS}
out['volatile_entry_marked_negative_all_clocks']=all(x<0 for x in out['volatile_entry_marked_usd'].values())
out['entry_calm_vs_volatile_not_clock_stable']=len({next(v['net_c_per_share'] for v in select('pf','entry','all',c)['rows'] if v['vol']=='calm')>next(v['net_c_per_share'] for v in select('pf','entry','all',c)['rows'] if v['vol']=='volatile') for c in C.CLOCKS})>1
out['btc_volatile_positive_share']={axis:{c:next(x['positive_share'] for x in select('btc',axis,'all',c)['rows'] if x['vol']=='volatile') for c in C.CLOCKS} for axis in ['entry','exit']}
out['named_market']=dict(entry=select('pf','entry','all',scope='3789484'),exit=select('pf','exit','all',scope='3789484'))
out['validation']=False
C.dump(C.H/'claims.json',out);print(json.dumps(C.clean(out),indent=2))
