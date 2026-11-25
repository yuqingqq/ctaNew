"""Post-run display of all frozen comparisons; no thresholds or fitted variants."""
import json
import pandas as pd
import common as C
r=json.loads((C.H/'readings.json').read_text());rows=[]
for x in r['attribution']:
 if x['scope']=='all19' and x['day']=='all' and x['role'] in ['all','MM']:
  for y in x['rows']:rows.append(dict(**{k:x[k] for k in ['asset','clock','role','axis']},vol=y['vol'],q=round(y['matched_shares'],1),net=round(y['net_usd'],2),c_share=round(y['net_c_per_share'],3) if y['net_c_per_share'] is not None else None,positive_pct=round(100*y['positive_share'],2) if y['positive_share'] is not None else None))
tab=pd.DataFrame(rows);tab.to_csv(C.H/'summary_clocks.csv',index=False)
print('CLOCKS all roles');print(tab[tab.role=='all'].to_string(index=False))
df=pd.read_parquet(C.OUT/'pairs.parquet');a=df[(df.clock=='pre')&(df.asset=='pf')];out=[]
for (day,axis,vol),g in a.groupby(['day','axis','vol']):out.append(dict(day=day,axis=axis,vol=vol,q=g.q.sum(),net=g.net_c.sum()/100,c_share=g.net_c.sum()/g.q.sum()))
dd=pd.DataFrame(out);dd.to_csv(C.H/'summary_days.csv',index=False);print('DAILY PF');print(dd.round(3).to_string(index=False))
for axis in ['entry','exit']:
 print('JOINT',axis)
 z=a[(a.axis==axis)&(a.role=='MM')];out=[]
 for (v,s),g in z.groupby(['vol','spread']):
  out.append(dict(vol=v,spread=s,q=g.q.sum(),net=g.net_c.sum()/100,c_share=g.net_c.sum()/g.q.sum(),positive=g.positive_c.sum()/100,edge=g.edge_c.sum()/100,drift=g.drift_c.sum()/100,markets=g.mid.nunique(),hashes=len(set(g.entry_hash)|set(g.exit_hash))))
 out=pd.DataFrame(out);out.to_csv(C.H/f'summary_joint_{axis}.csv',index=False);print(out.round(3).to_string(index=False))
print('PHASE');ph=pd.read_csv(C.H/'phase.csv',dtype={'scope':str});print(ph[(ph.scope=='all19')&(ph.role=='all')&(ph.clock=='pre')&(ph.asset=='pf')].round(3).to_string(index=False))
print('ACQUISITION CLOCKS');print(pd.DataFrame([x for x in r['acquisition'] if x['scope']=='all19' and x['day']=='all']).round(3).to_string(index=False))
print('PF volatile-exit net by market');print(a[(a.axis=='exit')&(a.vol=='volatile')].groupby(['day','mid']).net_c.sum().div(100).round(2).to_string())
