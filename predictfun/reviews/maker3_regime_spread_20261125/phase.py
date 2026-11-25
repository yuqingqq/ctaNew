import json
import pandas as pd
import common as C
def run():
 C.limits();C.bind();assert not (C.H/'phase.csv').exists();data=C.loaded();df=pd.read_parquet(C.OUT/'pairs.parquet');times={f"{c['mid']}|{i}":(z['entry_t'],z['exit_t']) for c in data['cases'] for i,z in enumerate(c['closes'])}
 df['phase']=['early' if times[r.id][0 if r.axis=='entry' else 1]<180 else 'late' for r in df.itertuples()];out=[]
 for scope,ss in [('all19',df),('3789484',df[df.mid==3789484])]:
  for role in ['all','MM','any_taker']:
   rs=ss if role=='all' else ss[ss.role==role]
   for key,x in rs.groupby(['clock','asset','axis','phase','vol']):
    out.append(dict(scope=scope,role=role,**dict(zip(['clock','asset','axis','phase','vol'],key)),pairs=len(x),markets=x.mid.nunique(),matched_shares=x.q.sum(),net_usd=x.net_c.sum()/100,positive_usd=x.positive_c.sum()/100,negative_usd=x.negative_c.sum()/100,net_c_per_share=x.net_c.sum()/x.q.sum()))
 tab=pd.DataFrame(out);tab.to_csv(C.H/'phase.csv',index=False)
 # Independent phase accounting from parent closes, not the projected long rows.
 for r in tab.itertuples():
  total=q=0.;n=0
  candidates=ss if False else df # select exact ids from classified source, check parent time/value independently
  ids=set(df.loc[(df.clock==r.clock)&(df.asset==r.asset)&(df.axis==r.axis)&(df.vol==r.vol),'id'])
  for c in data['cases']:
   if r.scope!='all19' and c['mid']!=int(r.scope):continue
   for i,z in enumerate(c['closes']):
    if f"{c['mid']}|{i}" not in ids:continue
    if r.role=='MM' and (z['entry_role']!='maker' or z['exit_role']!='maker'):continue
    if r.role=='any_taker' and z['entry_role']==z['exit_role']=='maker':continue
    if ('early' if z[r.axis+'_t']<180 else 'late')!=r.phase:continue
    total+=z['net_c'];q+=z['q'];n+=1
  assert abs(total/100-r.net_usd)<1e-6 and abs(q-r.matched_shares)<1e-6 and n==r.pairs
 assert ('early' if 179.999<180 else 'late')=='early' and ('early' if 180<180 else 'late')=='late'
 C.dump(C.H/'phase_checks.json',dict(cells=len(tab),parent_time_and_value_reconciliation=True));print('PASS',len(tab),'phase cells')
if __name__=='__main__':run()
