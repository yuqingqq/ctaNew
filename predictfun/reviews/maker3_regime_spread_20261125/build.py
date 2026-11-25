import collections,json,time
import numpy as np
import pandas as pd
import common as C
def rows(data,f,bounds):
 pair=[];fill=[];res=[];exposure=[]
 refs={int(x.m):x for x in C.refs().itertuples()}
 for case in data['cases']:
  mid=case['mid'];day=case['day'];ws=case['ws'];g=f[(f.mid==mid)&(f.day==day)];grid={int(x['k']):x for x in g.to_dict('records')};fs={x['id']:x for x in case['fills']};used=collections.Counter()
  for z in case['closes']:used[z['entry']]+=z['q'];used[z['exit']]+=z['q']
  ctx={(x['id'],c,a):C.context(x,ws,grid,c,a,bounds) for x in case['fills'] for c in C.CLOCKS for a in C.ASSETS}
  for x in case['fills']:
   remaining=x['q']-used[x['id']];assert remaining>=-1e-7
   for clock in C.CLOCKS:
    for asset in C.ASSETS:
     t=ctx[x['id'],clock,asset];v=x['book'][clock]['mid'];base=dict(day=day,mid=mid,id=x['id'],hash=x['hash'],role=x['role'],clock=clock,asset=asset,vol=t['vol'],spread=t['spread'],anchor_ms=t['anchor_ms'],condition_ms=C.clock_ms(x,clock),value=t['value'])
     fill.append(dict(**base,q=x['q'],fee_c=x['fee_c'],edge_c=x['s']*x['q']*(v-x['price']) if v is not None else 0,valid_q=x['q'] if v is not None else 0))
     if remaining>1e-8:
      value=x['s']*remaining*(case['mark']-x['price'])-remaining*x['fee_c']/x['q']
      res.append(dict(**base,q=remaining,net_c=value,positive_c=max(0,value),negative_c=min(0,value)))
  for j,z in enumerate(case['closes']):
   a,b=fs[z['entry']],fs[z['exit']]
   for clock in C.CLOCKS:
    value=z['by_clock'][clock];valid=value['kind']!='unknown'
    for asset in C.ASSETS:
     ca,cb=ctx[a['id'],clock,asset],ctx[b['id'],clock,asset]
     for axis,state in [('entry',ca),('exit',cb)]:
      pair.append(dict(day=day,mid=mid,id=f'{mid}|{j}',entry=a['id'],exit=b['id'],entry_hash=a['hash'],exit_hash=b['hash'],role='MM' if a['role']==b['role']=='maker' else 'any_taker',clock=clock,asset=asset,axis=axis,vol=state['vol'],spread=state['spread'],entry_vol=ca['vol'],exit_vol=cb['vol'],entry_spread=ca['spread'],exit_spread=cb['spread'],q=z['q'],net_c=z['net_c'],positive_c=max(0,z['net_c']),negative_c=min(0,z['net_c']),gross_c=z['gross_c'],fee_c=z['fee_c'],hold_qs=z['q']*(z['exit_t']-z['entry_t']),edge_c=value['entry_edge_c']+value['exit_edge_c'] if valid else 0,drift_c=value['drift_c'] if valid else 0,valid_q=z['q'] if valid else 0))
  ref=refs[mid]
  for r in g.to_dict('records'):
   sec=max(0,min(r['t_ms']+1000,ref.end_ms)-max(r['t_ms'],ref.start_ms))/1000
   if not sec:continue
   for asset in C.ASSETS:exposure.append(dict(day=day,mid=mid,k=r['k'],asset=asset,vol=C.vb(r[asset],bounds[asset]),spread=C.sb(r['spread']),seconds=sec))
 return {k:pd.DataFrame(v) for k,v in [('pairs',pair),('fills',fill),('residual',res),('exposure',exposure)]}
def views(df,roles):
 for scope,ss in [('all19',df),('3789484',df[df.mid==3789484])]:
  for day in ['all']+sorted(ss.day.unique()):
   ds=ss if day=='all' else ss[ss.day==day]
   for role in ['all']+roles:
    g=ds if role=='all' else ds[ds.role==role]
    if len(g):yield dict(scope=scope,day=day,role=role),g
def aggregate(df,kind):
 keys={'pairs':['clock','asset','axis','vol','spread'],'transitions':['clock','asset','entry_vol','exit_vol'],'fills':['clock','asset','vol','spread'],'residual':['clock','asset','vol','spread'],'exposure':['asset','vol','spread']}[kind]
 metrics=[k for k in ['q','net_c','positive_c','negative_c','gross_c','fee_c','hold_qs','edge_c','drift_c','valid_q','seconds'] if k in df]
 roles=['MM','any_taker'] if kind in ['pairs','transitions'] else ['maker','taker'] if kind in ['fills','residual'] else []
 out=[]
 for prefix,g in views(df,roles):
  for key,x in g.groupby(keys,sort=True,dropna=False):
   v=dict(**prefix,**dict(zip(keys,key)),rows=len(x),markets=x.mid.nunique(),**{m:float(x[m].sum()) for m in metrics})
   if kind in ['pairs','transitions']:
    v['hashes']=len(set(x.entry_hash)|set(x.exit_hash));v['leg_ids']=len(set(x.entry)|set(x.exit));v['thin']=v['markets']<3 or v['hashes']<30
    mm=x.groupby('mid').net_c.sum().abs();v['top3_abs_market_share']=float(mm.nlargest(3).sum()/mm.sum()) if mm.sum()>1e-9 else None
   elif kind in ['fills','residual']:v['hashes']=x.hash.nunique()
   if 'net_c' in v:v['net_c_per_share']=v['net_c']/v['q'] if v['q'] else None
   if 'hold_qs' in v:v['mean_hold_s']=v['hold_qs']/v['q'] if v['q'] else None
   out.append(v)
 return pd.DataFrame(out)
def readings(pair,residual,exp):
 out=[];acq=[]
 for pref,g in views(pair,['MM','any_taker']):
  for (clock,asset,axis),x in g.groupby(['clock','asset','axis']):
   den=x.positive_c.sum();qden=x.q.sum();records=[]
   for vol in C.VOLS:
    v=x[x.vol==vol];n=v.net_c.sum();q=v.q.sum();pos=v.positive_c.sum()
    rows=dict(vol=vol,matched_shares=q,net_usd=n/100,positive_usd=pos/100,negative_usd=v.negative_c.sum()/100,positive_share=pos/den if den else None,matched_share=q/qden if qden else None,net_c_per_share=n/q if q else None,markets=v.mid.nunique(),hashes=len(set(v.entry_hash)|set(v.exit_hash)))
    records.append(rows)
   leader=max(records,key=lambda r:r['positive_usd']);out.append(dict(**pref,clock=clock,asset=asset,axis=axis,rows=records,positive_leader=leader['vol'],majority=leader['positive_share'] is not None and leader['positive_share']>.5))
 # Full-book acquisition view includes all residuals, never a winner-only cohort.
 for pref,g in views(pair[pair.axis=='entry'],[]):
  for (clock,asset),x in g.groupby(['clock','asset']):
   r=residual[(residual.clock==clock)&(residual.asset==asset)]
   e=exp[exp.asset==asset]
   if pref['scope']!='all19':r=r[r.mid==int(pref['scope'])];e=e[e.mid==int(pref['scope'])]
   if pref['day']!='all':r=r[r.day==pref['day']];e=e[e.day==pref['day']]
   for vol in C.VOLS:
    xx=x[x.vol==vol];rr=r[r.vol==vol];ee=e[e.vol==vol];net=xx.net_c.sum()+rr.net_c.sum()
    acq.append(dict(**pref,clock=clock,asset=asset,vol=vol,completed_usd=xx.net_c.sum()/100,residual_usd=rr.net_c.sum()/100,marked_usd=net/100,reference_seconds=ee.seconds.sum()))
 return dict(attribution=out,acquisition=acq,validation=False)
def run():
 C.limits();start=time.process_time();C.bind();bounds=json.loads((C.H/'bounds.json').read_text());bm=json.loads((C.H/'bounds_manifest.json').read_text());assert C.sha(C.H/'bounds.json')==bm['sha256'];assert C.sha(C.OUT/'features.parquet')==bm['features_sha256']
 assert (C.H/'bounds.json').read_bytes()==__import__('subprocess').check_output(['git','show','HEAD:'+str((C.H/'bounds.json').relative_to(C.R))],cwd=C.R)
 assert not (C.H/'readings.json').exists();C.OUT.mkdir(exist_ok=True);f=pd.read_parquet(C.OUT/'features.parquet');data=C.loaded();rr=rows(data,f,bounds)
 for kind,d in rr.items():d.to_parquet(C.OUT/(kind+'.parquet'),index=False);aggregate(d,kind).to_csv(C.H/(kind+'_cells.csv'),index=False)
 aggregate(rr['pairs'][rr['pairs'].axis=='entry'],'transitions').to_csv(C.H/'transitions.csv',index=False)
 result=readings(rr['pairs'],rr['residual'],rr['exposure']);C.dump(C.H/'readings.json',result)
 C.dump(C.H/'build.json',dict(raw_rows={k:len(v) for k,v in rr.items()},markets=len(data['cases']),cpu_s=time.process_time()-start,hashes={p.name:C.sha(p) for p in C.OUT.glob('*.parquet')}))
 print('built',[(k,len(v)) for k,v in rr.items()]);assert __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss<1500*1024
if __name__=='__main__':run()
