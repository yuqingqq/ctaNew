import collections,copy,json,math,time
import numpy as np
import pandas as pd
import common as C
def close(a,b):
 if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
 else:assert b is not None and not pd.isna(b) and abs(float(a)-float(b))<1e-7*max(1,abs(float(a))),(a,b)
def feature_check(p,f):
 for key,g in p.groupby(['day','mid']):
  ff=f[(f.day==key[0])&(f.mid==key[1])].sort_values('k');assert len(ff)==300
  for i,r in enumerate(ff.itertuples()):
   assert r.t_ms==g.t_ms.iloc[i]
   w=g.iloc[max(0,i-10):i+1];m=(w.bid0+w.ask0).to_numpy()*50
   pf=math.sqrt(sum((m[j]-m[j-1])**2 for j in range(1,len(m)))/10) if i>=10 and w.kok0.all() and np.isfinite(m).all() else None
   b=g.iloc[max(0,i-9):i+1];btc=math.sqrt(sum(x*x for x in b.r_1s0)/10) if i>=10 and b.bok0.all() and b.r_1s0.notna().all() else None
   close(r.pf,pf);close(r.btc,btc)
def expected_context(row,f,features,bounds):
 t=f['hi_ms'] if row.clock=='upper' else None if f['lo_ms'] is None else f['lo_ms']-(1 if row.clock=='pre' else 1000)
 if t is None:val=None;anchor=None
 else:
  ix=np.searchsorted(features.t_ms.to_numpy(),t,side='right')-1
  val=None if ix<0 else features[row.asset].iloc[ix];anchor=None if ix<0 else int(features.t_ms.iloc[ix])
 b=bounds[row.asset];v='unknown' if val is None or not np.isfinite(val) else 'calm' if val<=b['q25'] else 'volatile' if val>b['q75'] else 'typical'
 s=f['book'][row.clock]['spread'];s='unknown' if s is None else 'narrow' if s<1+1e-7 else 'wide' if s>4+1e-7 else 'intermediate'
 return v,s,val,anchor,t
def verify_table(df,path,keys,roles):
 actual=pd.read_csv(path,float_precision='round_trip',keep_default_na=False);found=0
 for scope in ['all19','3789484']:
  ss=df if scope=='all19' else df[df.mid==3789484]
  for day in ['all']+sorted(ss.day.unique()):
   ds=ss if day=='all' else ss[ss.day==day]
   for role in ['all']+roles:
    rs=ds if role=='all' else ds[ds.role==role]
    aa=actual[(actual.scope.astype(str)==scope)&(actual.day.astype(str)==day)&(actual.role==role)]
    groups={k:x for k,x in rs.groupby(keys,dropna=False,sort=True)};assert len(aa)==len(groups)
    for r in aa.to_dict('records'):
     g=groups[tuple(r[k] for k in keys)];found+=1;assert r['rows']==len(g) and r['markets']==g.mid.nunique()
     for field in ['q','net_c','positive_c','negative_c','gross_c','fee_c','hold_qs','edge_c','drift_c','valid_q','seconds']:
      if field in g:close(r[field],sum(g[field]))
     if 'entry_hash' in g:
      assert r['hashes']==len(set(g.entry_hash)|set(g.exit_hash));assert r['leg_ids']==len(set(g.entry)|set(g.exit));assert str(r['thin'])==str(g.mid.nunique()<3 or len(set(g.entry_hash)|set(g.exit_hash))<30)
      vals=[abs(sum(x.net_c)) for _,x in g.groupby('mid')];den=sum(vals);ex=sum(sorted(vals,reverse=True)[:3])/den if den>1e-9 else None
      close(None if r['top3_abs_market_share']=='' else r['top3_abs_market_share'],ex)
     elif 'hash' in g:assert r['hashes']==g.hash.nunique()
     if 'net_c' in g:close(r['net_c_per_share'],sum(g.net_c)/sum(g.q))
     if 'hold_qs' in g:close(r['mean_hold_s'],sum(g.hold_qs)/sum(g.q))
 assert found==len(actual);return found
def run():
 C.limits();start=time.process_time();C.bind();data=C.loaded();bounds=json.loads((C.H/'bounds.json').read_text());f=pd.read_parquet(C.OUT/'features.parquet');p=C.neutral();feature_check(p,f)
 for asset in C.ASSETS:
  x=f.loc[f.day.isin(['20261024','20261025']),asset].dropna();assert len(x)==bounds[asset]['n']
  for key,q in [('q25',.25),('q75',.75)]:close(bounds[asset][key],x.quantile(q))
 rr={kind:pd.read_parquet(C.OUT/(kind+'.parquet')) for kind in ['pairs','fills','residual','exposure']};cases={c['mid']:c for c in data['cases']};source={(c['mid'],a['id']):a for c in data['cases'] for a in c['fills']};gr={int(mid):g for mid,g in f.groupby('mid')};allocation=collections.Counter()
 for c in data['cases']:
  for z in c['closes']:allocation[c['mid'],z['entry']]+=z['q'];allocation[c['mid'],z['exit']]+=z['q']
 assert len(rr['fills'])==6*len(source);assert not rr['fills'].duplicated(['mid','id','clock','asset']).any()
 for row in rr['fills'].itertuples():
  a=source[row.mid,row.id];close(row.q,a['q']);close(row.fee_c,a['fee_c']);v,s,value,anchor,t=expected_context(row,a,gr[row.mid],bounds);assert(row.vol,row.spread)==(v,s);close(row.value,value);close(row.anchor_ms,anchor);close(row.condition_ms,t);assert anchor is None or anchor<=t
  z=a['book'][row.clock]['mid'];close(row.edge_c,a['s']*a['q']*(z-a['price']) if z is not None else 0);close(row.valid_q,a['q'] if z is not None else 0)
 nclose=sum(len(c['closes']) for c in data['cases']);assert len(rr['pairs'])==12*nclose;assert not rr['pairs'].duplicated(['id','clock','asset','axis']).any()
 for row in rr['pairs'].itertuples():
  z=cases[row.mid]['closes'][int(row.id.split('|')[1])];a,b=source[row.mid,z['entry']],source[row.mid,z['exit']]
  for key in ['q','net_c','gross_c','fee_c']:close(getattr(row,key),z[key])
  ca=expected_context(row,a,gr[row.mid],bounds);cb=expected_context(row,b,gr[row.mid],bounds);state=ca if row.axis=='entry' else cb
  assert(row.vol,row.spread,row.entry_vol,row.exit_vol,row.entry_spread,row.exit_spread)==(state[0],state[1],ca[0],cb[0],ca[1],cb[1]);assert row.role==('MM' if a['role']==b['role']=='maker' else 'any_taker')
  close(row.positive_c,max(0,z['net_c']));close(row.negative_c,min(0,z['net_c']));bcl=z['by_clock'][row.clock];valid=bcl['kind']!='unknown';close(row.edge_c,bcl['entry_edge_c']+bcl['exit_edge_c'] if valid else 0);close(row.drift_c,bcl['drift_c'] if valid else 0);close(row.valid_q,z['q'] if valid else 0)
 for row in rr['residual'].itertuples():
  a=source[row.mid,row.id];rem=a['q']-allocation[row.mid,row.id];close(row.q,rem);close(row.net_c,a['s']*rem*(cases[row.mid]['mark']-a['price'])-rem*a['fee_c']/a['q']);v,s,*_=expected_context(row,a,gr[row.mid],bounds);assert(row.vol,row.spread)==(v,s)
 assert len(rr['residual'])==6*sum(a['q']-allocation[key]>1e-8 for key,a in source.items());assert not rr['residual'].duplicated(['mid','id','clock','asset']).any()
 refs={int(x.m):x for x in C.refs().itertuples()}
 for r in rr['exposure'].itertuples():
  a=gr[r.mid].iloc[r.k];ref=refs[r.mid];sec=max(0,min(a.t_ms+1000,ref.end_ms)-max(a.t_ms,ref.start_ms))/1000;close(r.seconds,sec);assert r.vol==C.vb(a[r.asset],bounds[r.asset]) and r.spread==C.sb(a.spread)
 for asset in C.ASSETS:close(rr['exposure'].loc[rr['exposure'].asset==asset,'seconds'].sum(),sum((r.end_ms-r.start_ms)/1000 for r in refs.values()))
 for (mid,clock,asset),g in rr['pairs'][rr['pairs'].axis=='entry'].groupby(['mid','clock','asset']):
  r=rr['residual'];rem=r[(r.mid==mid)&(r.clock==clock)&(r.asset==asset)].net_c.sum();close(g.net_c.sum(),cases[mid]['totals']['completed_c']);close(g.net_c.sum()+rem,cases[mid]['totals']['net_c'])
 checks={}
 for kind,keys,roles in [('pairs',['clock','asset','axis','vol','spread'],['MM','any_taker']),('fills',['clock','asset','vol','spread'],['maker','taker']),('residual',['clock','asset','vol','spread'],['maker','taker']),('exposure',['asset','vol','spread'],[])]:checks[kind]=verify_table(rr[kind],C.H/(kind+'_cells.csv'),keys,roles)
 checks['transitions']=verify_table(rr['pairs'][rr['pairs'].axis=='entry'],C.H/'transitions.csv',['clock','asset','entry_vol','exit_vol'],['MM','any_taker'])
 reading=json.loads((C.H/'readings.json').read_text())
 for r in reading['attribution']:
  a=rr['pairs'];a=a[(a.clock==r['clock'])&(a.asset==r['asset'])&(a.axis==r['axis'])]
  if r['scope']!='all19':a=a[a.mid==int(r['scope'])]
  if r['day']!='all':a=a[a.day==r['day']]
  if r['role']!='all':a=a[a.role==r['role']]
  for cell in r['rows']:
   v=a[a.vol==cell['vol']];close(cell['matched_shares'],v.q.sum());close(cell['net_usd'],v.net_c.sum()/100);close(cell['positive_share'],v.positive_c.sum()/a.positive_c.sum() if a.positive_c.sum() else None);close(cell['net_c_per_share'],v.net_c.sum()/v.q.sum() if v.q.sum() else None)
  leader=max(r['rows'],key=lambda x:x['positive_usd']);assert r['positive_leader']==leader['vol'];assert r['majority']==(leader['positive_share'] is not None and leader['positive_share']>.5)
 for r in reading['acquisition']:
  a=rr['pairs'];a=a[(a.axis=='entry')&(a.clock==r['clock'])&(a.asset==r['asset'])&(a.vol==r['vol'])];b=rr['residual'];b=b[(b.clock==r['clock'])&(b.asset==r['asset'])&(b.vol==r['vol'])]
  if r['scope']!='all19':a=a[a.mid==int(r['scope'])];b=b[b.mid==int(r['scope'])]
  if r['day']!='all':a=a[a.day==r['day']];b=b[b.day==r['day']]
  close(r['completed_usd'],a.net_c.sum()/100);close(r['residual_usd'],b.net_c.sum()/100);close(r['marked_usd'],(a.net_c.sum()+b.net_c.sum())/100)
 # Real-copy refusals exercise raw-feature and aggregate accounting instruments.
 bad=f.copy();i=bad.pf.first_valid_index();bad.loc[i,'pf']+=1;caught=0
 try:feature_check(p,bad)
 except AssertionError:caught+=1
 for path,col in [(C.H/'pairs_cells.csv','net_c'),(C.H/'residual_cells.csv','q'),(C.H/'transitions.csv','markets')]:
  tab=pd.read_csv(path,float_precision='round_trip',keep_default_na=False);tab.loc[0,col]=float(tab.loc[0,col])+123;tmp=C.OUT/('corrupt_'+path.name);tab.to_csv(tmp,index=False)
  kind='transitions' if 'transitions' in path.name else 'pairs' if 'pairs' in path.name else 'residual';df=rr['pairs'][rr['pairs'].axis=='entry'] if kind=='transitions' else rr[kind];keys=['clock','asset','entry_vol','exit_vol'] if kind=='transitions' else ['clock','asset','axis','vol','spread'] if kind=='pairs' else ['clock','asset','vol','spread']
  try:verify_table(df,tmp,keys,['MM','any_taker'] if kind in ['pairs','transitions'] else ['maker','taker'])
  except AssertionError:caught+=1
 assert caught==4;C.dump(C.H/'checks.json',dict(pass_all=True,feature_rows=len(f),parent_fills=len(source),parent_closes=nclose,tables=checks,corruptions_refused=caught,cpu_s=time.process_time()-start));print('PASS',checks,'corruptions',caught)
if __name__=='__main__':run()
