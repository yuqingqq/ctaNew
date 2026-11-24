"""Saved-data economic decomposition and chart payload. Source contract frozen."""
import collections,gzip,json,resource,subprocess,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import core as C
def bound():
 m=json.loads((C.H/'manifest.json').read_text())
 for p,h in m['inputs'].items():assert C.sha(C.R/p)==h,p
 for p,h in m['sources'].items():
  assert C.sha(C.R/p)==h
  assert subprocess.check_output(['git','show','HEAD:'+p],cwd=C.R)==(C.R/p).read_bytes()
 return m
def build():
 resource.setrlimit(resource.RLIMIT_CPU,(180,181));start=time.process_time();bound();C.OUT.mkdir(exist_ok=True);assert not (C.OUT/'data.json.gz').exists()
 pop=C.population();allf=C.load_fills(pop.m);allf.day=allf.day.astype(str)
 p=pd.read_parquet(C.H.parent/'maker_trace_20261122/v2/chains/take_logic_v2/panel.parquet',filters=[('rank','=',3)]);p.day=p.day.astype(str)
 ev=pd.read_parquet(C.C/'take_logic_v2/events.parquet',filters=[('rank','=',3)]);evmap={(str(r.day),int(r.mid),r.sid):r for r in ev.itertuples()}
 sizes=pd.read_csv(C.H.parent/'maker_action_population_20261124/outputs/orders.csv',float_precision='round_trip');sizes=sizes[sizes['rank']==3];ss={r.order_hash:r for r in sizes.itertuples()}
 posts=pd.read_csv(C.H.parent/'maker_timing_estimates_20261124/outputs/post_estimates.csv.gz',float_precision='round_trip');posts=posts[posts['rank']==3];post={r.order_hash:r for r in posts.itertuples()}
 cases=[];rawstates=[];cells=collections.defaultdict(lambda:collections.Counter());excl=[];trends=[];books={}
 for day in C.DAYS:
  tob=pd.read_parquet(C.V/f'tob_{day}.parquet',filters=[('mid','in',pop[pop.day==day].m.tolist())])
  books.update({(day,int(mid)):C.Book(g) for mid,g in tob.groupby('mid')})
 for ref in pop.itertuples():
  day=ref.day;mid=int(ref.m);pg=p[(p.day==day)&(p.mid==mid)].sort_values('k');assert len(pg)==300;ws=int(pg.t_ms.iloc[0]);book=books[(day,mid)]
  lf=pd.read_parquet(C.V/f'life/{day}_{mid}_fills.parquet');assert not lf.duplicated(['sid','mk_idx']).any();lm={(r.sid,int(r.mk_idx),r.mk_hash):r for r in lf.itertuples()}
  src=allf[(allf.day==day)&(allf.mid==mid)].sort_values(['settle_ms','sid','px_rank','mk_idx','role'],kind='stable');fills=[];statefills=[]
  for r in src.itertuples():
   if r.status!='ok' or r.self_trade or r.q<=0:
    excl.append(dict(day=day,mid=mid,sid=r.sid,role=r.role,status=str(r.status),self_trade=bool(r.self_trade)));continue
   l=lm.get((r.sid,int(r.mk_idx),r.hash)) if r.role=='maker' else evmap.get((day,mid,r.sid))
   hi=int(l.hi_ms) if l is not None else int(r.settle_ms);lo=int(l.x_lo_hard_ms) if l is not None and l.x_lo_hard_ms>=0 else None
   if r.role=='maker' and l is not None:assert abs(l.shares-r.q)<1e-7 and l.y_side==r.y_side
   po=post.get(r.hash);so=ss.get(r.hash);ow=None
   if po is not None and str(po.estimate_enabled)=='True':ow=[(po.effect_lo_ms-ws)/1000,(po.effect_hi_ms-ws)/1000]
   times={'pre':lo-1 if lo is not None else None,'upper':hi,'lag1':lo-1000 if lo is not None else None};bs={k:book.at(t) for k,t in times.items()}
   action=('BUY' if (r.s==1 and r.own_index_set==1) or (r.s==-1 and r.own_index_set==2) else 'SELL') if r.own_index_set in [1,2] else 'UNKNOWN'
   f=dict(id=f'{r.role}|{r.sid}|{r.mk_idx}',sid=r.sid,mk_idx=int(r.mk_idx),hash=r.hash,t=(int(r.settle_ms)-ws)/1000,hi_ms=hi,lo_ms=lo,lo=(lo-ws)/1000 if lo is not None else None,
    s=int(r.s),price=float(r.p*100),q=float(r.q),fee_c=float(r.fee_fill*100),role=r.role,book=bs,original_size=float(so.original_shares) if so is not None else None,
    post=ow,post_grade=str(po.grade) if po is not None else 'unknown',native=action+(' YES' if r.own_index_set==1 else ' NO' if r.own_index_set==2 else ' unknown'))
   statefills.append(f)
   if ref.start_ms<=r.settle_ms<=ref.end_ms:fills.append(f)
  cl,lots,econ=C.fifo(fills,ref.end_mark*100)
  if ref.status=='ok':
   for key,val in [('net_c',ref.total*100),('completed_c',ref.close_net*100),('matched_shares',ref.close_sh),('position',ref.end_pos)]:assert abs(econ[key]-val)<1e-5*max(1,abs(val)),(mid,key,econ[key],val)
  else:assert not fills and ref.status=='no_activity',ref.status
  occ=[]
  for label,lo,hi in [('all',0,298),('early',0,180),('late',180,298),('last60',240,298)]:occ.append(dict(phase=label,seconds=book.occupancy(max(ws+lo*1000,ref.start_ms),min(ws+hi*1000,ref.end_ms))))
  for f in fills:
   for clock in C.CLOCKS:
    phases=['all','early' if f['t']<180 else 'late']+(['last60'] if f['t']>=240 else [])
    for phase in phases:
     key=('fills',clock,phase,C.band(f['book'][clock]['spread']),f['role']);x=cells[key];x['legs']+=1;x['shares']+=f['q'];x['fee_c']+=f['fee_c']
  for z in cl:
   for clock in C.CLOCKS:
    a=z['by_clock'][clock];phases=['all','early' if z['exit_t']<180 else 'late']+(['last60'] if z['exit_t']>=240 else [])
    for phase in phases:
     key=('closes',clock,phase,a['kind'],z['entry_role']+'->'+z['exit_role']);x=cells[key];x['closes']+=1;x['shares']+=z['q'];x['net_c']+=z['net_c'];x['gross_c']+=z['gross_c'];x['positive_c']+=max(0,z['net_c']);x['negative_c']+=min(0,z['net_c'])
     if a['drift_c'] is not None:
      for name in ['entry_edge_c','exit_edge_c','drift_c']:x[name]+=a[name]
  grid=[]
  for t in np.arange(0,298.001,.25):
   z=book.at(ws+t*1000);grid.append([float(t),z['bid'],z['ask'],(z['frame']-ws)/1000 if z['frame'] is not None else None,z['age']])
  tr=C.trend([[float(r.k),(r.bid0+r.ask0)*50 if r.kok0 else None] for r in pg.itertuples()]) if fills else []
  trends.extend(dict(t,day=day,mid=mid) for t in tr)
  curves=[];cash=pos=fees=0.;ii=0
  for g in grid:
   while ii<len(fills) and fills[ii]['t']<=g[0]:
    f=fills[ii];cash-=f['s']*f['q']*f['price'];pos+=f['s']*f['q'];fees+=f['fee_c'];ii+=1
   mark=(g[1]+g[2])/2 if g[1] is not None else None;curves.append([g[0],pos,(cash+pos*mark-fees)/100 if mark is not None else None])
  own={}
  for name,folder,fn in [('BASE','maker3_cap_skew_20261124','BASE'),('WIDE','maker3_cap_skew_20261124','WIDE_PASSIVE')]:
   op=C.D/folder/f'{mid}_{fn}.json.gz'
   if op.exists():
    rr=json.loads(gzip.decompress(op.read_bytes()));offset=(rr['ident']['start_abs_ms']-ws)/1000
    own[name]=dict(metrics=rr['metrics'],fills=[dict(t=offset+t/1000,price=px,s=1-2*s,q=q,gid=gid) for t,gid,s,q,px,it in rr['fills']],curve=[[offset+x[0]/1000,x[4]] for x in rr['grid'][::5]])
  case=dict(mid=mid,day=day,ws=ws,end=(ref.end_ms-ws)/1000,mark=ref.end_mark*100,fills=fills,closes=cl,totals=econ,book=grid,curve=curves,occupancy=occ,own=own,
   btc=[[float(r.k),float(r.r_5s0) if np.isfinite(r.r_5s0) else None] for r in pg.itertuples()])
  cases.append(case);rawstates.append(dict(day=day,mid=mid,fills=statefills));print('market',mid,len(fills),flush=True)
 qualified=[t for t in trends if t['qualifies']];trend=sorted(qualified or trends,key=lambda t:(-abs(t['move_c']),t['day'],t['mid'],t['start_s']))[0]
 tables=[dict(family=k[0],clock=k[1],phase=k[2],band=k[3],role=k[4],**v) for k,v in sorted(cells.items())]
 data=dict(cases=cases,trend=trend,spread_cells=tables,excluded=excl,asof='2026-11-24',clock_note='Maker markers are record upper bounds; horizontal bars are match bounds, not resting times.',signed_sizes=json.loads((C.H.parent/'maker3_cap_skew_20261124/observations.json').read_text()))
 for fn,x in [('data.json.gz',data),('state_fills.json.gz',rawstates)]:
  path=C.OUT/fn;assert not path.exists();path.write_bytes(gzip.compress(json.dumps(C.clean(x),separators=(',',':'),allow_nan=False).encode(),mtime=0))
 pd.DataFrame(tables).to_csv(C.H/'spread_cells.csv',index=False);C.dump(C.H/'build.json',dict(markets=len(cases),trend=trend,excluded=excl,cpu_s=time.process_time()-start,hashes={f:C.sha(C.OUT/f) for f in ['data.json.gz','state_fills.json.gz']}))
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1500*1024
if __name__=='__main__':build()
