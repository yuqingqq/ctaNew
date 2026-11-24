"""Independent ledgers and saved-state recomputation; planted corruptions."""
import ast,collections,copy,gzip,json,math,resource,time
import numpy as np
import pandas as pd
import experiment as E
import observe as A
H,R,T=E.H,E.R,E.T
def verify(r,arm):
 C,S,R_,active=E.ARMS[arm];f=r['fills'];g=np.array(r['grid']);assert r['settings']['cap']==C
 assert r['settings']['q0']==100 and r['settings']['inventory']['S']==S and r['settings']['inventory']['Qside']==200
 cash=sum(-(1-2*s)*q*p for t,i,s,q,p,it in f);pos=sum((1-2*s)*q for t,i,s,q,p,it in f)
 assert abs(cash-r['cash'])<1e-5 and abs(pos-r['position'])<1e-7
 fee=sum(.02*min(p,100-p)*q for t,i,s,q,p,it in f if it<0)
 assert abs(fee-r['fees_c'])<1e-7 and abs(cash+pos*r['mark']-fee-r['net_value_c'])<1e-5
 lots=collections.deque();match=0.
 for t,i,s,q,p,it in f:
  assert 0<=t<=r['end'] and q>0 and s in [0,1]
  sg=1-2*s
  while q>1e-9 and lots and lots[0][0]!=sg:
   ss,n,pp=lots[0];v=min(q,n);match+=ss*v*(p-pp);n-=v;q-=v
   if n<1e-9:lots.popleft()
   else:lots[0]=(ss,n,pp)
  if q>1e-9:lots.append((sg,q,p))
 residual=sum(s*q*(r['mark']-p) for s,q,p in lots)
 assert abs(match-r['metrics']['completed_c'])<1e-5 and abs(residual-r['metrics']['residual_c'])<1e-5
 assert abs(match+residual-fee-r['metrics']['net_c'])<1e-5
 ts=np.array([v[0] for v in f]);dq=np.array([(1-2*v[2])*v[3] for v in f]);cum=np.r_[0,np.cumsum(dq)];assert np.all(np.diff(ts)>=0)
 assert np.allclose(g[:,1],cum[np.searchsorted(ts,g[:,0]-100,'right')],atol=1e-6,rtol=0)
 assert np.allclose(g[:,4],cum[np.searchsorted(ts,g[:,0],'right')],atol=1e-6,rtol=0)
 assert np.max(abs(g[:,4]))<=C+1e-6 and np.max(g[:,1]+g[:,2])<=C+1e-6 and np.max(-g[:,1]+g[:,3])<=C+1e-6
 assert np.all(g[:,4]>=g[:,1]-g[:,3]-1e-6) and np.all(g[:,4]<=g[:,1]+g[:,2]+1e-6)
 for k,w in [('peak_inventory',max(abs(g[:,4]))),('mean_abs_inventory',np.mean(abs(g[:,4]))),('both_grid_fraction',np.mean((g[:,6]>0)&(g[:,7]>0)))]:assert abs(r['metrics'][k]-w)<1e-9
 orders={o['gid']:o for o in r['orders']};by=collections.defaultdict(float)
 for t,i,s,q,p,it in f:by[i]+=q;assert orders[i]['side']==s
 for o in orders.values():assert by[o['gid']]<=o['size']+1e-7 and o['size']<=100
 sent={x['gid']:x for x in r['ioc_decisions'] if x['status']=='sent'};assert len(sent)==len(r['iocs'])
 for key,a in r['iocs'].items():
  x=sent[int(key)];z=next(v for v in r['ioc_results'] if v['gid']==int(key));s=1 if x['known_I']>0 else 0
  assert abs(x['known_I'])>=S-1e-9 and s==a['side'] and x['book_received_ms']<=x['t']
  room=C+x['known_I']-x['A'] if s else C-x['known_I']-x['B'];q=max(0,math.floor(min(100,abs(x['known_I'])-R_,room)+1e-9))
  assert a['q']==q and a['act']==a['submit']+10 and a['ack']==a['act']+100
  assert abs(by[int(key)]-z['filled'])<1e-7
 if not active:assert not r['iocs']
 return True
def check_sizes(s):
 p,o=A.rows_and_orders();ref=A.size_rows(p,o)
 # This recomputes all output fields; separate hand checks below guard signs/time.
 pd.testing.assert_frame_equal(s,ref,check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
 assert len(s)==2*len(o) and not s.duplicated(['hash','lag_ms']).any()
 for r in s[s.status.eq('sampled_window')].itertuples():
  assert r.n>=1 and r.lo_ms<=r.hi_ms
  if r.inventory=='reducing':assert (r.I_max<0 if r.side=='bid' else r.I_min>0)
  if r.touch=='at_touch':assert abs(r.distance_min)<1e-7 and abs(r.distance_max)<1e-7
 return True
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(70,71));E.bound();summary=json.loads((H/'replay.json').read_text());assert len(summary['rows'])==12
 assert len({(x['mid'],x['arm']) for x in summary['rows']})==12
 records={};corrupt=0
 for row in summary['rows']:
  p=E.OUT/f"{row['mid']}_{row['arm']}.json.gz";assert T.sha(p)==row['sha256'];r=json.loads(gzip.decompress(p.read_bytes()))
  assert r['metrics']==row['metrics'];verify(r,row['arm']);records[(row['mid'],row['arm'])]=r
 b=records[(3789484,'BASE')]
 for fn in [lambda r:r.__setitem__('net_value_c',r['net_value_c']+1),lambda r:r['grid'][0].__setitem__(1,999),lambda r:r['metrics'].__setitem__('completed_c',0),lambda r:r['settings'].__setitem__('cap',999)]:
  z=copy.deepcopy(b);fn(z)
  try:verify(z,'BASE')
  except AssertionError:corrupt+=1
  else:raise AssertionError('ledger corruption escaped')
 # Load ONLY the previously accepted independent checker function, no top-level old runs.
 source=(T.H/'execution_check.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify');ns={'np':np}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<accepted IOC verifier>','exec'),ns);nexec=0
 for day,mid,part in E.X.CASES:
  r=records[(mid,'WIDE_TAKE')]
  if r['iocs']:
   old=E.X.records(day,part)[mid]['CURRENT_ASYNC_FAST10'];path,_=E.X.get_path(day,mid,part,old);nexec+=ns['verify'](r,path['frames'],path['trades'])
 obs=json.loads((H/'observations.json').read_text())
 for fn,h in obs['output_hashes'].items():assert T.sha(E.OUT/fn)==h
 s=pd.read_csv(E.OUT/'sizes.csv',float_precision='round_trip',dtype={'day':str});check_sizes(s)
 for field in ['size','I_min']:
  bad=s.copy();idx=bad[bad.status.eq('sampled_window')].index[0];bad.loc[idx,field]+=1
  try:check_sizes(bad)
  except AssertionError:corrupt+=1
  else:raise AssertionError('size/state corruption escaped')
 p0,o=A.rows_and_orders();p,cov=A.passive_context(p0);stored=pd.read_parquet(E.OUT/'take_context_panel.parquet');pd.testing.assert_frame_equal(stored,p)
 assert A.matched(p)==obs['take_context'] and cov==obs['life_coverage']
 ev=pd.read_csv(E.OUT/'take_events.csv',float_precision='round_trip');inpop=ev.t_ms.notna();assert (ev.loc[inpop,'t_ms']<ev.loc[inpop,'x_lo_hard_ms']).all()
 # Independent direct prior-window sum for all seven named-market takes.
 actor=pd.read_parquet(E.X.DATA/'maker3_late_activity_20261124/actor_legs.parquet');a=actor[(actor.mid==3789484)&actor.role.eq('maker')]
 life=pd.read_parquet(E.X.DATA/'maker_trace_v2_20261123/life/20261026_3789484_fills.parquet');a=a.merge(life[['sid','mk_idx','hi_ms']],on=['sid','mk_idx'],validate='one_to_one')
 for r in ev[(ev.mid==3789484)&inpop].itertuples():
  for sec in [5,30]:
   if abs(r.q0)<1e-6:continue
   v=a[(a.hi_ms<=r.t_ms)&(a.hi_ms>r.t_ms-sec*1000)&a.y_side.eq('ask' if r.q0>0 else 'bid')].q.sum()
   assert abs(v-getattr(r,f'reduce_{sec}_0'))<1e-6
 T.dump(H/'checks.json',dict(pass_all=True,records=12,prior_parity_markets=3,independent_ioc_requests=nexec,corruptions_refused=corrupt,signed_state_rows=len(s),neutral_seconds=len(p),named_take_records=int((ev.mid==3789484).sum())))
 print('PASS',12,'ledgers;',len(s),'size/state rows;',len(p),'neutral seconds;',nexec,'raw IOC checks;',corrupt,'refusals')
if __name__=='__main__':run()
