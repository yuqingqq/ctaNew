"""CAP-AND-MAKER-SIZING v1: isolated risk-cap sensitivities; contract.md."""
import collections,copy,gzip,json,math,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2]
sys.path.insert(0,str(H.parent/'inventory_take_pilot_20261124'))
import pilot as T
X,O,LB,BE=T.X,T.O,T.LB,T.BE
OUT=X.DATA/'maker3_cap_skew_20261124'
ARMS={'BASE':(200,100,50,False),'CAP_ONLY':(1000,100,50,False),
      'WIDE_PASSIVE':(1000,500,250,False),'WIDE_TAKE':(1000,500,250,True)}
def qty(I,B,A,C,S,R_):
 if abs(I)<S-1e-9:return None,0.
 s=1 if I>0 else 0;room=C+I-A if s else C-I-B
 return s,max(0.,float(math.floor(min(100.,abs(I)-R_,room)+1e-9)))
class Branch(T.TakeBranch):
 def __init__(self,path,end,ident,bps,config):
  cap,S,R_,enabled=config;super().__init__(path,end,ident,100,bps,enabled)
  self.cap=float(cap);self.inv=dict(self.inv,Qside=200.,S=float(S),R=float(R_))
  self.settings=dict(self.settings,cap=self.cap,inventory=self.inv,cap_experiment='v1')
 def take_step(self,g):
  if not self.take_enabled:return
  I,B,A=self.reserve(g);s,q=qty(I,B,A,self.cap,self.inv['S'],self.inv['R'])
  if q<1:return
  row=dict(t=float(g),known_I=I,B=B,A=A,side=s,q=q,status='candidate',gid=None)
  if self.take_open:row['status']='pending_ioc'
  elif g+10>=self.end:row['status']='near_end'
  else:
   v=self.side_view(g,s);limit=v['ask'] if s==0 else v['bid'];row.update(limit=limit,book_received_ms=v['received'])
   signature=(I,s,limit,*self.E.f[self.E.known,2:].tolist())
   if not v['book_ok'] or g-v['received']>1000:row['status']='no_fresh_book'
   elif any(limit>=p if s==0 else limit<=p for p in v['opp_prices']):row['status']='own_cross'
   elif signature==self.last_take_state:row['status']='unchanged_state'
   else:
    gid=self.E.submit_ioc(g,s,q,limit);self.take_open.append(gid);self.last_take_state=signature;row.update(gid=gid,status='sent')
  self.take_decisions.append(row);self.check_position(g)
def one(path,cache,old,edges,bps,config):
 B=Branch(path,old['end'],old['ident'],bps,config);pol=O.OpeningGate(B,cache,edges);grid=[]
 for k in range(LB.n_ticks(B)):
  LB.step(B,k,pol);g=k*BE.STEP;B.take_step(g);cash,pos,_=B.E.value(B.mark);I,RB,RA=B.reserve(g)
  live=[sum(o['side']==s and o['gid'] not in B.E.iocs and not o['refused'] and o['act']<=g<o['remove'] and o['rem']>1e-9 for o in B.E.o) for s in (0,1)]
  grid.append([g,I,RB,RA,pos,cash,*live])
 lg=LB.finish(B);rec=T.M.ledger_record(B,lg)
 rec.update(commands=T.clean(B.cmd_log),guard_log=T.clean(pol.log),scope_log=pol.scope_log,
  iocs=T.clean(B.E.iocs),ioc_decisions=B.take_decisions,ioc_results=B.E.exec_log,fees_c=sum(x[2] for x in B.E.fees),grid=grid)
 rec['net_value_c']=rec['value']-rec['fees_c'];rec['fee2_net_value_c']=rec['value']-2*rec['fees_c']
 metric=X.fifo(rec['fills'],rec['mark']);g=np.array(grid)
 metric.update(net_c=rec['net_value_c'],fees_c=rec['fees_c'],maker_shares=sum(f[3] for f in rec['fills'] if f[5]>=0),taker_shares=sum(f[3] for f in rec['fills'] if f[5]<0),
  take_requests=len(rec['iocs']),takes_filled=sum(z['filled']>1e-9 for z in rec['ioc_results']),peak_inventory=float(max(abs(g[:,4]))),mean_abs_inventory=float(np.mean(abs(g[:,4]))),
  both_grid_fraction=float(np.mean((g[:,6]>0)&(g[:,7]>0))),passive_generations=len(B.E.o)-len(B.E.iocs),passive_pulls=sum(c[7]=='PULL' for c in B.cmd_log),
  wait_reasons=dict(collections.Counter(c[8] for c in B.cmd_log if c[7]=='WAIT')),max_reserved_bid=float(max(g[:,1]+g[:,2])),max_reserved_ask=float(max(-g[:,1]+g[:,3])))
 rec['metrics']=metric;return T.clean(rec)
def sources():
 old=json.loads((T.H/'manifest.json').read_text());paths={R/p for p in old['sources']}
 paths.update([T.H/'pilot.py',T.H/'manifest.json',T.H/'check.py',T.H/'execution_check.py'])
 paths.update(H/p for p in ['contract.md','experiment.py','observe.py','check.py','selftest.py'])
 C=H.parent/'maker_trace_20261122/v2/chains'
 paths.update([C/'take_logic_v2/panel.parquet',C/'take_logic_v2/events.parquet',H.parent/'maker_action_population_20261124/outputs/orders.csv',H.parent/'maker_action_population_20261124/outputs/contexts.csv'])
 paths.add(X.DATA/'maker3_late_activity_20261124/actor_legs.parquet')
 for _,m,_ in X.CASES:paths.add(T.OUT/f'{m}_PASSIVE_100.json.gz')
 # All19 lifecycle files used to time the passive context (not a new tape load).
 keys=T.pd.read_parquet(C/'take_logic_v2/panel.parquet',columns=['day','mid','rank']).query('rank == 3')[['day','mid']].drop_duplicates()
 paths.update(X.DATA/f'maker_trace_v2_20261123/life/{d}_{mid}_fills.parquet' for d,mid in keys.itertuples(index=False,name=None))
 return sorted(paths)
def prepare():
 assert not (H/'manifest.json').exists();old=json.loads((T.H/'manifest.json').read_text())
 T.dump(H/'manifest.json',dict(arms=ARMS,cases=X.CASES,fees_bps=old['fees_bps'],sources={str(p.relative_to(R)):T.sha(p) for p in sources()},new_candidates=3,asof='2026-11-24',validation=False))
def bound():
 m=json.loads((H/'manifest.json').read_text())
 for p,h in m['sources'].items():assert T.sha(R/p)==h,p
 for n in ['contract.md','experiment.py','observe.py','check.py','selftest.py','manifest.json']:
  p=H/n;assert subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(R))],cwd=R)==p.read_bytes(),n
 return m
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(180,181));start=time.process_time();m=bound();OUT.mkdir(exist_ok=True);rows=[];par=[];edges=T.P.frozen_edges()
 assert not (OUT/'summary.json').exists()
 for day,mid,part in X.CASES:
  old=X.records(day,part)[mid]['CURRENT_ASYNC_FAST10'];path,cache=X.get_path(day,mid,part,old)
  for arm,cf in ARMS.items():
   rec=one(path,cache,old,edges,m['fees_bps'][str(mid)],cf);rec.update(arm=arm,manifest_sha256=T.sha(H/'manifest.json'))
   if arm=='BASE':
    prior=json.loads(gzip.decompress((T.OUT/f'{mid}_PASSIVE_100.json.gz').read_bytes()));ff=['fills','orders','requests','value','cash','position','commands','guard_log','closed','mode_log']
    eq={k:rec[k]==prior[k] for k in ff};assert all(eq.values()),eq;par.append(dict(mid=mid,fields=eq))
   p=OUT/f'{mid}_{arm}.json.gz';assert not p.exists();p.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False).encode(),mtime=0))
   rows.append(dict(mid=mid,day=day,arm=arm,sha256=T.sha(p),metrics=rec['metrics']))
   T.dump(OUT/'progress.json',dict(rows=rows,cpu_s=time.process_time()-start));print(mid,arm,round(time.process_time()-start,2),flush=True)
 T.dump(H/'replay.json',dict(rows=rows,parity=par,cpu_s=time.process_time()-start));assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1500*1024
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
