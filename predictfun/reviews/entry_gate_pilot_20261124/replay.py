"""DIRECTIONAL-ENTRY-PILOT-v1; declaration.md is the pre-run contract.

One unfitted admission candidate on three already consumed dashboard markets.
Frozen research modules remain unchanged. Latest USER asks to set an entry rule
and replay; total work stays inside the dashboard's unused CPU allocation.
"""
import collections,copy,gzip,hashlib,json,math,resource,signal,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];HERE=R/'predictfun/research'
sys.path.insert(0,str(HERE))
import pf_ladder_async_fast as AF
import pf_stop_ev_minutes_inputs as I
import numpy as np
P,M,LB,BE,QP=AF.P,AF.M,AF.LB,AF.BE,AF.M.QP
DATA=R/'data/predict_fun/research';OUT=DATA/'entry_gate_pilot_20261124'
CASES=[('20261024',3702960,'rest'),('20261026',3789484,'pilot'),('20261027',3877472,'pilot')]
ARMS=('ORIGINAL','FIXED_DEFAULT','DIRECTIONAL_ENTRY')
CAP_RUN_CPU=75.;TOL=1e-9
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def fixed_mode(old,pos,S,R_):
 if pos>=S-TOL:return 'LONG_RESTRICTED'
 if pos<=-S+TOL:return 'SHORT_RESTRICTED'
 if old=='LONG_RESTRICTED' and pos>R_+TOL:return old
 if old=='SHORT_RESTRICTED' and pos< -R_-TOL:return old
 return 'UNRESTRICTED'
class FixedBranch(AF.ProfileBranch):
 def update_mode(self,g):
  pos=self.reserve(g)[0];new=fixed_mode(self.mode,pos,self.inv['S'],self.inv['R'])
  if new!=self.mode:self.mode_log.append((float(g),float(pos),self.mode,new));self.mode=new
  assert not (self.mode=='LONG_RESTRICTED' and pos<0)
  assert not (self.mode=='SHORT_RESTRICTED' and pos>0)
def admission(side,pos,B,A,q,signal_value,status,other_edges):
 """Full proposed size on favourable opposite-side pressure; otherwise only
 unreserved reduction of known net inventory. No future data or clock timer."""
 favourable=status=='ok' and P.on_value(signal_value,other_edges)
 if favourable:return float(q),'favourable'
 reducing=max(0.,-pos-B) if side==0 else max(0.,pos-A)
 n=float(math.floor(min(float(q),reducing)+TOL))
 return (n,'reduce_only') if n>=LB.QMIN else (0.,'entry_wait')
class EntryGate(P.PressureGuard):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.entry_log=[]
 def slot(self,v,r,price):
  action,q,why=super().slot(v,r,price)
  if action!='POST':return action,q,why
  other=dict(v,side=1-int(v['side']));x,st,age=self.value(other)
  qnew,reason=admission(int(v['side']),v['position'],v['B'],v['A'],q,x,st,self.edges[other['side']])
  self.entry_log.append(dict(command_index=len(self.B.cmd_log),t=float(v['t']),side=int(v['side']),rung=int(r),price=float(price),position=float(v['position']),B=float(v['B']),A=float(v['A']),
    proposed=float(q),quantity=qnew,other_signal=float(x) if np.isfinite(x) else None,status=st,age=float(age) if np.isfinite(age) else None,reason=reason))
  return ('POST',qnew,'post') if qnew>=LB.QMIN else ('WAIT',None,'entry_wait')
def records(day,part):
 with gzip.open(AF.OUT/part/f'{day}.jsonl.gz','rt') as f:return {int(x['m']):x for x in map(json.loads,f)}
def inputs():
 """All files read by the unchanged loader plus saved-control and cache bytes."""
 paths={H/'replay.py',H/'declaration.md',H/'check.py',R/P.RECEIPT/'manifest.json',R/AF.RECEIPT/'manifest.json',DATA/'step4/minute_lines_t60k.json',P.M.OUT/'measure/audit.json'}
 for day,mid,part in CASES:
  old=records(day,part)[mid]['CURRENT_ASYNC_FAST10'];ws=old['ident']['opening_ms'];s1=DATA/'step1'/day;s5=DATA/'step5'/day
  paths.update(s1/n for n in ['registry.parquet','clock_rl_d2.json','legs_timed2.parquet','legs_tk.parquet'])
  paths.update(s5/n for n in ['preds_t60k.parquet','rows_rl_d2_pk.parquet'])
  paths.update(p for p in sorted((s1/'books').glob('part_*.parquet')) if I.part_allowed(str(p),day) and I.part_overlaps(str(p),{mid:ws}))
  paths.add(Path(AF.M.KP.QL.C.V.E.OUTD)/f'day_{day}_imp.json')
  paths.update(AF.OUT/part/f'{day}{s}' for s in ['.json','.jsonl.gz','_arrays.npz'])
  pp='pilot' if part=='pilot' else 'days';paths.update(P.OUT/pp/f'{day}{s}' for s in ['.json','.jsonl.gz','_arrays.npz'])
 for mod in list(sys.modules.values()):
  fn=getattr(mod,'__file__',None)
  if fn and str(HERE)+'/' in str(Path(fn).resolve()) and fn.endswith('.py'):paths.add(Path(fn))
 return sorted(paths)
def freeze():
 selftest();assert not (H/'manifest.json').exists()
 dump(H/'manifest.json',dict(version='directional_entry_pilot_v1',cases=CASES,arms=ARMS,run_cpu_cap=CAP_RUN_CPU,
  sources={str(p.relative_to(R)):sha(p) for p in inputs()},multiplicity=dict(new_candidates=['DIRECTIONAL_ENTRY'],controls=['ORIGINAL','FIXED_DEFAULT','NO_TRADE accounting zero']),
  asof='2026-11-24',new_days=False,fit=False,validation=False))
 print('manifest prepared; commit full pipeline before run')
def verify_freeze():
 man=json.loads((H/'manifest.json').read_text())
 for p,h in man['sources'].items():assert sha(R/p)==h,('source_changed',p)
 import subprocess
 for p in ['replay.py','check.py','declaration.md','manifest.json']:
  committed=subprocess.check_output(['git','show','HEAD:'+str((H/p).relative_to(R))],cwd=R)
  assert committed==(H/p).read_bytes(),('uncommitted',p)
 return man
def get_path(day,mid,part,old):
 src=AF.M.KP.Tape();data=src.load(day,[mid],src.offset(day));W=next(src.pairs(data));assert int(W[0]['m'])==mid
 path,status,info=M.market_path(W,old['ident']['opening_ms']);assert status=='complete'
 assert BE.tape_id(path['frames'],path['trades'],path['mark'])==old['tape_id'],'tape identity'
 assert path['mark']==old['mark'] and info['end_rel']==old['end']
 pp='pilot' if part=='pilot' else 'days';z=np.load(P.OUT/pp/f'{day}_arrays.npz');cache=AF.LG.cache_of(z,mid)
 return path,cache
def fifo(fills,mark):
 lots=collections.deque();matched=value=cash=pos=0.
 for t,g,s,q,p,it in fills:
  sg=1. if s==0 else -1.;cash-=sg*q*p;pos+=sg*q;n=q
  while n>1e-9 and lots and lots[0][0]!=sg:
   ls,lq,lp=lots[0];x=min(n,lq);matched+=x;value+=ls*x*(p-lp);n-=x;lq-=x
   if lq<1e-9:lots.popleft()
   else:lots[0]=(ls,lq,lp)
  if n>1e-9:lots.append((sg,n,p))
 residual=sum(s*q*(mark-p) for s,q,p in lots)
 assert abs(value+residual-cash-pos*mark)<1e-5
 return dict(matched_shares=matched,completed_c=value,residual_c=residual,total_c=cash+pos*mark,net_shares=pos)
def measure(rec,arr):
 f=rec['fills'];mm=rec['fill_mids'];out=fifo(f,rec['mark']);cmd=rec['commands'];orders=rec['orders'];pos=arr['pose']
 out.update(filled_shares=sum(x[3] for x in f),generations=len(orders),posts=sum(x[7]=='POST' for x in cmd),pulls=sum(x[7]=='PULL' for x in cmd),
  peak_abs_inventory=float(np.max(np.abs(pos))),mean_abs_inventory=float(np.mean(np.abs(pos))),both_active_grid_share=float(np.mean(np.all(arr['nva']>0,axis=1))),
  first_post_ms=min([o['submit'] for o in orders],default=None),first_post_sides=sorted(set(o['side'] for o in orders if o['submit']==min([z['submit'] for z in orders],default=-1))))
 out['markouts']={}
 for h in [50,1000,5000,30000]:
  idx=[0,50,100,1000,5000,30000].index(h)*2
  vals=[((1 if fl[2]==0 else -1)*fl[3]*(row[idx]-fl[4]),fl[3]) for fl,row in zip(f,mm) if row[idx] is not None]
  common=[((1 if fl[2]==0 else -1)*fl[3]*(row[idx]-fl[4]),fl[3]) for fl,row in zip(f,mm) if fl[0]<=rec['end']-30000 and row[idx] is not None]
  out['markouts'][str(h)]=dict(value_c=sum(x[0] for x in vals),shares=sum(x[1] for x in vals),common_cut_value_c=sum(x[0] for x in common),common_cut_shares=sum(x[1] for x in common))
 out['first_minute']=fifo([x for x in f if x[0]<60000],rec['minute_mark'])
 return out
def run():
 man=verify_freeze();OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'summary.json').exists()
 resource.setrlimit(resource.RLIMIT_CPU,(int(CAP_RUN_CPU),int(CAP_RUN_CPU)+1))
 signal.signal(signal.SIGXCPU,lambda *_: (_ for _ in ()).throw(RuntimeError('CPU_STOP')))
 start=time.process_time();stats=[];edges=P.frozen_edges()
 try:
  for day,mid,part in CASES:
   old=records(day,part)[mid]['CURRENT_ASYNC_FAST10'];path,cache=get_path(day,mid,part,old)
   for arm in ARMS:
    fn=OUT/f'{mid}_{arm}.json.gz';assert not fn.exists(),'no unverified resume'
    B=(AF.ProfileBranch if arm=='ORIGINAL' else FixedBranch)(path,old['end'],old['ident'],False,'ASYNC_FAST10')
    pol=(EntryGate if arm=='DIRECTIONAL_ENTRY' else P.PressureGuard)(B,cache,edges);obs=P.QueueObs(B,pol)
    LB.run_episode(B,pol,observe=obs);lg=LB.finish(B);rec=M.ledger_record(B,lg)
    rec.update(commands=[[None if isinstance(v,float) and not math.isfinite(v) else v for v in row] for row in M.jl(B.cmd_log)],entry_log=getattr(pol,'entry_log',[]),fill_mids=M.mids_of(B.E.f,B.E.fills,B.end),minute_mark=M.AR.mid_at(B.E.f,60000.),guard_log=M.jl(pol.log))
    if arm=='ORIGINAL':
     for key in ['fingerprint','fills','orders','requests','value','guard_log','closed','mode_log']:assert rec[key]==old[key],('parity',mid,key)
    metrics=measure(rec,obs.arrays());assert abs(metrics['total_c']-rec['value'])<1e-6
    rec['arm']=arm;rec['metrics']=metrics;rec['manifest_sha256']=sha(H/'manifest.json');fn.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False).encode(),mtime=0))
    np.savez_compressed(OUT/f'{mid}_{arm}.npz',**obs.arrays())
    stats.append(dict(day=day,m=mid,arm=arm,metrics=metrics,sha256=sha(fn)))
    dump(OUT/'progress.json',dict(completed=stats,cpu_s=time.process_time()-start,state='RUNNING'))
    print(mid,arm,'checkpoint cpu',round(time.process_time()-start,2),flush=True)
  dump(OUT/'summary.json',dict(rows=stats,cpu_s=time.process_time()-start,manifest_sha256=sha(H/'manifest.json'),status='COMPLETE'))
 except BaseException as e:
  dump(OUT/'stop.json',dict(error=repr(e),completed=len(stats),cpu_s=time.process_time()-start));raise
def selftest():
 cases=[('LONG_RESTRICTED',-97.,'UNRESTRICTED'),('SHORT_RESTRICTED',97.,'UNRESTRICTED'),('LONG_RESTRICTED',70.,'LONG_RESTRICTED'),
  ('SHORT_RESTRICTED',-70.,'SHORT_RESTRICTED'),('LONG_RESTRICTED',-101.,'SHORT_RESTRICTED'),('UNRESTRICTED',100.,'LONG_RESTRICTED'),('LONG_RESTRICTED',50.,'UNRESTRICTED')]
 for old,p,new in cases:assert fixed_mode(old,p,100,50)==new
 assert LB.next_mode('LONG_RESTRICTED',-97,100,50)=='LONG_RESTRICTED','known-bad control must fire'
 e=[.05,.1,.2,.3]
 assert admission(0,0,0,0,100,0,'ok',e)==(0.,'entry_wait')
 assert admission(0,0,0,0,100,.4,'ok',e)==(100.,'favourable')
 assert admission(1,0,0,0,100,-.4,'ok',e)==(0.,'entry_wait')
 assert admission(0,0,0,0,100,.4,'nan',e)==(0.,'entry_wait')
 assert admission(0,-80,30,0,100,0,'ok',e)==(50.,'reduce_only')
 assert admission(1,80,0,30,100,0,'ok',e)==(50.,'reduce_only')
 assert admission(0,-20,20,0,100,0,'ok',e)==(0.,'entry_wait')
 assert EntryGate.live is P.PressureGuard.live
 # Future values are outside the current tick's input; no smoothing/lookahead.
 class Dummy:
  cache={'x':{0:[0.,1.],1:[0.,-1.]},'status':{0:[0,0],1:[0,0]},'age':{0:[0.,0.],1:[0.,0.]}}
 d=Dummy();before=P.PressureGuard.value(d,{'side':0,'t':0});d.cache['x'][0][1]=99.
 assert P.PressureGuard.value(d,{'side':0,'t':0})==before
 assert fifo([(0,0,0,10,40,0),(20,1,1,10,45,1)],50)['total_c']==50
 assert fifo([(0,0,1,10,40,0),(20,1,0,10,45,1)],50)['total_c']==-50
 print('mode crossing, gate sign/tie/missing/reducing reservation, unchanged live, future input, FIFO controls PASS')
if __name__=='__main__':{'selftest':selftest,'freeze':freeze,'run':run}[sys.argv[1]]()
