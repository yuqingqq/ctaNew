"""OPENING-ENTRY-v1. USER scope correction: initial quote batch only.
After its submission tick, permanently delegate every slot to the original
PressureGuard. Frozen parent experiment and all engine modules stay unchanged.
"""
import gzip,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];PARENT=H.parent/'entry_gate_pilot_20261124'
sys.path.insert(0,str(PARENT))
import replay as X
P,M,LB,BE,np=X.P,X.M,X.LB,X.BE,X.np
OUT=X.DATA/'opening_entry_pilot_20261124';CAP_CPU=28.;ARM='OPENING_ONLY'
def initial_tick(orders):return min((float(o['submit']) for o in orders),default=None)
def in_opening(t,orders):
 first=initial_tick(orders)
 return first is None or float(t)<=first
class OpeningGate(X.EntryGate):
 def __init__(self,*a,disabled=False,**kw):super().__init__(*a,**kw);self.disabled=disabled;self.scope_log=[]
 def slot(self,v,r,price):
  active=not self.disabled and in_opening(v['t'],self.B.E.o)
  result=super().slot(v,r,price) if active else P.PressureGuard.slot(self,v,r,price)
  # Read-only audit at the policy call, prior to any submission.
  x,st,age=self.value(v)
  self.scope_log.append(dict(command_index=len(self.B.cmd_log),t=float(v['t']),side=int(v['side']),rung=int(r),price=float(price),
    initial_tick=initial_tick(self.B.E.o),gate_active=active,position=float(v['position']),B=float(v['B']),A=float(v['A']),mode=v['mode'],
    action=result[0],q=result[1],why=result[2],own_signal=float(x) if np.isfinite(x) else None,status=st))
  return result
def files():
 old=json.loads((PARENT/'manifest.json').read_text())['sources'];paths={R/p for p in old}
 paths.update(H/p for p in ['opening.py','check_opening.py','declaration.md'])
 paths.update(PARENT/p for p in ['manifest.json','checks.json','summary.json'])
 for _,m,_ in X.CASES:
  paths.update(X.OUT/f'{m}_FIXED_DEFAULT{s}' for s in ['.json.gz','.npz'])
 return sorted(paths)
def freeze():
 selftest();assert not (H/'manifest.json').exists()
 X.dump(H/'manifest.json',dict(version='opening_entry_v1',cases=X.CASES,arm=ARM,cap_run_cpu=CAP_CPU,
  sources={str(p.relative_to(R)):X.sha(p) for p in files()},reused_controls='FIXED_DEFAULT from fa9d035d2/de825a07a',
  new_candidate=ARM,candidate_count_this_entry_family=2,scope='all proposals on first actual submission tick; original entries at every later tick',validation=False))
def verify_freeze():
 man=json.loads((H/'manifest.json').read_text())
 for p,h in man['sources'].items():assert X.sha(R/p)==h,('source_identity',p)
 for p in ['opening.py','check_opening.py','declaration.md','manifest.json']:
  assert subprocess.check_output(['git','show','HEAD:'+str((H/p).relative_to(R))],cwd=R)==(H/p).read_bytes()
 return man
def one(path,cache,old,edges,disabled=False):
 B=X.FixedBranch(path,old['end'],old['ident'],False,'ASYNC_FAST10');pol=OpeningGate(B,cache,edges,disabled=disabled);obs=P.QueueObs(B,pol)
 LB.run_episode(B,pol,observe=obs);lg=LB.finish(B);rec=M.ledger_record(B,lg)
 rec.update(commands=[[None if isinstance(v,float) and not math.isfinite(v) else v for v in row] for row in M.jl(B.cmd_log)],
  entry_log=pol.entry_log,scope_log=pol.scope_log,fill_mids=M.mids_of(B.E.f,B.E.fills,B.end),minute_mark=M.AR.mid_at(B.E.f,60000.),guard_log=M.jl(pol.log),
  arm='DISABLED_CONTROL' if disabled else ARM,opening_batch_ms=initial_tick(B.E.o),manifest_sha256=X.sha(H/'manifest.json'))
 rec['metrics']=X.measure(rec,obs.arrays())
 return rec,obs.arrays()
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(int(CAP_CPU),int(CAP_CPU)+1))
 signal.signal(signal.SIGXCPU,lambda *_:(_ for _ in ()).throw(RuntimeError('CPU_STOP')))
 verify_freeze();OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'summary.json').exists()
 stats=[];parity=[];start=time.process_time();edges=P.frozen_edges()
 try:
  for ix,(day,mid,part) in enumerate(X.CASES):
   old=X.records(day,part)[mid]['CURRENT_ASYNC_FAST10'];path,cache=X.get_path(day,mid,part,old)
   if ix==0:
    ctrl,_=one(path,cache,old,edges,disabled=True);baseline=json.loads(gzip.decompress((X.OUT/f'{mid}_FIXED_DEFAULT.json.gz').read_bytes()))
    fields=['fingerprint','fills','orders','requests','value','guard_log','commands','closed','mode_log']
    flags={k:ctrl[k]==baseline[k] for k in fields};assert all(flags.values()),flags
    parity.append(dict(m=mid,fields=flags));X.dump(OUT/'disabled_parity.json',parity)
   rec,arr=one(path,cache,old,edges);fn=OUT/f'{mid}_{ARM}.json.gz';assert not fn.exists()
   fn.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False).encode(),mtime=0));ap=OUT/f'{mid}_{ARM}.npz';np.savez_compressed(ap,**arr)
   stats.append(dict(day=day,m=mid,arm=ARM,metrics=rec['metrics'],opening_batch_ms=rec['opening_batch_ms'],sha256=X.sha(fn),arrays_sha256=X.sha(ap)))
   X.dump(OUT/'progress.json',dict(rows=stats,cpu_s=time.process_time()-start,state='RUNNING'));print(mid,'checkpoint',round(time.process_time()-start,2),flush=True)
  X.dump(OUT/'summary.json',dict(rows=stats,parity=parity,cpu_s=time.process_time()-start,status='COMPLETE',manifest_sha256=X.sha(H/'manifest.json')))
 except BaseException as e:
  X.dump(OUT/'stop.json',dict(error=repr(e),completed=len(stats),cpu_s=time.process_time()-start));raise
def selftest():
 assert in_opening(0,[]) and in_opening(100,[])
 history=[{'submit':100.,'rem':0.,'refused':True}]
 assert in_opening(100,history) and not in_opening(150,history)
 assert not in_opening(200,history+[{'submit':200.}]),'must never rearm'
 # Real policy callbacks: first tick admits both eligible rungs, next tick
 # allows default flat-signal entries that the all-entry gate would withhold.
 class Inner:
  def slot(self,v,r,p):return 'POST',100.,'post'
  def live(self,v,o):return 'KEEP','keep'
 class Engine:o=[]
 class Branch:E=Engine();cmd_log=[]
 pol=object.__new__(OpeningGate);pol.B=Branch();pol.inner=Inner();pol.enabled=True;pol.disabled=False;pol.entry_log=[];pol.scope_log=[]
 pol.log=[];pol.missing={};pol.edges={0:[.05,.1,.2,.3],1:[.05,.1,.2,.3]}
 pol.cache={'x':{0:[-.4,-.4,0.,0.],1:[.4,.4,0.,0.]},'status':{0:[0]*4,1:[0]*4},'age':{0:[0.]*4,1:[0.]*4}}
 v=dict(side=0,t=50.,position=0.,B=0.,A=0.,mode='UNRESTRICTED')
 assert pol.slot(v,0,49)[0]=='POST'
 pol.B.E.o=[{'submit':50.,'refused':True,'rem':0.}]
 assert pol.slot(v,1,48)[0]=='POST' and pol.scope_log[-1]['gate_active']
 v['t']=100.;assert pol.slot(v,0,49)==('POST',100.,'post') and not pol.scope_log[-1]['gate_active']
 assert not pol.entry_log[-1]['t']>50
 v['t']=150.;assert pol.slot(v,1,48)[0]=='POST' and not pol.scope_log[-1]['gate_active']
 assert OpeningGate.live is P.PressureGuard.live
 print('PASS: first batch, both rungs, next-tick delegation on flat signal, refusal/no-fill/no-rearm, live handling')
if __name__=='__main__':{'selftest':selftest,'freeze':freeze,'run':run}[sys.argv[1]]()
