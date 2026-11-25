"""INVENTORY-EXIT-EXCEPTION v1. Frozen contract.md defines the whole experiment."""
import collections,copy,gzip,json,math,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2]
sys.path.insert(0,str(H.parent/'maker3_cap_skew_20261124'))
import experiment as CAP
# Loaded by filename under a distinct module name by launch.py to avoid itself.
T,X,O,LB,BE=CAP.T,CAP.X,CAP.O,CAP.LB,CAP.BE
OUT=X.DATA/'inventory_exit_exception_20261125';EPS=1e-8
ARMS={'BASE':((200,100,50,False),'none'),'WIDE':((1000,500,250,False),'none'),
 'WIDE_REDUCE':((1000,500,250,False),'reduce'),'WIDE_COST':((1000,500,250,False),'cost')}
dump,sha,clean=T.dump,T.sha,T.clean
class KnownLots:
 def __init__(self):self.cursor=0;self.lots=collections.deque()
 def view(self,fills,g):
  while self.cursor<len(fills) and fills[self.cursor][0]+BE.FILL_NOTICE<=g:
   t,i,s,q,p,it=fills[self.cursor];self.cursor+=1;sg=1-2*s
   while q>EPS and self.lots and self.lots[0][0]!=sg:
    ss,n,pp=self.lots[0];d=min(q,n);q-=d;n-=d
    if n<=EPS:self.lots.popleft()
    else:self.lots[0]=(ss,n,pp)
   if q>EPS:self.lots.append((sg,q,p))
  pos=sum(s*q for s,q,p in self.lots);n=sum(q for s,q,p in self.lots)
  return pos,sum(q*p for s,q,p in self.lots)/n if n>EPS else None
def eligibility(v,price,q,own,basis,kind,mode):
 s=v['side'];I=v['position'];res=v['B'] if s==0 else v['A']
 room=max(0.,(-I if s==0 else I)-max(0.,res-own))
 size=float(math.floor(min(q,room)+EPS)) if kind=='slot' else float(q)
 reducing=(I< -EPS if s==0 else I>EPS)
 margin=None if basis is None else (basis-price if s==0 else price-basis)
 eligible=reducing and size>EPS and size<=room+EPS and (kind!='slot' or size>=LB.QMIN)
 cost_ok=margin is not None and margin>=1.-EPS
 return size,room,margin,bool(eligible and (mode=='reduce' or (mode=='cost' and cost_ok))),bool(eligible),bool(cost_ok)
class ExitPolicy(O.OpeningGate):
 def __init__(self,*args,mode='none',**kw):super().__init__(*args,**kw);self.mode=mode;self.known=KnownLots();self.exit_log=[]
 def proposal(self,v,price,q,own,kind,gid,parent):
  pos,basis=self.known.view(self.B.E.fills,v['t']);assert abs(pos-v['position'])<1e-6
  size,room,margin,override,reducing,cost_ok=eligibility(v,price,q,own,basis,kind,self.mode)
  z=dict(command_index=len(self.B.cmd_log),t=v['t'],side=v['side'],kind=kind,gid=gid,price=price,q=q,own=own,
   I=pos,B=v['B'],A=v['A'],basis=basis,room=room,size=size,margin=margin,reducing=reducing,cost_ok=cost_ok,
   override=override,mode=self.mode,parent=parent)
  self.exit_log.append(z);return override,size
 def slot(self,v,r,price):
  result=super().slot(v,r,price)
  if result[2]!='pressure_withhold':return result
  a,q,w=self.inner.slot(v,r,price);assert a=='POST'
  yes,n=self.proposal(v,price,q,0.,'slot',None,list(result))
  return ('POST',n,'exit_override') if yes else result
 def live(self,v,o):
  result=super().live(v,o)
  if result[1]!='pressure_pull':return result
  q=float(o['size'])-float(o['known_filled'])
  yes,n=self.proposal(v,o['price'],q,q,'live',o['gid'],list(result))
  return ('KEEP','exit_override') if yes else result
def run_one(path,cache,old,edges,arm,bps):
 config,mode=ARMS[arm];B=CAP.Branch(path,old['end'],old['ident'],bps,config);p=ExitPolicy(B,cache,edges,mode=mode);grid=[]
 for k in range(LB.n_ticks(B)):
  LB.step(B,k,p);g=k*BE.STEP;cash,pos,_=B.E.value(B.mark);I,RB,RA=B.reserve(g)
  live=[sum(o['side']==s and not o['refused'] and o['act']<=g<o['remove'] and o['rem']>1e-9 for o in B.E.o) for s in (0,1)]
  grid.append([g,I,RB,RA,pos,cash,*live])
 lg=LB.finish(B);rec=T.M.ledger_record(B,lg)
 rec.update(commands=clean(B.cmd_log),guard_log=clean(p.log),scope_log=p.scope_log,exit_log=p.exit_log,
  iocs={},ioc_decisions=[],ioc_results=[],fees_c=0.,grid=grid,arm=arm)
 rec['net_value_c']=rec['value'];rec['metrics']=X.fifo(rec['fills'],rec['mark']);g=np.array(grid)
 rec['metrics'].update(net_c=rec['value'],fees_c=0.,maker_shares=sum(f[3] for f in rec['fills']),taker_shares=0.,take_requests=0,
  peak_inventory=float(max(abs(g[:,4]))),mean_abs_inventory=float(np.mean(abs(g[:,4]))),both_grid_fraction=float(np.mean((g[:,6]>0)&(g[:,7]>0))),
  passive_generations=len(B.E.o),passive_pulls=sum(c[7]=='PULL' for c in B.cmd_log))
 return clean(rec)
def cases():
 m=json.loads((R/X.AF.RECEIPT/'manifest.json').read_text())
 return [(d,int(mid),part) for d in sorted(m['days']) for part in ['pilot','rest'] for mid in m['parts'][part][d]]
def prepare():
 assert not (H/'manifest.json').exists();cs=cases();assert len(cs)==19
 old=X.CASES;X.CASES=cs
 try:paths=set(X.inputs())
 finally:X.CASES=old
 paths.update([CAP.H/'experiment.py',T.H/'pilot.py',O.H/'opening.py',CAP.H/'manifest.json'])
 for _,mid,_ in old:
  paths.update(CAP.OUT/f'{mid}_{a}.json.gz' for a in ['BASE','WIDE_PASSIVE'])
 src={str(p.relative_to(R)):sha(p) for p in paths};files=['contract.md','experiment.py','launch.py','selftest.py','check.py','report.py']
 src.update({str((H/p).relative_to(R)):sha(H/p) for p in files})
 bps=json.loads((CAP.H/'manifest.json').read_text())['fees_bps'];assert len(set(bps.values()))==1
 dump(H/'manifest.json',dict(asof='2026-11-25',cases=cs,arms=ARMS,sources=src,fees_bps=next(iter(bps.values())),new_candidates=2,validation=False))
def bound():
 m=json.loads((H/'manifest.json').read_text())
 for p,h in m['sources'].items():assert sha(R/p)==h,p
 for p in ['contract.md','experiment.py','launch.py','selftest.py','check.py','report.py','manifest.json']:
  assert subprocess.check_output(['git','show','HEAD:'+str((H/p).relative_to(R))],cwd=R)==(H/p).read_bytes(),p
 return m
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(400,401));m=bound();OUT.mkdir(exist_ok=True);assert not (OUT/'summary.json').exists()
 start=time.process_time();edges=T.P.frozen_edges();rows=[];parity=[]
 for ix,(day,mid,part) in enumerate(m['cases']):
  old=X.records(day,part)[mid]['CURRENT_ASYNC_FAST10'];path,cache=X.get_path(day,mid,part,old)
  for arm in ARMS:
   rec=run_one(path,cache,old,edges,arm,m['fees_bps']);rec.update(day=day,mid=mid,manifest_sha256=sha(H/'manifest.json'))
   if mid in [v[1] for v in X.CASES] and arm in ['BASE','WIDE']:
    prev=json.loads(gzip.decompress((CAP.OUT/f"{mid}_{'BASE' if arm=='BASE' else 'WIDE_PASSIVE'}.json.gz").read_bytes()))
    fs=['fills','orders','requests','value','cash','position','commands','guard_log','grid','closed','mode_log']
    same={f:rec[f]==prev[f] for f in fs};assert all(same.values()),(mid,arm,same);parity.append(dict(mid=mid,arm=arm,same=same))
   fn=OUT/f'{mid}_{arm}.json.gz';assert not fn.exists();fn.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False).encode(),mtime=0))
   rows.append(dict(day=day,mid=mid,arm=arm,metrics=rec['metrics'],sha256=sha(fn)))
  elapsed=time.process_time()-start
  if ix==0:
   projection=elapsed*len(m['cases'])*1.2;dump(H/'projection.json',dict(first_mid=mid,cpu_s=elapsed,projection_cpu_s=projection,limit=400,proceed=projection<=400));assert projection<=400,'projection stop'
  dump(OUT/'progress.json',dict(markets=ix+1,rows=rows,cpu_s=elapsed));print(day,mid,'done',round(elapsed,2),flush=True)
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024
 dump(OUT/'summary.json',dict(rows=rows,parity=parity,cpu_s=time.process_time()-start,status='COMPLETE'))
