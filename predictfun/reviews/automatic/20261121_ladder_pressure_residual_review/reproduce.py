"""Pinned residual audit review: saved artifacts and synthetic controls only."""
from pathlib import Path
import collections, copy, hashlib, json, os, subprocess, sys, types
import numpy as np
import pandas as pd
ROOT=Path('/home/yuqing/ctaNew');PIN=Path('/tmp/pf-residual-review-2063d31a')
TARGET='2063d31a91ab7e6124d7b8b7b7ec5df01ea66711';OUT=Path(__file__).resolve().parent
assert subprocess.check_output(['git','-C',str(PIN),'rev-parse','HEAD']).decode().strip()==TARGET
os.environ['PF_TEACHER_SYN']=str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_ladder_pressure_residual as X
import pf_ladder_pressure_residual_selftest as ST
ST.main(X)
R=PIN/X.RECEIPT;p=R/'residual_checks.py';CK=types.ModuleType('review_residual_checks');CK.__file__=str(p)
src=p.read_text().replace("REPO = Path('/home/yuqing/ctaNew')",f"REPO = Path('{PIN}')")
exec(compile(src,str(p),'exec'),CK.__dict__)
for n,path in [('D','ladder_pressure_residual_v1_20261121'),('AD','ladder_pressure_ablation_v1_20261121'),('PD_','ladder_pressure_v1_20261121'),('RUN','ladder_rest_v1_20261121')]:
 setattr(CK,n,ROOT/'data/predict_fun/research'/path)
CK.EDGE_FILE=CK.RUN/'measure/audit.json'
def resources(E,root=None):
 own,peak=CK.cpu(list((root or ROOT/X.RECEIPT/'logs').rglob('*.log')))
 prev,_=CK.cpu([p for n in CK.LINE for p in (ROOT/'predictfun/reviews'/n/'logs').rglob('*.log')])
 rev,_=CK.cpu([p for n in CK.REVIEWS for p in (ROOT/'predictfun/reviews/automatic'/n).glob('*.log')])
 return dict(I_resources=own<=CK.CAP and peak<1536 and own+prev+rev<=CK.LINE_CAP),dict(I=dict(round_core_hours=own,line_core_hours=own+prev+rev,peak_mib=peak))
CK.FAMS['I']=resources
E=CK.load();checks=CK.run_checks(E);assert checks['all_ok'],checks
out=dict(target=TARGET,selftest='passed',checks=checks)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={}
for ln in (R/'outputs/SHA256SUMS').read_text().splitlines():
 h,n=ln.split('  ',1);assert sha(R/'outputs'/n)==h,n
 assert (R/'outputs'/n).read_bytes()==(CK.D/n).read_bytes(),n
 hashes[n]=h
out['receipt_hashes']=hashes
for n,h in E['gm']['sources'].items():assert sha(PIN/'predictfun/research'/n)==h,n
assert sha(PIN/X.AB.RECEIPT/'manifest.json')==E['gm']['ablation']['manifest_sha256']
assert sha(CK.AD/'measure/arms.csv')==E['gm']['ablation']['arms_csv_sha256']
assert (CK.AD/'measure/arms.csv').read_bytes()==(PIN/X.AB.RECEIPT/'outputs/arms.csv').read_bytes()
assert sha(CK.EDGE_FILE)==E['gm']['edges_sha256']
assert E['audit']['manifest_sha256']==sha(R/'manifest.json') and E['audit']['sources']==E['gm']['sources']
expected=set()
for day in E['gm']['days']:
 X.LG.parent_day(day,E['gm']['parent_day_records'][day])
 for part in ('pilot','rest'):
  rec,mk,za=X.ablation_part(day,part)
  assert sha(CK.AD/part/f'{day}.json')==E['gm']['ablation']['part_records'][f'{part}|{day}']
  assert sorted(mk)==sorted(rec['members'])==sorted(E['gm']['parts'][part][day])
  expected.update((day,m) for m in mk)
assert set(E['led'])==expected and len(expected)==19
out['all_input_identities_verified']=True
print('Pinned selftest, ten saved checks, artifact hashes and complete membership passed',flush=True)

# Independently classify from the last grid able to remove at the fill time.
# Also inspect all feasible callbacks even for categories whose builder returns early.
rows=[];gens=[];ds=[];t1_future=0;pending_with_feasible=0;early_t1_delayed=[];all_timed_on=0;req_reasons=collections.Counter()
for (day,m),L in E['led'].items():
 cz=E['cache'][(day,m)];cm=E['cmd'][(day,m)];end=L['end'];mark=L['mark'];fills=L['fills'];mids=L['fill_mids']
 assert len(fills)==len(mids)
 assert all(np.isfinite(float(z)) for v in mids for z in v if z is not None)
 od={int(o['gid']):o for o in L['orders']};fg=collections.defaultdict(list);cgs={};first={};reason={}
 for gid,o in od.items():
  cg=cm[cm[:,3]==gid];live=cg[cg[:,2]==0];cgs[gid]=live
  if len(live):
   assert (np.diff(live[:,0])>=0).all();first[gid]=float(live[0,0]);assert first[gid]>=np.ceil(o['act']/50)*50
   if first[gid]>np.ceil(o['act']/50)*50:early_t1_delayed.append([day,m,gid,o['act'],first[gid]])
  if o['req'] is not None:
   if o['why']=='structural':rr='structural'
   else:
    ev=live[(live[:,0]==o['req'])&(live[:,7]==3)];assert len(ev)==1
    rr={2:'inventory',12:'pressure'}[int(ev[0,8])]
   reason[gid]=rr;req_reasons[rr]+=1
   effect=o['req']+37 if o['req']>=o['act'] else max(o['req']+37,o['submit']+95)
   assert o['remove']==effect,(day,m,gid,o)
 for i,(f,mm) in enumerate(zip(fills,mids)):
  t,gid,s,q,price,_=f;gid=int(gid);s=int(s);o=od[gid];fg[gid].append(f)
  assert not o['refused'] and o['act']<=t<o['remove'] and 0<=t<=end and o['side']==s and o['price']==price
  live=cgs[gid];t1=first.get(gid);deadline=50*np.floor((t-37)/50)
  feas=live[live[:,0]<=deadline];ks=(feas[:,0]/50).astype(int)
  good=(feas[:,8]!=11)&(cz['status'][s][ks]==0)
  on=good&np.isfinite(cz['x'][s][ks])&(cz['x'][s][ks]>0)&(cz['x'][s][ks]>=E['last'][s])
  all_timed_on+=int(on.sum())
  pending=o['req'] is not None and o['req']<t
  if pending:st='pending_'+reason[gid]
  elif t1 is None or t<t1+37:st='before_feasible_effect'
  elif on.any():st='event_or_action_mismatch'
  elif not good.any():st='nonevaluable'
  else:st='unflagged_later'
  pending_with_feasible+=int(pending and len(feas)>0)
  t1_future+=int(t1 is not None and t1>t)
  sg=1 if s==0 else -1
  val=lambda j:np.nan if mm[j] is None else sg*q*(mm[j]-price)
  eligible=t<=end-1000 and mm[6] is not None;v=sg*q*(mark-price)
  row=dict(day=day,m=m,fill=i,gid=gid,side=s,rung=int(o['rung']),status=st,t=t,q=q,p=price,edge=val(0),mo50=val(2),mo100=val(4),mo1s=val(6) if eligible else np.nan,
           eligible=eligible,move_end=sg*q*(mark-mm[6]) if eligible else 0.,late_end=0. if eligible else v,end=v,exposure_ms=t-o['act'],T1=t1)
  rows.append(row)
  ds.append(dict(day=day,m=m,fill=i,status=st,actual_n_feasible=len(feas),actual_n_evaluable=int(good.sum()),actual_n_on=int(on.sum()),actual_k_last=float(feas[-1,0]) if len(feas) else None))
 for gid,o in od.items():
  ff=fg[gid];filled=sum(f[3] for f in ff);assert filled<=o['size']+1e-7
  full=ff[-1][0] if ff and filled>=o['size']-1e-9 else end
  life=None if o['refused'] else max(0,min(o['remove'],full,end)-o['act'])
  gens.append(dict(day=day,m=m,gid=gid,side=int(o['side']),rung=int(o['rung']),gen_status='refused' if o['refused'] else 'filled' if ff else 'no_fill',fills=len(ff),live_ms=life,req_reason=reason.get(gid)))
F=pd.DataFrame(rows);G=pd.DataFrame(gens);D=pd.DataFrame(ds)
def equal_df(a,b,key,cols=None):
 a=a.sort_values(key).reset_index(drop=True);b=b.sort_values(key).reset_index(drop=True);assert len(a)==len(b)
 for c in cols or list(a.columns):
  if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
   assert np.allclose(a[c].astype(float),b[c].astype(float),atol=1e-7,rtol=0,equal_nan=True),c
  else:assert a[c].fillna('__missing__').equals(b[c].fillna('__missing__')),c
equal_df(F,E['fills'],['day','m','fill']);equal_df(G,E['generations'],['day','m','gid'])
# Complete diagnostic row parity against the inspected transformation, separate
# from independent status/value parity above. Includes omitted CSV fields.
builder=[]
for (day,m),L in E['led'].items():
 f,g=X.market_rows(day,m,L,E['cmd'][(day,m)],E['cache'][(day,m)],{s:E['gm']['edges'][str(s)] for s in (0,1)});builder+=f
equal_df(pd.DataFrame(builder),E['fills'],['day','m','fill'])
# Every table field, concentration and reading is rederived; raw values and
# memberships have already been independently checked above.
T,GS,conc=X.table(F,G);rd=X.reading(F)
equal_df(T,E['status_table'],['status','day','side','rung']);equal_df(GS,E['generation_status'],['gen_status','day','side','rung'])
assert conc==E['audit']['concentration'] and rd==E['audit']['reading']
J=D.merge(E['fills'][['day','m','fill','n_feasible','n_evaluable','n_on','k_last']],on=['day','m','fill'],validate='one_to_one')
affected=J[J.actual_n_feasible!=J.n_feasible]
diag=dict(pending_with_feasible=pending_with_feasible,affected_by_status=affected.status.value_counts().to_dict(),all_timed_on=all_timed_on,T1_after_fill=t1_future,
          delayed_T1_count=len(early_t1_delayed),delayed_T1_examples=early_t1_delayed[:5])
J.to_csv(OUT/'independent_tick_counts.csv',index=False)
summary=F.assign(negative=F.mo1s.clip(upper=0),positive=F.mo1s.clip(lower=0)).groupby('status').agg(fills=('fill','size'),q=('q','sum'),valid_q=('q',lambda q:float(q[F.loc[q.index,'eligible']].sum())),negative=('negative','sum'),positive=('positive','sum'),mo1s=('mo1s','sum'),end=('end','sum')).reset_index()
summary.to_csv(OUT/'independent_status.csv',index=False)
out.update(fills=len(F),generations=len(G),status_counts=F.status.value_counts().to_dict(),generation_counts=G.gen_status.value_counts().to_dict(),request_reason_counts=dict(req_reasons),diagnostics=diag,reading=rd,concentration=conc)
out['falsifier']=CK.falsify(E,checks);assert out['falsifier']['falsifier_ok']
# The checker also needs to distinguish independently correct derived fields
# from fields whose current correctness relies on this review's workaround.
probes={}
for name in ('diagnostic','concentration','generation_status','positive_reading'):
 B=dict(E)
 if name=='diagnostic':B['fills']=E['fills'].assign(T1_x=E['fills'].T1_x+1000.)
 if name=='concentration':
  B['audit']=copy.deepcopy(E['audit']);B['audit']['concentration']['unflagged_later']['top3_share']=.999
 if name=='generation_status':B['generation_status']=E['generation_status'].assign(generations=0)
 if name=='positive_reading':
  B['audit']=copy.deepcopy(E['audit']);B['audit']['reading']['by_day']['all']['unflagged']['pos']=-999999
 c,_=CK.fam_F(B);probes[name]=bool(all(c.values()))
out['checker_omission_probes_family_F_passes']=probes
# One explicit synthetic pending case proves the zero counters are not facts.
cache={k:{s:np.zeros(50,np.float32 if k!='status' else np.int8) for s in (0,1)} for k in ('x','age','status')}
order=dict(gid=0,side=0,submit=0.,act=58.,req=1000.,remove=1037.,why='structural')
cmd=[(float(t),'live','KEEP','keep') for t in range(100,1000,50)]
st,dd=X.classify_fill(1010.,order,cmd,cache,[-.6,-.2,.2,.46]);assert st=='pending_structural' and dd['n_feasible']==0 and len(cmd)==18
out['synthetic_pending_counter']=dict(status=st,stored_n_feasible=dd['n_feasible'],actual_n_feasible=18)
(OUT/'evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('checks','receipt_hashes','falsifier','reading','concentration')},indent=2,default=str),flush=True)
