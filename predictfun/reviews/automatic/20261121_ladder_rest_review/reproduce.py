"""Bounded review of pinned LADDER_REST artifacts; no real-tape load/replay or fit.
Run via PF_SLOT=codex run_light.sh. Synthetic controls and saved-ledger arithmetic only.
"""
from pathlib import Path
import copy, gzip, hashlib, importlib.util, json, math, os, pickle, subprocess, sys
import numpy as np
import pandas as pd

TARGET = '869fa85ffc158b070e63c1b89af4cea3c82c3abf'
ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-lr-review-869fa85f')
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git','-C',str(PIN),'rev-parse','HEAD']).decode().strip() == TARGET
sys.path.insert(0, str(PIN/'predictfun/research'))
import pf_ladder_rest as M
import pf_ladder_rest_selftest as ST
LB, BE = M.LB, M.BE
os.environ['PF_TEACHER_SYN'] = str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
selftest_result = {}
try:
 ST.main(M); selftest_result['unmodified'] = 'pass'
except SystemExit as e:
 selftest_result['unmodified'] = str(e)
 assert 'the binder was reached: a guard did not refuse' in str(e)
 # Its two negative guard fixtures assume an uncommitted module and no projection.
 # Isolate those fixture states in this review process, preserving production files.
 old_frozen, old_projection = M.frozen, M.projection_allows
 try:
  M.frozen=lambda:False; M.projection_allows=lambda:False
  ST.main(M); selftest_result['isolated_negative_fixtures']='pass'
 finally:
  M.frozen, M.projection_allows = old_frozen, old_projection

spec=importlib.util.spec_from_file_location('review_ladder_checks', PIN/'predictfun/reviews/ladder_rest_20261121/ladder_checks_v2.py')
CK=importlib.util.module_from_spec(spec); spec.loader.exec_module(CK)
CK.REPO=PIN; CK.R=PIN/'predictfun/reviews/ladder_rest_20261121'; CK.HERE=PIN/'predictfun/research'
CK.D=ROOT/'data/predict_fun/research/ladder_rest_v1_20261121'
assert CK.selftest()
E=CK.load('days'); E['pilot']=CK.load('pilot')
base=CK.run_checks(E)
assert base['all_ok'], base
fz=CK.falsify(E)
assert fz['falsifier_ok']
out={'target':TARGET,'baseline_selftest':selftest_result,'saved_checks':base,'saved_falsifier':fz}
out['imported_sources']={n:hashlib.sha256((PIN/'predictfun/research'/n).read_bytes()).hexdigest() for n in M.SOURCES}
out['receipt_hashes_ok']=all(hashlib.sha256((CK.R/'outputs'/n).read_bytes()).hexdigest()==h for h,n in (ln.split('  ',1) for ln in (CK.R/'outputs/SHA256SUMS').read_text().splitlines()))
assert out['receipt_hashes_ok']

# A structural request is legal at the first post-pending tick, before first live review.
fr=ST.fr
path=BE.tape([fr(0,0,[(50,0),(49,0)],[(52,0),(53,0)]),fr(50,50,[(54,0),(53,0)],[(56,0),(57,0)])],[(100,0,50,10)],51.)
B=LB.LadderBranch(path,LB.LADDER,end=1000.)
gid=B.E.add_order(0,50.,0.,100.,0);B.unclosed[0].append(gid)
B.pre_tick(50.);B.E.advance(110.)
control=LB.LadderBranch(path,LB.LADDER,end=1000.)
control.E.add_order(0,50.,0.,100.,0);control.E.advance(110.)
out['pending_cancel_control']={'request':B.E.o[gid]['req'],'effect':B.E.o[gid]['remove'],
 'reported_earliest':math.ceil(B.E.o[gid]['act']/M.STEP)*M.STEP+M.CANCEL,
 'controlled_fill_q':sum(f[3] for f in B.E.fills),'no_request_fill_q':sum(f[3] for f in control.E.fills)}
assert out['pending_cancel_control']==dict(request=50.,effect=95.,reported_earliest=137.,controlled_fill_q=0,no_request_fill_q=10.)

# Resume checks: the unmodified positive, changed blob refusal, then envelope/runtime holes.
p=BE.tape([fr(0,0,[(50,0)],[(52,0)])],[],51.)
b=LB.LadderBranch(p,LB.LADDER,end=1000.);LB.run_episode(b,LB.LadderInventory(b),until=2)
sv=LB.save_state(b,2);br,k=LB.resume(sv,LB.LADDER,p,1000.);assert k==2 and LB.fingerprint(br)==LB.fingerprint(b)
def accepted(s,path=p):
 try:
  z,k=LB.resume(s,LB.LADDER,path,1000.);return {'accepted':True,'next_k':k,'cap':z.cap,'executed_mark':z.mark}
 except ValueError as e:return {'accepted':False,'reason':str(e)[:160]}
bad=copy.deepcopy(sv);zz=pickle.loads(bad['blob']);zz.mode='LONG_RESTRICTED';bad['blob']=pickle.dumps(zz)
assert not accepted(bad)['accepted']
out['resume_controls']={'unchanged_accepted':True,'changed_bound_field_refused':True}
s=copy.deepcopy(sv);s['next_k']=11;out['resume_next_k']=accepted(s)
s=copy.deepcopy(sv);s['ident']={'instrument_id':'wrong'};out['resume_ident']=accepted(s)
s=copy.deepcopy(sv);zz=pickle.loads(s['blob']);zz.cap=999.;s['blob']=pickle.dumps(zz);out['resume_runtime_cap']=accepted(s)
s=copy.deepcopy(sv);p2=dict(p,mark=52.);s['tape_id']=BE.tape_id(p2['frames'],p2['trades'],p2['mark']);out['resume_tape_envelope']=accepted(s,p2)

# Independent saved-fill reconciliation, exact duration and inventory integration.
comp=[r for r in E['recs'] if r['status']=='complete']; diag=[]; unlocked=[]; state_fill=[]
for r in comp:
 L=r['ledger'];end=L['end'];os_={o['gid']:o for o in L['orders']};fills=L['fills'];gids={}
 for f in fills:gids.setdefault(f[1],[]).append(f)
 cash=-sum(M.sgn(f[2])*f[3]*f[4] for f in fills);pos=sum(M.sgn(f[2])*f[3] for f in fills)
 assert abs(cash+pos*L['mark']-L['value'])<1e-6
 intervals=[]
 for o in L['orders']:
  fs=gids.get(o['gid'],[]);full=max(f[0] for f in fs) if sum(f[3] for f in fs)>=o['size']-1e-9 else end
  hi=min(o['remove'],full,end)
  if not o['refused'] and hi>o['act']:intervals.append((o['act'],hi,o['side']))
  early=max(o['submit']+M.STEP+M.CANCEL,o['submit']+BE.PENDING)
  reported=math.ceil(o['act']/M.STEP)*M.STEP+M.CANCEL
  for f in fs:
   if early<=f[0]<reported:unlocked.append((r['day'],r['m'],o['gid'],f[3],M.sgn(f[2])*f[3]*(L['mark']-f[4])))
 events={0.:[0,0],end:[0,0]}
 for lo,hi,s in intervals:
  events.setdefault(lo,[0,0])[s]+=1;events.setdefault(hi,[0,0])[s]-=1
 active=[0,0];both=either=0.;prev=0.
 for t,delta in sorted(events.items()):
  both+=(t-prev)*int(active[0]>0 and active[1]>0);either+=(t-prev)*int(any(active))
  active=[a+d for a,d in zip(active,delta)];prev=t
 p=0.;area=0.;last=0.;peak=0.
 for f in sorted(fills,key=lambda f:(f[0],f[5])):
  area+=abs(p)*(f[0]-last);p+=M.sgn(f[2])*f[3];peak=max(peak,abs(p));last=f[0]
 area+=abs(p)*(end-last)
 ar=lambda n:E['arrs'][f"{r['day']}|{r['m']}__{n}"]
 diag.append(dict(day=r['day'],m=r['m'],opening_wait=r['path_info']['opening_wait_ms'],end=end,both_ms_exact=both,either_ms_exact=either,
  both_grid=float(((ar('nva')[:,0]>0)&(ar('nva')[:,1]>0)).mean()),mean_abs_exact=area/end,mean_abs_grid=float(np.abs(ar('pose')).mean()),peak_exact=peak,peak_grid=float(np.abs(ar('pose')).max())))
 for sn in L['snapshots']:
  if sn['kind']=='entry':continue
  fs=gids.get(sn['gid'],[]);past=any(f[0]<sn['t']+M.CANCEL for f in fs);future=any(f[0]>=sn['t']+M.CANCEL for f in fs)
  state_fill.append((sn['kind'],past,future))
out['durations_inventory']=diag
out['locked_window']={'fills':len(unlocked),'generations':len(set(x[:3] for x in unlocked)),'shares':sum(x[3] for x in unlocked),'end_value':sum(x[4] for x in unlocked)}
out['live_fill_denominators']={kind:dict(n=sum(x[0]==kind for x in state_fill),any_generation_fill=sum(x[0]==kind and (x[1] or x[2]) for x in state_fill),future_fill=sum(x[0]==kind and x[2] for x in state_fill)) for kind in ('first','live')}
out['headline']={'markets':len(comp),'value':sum(r['ledger']['value'] for r in comp),'filled_shares':sum(f[3] for r in comp for f in r['ledger']['fills']),
 'both_exact':sum(x['both_ms_exact'] for x in diag)/sum(x['end']+x['opening_wait'] for x in diag),
 'both_grid':sum(x['both_grid']*x['end'] for x in diag)/sum(x['end'] for x in diag)}
out['complete_statuses']=E['markets'].status.value_counts().to_dict()
# Binding the saved receipt copies to their uncompressed data counterparts.
out['receipt_matches_raw']={}
for pth in (CK.R/'outputs').iterdir():
 if pth.is_file() and pth.name.endswith(('.csv','.csv.gz')):
  raw=CK.D/'measure'/pth.name.removesuffix('.gz')
  b=gzip.decompress(pth.read_bytes()) if pth.suffix=='.gz' else pth.read_bytes()
  out['receipt_matches_raw'][pth.name]=raw.exists() and raw.read_bytes()==b
assert all(out['receipt_matches_raw'].values())
(OUT/'evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps({k:out[k] for k in ('headline','pending_cancel_control','locked_window','live_fill_denominators','resume_next_k','resume_runtime_cap','resume_tape_envelope')},indent=2))
