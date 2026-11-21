"""Pinned saved-artifact review. No real tape replay, fit, labels or new days.
Only checker filesystem roots are relocated; scientific source is unmodified.
"""
from pathlib import Path
import collections, copy, hashlib, json, os, subprocess, sys, types
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-ablation-review-54eab545')
TARGET='54eab5455dd4c6e61c79d3543e3dd8a28e801ed0'; OUT=Path(__file__).resolve().parent
assert subprocess.check_output(['git','-C',str(PIN),'rev-parse','HEAD']).decode().strip()==TARGET
os.environ['PF_TEACHER_SYN']=str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_ladder_pressure_ablation as AB
import pf_ladder_pressure_ablation_selftest as ST
ST.main(AB)
R=PIN/AB.RECEIPT; p=R/'ablation_checks_v2.py'; CK=types.ModuleType('ablation_review_checks'); CK.__file__=str(p)
src=p.read_text().replace("REPO = Path('/home/yuqing/ctaNew')",f"REPO = Path('{PIN}')")
exec(compile(src,str(p),'exec'),CK.__dict__)
CK.D=ROOT/'data/predict_fun/research/ladder_pressure_ablation_v1_20261121'
CK.PD_=ROOT/'data/predict_fun/research/ladder_pressure_v1_20261121'
CK.RUN=ROOT/'data/predict_fun/research/ladder_rest_v1_20261121'; CK.EDGE_FILE=CK.RUN/'measure/audit.json'
CK.PC.REPO=PIN
# Resource logs belong to the shared run, not to the sparse source checkout.
def resources(E, root=None):
 own,peak=CK.logs_cpu(list((root or ROOT/AB.RECEIPT/'logs').rglob('*.log')))
 prev,_=CK.logs_cpu([p for n in CK.LINE for p in (ROOT/'predictfun/reviews'/n/'logs').rglob('*.log')])
 rev,_=CK.logs_cpu([p for n in CK.REVIEWS for p in (ROOT/'predictfun/reviews/automatic'/n).glob('*.log')])
 return dict(I_resources=own<=CK.CAP and peak<1536 and own+prev+rev<=CK.LINE_CAP),dict(I=dict(round_core_hours=own,line_core_hours=own+prev+rev,peak_mib=peak))
CK.FAMS['I']=resources
E=CK.load(); checks=CK.run_checks(E); assert checks['all_ok'],checks
out=dict(target=TARGET,selftest='passed',checks=checks)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={}
for ln in (R/'outputs/SHA256SUMS').read_text().splitlines():
 h,n=ln.split('  ',1);p=R/'outputs'/n;assert sha(p)==h,n
 raw=CK.D/n
 if n in ('arms.csv','audit.json','control.csv','lost_added.csv','reentry.csv'):raw=CK.D/'measure'/n
 assert raw.read_bytes()==p.read_bytes(),n
 hashes[n]=h
out['receipt_hashes']=hashes
for n,h in E['gm']['sources'].items(): assert sha(PIN/'predictfun/research'/n)==h,n
assert sha(PIN/AB.P.RECEIPT/'manifest.json')==E['gm']['parent']['manifest_sha256']
for day in E['gm']['days']:
 AB.LG.parent_day(day,E['gm']['parent']['day_records'][day])
out['parent_and_source_hashes_verified']=True
print('Pinned selftest, 17 checks and receipt/source/parent hashes passed',flush=True)

derived=[]; commands=collections.Counter(); queue_rows=0; first_reviews=0; all_la=[]; array_controls=0
for r in E['recs']:
 day,m,part=r['_day'],int(r['m']),r['_part'];end=float(r['FULL_11']['end']);parent=E['parent'][(day,m)]
 for nm in ('commands','decisions','pose','posc','res','cash','mid','nva'):
  assert np.array_equal(E['arr'][(day,part)][f'{m}__FULL_11__{nm}'],E['parr'][day][f'{m}__pressure__{nm}'],equal_nan=True),(day,m,nm)
  array_controls+=1
 cz=CK.cache(E,day,m)
 for arm in CK.ARMS:
  rec=CK.ledger(E,r,arm);f=np.asarray(rec['fills'],float).reshape(-1,6);mm=np.asarray(rec['fill_mids'],float)
  assert rec['ident']==parent['pressure']['ident'] and rec['tape_id']==parent['pressure']['tape_id'] and rec['mark']==parent['pressure']['mark']
  assert len(f)==len(mm) and (f[:,0]>=0).all() and (f[:,0]<=end).all()
  row=E['arms'].query('day==@day and m==@m and arm==@arm').iloc[0]
  q=f[:,3];sg=np.where(f[:,2]==0,1.,-1.);v=sg*q*(rec['mark']-f[:,4]);el=(f[:,0]<=end-1000)&np.isfinite(mm[:,6]);mo=sg*q*(mm[:,6]-f[:,4])
  ff=collections.defaultdict(list)
  for z in rec['fills']:ff[int(z[1])].append(z)
  vals=dict(value=float(v.sum()),all_q=float(q.sum()),all_cut_mo1s=float(mo[el].sum()),all_cut_valid_q=float(q[el].sum()),
            all_id_move=float((sg*q*(rec['mark']-mm[:,6]))[el].sum()),all_id_late=float(v[~el].sum()),all_mo50=float(np.nansum(sg*q*(mm[:,2]-f[:,4]))))
  assert abs(vals['value']-vals['all_cut_mo1s']-vals['all_id_move']-vals['all_id_late'])<1e-6
  pos=last=area=high=peak=0.
  for z in rec['fills']:
   area+=abs(pos)*(z[0]-last);high+=(abs(pos)>=100-1e-9)*(z[0]-last);pos+=(1 if z[2]==0 else -1)*z[3];last=z[0];peak=max(peak,abs(pos))
  area+=abs(pos)*(end-last);high+=(abs(pos)>=100-1e-9)*(end-last)
  vals.update(mean_abs_inv=area/end,share_abs_ge_S=high/end,peak_abs_inv=peak,terminal_inv=pos)
  ev={0.:[0,0],end:[0,0]}
  for o in rec['orders']:
   if o['refused']:continue
   fs=ff[o['gid']];full=fs[-1][0] if fs and sum(x[3] for x in fs)>=o['size']-1e-9 else end
   lo,hi=o['act'],min(o['remove'],full,end)
   if hi>lo:ev.setdefault(lo,[0,0])[o['side']]+=1;ev.setdefault(hi,[0,0])[o['side']]-=1
  count=[0,0];last=0.;times=np.zeros(3)
  for t,dv in sorted(ev.items()):
   times+=(t-last)*np.array([count[0]>0,count[1]>0,all(x>0 for x in count)]);count=[x+y for x,y in zip(count,dv)];last=t
  vals.update(zip(('bid_exact','ask_exact','both_exact'),times/end))
  for k,vv in vals.items():assert np.isclose(row[k],vv,atol=1e-6,rtol=0),(day,m,arm,k,row[k],vv)
  derived.append(dict(day=day,m=m,arm=arm,**vals))
  if arm=='BASELINE_00':continue
  logged=collections.Counter()
  for z in rec['guard_log']:
   k=int(z[0]/50);s=int(z[1]);assert z[5]==float(cz['x'][s][k]) and z[6]==float(cz['age'][s][k])
   assert abs(z[8]-sum((1 if a[2]==0 else -1)*a[3] for a in rec['fills'] if a[0]+100<=z[0]))<1e-7
   if z[10]=='pressure_pull':
    o=rec['orders'][int(z[3])];assert o['req']==z[0] and o['remove']==z[0]+37 and o['act']<=z[0]
   logged[(z[0],z[1],z[2],z[3],z[4],z[10])]+=1
  seen=collections.Counter();first={}
  for z in CK.cmds(E,r,arm):
   why=AB.P.REASONS[int(z[8])];kind=AB.M.CMD_KINDS[int(z[2])]
   if why.startswith('pressure'):
    seen[(z[0],int(z[1]),kind,int(z[3]),z[4],why)]+=1;commands[arm+'|'+why]+=1
   if kind=='live' and why!='fallback':first.setdefault(int(z[3]),z[0])
  assert seen==logged
  for gid,t in first.items(): assert rec['queue_venue']['first'][str(gid)][0]==t
  first_reviews+=len(first)
  ex=CK.reentry_expect(rec);stored=E['reentry'].query('day==@day and m==@m and arm==@arm').set_index('pull_gid')
  assert len(ex)==len(stored)
  for gid,z in ex.items():
   sr=stored.loc[gid]
   for k,vv in z.items():assert CK.same(vv,sr[k]),(day,m,arm,gid,k,vv,sr[k])
  queue_rows+=len(ex)
 # Independently match the fill multisets; check every exported lost/added field.
 key=lambda z:(round(float(z[0]),6),int(z[2]),round(float(z[4]),6),round(float(z[3]),6))
 for name,(a1,a0) in CK.CONTRASTS.items():
  ll={a:CK.ledger(E,r,a) for a in (a0,a1)};xx={}
  for nm,arm,other in [('lost',a0,a1),('added',a1,a0)]:
   rem=collections.Counter(map(key,ll[other]['fills']));fs=[];mi=[]
   for z,mid in zip(ll[arm]['fills'],ll[arm]['fill_mids']):
    if rem[key(z)]:rem[key(z)]-=1
    else:fs.append(z);mi.append(mid)
   f=np.asarray(fs,float).reshape(-1,6);q=f[:,3];sg=np.where(f[:,2]==0,1.,-1.);v=sg*q*(ll[arm]['mark']-f[:,4]);mm=np.asarray(mi,float)
   mo=sg*q*(mm[:,6]-f[:,4]) if len(f) else np.array([]);el=(f[:,0]<=end-1000)&np.isfinite(mo)
   xx.update({f'{nm}_n':len(f),f'{nm}_q':float(q.sum()),f'{nm}_end':float(v.sum()),f'{nm}_end_pos':float(v[v>0].sum()),f'{nm}_end_neg':float(v[v<0].sum()),
              f'{nm}_mo1s_pos':float(mo[el&(mo>0)].sum()),f'{nm}_mo1s_neg':float(mo[el&(mo<0)].sum())})
  xx['delta_value']=ll[a1]['value']-ll[a0]['value'];xx['identity_gap']=xx['delta_value']-xx['added_end']+xx['lost_end']
  row=E['lost_added'].query('day==@day and m==@m and contrast==@name').iloc[0]
  for k,vv in xx.items():assert np.isclose(vv,row[k],atol=1e-6,rtol=0),(day,m,name,k,vv,row[k])
  all_la.append(dict(day=day,m=m,contrast=name,**xx))
df=pd.DataFrame(derived);df.to_csv(OUT/'independent_arms.csv',index=False)
pd.DataFrame(all_la).to_csv(OUT/'independent_lost_added.csv',index=False)
out.update(arm_ledgers=len(derived),full_array_controls=array_controls,commands=dict(commands),first_reviews=first_reviews,reentry_rows=queue_rows,lost_added_contrasts=len(all_la))
out['predicates']=AB.predicates(E['arms']);assert out['predicates']==E['maudit']['predicates']
out['falsifier']=CK.falsify(E,checks);assert out['falsifier']['falsifier_ok']
# Direct controls for disabled components: a signal read must never occur.
from types import SimpleNamespace
edges={s:E['gm']['edges'][str(s)] for s in (0,1)}
v=dict(t=100.,side=0,position=0.,B=0.,A=0.,mode='UNRESTRICTED',avail=200.)
o=dict(gid=7,price=50.,rung=0.,size=100.,state='live',known_filled=0.,act=58.,submit=0.)
count=0
for C,W in ((0,0),(0,1),(1,0)):
 g=AB.AblatedPressureGuard(SimpleNamespace(end=1000.,inv={'Qside':200.,'S':100.},q0=100.,cap=200.),{},edges,C,W)
 def fail(*a,**k):raise AssertionError('disabled component read the signal')
 g.value=fail
 if not C:assert g.live(v,o)==('KEEP','keep');assert g.q_first==[(7,100.)];count+=1
 if not W:assert g.slot(v,0,50.)[0]=='POST';count+=1
out['disabled_read_barriers']=count
# Bound the observed minor checker omissions, without changing production code.
probes={}
for col in ('lost_q','lost_mo1s_neg'):
 X=dict(E);X['lost_added']=E['lost_added'].copy();X['lost_added'].loc[0,col]+=1000
 c,_=CK.fam_F(X);probes[col]=dict(family_still_passes=all(c.values()))
X=dict(E);X['maudit']=copy.deepcopy(E['maudit']);X['maudit']['predicates']['contrasts']['cancel_alone']['mo1s']['total']+=1000
c,_=CK.fam_F(X);probes['contrast_mo1s_total']=dict(family_still_passes=all(c.values()))
out['checker_coverage_probes']=probes
(OUT/'evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('checks','receipt_hashes','predicates','falsifier')},indent=2,default=str),flush=True)
