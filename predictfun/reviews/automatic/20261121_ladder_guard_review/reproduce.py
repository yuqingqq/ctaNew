"""Bounded pinned review: synthetic controls and saved ledgers; no real-tape replay or fit."""
from pathlib import Path
import collections, copy, gzip, hashlib, importlib.util, json, os, subprocess, sys
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-lg-review-84a4f999')
TARGET='84a4f999eabdfbe1e756b3f59213a641dc343a4e'; OUT=Path(__file__).resolve().parent
assert subprocess.check_output(['git','-C',str(PIN),'rev-parse','HEAD']).decode().strip()==TARGET
os.environ['PF_TEACHER_SYN']=str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_ladder_guard as G
import pf_ladder_guard_selftest as ST
ST.main(G)
R=PIN/'predictfun/reviews/ladder_guard_20261121'
spec=importlib.util.spec_from_file_location('guard_checks_review',R/'guard_checks.py')
CK=importlib.util.module_from_spec(spec);spec.loader.exec_module(CK)
CK.REPO=PIN;CK.R=R
E=CK.load(); checks=CK.run_checks(E); assert checks['all_ok'], checks
print('Pinned selftest and 21 saved checks passed.',flush=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=dict(target=TARGET,selftest='passed',saved_checks=checks)
hashes={}
for ln in (R/'outputs/SHA256SUMS').read_text().splitlines():
 h,n=ln.split('  ',1);p=R/'outputs'/n;assert sha(p)==h,n
 raw=CK.D/n
 if n in ('arms.csv','audit.json','control.csv'):raw=CK.D/'measure'/n
 assert raw.read_bytes()==p.read_bytes(),n
 hashes[n]=h
out['receipt_hashes']=hashes
assert sha(PIN/'predictfun/reviews/ladder_rest_20261121/manifest.json')==E['gm']['baseline']['manifest_sha256']
assert sha(CK.RUN/'measure/audit.json')==E['gm']['edges_sha256']
assert E['selection']['inputs']==dict(states_sha256=sha(CK.AUD/'states.csv'),generations_sha256=sha(CK.AUD/'generations_v2.csv'),edges_sha256=sha(CK.RUN/'measure/audit.json'))
sel=G.selection(E['states'],E['edges'])
for k,v in sel.items():
 if isinstance(v,float):assert np.isclose(v,E['selection'][k],atol=1e-10,rtol=0,equal_nan=True),k
 else:assert v==E['selection'][k],k
assert G.predicates(E['arms'],sel)==E['maudit']['predicates']
out['selection_reconciled']=sel

arrays={day:np.load(CK.D/'days'/f'{day}_arrays.npz') for day in E['gm']['days']}
baseline_arrays={day:np.load(CK.RUN/'days'/f'{day}_arrays.npz') for day in E['gm']['days']}
diag=[]; cmd_counts=collections.Counter(); derived=[]
for r in E['recs']:
 day,m=r['_day'],int(r['m']); ref=E['base'][(day,m)]['ledger']; d=r['disabled']; end=float(d['end'])
 assert d['fingerprint']==ref['fingerprint'] and d['ident']==ref['ident']
 assert d['tape_id']==r['guard']['tape_id']==ref['tape_id']
 for nm in ('commands','decisions','pose','posc','res','cash','mid'):
  a=arrays[day][f'{m}__disabled__{nm}'];b=baseline_arrays[day][f'{m}__{nm}']
  assert np.array_equal(a,b,equal_nan=True),(day,m,nm)
 metrics={}
 for arm in ('disabled','guard'):
  L=r[arm];f=np.asarray(L['fills'],float).reshape(-1,6);mm=np.asarray(L['fill_mids'],float)
  assert len(f)==len(mm)
  sign=np.where(f[:,2]==0,1.,-1.);q=f[:,3];p=f[:,4];mark=L['mark']
  ve=sign*q*(mark-p);cut=f[:,0]<=end-1000;valid=cut&np.isfinite(mm[:,6]);mo=sign*q*(mm[:,6]-p)
  row=E['arms'][(E['arms'].day==day)&(E['arms'].m==m)&(E['arms'].arm==arm)].iloc[0]
  assert abs(sum(ve)-row.value)<1e-6
  for nm,s in [('all',None),('bid',0),('ask',1)]:
   use=np.ones(len(f),bool) if s is None else f[:,2]==s
   vals={'q':q[use].sum(),'end':ve[use].sum(),'cut_q':q[use&cut].sum(),'cut_valid_q':q[use&valid].sum(),
    'cut_mo1s':mo[use&valid].sum(),'cut_censored':int((use&cut&~valid).sum()),
    'good_mo1s':mo[use&valid&(mo>0)].sum(),'bad_mo1s':mo[use&valid&(mo<0)].sum()}
   for fld,ix in [('edge',0),('mo50',2),('mo100',4)]: vals[fld]=np.nansum((sign*q*(mm[:,ix]-p))[use])
   for fld,v in vals.items():assert np.isclose(v,row[f'{nm}_{fld}'],atol=1e-6,rtol=0),(day,m,arm,nm,fld)
  ar={k.split('__')[2]:arrays[day][k] for k in arrays[day].files if k.startswith(f'{m}__{arm}__')}
  fills_by_gid={}
  for ff in L['fills']:fills_by_gid.setdefault(ff[1],[]).append(ff)
  # Exact inventory integrals, plus reconstruction of the saved grid inventory.
  area=bandtime=0.;pos=0.;last=0.;peak=0.
  for ff in L['fills']:
   area+=abs(pos)*(ff[0]-last);bandtime+=(abs(pos)>=100.-1e-9)*(ff[0]-last)
   pos+=(1 if ff[2]==0 else -1)*ff[3];last=ff[0];peak=max(peak,abs(pos))
  area+=abs(pos)*(end-last);bandtime+=(abs(pos)>=100.-1e-9)*(end-last)
  assert np.isclose(row.mean_abs_inv,np.mean(np.abs(ar['pose'])))
  assert np.isclose(row.share_abs_ge_S,np.mean(np.abs(ar['pose'])>=100.-1e-9))
  order={o['gid']:o for o in L['orders']}
  # Independently rebuild known B and position from submission and notice clocks.
  cache={}
  def known(t):
   if t not in cache:
    I=sum((1 if ff[2]==0 else -1)*ff[3] for ff in L['fills'] if ff[0]+100<=t)
    B=0.
    for o in L['orders']:
     if o['side']!=0 or o['submit']>t or (o['submit']==t and int(round(t/50))%2==1):continue
     kf=sum(ff[3] for ff in fills_by_gid.get(o['gid'],[]) if ff[0]+100<=t)
     if (o['refused'] and o['act']<=t) or kf>=o['size']-1e-9 or (o['req'] is not None and o['remove']+100<=t):continue
     B+=o['size']-kf
    cache[t]=(I,float(np.float32(B)))
   return cache[t]
  logged=collections.Counter()
  for z in L['guard_log']:
   I,B=known(z[0]);assert abs(z[6]-I)<1e-7 and z[5]==B and B>=200,(day,m,z,I,B)
   logged[(z[0],z[1],z[2],z[3],z[4],z[8])]+=1
   if z[8]=='guard_withhold' and z[0]+58>=end:cmd_counts['near_end_withholds']+=1
  seen=collections.Counter()
  for c in ar['commands']:
   kind=G.M.CMD_KINDS[int(c[2])];reason=G.REASONS[int(c[8])];act=G.M.CMD_ACTIONS[int(c[7])]
   if reason.startswith('guard_'):
    seen[(c[0],int(c[1]),kind,int(c[3]),c[4],reason)]+=1;cmd_counts[reason]+=1
   if arm=='guard' and int(c[1])==1 and reason in ('keep','post'):
    assert known(c[0])[1]<200,(day,m,c)
  assert logged==seen,(day,m,arm)
  # Exact active durations independently from event sweeps (no frozen helper).
  ev={0.:[0,0],end:[0,0]}
  for o in L['orders']:
   if o['refused']:continue
   fs=fills_by_gid.get(o['gid'],[]);filled=sum(x[3] for x in fs)
   full=max(x[0] for x in fs) if fs and filled>=o['size']-1e-9 else end
   lo,hi=o['act'],min(o['remove'],full,end)
   if hi<=lo:continue
   ev.setdefault(lo,[0,0])[o['side']]+=1;ev.setdefault(hi,[0,0])[o['side']]-=1
  active=[0,0];last=0.;both=ask=bid=0.
  for t,v in sorted(ev.items()):
   both+=(t-last)*int(all(x>0 for x in active));bid+=(t-last)*(active[0]>0);ask+=(t-last)*(active[1]>0)
   active=[x+y for x,y in zip(active,v)];last=t
  for nm,v in [('both',both),('ask',ask),('bid',bid)]:assert np.isclose(v/end,row[nm+'_exact'],atol=1e-12)
  v1=float(mo[valid].sum());residual=float((sign*q*(mark-mm[:,6]))[valid].sum());late=float(ve[~valid].sum())
  assert abs(v1+residual+late-L['value'])<1e-6
  metrics[arm]=dict(end=L['value'],q=float(q.sum()),mo1s=v1,post1s_to_END=residual,late_or_missing_END=late,
    mean_abs_exact=area/end,mean_abs_grid=row.mean_abs_inv,band_fraction_exact=bandtime/end,band_fraction_grid=row.share_abs_ge_S,
    terminal_inventory=L['position'],peak_abs_exact=peak,both_exact=both/end,
    bid_q=float(q[f[:,2]==0].sum()),ask_q=float(q[f[:,2]==1].sum()))
  derived.append(dict(day=day,m=m,arm=arm,**metrics[arm]))
 a,b=metrics['guard'],metrics['disabled']
 diag.append(dict(day=day,m=m,delta={k:a[k]-b[k] for k in a},arms=metrics))
pd.DataFrame(derived).to_csv(OUT/'independent_arms.csv',index=False)
out['paired_diagnostics']=diag;out['guard_commands_verified']=dict(cmd_counts)
out['all_pairs_checked']=len(diag)
print('All arm metrics, actual guard reservations and commands reconciled.',flush=True)
out['falsifiers']=CK.falsify(E);assert out['falsifiers']['falsifier_ok']
# A checker coverage probe: changing a markout in arms.csv and its matching reading escapes F.
X=copy.deepcopy(E);i=X['arms'].index[X['arms'].arm=='guard'][0]
X['arms'].loc[i,'all_cut_mo1s']+=10000.
X['maudit']['predicates']=G.predicates(X['arms'],X['selection'])
out['coordinated_markout_corruption_accepted']=CK.run_checks(X)['all_ok']
assert out['coordinated_markout_corruption_accepted']
(OUT/'evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps(dict(pairs=len(diag),commands=out['guard_commands_verified'],worst=min(diag,key=lambda x:x['delta']['end']),
 coordinated_markout_corruption_accepted=out['coordinated_markout_corruption_accepted']),indent=2))
