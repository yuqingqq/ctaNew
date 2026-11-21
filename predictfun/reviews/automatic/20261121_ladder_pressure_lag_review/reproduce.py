"""Pinned, bounded review: synthetic controls and saved artifacts, no real replay/fit.
Checker paths are relocated to the pinned checkout; data paths stay read-only.
"""
from pathlib import Path
import collections, copy, hashlib, json, os, subprocess, sys, types
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-lag-review-6b6f977b')
TARGET='6b6f977bac5eee954033c7d9326506832d935b31'; OUT=Path(__file__).resolve().parent
assert subprocess.check_output(['git','-C',str(PIN),'rev-parse','HEAD']).decode().strip()==TARGET
os.environ['PF_TEACHER_SYN']=str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_ladder_pressure_lag as L
import pf_ladder_pressure_lag_selftest as ST
ST.main(L)
R=PIN/L.RECEIPT; p=R/'lag_checks_v2.py'; CK=types.ModuleType('lag_review_checks'); CK.__file__=str(p)
src=p.read_text().replace("REPO = Path('/home/yuqing/ctaNew')",f"REPO = Path('{PIN}')")
exec(compile(src,str(p),'exec'),CK.__dict__)
CK.D=ROOT/'data/predict_fun/research/ladder_pressure_lag_v1_20261121'
CK.PD_=ROOT/'data/predict_fun/research/ladder_pressure_v1_20261121'
CK.RUN=ROOT/'data/predict_fun/research/ladder_rest_v1_20261121'
CK.AUD=ROOT/'data/predict_fun/research/ladder_audit_v1_20261121'; CK.EDGE_FILE=CK.RUN/'measure/audit.json'
CK.PC.REPO=PIN
E=CK.load();checks=CK.run_checks(E);assert checks['all_ok'],checks
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
for n,h in E['gm']['sources'].items():assert sha(PIN/'predictfun/research'/n)==h,n
assert sha(PIN/L.P.RECEIPT/'manifest.json')==E['gm']['parent']['manifest_sha256']
for day in E['gm']['days']:
 rec,_,_=L.parent_day(day,E['gm']['parent']['day_records'][day])
 assert rec['sha256']['npz']==sha(CK.PD_/'days'/f'{day}_arrays.npz')
 assert rec['sha256']['gz']==sha(CK.PD_/'days'/f'{day}.jsonl.gz')
out['parent_hashes_verified']=True
caches={(r['_day'],int(r['m'])):CK.cache(E,r['_day'],int(r['m'])) for r in E['recs']}
joined,join=L.lag_join(E['states'],caches)
sel=L.P.selection(joined,{s:E['gm']['edges'][str(s)] for s in (0,1)})
for k,v in sel.items():
 if isinstance(v,float):assert np.isclose(v,E['selection'][k],atol=1e-9,rtol=0,equal_nan=True),k
 else:assert v==E['selection'][k],k
assert join==E['selection']['join'];out['selection']=sel;out['join']=join
assert L.predicates(E['arms'],sel['answer'],E['parent_selection']['answer'])==E['maudit']['predicates']
assert (CK.PD_/'selection.json').read_bytes()==(PIN/L.P.RECEIPT/'outputs/selection.json').read_bytes()
print('Pinned selftest, 21 checks, all receipt/source/parent hashes and selection passed',flush=True)

derived=[];commands=collections.Counter();queue_rows=0;warm=collections.Counter();future_checks=0
for r in E['recs']:
 day,m=r['_day'],int(r['m']);end=float(r['lag1']['end']);parent=E['parent'][(day,m)]
 assert r['lag1']['ident']==r['lag0']['ident']==parent['pressure']['ident']
 assert r['lag1']['mark']==r['lag0']['mark']==parent['disabled']['mark']
 assert r['lag1']['tape_id']==r['lag0']['tape_id']==parent['pressure']['tape_id']
 for nm in ('commands','decisions','pose','posc','res','cash','mid'):
  assert np.array_equal(E['arr'][day][f'{m}__lag0__{nm}'],E['parr'][day][f'{m}__pressure__{nm}'],equal_nan=True),(day,m,nm)
 for arm in ('baseline','lag0','lag1'):
  rec=CK.arm_ledger(E,r,arm);f=np.asarray(rec['fills'],float).reshape(-1,6);mm=np.asarray(rec['fill_mids'],float)
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
  for k,vv in vals.items():assert np.isclose(row[k],vv,atol=1e-6,rtol=0),(day,m,arm,k)
  derived.append(dict(day=day,m=m,arm=arm,**vals))
  if arm=='baseline':continue
  logged=collections.Counter();lag=0 if arm=='lag0' else 1;c=caches[(day,m)]
  for z in rec['guard_log']:
   k=int(z[0]/50);s=int(z[1]);j=k-lag
   assert j>=0 and z[5]==float(c['x'][s][j]) and z[6]==float(c['age'][s][j]) and z[11]==50*j and z[12]==z[6]+lag*50
   assert z[8]==sum((1 if a[2]==0 else -1)*a[3] for a in rec['fills'] if a[0]+100<=z[0])
   if z[10]=='pressure_pull':
    o=rec['orders'][int(z[3])];assert o['req']==z[0] and o['remove']==z[0]+37 and o['act']<=z[0]
   logged[(z[0],z[1],z[2],z[3],z[4],z[10])]+=1
  seen=collections.Counter()
  for z in CK.arm_cmds(E,r,arm):
   why=L.P.REASONS[int(z[8])]
   if why.startswith('pressure'):
    seen[(z[0],int(z[1]),L.M.CMD_KINDS[int(z[2])],int(z[3]),z[4],why)]+=1;commands[arm+'|'+why]+=1
  assert seen==logged
  warm[arm]+=sum(vv for kk,vv in rec['missing'].items() if kk.endswith('|warmup'))
  rp=L.reentry_pairs(rec);stored=E['reentry'].query('day==@day and m==@m and arm==@arm').set_index('pull_gid')
  assert len(rp)==len(stored)
  for z in rp:
   sr=stored.loc[z['pull_gid']]
   for k,vv in z.items():
    if k=='pull_gid':continue
    if vv is None or (isinstance(vv,float) and np.isnan(vv)):assert pd.isna(sr[k]),k
    elif isinstance(vv,bool):assert sr[k]==vv,k
    else:assert np.isclose(sr[k],vv,atol=1e-7,rtol=0),(k,sr[k],vv)
  queue_rows+=len(rp)
 # A direct read-barrier control: changing tick k onward cannot change value(k).
 for s in (0,1):
  for k in (1,20,100,len(caches[(day,m)]['x'][s])-1):
   guard=object.__new__(L.LaggedPressureGuard);guard.cache=caches[(day,m)];guard.lag=1
   v=dict(t=k*50.,side=s);before=guard.value(v)
   z=copy.deepcopy(guard.cache)
   for ss in (0,1):z['x'][ss][k:]=123.;z['age'][ss][k:]=999.;z['status'][ss][k:]=1
   guard.cache=z;assert guard.value(v)==before
   z['x'][s][k-1]+=np.float32(10.);assert guard.value(v)[0]!=before[0];future_checks+=1
print('57 ledgers, exact exposure, all intervention logs and re-entry rows reconciled',flush=True)
D=pd.DataFrame(derived);D.to_csv(OUT/'independent_arms.csv',index=False)
out.update(interventions=dict(commands),warmup_policy_evaluations=dict(warm),reentry_rows_checked=queue_rows,direct_read_barrier_controls=future_checks)
tot=D.groupby('arm').sum(numeric_only=True);out['totals']={arm:{k:float(tot.loc[arm,k]) for k in ('value','all_q','all_cut_mo1s','all_cut_valid_q','all_id_move','all_id_late','all_mo50')} for arm in tot.index}
out['retained']=E['maudit']['predicates']['retained']
out['falsifiers']=CK.falsify(E,checks);assert out['falsifiers']['falsifier_ok']
# A low-priority residual checker gap; no disk output is altered.
X=dict(E);X['reentry']=E['reentry'].copy();X['reentry']['next_price']=X['reentry']['next_price']+1
out['changed_reentry_prices_pass_H']=CK.fam_H(X)[0]['H_queue']
assert out['changed_reentry_prices_pass_H']
(OUT/'evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps(dict(totals=out['totals'],retained=out['retained'],interventions=out['interventions'],queue_rows=queue_rows,future_checks=future_checks),indent=2))
