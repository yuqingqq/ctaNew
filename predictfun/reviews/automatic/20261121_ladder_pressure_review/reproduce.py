"""Bounded review of pinned source and saved evidence; no replay, fit or new day.

Reconcile all saved arms, cache rows, participation and intervention clocks;
probe the remaining checker boundary. Runs only in PF_SLOT=codex.
"""
from pathlib import Path
import collections, copy, hashlib, importlib.util, json, os, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-lp-review-40a3ee8d')
TARGET = '40a3ee8d4cac39e98ee065845df63f853544c855'
OUT = Path(__file__).resolve().parent
assert subprocess.check_output(['git', '-C', str(PIN), 'rev-parse', 'HEAD']).decode().strip() == TARGET
os.environ['PF_TEACHER_SYN'] = str(ROOT / 'data/predict_fun/research/r2_teacher_selftest_world')
sys.path.insert(0, str(PIN / 'predictfun/research'))
import pf_ladder_pressure as P
import pf_ladder_pressure_selftest as ST
ST.main(P)
R = PIN / 'predictfun/reviews/ladder_pressure_20261121'
sp = importlib.util.spec_from_file_location('pressure_review_checks', R / 'pressure_checks.py')
CK = importlib.util.module_from_spec(sp); sp.loader.exec_module(CK)
CK.REPO = PIN; CK.R = R
E = CK.load(); checks = CK.run_checks(E)
assert checks['all_ok'], checks
print('Pinned selftest and saved checks passed', flush=True)
out = dict(target=TARGET, selftest='passed', checks=checks)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out['receipt_hashes'] = {}
for ln in (R / 'outputs/SHA256SUMS').read_text().splitlines():
    h, n = ln.split('  ', 1); p = R / 'outputs' / n
    assert sha(p) == h, n
    raw = CK.D / n
    if n in ('arms.csv', 'audit.json', 'control.csv', 'lost_added.csv', 'causal.csv'):
        raw = CK.D / 'measure' / n
    assert raw.read_bytes() == p.read_bytes(), n
    out['receipt_hashes'][n] = h
for n, h in E['gm']['sources'].items():
    assert sha(PIN / 'predictfun/research' / n) == h, n
assert E['selection']['inputs'] == dict(states_sha256=sha(CK.AUD/'states.csv'), edges_sha256=sha(CK.EDGE_FILE))
sel = P.selection(E['states'], {s:E['gm']['edges'][str(s)] for s in (0,1)})
for k,v in sel.items():
    if isinstance(v,float): assert np.isclose(v,E['selection'][k],atol=1e-9,rtol=0,equal_nan=True),k
    else: assert v == E['selection'][k], k
assert P.predicates(E['arms'],sel['answer']) == E['maudit']['predicates']
out['selection'] = sel
derived=[]; commands=collections.Counter(); clocks=[]; queue=[]
for day in E['gm']['days']:
    # Filtered reads of the already-built step rows; no feature rebuild or tape load.
    members=E['gm']['complete'][day]
    raw=pd.read_parquet(ROOT/f'data/predict_fun/research/step5/{day}/rows_rl_d2_pk.parquet',
                        columns=['mid','tau','bn_r250','bn_age'],filters=[('mid','in',members)])
    ck=json.loads((ROOT/f'data/predict_fun/research/step1/{day}/clock_rl_d2.json').read_text())
    assert ck['d_dec']==2 and ck['bn_in']==1.5
    bas=np.load(CK.RUN/'days'/f'{day}_arrays.npz')
    for r in [r for r in E['recs'] if r['_day']==day]:
        m=int(r['m']); end=float(r['disabled']['end'])
        assert r['pressure']['mark']==r['disabled']['mark']
        assert r['pressure']['ident']==r['disabled']['ident']==E['base'][(day,m)]['ledger']['ident']
        assert r['pressure']['tape_id']==r['disabled']['tape_id']==E['base'][(day,m)]['ledger']['tape_id']
        for nm in ('commands','decisions','pose','posc','res','cash','mid'):
            assert np.array_equal(CK.arr(E,day,m,'disabled__'+nm),bas[f'{m}__{nm}'],equal_nan=True),(day,m,nm)
        start=r['disabled']['ident']['start_abs_ms']
        for side in (0,1):
            xs=CK.arr(E,day,m,f'cache__x{side}'); ages=CK.arr(E,day,m,f'cache__age{side}')
            rr=raw[raw.mid==m].set_index('tau').loc[start+50*np.arange(len(xs))]
            want=(rr.bn_r250.to_numpy(np.float32)*(-1 if side==0 else 1)).astype(np.float32)
            assert np.array_equal(want.view(np.uint32),xs.view(np.uint32))
            assert np.array_equal(rr.bn_age.to_numpy(np.float32).view(np.uint32),ages.view(np.uint32))
            assert np.isfinite(ages).all() and (ages>=0).all()
            clocks.append(dict(day=day,m=m,side=side,rows=len(xs),min_age=float(ages.min()),max_age=float(ages.max())))
        for arm in ('disabled','pressure'):
            L=r[arm]; f=np.asarray(L['fills'],float).reshape(-1,6); mm=np.asarray(L['fill_mids'],float)
            assert len(f)==len(mm) and (f[:,0]>=0).all() and (f[:,0]<=end).all()
            row=E['arms'].query('day==@day and m==@m and arm==@arm').iloc[0]
            q=f[:,3]; sg=np.where(f[:,2]==0,1.,-1.); ve=sg*q*(L['mark']-f[:,4])
            valid=(f[:,0]<=end-1000)&np.isfinite(mm[:,6]); mo=sg*q*(mm[:,6]-f[:,4])
            assert np.isclose(ve.sum(),row.value,atol=1e-6,rtol=0)
            fills=collections.defaultdict(list)
            for ff in L['fills']: fills[int(ff[1])].append(ff)
            area=high=pos=last=peak=0.
            for ff in L['fills']:
                area+=abs(pos)*(ff[0]-last); high+=(abs(pos)>=100-1e-9)*(ff[0]-last)
                pos+=(1 if ff[2]==0 else -1)*ff[3];last=ff[0];peak=max(peak,abs(pos))
            area+=abs(pos)*(end-last);high+=(abs(pos)>=100-1e-9)*(end-last)
            for name,val in [('mean_abs_inv',area/end),('share_abs_ge_S',high/end),('peak_abs_inv',peak),('terminal_inv',pos)]:
                assert np.isclose(row[name],val,atol=1e-8,rtol=0),(day,m,arm,name)
            ev={0.:[0,0],end:[0,0]}
            for o in L['orders']:
                if o['refused']:continue
                fs=fills[o['gid']]; full=fs[-1][0] if fs and sum(x[3] for x in fs)>=o['size']-1e-9 else end
                lo,hi=o['act'],min(o['remove'],full,end)
                if hi>lo:
                    ev.setdefault(lo,[0,0])[o['side']]+=1;ev.setdefault(hi,[0,0])[o['side']]-=1
                birth=L['queue_venue']['ahead0'][str(o['gid'])]
                assert birth[0]==o['submit'] and o['act']==o['submit']+58
            counts=[0,0];last=0.;times=np.zeros(3)
            for t,dv in sorted(ev.items()):
                times+=(t-last)*np.array([counts[0]>0,counts[1]>0,all(x>0 for x in counts)])
                counts=[x+y for x,y in zip(counts,dv)];last=t
            for name,val in zip(('bid_exact','ask_exact','both_exact'),times/end):
                assert np.isclose(row[name],val,atol=1e-12,rtol=0),(day,m,arm,name)
            logged=collections.Counter()
            for z in L['guard_log']:
                I=sum((1 if ff[2]==0 else -1)*ff[3] for ff in L['fills'] if ff[0]+100<=z[0])
                assert np.isclose(z[8],I,atol=1e-8,rtol=0)
                logged[(z[0],z[1],z[2],z[3],z[4],z[10])]+=1
                if z[10]=='pressure_pull':
                    o=L['orders'][int(z[3])]
                    assert o['req']==z[0] and o['remove']==z[0]+37 and o['act']<=z[0]
            seen=collections.Counter()
            for c in CK.arr(E,day,m,arm+'__commands'):
                why=P.REASONS[int(c[8])]
                if why.startswith('pressure'):
                    seen[(c[0],int(c[1]),P.M.CMD_KINDS[int(c[2])],int(c[3]),c[4],why)]+=1;commands[why]+=1
            assert seen==logged
            od={o['gid']:o for o in L['orders']}
            for gid,v in L['queue_venue']['pull'].items():
                o=od[int(gid)];nxt=sorted((x['submit'],x['gid']) for x in L['orders'] if x['side']==o['side'] and x['submit']>v[0])
                if nxt:
                    oo=od[nxt[0][1]]
                    queue.append(dict(day=day,m=m,gid=int(gid),new_gid=oo['gid'],same_rung=oo['rung']==o['rung'],same_price=oo['price']==o['price'],
                                      delay=oo['submit']-v[0],before_old_removal=oo['submit']<o['remove']))
            derived.append(dict(day=day,m=m,arm=arm,end_value=float(ve.sum()),q=float(q.sum()),mo1s=float(mo[valid].sum()),valid_q=float(q[valid].sum()),
                                 mo50=float(np.nansum(sg*q*(mm[:,2]-f[:,4]))),bid_exact=times[0]/end,ask_exact=times[1]/end,both_exact=times[2]/end,
                                 mean_abs_inv=area/end,peak_abs_inv=peak,terminal_inv=pos))
print('All cache rows, disabled paths, 38 ledgers, exact exposure and intervention times reconciled',flush=True)
out['clocks']=clocks;out['commands']=dict(commands);out['queue_reentries']=dict(n=len(queue),different_rung=sum(not x['same_rung'] for x in queue),
    same_price=sum(x['same_price'] for x in queue),before_old_removal=sum(x['before_old_removal'] for x in queue))
pd.DataFrame(derived).to_csv(OUT/'independent_arms.csv',index=False)
out['day_totals']=pd.DataFrame(derived).groupby(['day','arm']).agg({'end_value':'sum','q':'sum','mo1s':'sum','valid_q':'sum','mo50':'sum','both_exact':'mean','bid_exact':'mean','ask_exact':'mean'}).reset_index().to_dict('records')
out['falsifiers']=CK.falsify(E,checks);assert out['falsifiers']['falsifier_ok']
# One unresolved acceptance-instrument boundary. No data on disk is changed.
X=dict(E);X['arms']=E['arms'].copy();ix=X['arms'].arm=='pressure'
X['arms'].loc[ix,'both_exact']+=.25
X['maudit']=dict(E['maudit'],predicates=P.predicates(X['arms'],sel['answer']))
out['coordinated_uptime_corruption_accepted']=CK.run_checks(X)['all_ok']
assert out['coordinated_uptime_corruption_accepted']
(OUT/'evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps(dict(arms=len(derived),cache_rows=sum(x['rows'] for x in clocks),commands=out['commands'],queue=out['queue_reentries'],
                     coordinated_uptime_corruption_accepted=out['coordinated_uptime_corruption_accepted']),indent=2))
