"""Pinned ENTRY-EXPOSURE review. Saved artifacts and synthetic controls only."""
from pathlib import Path
import collections as co
import copy, gzip, hashlib, io, json, math, os, subprocess, sys
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-review-73f6cfa6')
OUT=ROOT/'predictfun/reviews/automatic/20261121_73f6cfa6'
RC=PIN/'predictfun/reviews/entry_exposure_20261121'; RES=PIN/'predictfun/research'
sys.path.insert(0,str(RES))
import numpy as np
import pandas as pd
import pf_entry_exposure as M
import pf_entry_exposure_summary as V
import pf_entry_exposure_selftest
import pf_entry_exposure_summary_selftest
sys.path.insert(0,str(RC))
import exposure_checks_v3 as Q
X=Q.X; X.RES=RES; X.LOGS=RC/'logs'
sys.path.insert(0,str(RES))
assert Path(M.__file__).parent==RES and Path(V.__file__).parent==RES
S,L,BE=M.LS,M.L,M.BE
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(PIN),*a])
def same(a,b):
    # Compare writer semantics: JSON None becomes CSV missing, never a false mismatch.
    a=pd.read_csv(io.StringIO(a.to_csv(index=False)),float_precision='round_trip')
    b=pd.read_csv(io.StringIO(b.to_csv(index=False)),float_precision='round_trip')
    if 'day' in a:a['day']=a.day.astype(str);b['day']=b.day.astype(str)
    pd.testing.assert_frame_equal(a.reset_index(drop=True),b.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-10)
def key(r):return (str(r['day']),int(r['m']),int(r['t_ms']),int(r['side']))

def metadata():
    n=0
    for line in (RC/'outputs/SHA256SUMS').read_text().splitlines():
        h,p=line.split();assert sha(RC/'outputs'/p)==h,p;n+=1
    assert (RC/'manifest.json').read_bytes()==(M.OUT/'manifest.json').read_bytes()==git('show','6ca42d25:'+M.RECEIPT_REL)
    for name in ('exposure.json','summary.csv','reasons.csv'):
        assert (RC/'outputs'/name).read_bytes()==(V.OUT2/name).read_bytes(),name
    assert gzip.decompress((RC/'outputs/forks.csv.gz').read_bytes())==(V.OUT2/'forks.csv').read_bytes()
    for name in ('exposure.json','summary.csv'):
        assert (RC/'outputs/frozen_v1'/name).read_bytes()==(M.OUT/name).read_bytes(),name
    assert gzip.decompress((RC/'outputs/frozen_v1/forks.csv.gz').read_bytes())==(M.OUT/'forks.csv').read_bytes()
    assert (V.OUT2/'manifest.json').read_bytes()==(M.OUT/'manifest.json').read_bytes()
    assert git('show','6ca42d25:predictfun/research/pf_entry_exposure.py')==(RES/'pf_entry_exposure.py').read_bytes()
    # The effective H comparison is the sole behavioral change from v2 to v3.
    import inspect
    a=inspect.getsource(Q.V2.check_H).replace('(F1.overrides.astype(str).values == F.overrides_counter_v1.astype(str).values).all()', 'counter_equal(F1.overrides, F.overrides_counter_v1)')
    b=inspect.getsource(Q.check_H)
    assert a[a.index('    out ='):]==b[b.index('    out ='):]
    _,_,fz=L.teacher_assets()
    assert all(Path(sys.modules[z].__file__).is_relative_to(RES) for z in sys.modules if z.startswith('pf_') and getattr(sys.modules[z],'__file__',None))
    r={};assert X.check_G(r) and Q.V2.check_G2(r)
    return dict(receipt_hashes=n,teacher_sources=len(fz['sources']),mutants=r,H_body_only_declared_comparison=True,original_runner_frozen=True)

def data():
    man=read(M.OUT/'manifest.json'); assert man['sources']==M.sources(); mf={key(f):f for v in man['sets'].values() for f in v['forks']}
    n=co.Counter(); allrows=[]; observed={}; original_marks={}; examples=[]
    for kind,p in M.PLANS.items():
        plan=read(Path(p['dir'])/'plan.json');assert sha(Path(p['dir'])/'plan.json')==man['sets'][kind]['plan_sha256']
        selected,pop=M.select_worlds(plan)
        assert pop==man['sets'][kind]['population']
        assert {(str(w['day']),w['m'],w['t_ms']) for w in selected}=={k[:3] for k in mf if k[0] in plan['days']}
        # Independent rank, not outcome-selected: exact four smallest canonical digests per hour.
        for d in plan['days']:
            for h in range(24):
                ws=[w for w in plan['sample'][d]['selected'] if w['hour']==h]
                def rank(w):
                    st=S.ckey(dict(day=d,m=int(w['m']),t_ms=int(w['t_ms'])))
                    return hashlib.sha256(('entry_exposure_v1|'+st).encode()).hexdigest(),st
                want=sorted(ws,key=rank)[:4]
                assert {(w['m'],w['t_ms']) for w in selected if w['day']==d and w['hour']==h}=={(w['m'],w['t_ms']) for w in want}
        for r in M.A.dataset_rows(p['dir'],p['dataset']):
            k=key(r)
            if k not in mf or r.get('head')!='post':continue
            if mf[k]['status']!='ok':
                assert r['status']==mf[k]['status'];observed[k]=True;continue
            if r.get('pick')!='mandatory':continue
            assert k not in observed;observed[k]=True; f=mf[k]
            assert r['status']==f['status']
            if f['status']=='ok':
                assert S.is_mandatory(r['row_key']) and r['fork_fingerprint']==f['fork_fingerprint'] and r['oid']==f['oid']
                assert dict(r['row_key'],day=k[0],side=k[3])==f['sel'] and r['result']['Y']==f['Y']
                for arm in ('POST','WAIT'):
                    assert all(r['result']['arms'][arm][h]==v for h,v in f['arms'][arm].items())
                original_marks[k]=r['result']['mark']
    assert set(observed)==set(mf)
    for i,pt in enumerate(man['parts']):
        rp=M.OUT/f'parts/part_{i:03d}.json';gp=M.OUT/f'parts/part_{i:03d}.jsonl.gz';rec=read(rp);b=gp.read_bytes()
        assert rp.read_bytes()==(RC/f'outputs/parts/part_{i:03d}.json').read_bytes()
        assert hashlib.sha256(b).hexdigest()==rec['sha256'] and rec['manifest_sha256']==sha(M.OUT/'manifest.json') and rec['sources']==M.sources()
        assert rec['part']==i and rec['day']==pt['day'] and rec['set']==pt['set'] and rec['worlds']==pt['worlds']
        rows=S.gz_rows(b);assert rec['rows']==len(rows)==2*len(pt['worlds'])
        assert [key(r) for r in rows]==[(w['day'],w['m'],w['t_ms'],s) for w in pt['worlds'] for s in (0,1)]
        for r in rows:
            k=key(r);n[r['status']]+=1;assert r['status']==mf[k]['status']; allrows.append(r)
            if r['status']!='ok':continue
            assert not Q.V2.raw_row_bad_v2(r)
            for name,a in r['arms'].items():
                n['arms']+=1;assert a['end']==5000 and a['settings']==BE.SETTINGS and a['mark']==original_marks[k] and math.isfinite(a['mark'])
                fills=a['fill_list'];orders=a['orders'];assert [o['gid'] for o in orders]==list(range(len(orders)))
                sizes=[o['size'] for o in orders]; assert all(z==30 for z in sizes)
                assert S._ledger_bad(dict(a,sizes=sizes,fills=len(fills))) is None
                assert S._arm_calls_bad(a,0.,k[3],0) is None
                assert a['causal']['checked']==math.ceil(a['teacher']['calls']/35)
                assert all(0<=f[0]<=5000 and f[3]>0 and math.isfinite(f[3]) and math.isfinite(f[4]) for f in fills)
                for o in orders:
                    fq=sum(f[3] for f in fills if f[1]==o['gid'])
                    assert abs(o['rem']-(0. if o['refused'] else o['size']-fq))<1e-7
                    assert not o['refused'] or fq==0
                    assert o['act']==o['submit']+58 and 0<=o['rem']<=o['size']+1e-9
                if name!='POST_KEEP_INITIAL':
                    old=mf[k]['arms']['POST' if name=='POST_PI0' else 'WAIT']
                    assert a['value']==old['value'] and a['fills_sha256']==old['fills_sha256'] and a['decisions_sha256']==old['decisions_sha256']
                    assert L._h(a['requests'])==old['requests_sha256']
            # Every stored component, initial generation and lifespan recomputed from the raw orders/fills.
            for lab,name in (('PI0','POST_PI0'),('KEEP','POST_KEEP_INITIAL')):
                a=r['arms'][name];init=r['init_'+lab];gid=init['gid0'];o=a['orders'][gid];fl=[f for f in a['fill_list'] if f[1]==gid]
                assert o['side']==k[3] and o['submit']==0
                assert init['act']==o['act'] and init['req']==o['req'] and init['remove']==o['remove']
                assert init['refused']==o['refused'] and init['size']==o['size'] and init['ahead_final']==o['ahead']
                full=None;fq=0.
                for f in sorted(fl):
                    fq+=f[3]
                    if full is None and fq>=o['size']-1e-9:full=f[0]
                assert init['filled']==fq and init['first_fill']==min((f[0] for f in fl),default=None) and init['full_fill']==full
                end=min(o['remove'],full if full is not None else math.inf,5000)
                assert init['active_ms']==(0. if o['refused'] else max(0.,end-o['act']))
                assert init['unfilled_at_end']==(o['rem'] if o['remove']>5000 and not o['refused'] else 0.)
                assert len(fl)==len(init['fills'])
                for f,z in zip(fl,init['fills']):
                    assert [z['t'],z['q'],z['price']]==[f[0],f[3],f[4]]
                    assert z['edge']==(1 if k[3]==0 else -1)*(a['mark']-f[4])*f[3]
                    for h in (50,100,1000):
                        assert z['m'+str(h)] is None if f[0]+h>5000 else (z['m'+str(h)] is None or math.isfinite(z['m'+str(h)]))
            p=r['arms']['POST_PI0'];b=r['arms']['POST_KEEP_INITIAL'];sh=r['shadow']
            assert len({(x['t'],x['gid']) for x in sh})==len(sh)
            assert all([x['t'],k[3],'live','KEEP',x['gid']] in b['decision_list'] for x in sh)
            if not sh:assert b['fill_list']==p['fill_list'] and b['decision_list']==p['decision_list'] and b['value']==p['value']
            else:
                j=next(j for j,(a,z) in enumerate(zip(b['decision_list'],p['decision_list'])) if a!=z)
                assert b['decision_list'][j]==[sh[0]['t'],k[3],'live','KEEP',r['init_KEEP']['gid0']]
                assert p['decision_list'][j]==[sh[0]['t'],k[3],'live','PULL',r['init_PI0']['gid0']]
                assert [f for f in b['fill_list'] if f[0]<=sh[0]['t']]==[f for f in p['fill_list'] if f[0]<=sh[0]['t']]
            n['counter_inflated']+=int(r['overrides']!=len(sh));n['with_override']+=bool(sh)
            # Economic comparison ignores branch-local generation ids.
            oa=[f for f in b['fill_list'] if f[2]!=k[3]];ob=[f for f in p['fill_list'] if f[2]!=k[3]]
            canon=lambda ff:[(f[0],f[2],f[3],f[4],f[5]) for f in ff]
            n['opposite_any']+=bool(oa or ob);n['opposite_gid_only']+=oa!=ob and canon(oa)==canon(ob)
            assert canon(oa)==canon(ob) and r['D_opposite']==0
    F=pd.DataFrame([V.flat_v2(r) for r in allrows]);sv=pd.read_csv(V.OUT2/'forks.csv',float_precision='round_trip');same(F,sv)
    for name,rows in (('summary',M.summaries(F)),('reasons',M.reasons(F))):
        fresh=pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)),float_precision='round_trip')
        same(fresh,pd.read_csv(V.OUT2/f'{name}.csv',float_precision='round_trip'))
    js=read(V.OUT2/'exposure.json');assert M.predicates(F)==js['predicates']
    assert js['summary']==M.summaries(F) and js['reasons']==M.reasons(F)
    F1=pd.read_csv(M.OUT/'forks.csv',float_precision='round_trip'); cols=[c for c in F1 if c not in ('overrides','keep_baseline_equal')]
    same(F[cols],F1[cols]);assert np.array_equal(F.overrides_counter_v1,F1.overrides,equal_nan=True)
    return dict(counts=dict(n),manifest_source_rows=len(observed),every_flat_and_summary_field_reproduced=True,predicates=js['predicates'],
                limits='Markout sums/denominators and time bounds checked; raw venue frames not reloaded. END ledgers are fully recomputed.')

def controls():
    os.environ['PF_TEACHER_SYN']=str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
    M.selftest();V.selftest();Q.V2.selftest_v2();Q.selftest_v3()
    R={};checks={}
    for c in 'ABCDEF':checks[c]=getattr(X,'check_'+c)(V.OUT2,R)
    checks['G']=X.check_G(R);checks['G2']=Q.V2.check_G2(R);checks['H']=Q.check_H(V.OUT2,R)
    assert all(checks.values()),R
    Q.HERE=OUT;Q.falsify_v3(V.OUT2);fz=read(OUT/'checks_falsifier_v3.json');assert fz['falsifier_ok']
    # Demonstrate the remaining common-END checker blind spot, on an in-memory copy.
    r=next(r for r in S.gz_rows((M.OUT/'parts/part_000.jsonl.gz').read_bytes()) if r['status']=='ok')
    good=copy.deepcopy(r);assert not Q.V2.raw_row_bad_v2(good)
    good['arms']['POST_KEEP_INITIAL']['end']=6000.
    blind=not Q.V2.raw_row_bad_v2(good)
    return dict(synthetic=True,checks=checks,readings=R,falsifier=fz,wrong_END_accepted_by_row_checker=blind,
                mitigation='data mode verifies END/settings/common marks/whole callback populations on all saved arms')

if __name__=='__main__':
    mode=sys.argv[1];r=globals()[mode]();(OUT/(mode+'.json')).write_text(json.dumps(r,indent=2,sort_keys=True,default=str)+'\n');print(json.dumps(r,sort_keys=True,default=str))
