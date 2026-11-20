"""Independent pinned R2-E review. Saved artifacts and small corruptions only.
Run each mode through PF_SLOT=codex run_light.sh; no fit or real tape replay.
"""
import ast
import collections
import copy
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-714ce759')
TARGET = '714ce75978ab8dfec29c1fc41fd0dd800b982ad8'
FREEZE = '56e740826d480afd252c487e591678a0faed05f5'
OUT = ROOT / 'predictfun/reviews/automatic/20261120_714ce759'
RC = PIN / 'predictfun/reviews/r2e_label_pilot_20261120'
DATA = ROOT / 'data/predict_fun/research/r2_label_sample_v2_20261120'
A, B = DATA / 'attempt4_pilot_A', DATA / 'attempt4_pilot_B'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import numpy as np
import pandas as pd
import pf_r2_label_sample as S


def read(p): return json.loads(Path(p).read_text())
def digest(b): return hashlib.sha256(b).hexdigest()
def canon(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
def rank(kind, x): return digest(('r2_label_sample_v2_20261120|' + kind + '|' + canon(x)).encode())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def parts():
    plan = read(A / 'plan.json')
    return plan, [(p, read(S.part_paths(A, p['part'])[0]), S.gz_rows(S.part_paths(A, p['part'])[1].read_bytes()),
                   S.gz_rows(S.part_paths(A, p['part'])[2].read_bytes())) for p in plan['partitions']]


def metadata():
    plan = read(A / 'plan.json'); pb = (A / 'plan.json').read_bytes()
    assert pb == (B / 'plan.json').read_bytes() == gzip.decompress((RC / 'plan/attempt4_plan.json.gz').read_bytes())
    assert digest(pb) == (RC / 'plan/attempt4_plan.sha256').read_text().split()[0]
    assert plan['sources'] == S.sources_now() and len(plan['sources']) == 81
    assert plan['freeze_sha256'] == S.L.FREEZE_SHA and plan['settings'] == S.BE.SETTINGS
    assert plan['feature_schema'] == dict(n=107, sha256=digest(canon(S.T.FEATS).encode()))
    for n, h in plan['sources'].items():
        b = subprocess.check_output(['git', 'show', f'{FREEZE}:predictfun/research/{n}'], cwd=ROOT)
        assert digest(b) == h
    fz = S.T.frozen(); S.L.libraries_enforced(fz)
    assert sha(S.T.REAL['out'] / 'freeze.json') == plan['freeze_sha256']
    for n, h in fz['models'].items(): assert sha(S.T.REAL['out'] / 'models' / (n + '.txt')) == h
    # Hash all producing files named by the plan, plus the upstream metadata receipt's own inputs.
    expected = {}
    def remember(p, h):
        p = str(p); assert p not in expected or expected[p] == h; expected[p] = h
    population = {}
    for d in plan['days']:
        sp = plan['sample'][d]; inp = sp['inputs']; kp = Path(plan['kp_dir']) / d
        remember(kp / 'receipt.json', inp['kp_receipt'])
        for n, h in inp['kp_outputs'].items(): remember(kp / n, h)
        for p, h in {**inp['files'], **inp['tracked']}.items(): remember(p, h)
        raw = read(Path(S.X.QP.JD.C.OUT) / d / 'receipt.json')
        for p, h in raw.get('input_hashes', {}).items(): remember(p, h)
        assert S.KP.Tape().offset(d) == inp['offset']
        assert S.X.I.X5.depth_hours(d) == inp['hours']
        O = pd.read_parquet(kp / 'orders.parquet', columns=['m', 't', 'side', 'oid'])
        candidates = []; by = collections.defaultdict(list)
        for (m, t), g in O.groupby(['m', 't'], sort=True):
            assert sorted(g.side.astype(int)) == [0, 1] and float(t) == int(t)
            hour = datetime.fromtimestamp(int(t)/1000, timezone.utc).hour
            assert datetime.fromtimestamp(int(t)/1000, timezone.utc).strftime('%Y%m%d') == d
            k = dict(day=d, m=int(m), t_ms=int(t)); o = dict(zip(g.side.astype(int), g.oid.astype(int)))
            by[hour].append((rank('world', k), canon(k), k, o))
        assert len(by) == 24
        for h, ws in sorted(by.items()):
            ws.sort(); n = min(64, len(ws)); st = next(x for x in sp['strata'] if x['hour'] == h)
            assert st == dict(day=d, hour=h, N=len(ws), n=n, pi=n/len(ws))
            for i, (rk, _, k, o) in enumerate(ws[:n]):
                candidates.append(dict(k, hour=h, rank=rk, pi=n/len(ws), pilot=i == 0, oid0=o[0], oid1=o[1]))
        candidates.sort(key=lambda x:(x['day'], x['m'], x['t_ms']))
        assert candidates == sp['selected'] and digest(canon(candidates).encode()) == sp['selected_sha256']
        assert sp['N_worlds'] == sum(map(len, by.values()))
        population[d] = dict(worlds=sp['N_worlds'], selected=len(candidates), pilot=sum(x['pilot'] for x in candidates),
                             selected_markets=len({x['m'] for x in candidates}), pilot_markets=len({x['m'] for x in candidates if x['pilot']}))
    for p, h in expected.items(): assert sha(p) == h, p
    for ln in (RC / 'checkpoint/checkpoint.sha256').read_text().splitlines():
        h, p = ln.split(); assert sha(RC / 'checkpoint' / p) == h
    # Audit every mutation record's identity, not just a success sentence.
    mut = {}
    for tag, stem, count in [('v1','pf_r2_labels',35), ('v2','pf_r2_label_sample',100)]:
        src = PIN / 'predictfun/research'; harness = src / (stem + '_mutants.py')
        tr = ast.parse(harness.read_text()); node = next(n for n in tr.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MUTANTS' for t in n.targets))
        ms = ast.literal_eval(node.value); assert len(ms) == count
        want = {n: sha(src/n)[:16] for n in (stem+'.py',stem+'_selftest.py',stem+'_mutants.py')}
        for i in ['control'] + list(range(count)):
            lines = (RC/f'logs/mutants_{tag}/mut_{tag}_{i}.log').read_text().splitlines()
            rr = [json.loads(x) for x in lines if x.startswith('{') and '"cls"' in x]; assert len(rr)==1
            r=rr[0]; assert r['index']==str(i) and r['sources']==want
            assert r['cls']==('PASSED' if i=='control' else 'CAUGHT')
            if i!='control': assert r['name']==ms[i][0]
        mut[tag]=dict(caught=count, identities_verified=True)
    old=read(RC/'checks_v3.json'); new=read(RC/'checks.json')
    assert old['checks']['H_suites'] is False and new['checks']['H_suites'] is True
    assert old['readings']==new['readings'] and old['accepted']==new['accepted']
    assert {k:v for k,v in old['checks'].items() if k!='H_suites'}=={k:v for k,v in new['checks'].items() if k!='H_suites'}
    return dict(plan_sha256=digest(pb), sources=81, models=len(fz['models']), population=population,
                input_files=len(expected), input_bytes=sum(Path(p).stat().st_size for p in expected), mutations=mut,
                v3_v4_readings_unchanged=True, all_declared_predicates=new['checks'])


def labels():
    plan, pp = parts(); psha=sha(A/'plan.json'); allrows=[]; cz=collections.Counter(); binding=0; maxhwm=0; controls=0
    for p, rec, rr, pools in pp:
        k=p['part']; ap=S.part_paths(A,k); bp=S.part_paths(B,k)
        for i in range(3): assert ap[i].read_bytes()==(RC/'checkpoint/parts'/ap[i].name).read_bytes()
        assert ap[1].read_bytes()==bp[1].read_bytes() and ap[2].read_bytes()==bp[2].read_bytes()
        for o in (A,B):
            rp,fp,qp=S.part_paths(o,k); r=read(rp); raw=(fp.read_bytes(),qp.read_bytes())
            assert not S.validate(plan,psha,p,S.gz_rows(raw[0]),S.gz_rows(raw[1]),r,raw)
            assert r['elapsed_s']<=plan['limits']['load_s']+plan['limits']['world_s']*len(p['members'])
            assert r['max_pair_s']<=120 and r['hwm_mib']<=1536; maxhwm=max(maxhwm,r['hwm_mib'])
        ct=read(A/'controls'/f'part_{k:03d}.json')
        assert (A/'controls'/f'part_{k:03d}.json').read_bytes()==(RC/'checkpoint/controls'/f'part_{k:03d}.json').read_bytes()
        assert ct['plan_sha256']==psha and ct['part_sha256']==rec['sha256'] and ct['sources']==plan['sources']
        assert ct['intact_valid'] and all(ct['corruptions_refused'].values()) and ct['stale_binding']['binding_bad']>0
        for rv in ct['reversal']['rows']:
            orig=next(r for r in rr if all(r[x]==rv[x] for x in ('m','t_ms','side','head','pick')))
            assert rv['equal'] and rv['Y']==rv['Y_reversed']==orig['result']['Y']; controls+=1
        allrows+=rr
    diffs=collections.Counter(); nz=[]; bygroup=collections.defaultdict(list); pools_by_source=collections.Counter(); pick_weight=collections.defaultdict(float)
    for p, rec, rr, pools in pp:
        sel={canon(S.wkey(w)):w for w in plan['sample'][p['day']]['selected']}
        for pool in pools:
            keys=sorted(pool['keys'],key=lambda k:(rank('row',k),canon(k))); n=len(keys)
            for k in keys: pools_by_source[(p['day'],pool['head'],k['source'])]+=1
            mine=[r for r in rr if (r['m'],r['t_ms'],r['side'],r['head'])==(pool['m'],pool['t_ms'],pool['side'],pool['head'])]
            mand=[k for k in keys if k['source']=='neutral' and k['shadow_source']=='entry' and k['decision_ms']==0]
            want=[('keep',keys[0],1/n)] if pool['head']=='keep' else [('mandatory',mand[0],1.),('extra',next(k for k in keys if k!=mand[0]),1/(n-1))] if mand else [('only',keys[0],1/n)]
            w=sel[canon(dict(day=p['day'],m=pool['m'],t_ms=pool['t_ms']))]
            assert len(want)==len(mine)
            for name,key,q in want:
                r=next(r for r in mine if r['pick']==name)
                assert r['row_key']==key and r['N']==n and r['q']==q and abs(r['w']-1/(w['pi']*n*q))<1e-12
            assert abs(sum(r['w'] for r in mine)-1/w['pi'])<1e-10
    for r in allrows:
        assert r['status']=='ok'; R=r['result']; n0,n1=S.NAMES[r['head']]; a,b=R['arms'][n0],R['arms'][n1]
        assert R['end']==5000 and math.isfinite(R['mark']) and 0<=R['mark']<=100
        assert a['mark']==b['mark']==R['mark'] and R['settings']==plan['settings']
        for arm in (a,b):
            fills=arm['fill_list']; size=arm['sizes']; qty=collections.Counter()
            for fill in fills:
                assert len(fill)==6  # venue trade index is the sixth field (pf_branch_engine.py:173).
                t,g,s,q,pr=fill[:5]
                assert all(math.isfinite(float(x)) for x in (t,g,s,q,pr)) and 0<=t<=5000 and s in (0,1) and q>0 and 0<=pr<=100
                assert g==int(g) and 0<=g<len(size); qty[int(g)]+=q
            assert all(math.isfinite(x) and 0<x<=30 for x in size)
            assert all(q<=size[g]+1e-9 for g,q in qty.items())
            cash=-sum((1 if f[2]==0 else -1)*f[3]*f[4] for f in fills); pos=sum((1 if f[2]==0 else -1)*f[3] for f in fills)
            assert abs(cash-arm['cash'])<1e-6 and abs(pos-arm['position'])<1e-9 and abs(cash+pos*R['mark']-arm['value'])<1e-6
            assert abs(pos)<=30+1e-9
            t=arm['teacher']; assert t['calls']==t['binding_checked']==sum(t['by_action'].values())==sum(t['by_status'].values()) and t['binding_bad']==0
            binding+=t['binding_checked']
        assert abs(R['Y']-(a['value']-b['value']))<1e-6
        assert r['fork']['hidden_equal'] and r['fork']['rebuild_equal']; assert len(r['fork']['inputs'])==107
        assert digest(np.asarray(r['fork']['inputs'],np.float32).tobytes())==r['fork']['inputs_sha256']
        for c in r['causal']:
            for name in ('checked','mismatch','parity_checked','parity_mismatch','delivered_changed'): cz[name]+=c[name]
        y=R['Y']; bygroup[(r['day'],r['head'])].append(r)
        if y!=0: nz.append(dict(day=r['day'],m=r['m'],t_ms=r['t_ms'],side=r['side'],head=r['head'],pick=r['pick'],Y=y))
        diffs['different_fill_lists']+=int(a['fill_list']!=b['fill_list'])
        diffs['different_fill_lists_zero_Y']+=int(a['fill_list']!=b['fill_list'] and y==0)
        econ=lambda arm: collections.Counter((f[2],f[3],f[4]) for f in arm['fill_list'])
        diffs['different_fill_economics']+=int(econ(a)!=econ(b))
        diffs['different_fill_economics_zero_Y']+=int(econ(a)!=econ(b) and y==0)
        diffs['different_cash_or_position_zero_Y']+=int((a['cash']!=b['cash'] or a['position']!=b['position']) and y==0)
    assert len(allrows)==controls==288
    saved=read(RC/'checks.json'); assert cz['mismatch']==cz['parity_mismatch']==0
    for name in cz: assert cz[name]==saved['readings']['causal'][name if name!='delivered_changed' else 'probe_delivered_changed']
    assert binding==saved['readings']['causal']['binding_checked']
    ys=pd.read_csv(RC/'y_summary.csv'); weights=pd.read_csv(RC/'weights.csv')
    for (d,h),rr in bygroup.items():
        y=np.array([r['result']['Y'] for r in rr]); w=np.array([r['w'] for r in rr]); yy=ys[(ys.day.astype(str)==d)&(ys['head']==h)].iloc[0]
        assert yy.n_ok==len(y) and yy.zero==sum(y==0) and yy.positive==sum(y>0) and yy.negative==sum(y<0)
        assert np.isclose(yy.mean_c,np.mean(y))
        ww=weights[(weights.day.astype(str)==d)&(weights['head']==h)].iloc[0]
        assert np.isclose(ww.raw_weight_total,w.sum()) and np.isclose(ww.weighted_mean_Y_c,np.average(y,weights=w)) and np.isclose(ww.ess,w.sum()**2/sum(w*w))
    for o,tag in ((A,'A'),(B,'B')):
        acc=read(o/'accepted.json'); assert (o/'accepted.json').read_bytes()==(RC/f'checkpoint/accepted_{tag}.json').read_bytes()
        assert acc['accepted'] and acc['plan_sha256']==psha and acc['rows']==288
        assert acc['dataset_sha256']==digest(canon(acc['parts']).encode())
        assert acc['statuses']==dict(sorted(collections.Counter(f"{r['day']} {r['head']} {r['status']}" for r in allrows).items()))
        assert not (o/'stops.jsonl').exists()
    tp=pd.read_csv(RC/'throughput.csv'); mat=np.column_stack([np.ones(len(tp)),tp.pairs,tp.worlds]); rank_x=np.linalg.matrix_rank(mat)
    coef=np.linalg.lstsq(mat,tp.wall,rcond=None)[0]; projected=float((coef@np.array([96,18432,3072]))/3600)
    assert np.isclose(projected,saved['readings']['throughput']['projected_core_hours'])
    return dict(pairs=288,zero=288-len(nz),positive=sum(x['Y']>0 for x in nz),negative=sum(x['Y']<0 for x in nz),
                nonzero_originals=len({(x['day'],x['m'],x['t_ms'],x['side']) for x in nz}),nonzero=nz,
                fill_path_differences=dict(diffs),binding_checked=binding,causal=dict(cz),reversals=controls,
                max_record_hwm_mib=maxhwm,pool_sources={'|'.join(k):v for k,v in sorted(pools_by_source.items())},
                throughput=dict(matrix_rank=int(rank_x),columns=3,pairs_per_world=sorted(set(tp.pairs/tp.worlds)),projected_core_hours=projected),
                all_numerical_and_identity_checks_pass=True)


def controls():
    plan, pp=parts(); p,rec,rows,pools=pp[0]; psha=sha(A/'plan.json')
    assert not S.validate(plan,psha,p,rows,pools)
    result={}
    mutations={'Y+1':lambda r:r['result'].__setitem__('Y',r['result']['Y']+1),
        'wrong_fork':lambda r:r['result'].__setitem__('fork_fingerprint','0'*64),
        'false_reconcile':lambda r:r['result'].__setitem__('reconciled',False),
        'binding_bad':lambda r:next(iter(r['result']['arms'].values()))['teacher'].__setitem__('binding_bad',1),
        'end_changed':lambda r:r['result'].__setitem__('end',4000),
        'failed_pair_disguised_as_feature_unavailable':lambda r:(r.__setitem__('status','feature_unavailable'),r['result'].__setitem__('reconciled',False)),
        'ok_disguised_as_pair_not_ok':lambda r:r.__setitem__('status','pair_not_ok')}
    for name,change in mutations.items():
        rr=copy.deepcopy(rows);change(rr[0]);bad=S.validate(plan,psha,p,rr,pools);result[name]=dict(refused=bool(bad),messages=bad[:2])
    assert all(result[n]['refused'] for n in ('Y+1','wrong_fork','false_reconcile','binding_bad'))
    # A different producing-data END/status is not present in the real pilot; the above are in-memory known-bad controls only.
    import pf_mutant_summary as MS
    MS.selftest()
    return dict(intact_pass=True,corruption_results=result)


if __name__=='__main__':
    mode=sys.argv[1]; obj=globals()[mode]()
    obj.update(target=TARGET,asof=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),mode=mode)
    (OUT/(mode+'.json')).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
    print(json.dumps(dict(mode=mode,result='PASS',output=str(OUT/(mode+'.json')))))
