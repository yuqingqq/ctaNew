"""Pinned launch-gate review: archived parts/checkpoints and bounded controls only.
Run through PF_SLOT=codex run_light.sh. Never reads/replays raw tape or fits.
"""
from pathlib import Path
from datetime import datetime, timezone
import contextlib, copy, hashlib, io, json, math, re, shutil, sys, tempfile, types
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-c8d8fc8d')
OUT = ROOT/'predictfun/reviews/automatic/20261120_c8d8fc8d'
TARGET = 'c8d8fc8d52faeacb9413b33a679f99de451b3624'
sys.path.insert(0, str(PIN/'predictfun/research'))
import pf_r2_runner as N
R = N.R2
DATA = ROOT/'data/predict_fun/research/r2_shadow_20261119'
RC = PIN/'predictfun/reviews/r2a_launch_gate_20261120'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text())
def save(name, d):
    d.update(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
             source_sha256=sha(R.__file__), runner_sha256=sha(N.__file__))
    (OUT/name).write_text(json.dumps(d, indent=2, allow_nan=False, default=str)+'\n')

def key(r):
    return json.dumps(['r2_full_causal_v1', str(r.day), int(r.instrument_id), int(r.original_episode_id), int(r.side),
        str(r.source), str(r.head), None if pd.isna(r.offset_ms) else int(r.offset_ms), int(r.generation), int(r.decision_ms)], separators=(',',':'))

def artifacts():
    h = {}
    for ln in (RC/'sources.sha256').read_text().splitlines():
        want, rel = ln.split(None,1)
        p = ROOT/rel if rel.startswith('data/') else (PIN/'predictfun/research'/rel if rel.startswith('pf_') else RC/rel)
        assert sha(p) == want, rel
        h[rel] = want
    aa, bb = [DATA/f'census_20261023_gate2{x}' for x in 'AB']
    ja, jb = [js(DATA/f'census_20261023_gate2{x}.json') for x in 'AB']
    pa, pb = js(aa/'plan.json'), js(bb/'plan.json')
    assert pa == pb and sha(aa/'plan.json') == ja['plan_sha256'] == jb['plan_sha256']
    assert N.sources(R) == pa['sources']
    assert pa['engine_settings'] == R.BE.SETTINGS and pa['schema']['inputs'] == R.INPUTS and pa['schema']['ctx'] == R.CTX
    assert pa['causal'] == dict(on=True,every=None,sample='r2_full_causal_v1',draws=8)
    assert pa['limits'] == N.LIMITS
    for label, d in [('A',aa),('B',bb)]:
        for pre in ['census','support']:
            f=f'{pre}_20261023_gate2{label}.json'; assert sha(DATA/f) == sha(RC/f)
        assert sha(d/'plan.json') == sha(RC/f'gate2{label}_checkpoint/plan.json')
        assert sha(d/'run_status.json') == sha(RC/f'gate2{label}_checkpoint/run_status.json')
        j = js(DATA/f'census_20261023_gate2{label}.json')
        for rel, want in j['outputs'].items(): assert sha(DATA/rel) == want, rel
    for ln in (RC/'gate2B_checkpoint/records.sha256').read_text().splitlines():
        want, rel=ln.split(None,1); assert sha(bb/'parts'/Path(rel).name) == want
    kp = Path(pa['inputs']['kp_dir'])/'20261023'
    assert sha(kp/'receipt.json') == pa['inputs']['kp_receipt']
    assert sha(kp/'orders.parquet') == pa['inputs']['kp_outputs']['orders.parquet']
    o = pd.read_parquet(kp/'orders.parquet', columns=['oid','side','m','status'])
    selected = pd.concat([o[o.side==s].sort_values('oid').head(128) for s in [0,1]])
    ids=sorted([int(s),int(i)] for s,i in zip(selected.side,selected.oid))
    assert ids == pa['population']['selected_ids']
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == pa['population']['selected_ids_sha256']
    parts = R.partitions(selected)
    records, seen, keys, n256, popstatus, sumrows, n_ok, weights = [], set(), [], 0, {}, 0, 0, 0
    for i in range(4):
        a,b=[js(d/'parts'/f'part_{i:04d}.json') for d in [aa,bb]]
        assert sha(aa/'parts'/f'part_{i:04d}.json') == sha(RC/f'gate2A_checkpoint/part_{i:04d}.json')
        strip=lambda r:{k:v for k,v in r.items() if k not in ['elapsed_s','part_peak_rss_mib','hwm_mib','asof','breach']}
        assert strip(a) == strip(b) and not any(a['breach'].values()) and not any(b['breach'].values())
        assert N.accept(aa,pa,sha(aa/'plan.json'),i) == a
        assert N.accept(bb,pb,sha(bb/'plan.json'),i) == b
        expect=sorted([int(s),int(x)] for s,x in zip(parts[i].side,parts[i].oid))
        assert a['members'] == pa['partitions'][i]['members'] == expect
        members=set(map(tuple,expect)); assert not seen & members; seen |= members
        f=aa/'parts'/a['file']; v=DATA/'census_20261023_r2av3'/a['file']
        assert sha(f)==sha(v)==a['sha256']==b['sha256']
        d=pd.read_parquet(f); assert len(d)==a['rows']==a['keys']
        assert not d.duplicated(R.SORT_KEY).any()
        assert set(zip(d.side.astype(int),d.original_episode_id.astype(int))) == members
        assert d[R.SORT_KEY].equals(R.canonical(d[R.SORT_KEY]))
        good=(d.status=='ok')&d.sampled.astype(bool); q=d[good]
        assert len(q)==a['sampled_ok']==a['recon']['checked']
        st={f'{x} {y} {z}':int(n) for (x,y,z),n in d.groupby(['source','head','status']).size().items()}
        assert st==a['status_counts']
        for x,n in st.items():popstatus[x]=popstatus.get(x,0)+n
        d_st=d[d.source=='original']; d_ep=d[d.source!='original']
        assert d_st.groupby(['side','original_episode_id']).size().eq(1).all()
        assert not set(zip(d_st.side,d_st.original_episode_id)) & set(zip(d_ep.side,d_ep.original_episode_id))
        assert all(str(d[c].dtype)=='float32' for c in R.INPUTS+R.CTX) and q[R.CTX].notna().all().all()
        gw=q.groupby(['side','original_episode_id','head']).weight.sum()
        assert len(gw)==a['weight_groups']==a['weight_groups_one'] and np.allclose(gw,1.,atol=1e-12,rtol=0) and (d.weight[~good]==0).all()
        sg=np.where(q.side==0,1.,-1.)
        assert np.max(np.abs(q.value-sg*(q.end_mark-q.price)*q.q))<1e-7
        assert np.max(np.abs(q.qm-sg*(q.end_mark-q.pf_mid_ref)*q.q))<1e-7
        assert np.max(np.abs(q.severe-np.maximum(-q.value-R.KP.SEVERE,0)))<1e-7
        assert np.allclose(q.end_abs_ms-q.t0,R.END,rtol=0,atol=1e-6)
        assert np.allclose(q.decision_abs_ms-q.t0,q.decision_ms,rtol=0,atol=1e-6)
        assert (q.t_to_end==R.END-q.decision_ms).all()
        assert R.support_part(d)==a['support']
        independent=[]
        for _,g in q.groupby(['original_episode_id','side','source','head']):
            ordered=sorted((hashlib.sha256(k.encode()).hexdigest(),k) for k in (key(r) for r in g.itertuples()))
            independent.extend(k for _,k in ordered[:max(1,math.ceil(len(g)/64))])
            n256+=max(1,math.ceil(len(g)/256))
        audit=a['causal']['audit']; actual=[x['key'] for x in audit]
        assert sorted(independent)==sorted(actual) and len(actual)==len(set(actual))==a['causal']['checked']
        assert all(x['rebuild'] and x['perturbed'] and x['visibility'] and x['fires'] for x in audit)
        for x in audit:
            row=json.loads(x['key']); oid,side,tdec=row[3],row[4],row[9]
            t0=float(q[(q.original_episode_id==oid)&(q.side==side)].t0.iloc[0]); absd=t0+tdec
            assert x['seed_market']==int(absd)%100003 and x['seed_hidden']==int(absd)%100043
            assert 1<=len(x['draws'])<=8 and x['draws'][-1][1]=='eligible'
            assert [z[0] for z in x['draws']]==list(range(int(absd)%100019,int(absd)%100019+len(x['draws'])))
        assert a['causal']['retried']==sum(len(x['draws'])>1 for x in audit)
        records.append(a); keys.extend(actual);sumrows+=len(d);n_ok+=len(q);weights+=len(gw)
    assert seen==set(map(tuple,ids)) and sumrows==213349 and n_ok==44959 and weights==508 and len(keys)==1598
    s=R.support_merge([a['support'] for a in records]); assert s==js(DATA/'support_20261023_gate2A.json')==js(DATA/'support_20261023_gate2B.json')
    assert s==js(DATA/'support_20261023_r2av3.json')
    strip_r=lambda r:{k:v for k,v in r.items() if k not in ['partitions','peak_rss_mib','finalize_hwm_mib']}
    assert ja['state']==jb['state']=='COMPLETE' and strip_r(ja['readings'])==strip_r(jb['readings'])
    assert ja['readings']['by_source_head_status']==popstatus
    cz=ja['readings']['causality'];assert cz['checked']==cz['audited']==1598 and cz['verdict']=='PASS' and not any(cz['components'].values())
    assert all(r['status']=='PASS' for r in records)
    assert strip_r(js(RC/'gate2_refinalize.json')['readings'])==strip_r(ja['readings'])
    refusal=js(RC/'gate2_refuse.json');assert len(refusal)==7 and all(v['refused'] for v in refusal.values())
    b1=(RC/'logs/gate2_B1.log').read_text();assert '"state": "STOPPED", "records": 2' in b1
    pka=[int(x) for x in re.findall(r'peak_rss_mib=(\d+)',(RC/'logs/gate2_A.log').read_text())]
    assert pka==[1179] and max(r['hwm_mib'] for r in records)<=1536 and max(r['elapsed_s'] for r in records)<=900
    fp1,fp2=[js(RC/f'first_run/plan_gate{x}.json') for x in 'AB']
    assert [k for k in fp1 if fp1[k]!=fp2[k]]==['tag'] and 'tag' not in pa
    assert 'resume_identity\': False' in (RC/'logs/gate_checks_FAIL_plan_binds_tag.log').read_text()
    import pf_r2_runner_mutants as RM, pf_r2_shadow_mutants as SM
    for module in [RM,SM]:
        for i in range(len(module.MUT)): module.source(i)
    logs=[ln for ln in (RC/'logs/runner_mutants_final.log').read_text().splitlines() if ' -> ' in ln]
    assert len(logs)==19 and 'rc 0 -> passed' in logs[0] and all('rc 1 -> caught: R2 RUNNER SELFTEST FAILED' in l for l in logs[1:])
    targeted=list((RC/'logs/shadow_targeted').glob('*.log'));assert len(targeted)==18
    assert all('SELFTEST FAILED' in p.read_text() for p in targeted)
    save('artifacts.json',dict(all_checks_pass=True,hashes=h,saved_asof=ja['asof'],resumed_saved_asof=jb['asof'],rows=sumrows,sampled_ok=n_ok,
        originals=len(ids),weight_groups=weights,audits=len(keys),audit_components=cz['components'],retry_count=cz['retried'],support_cells=len(s['cells']),
        parts_identical_to_r2av3=True,resumed_records_identical_except_resources=True,gate_checks=8,saved_peak_rss_mib=pka[0],
        partition_seconds=[r['elapsed_s'] for r in records],old_plan_difference=['tag'],
        mutation_targets=dict(runner=len(RM.MUT)-1,shadow=len(SM.MUT)-1),developer_logs=dict(runner=18,shadow_targeted=18),
        prospective_div256_slice_count=n256,prospective_div256_full_count=n256*30466/256,
        prospective_div256_full_hours=3.3+n256*30466/256*.32/3600))
    print('Pinned source/output/checkpoint hashes, gate2 identities, eight predicates, keys/seeds, labels and support verified.')

def controls():
    # Bounded current visibility fixture, including the previous independent leak.
    path=R.BE.tape([R.BE.frame(0,0,50,52),R.BE.frame(0,500,64,66)],[],50.)
    o=pd.Series(dict(side=0,oid=1,m=7,t=0.,phase=1)); result=[]
    for g in [0.,150.]:
        row=dict(source='entry' if g==0 else 'retain',offset_ms=0.,decision_ms=g)
        b=R.resim(path,o,R.DAY,row); h=R.hidden_copy(b,g,np.random.default_rng(1))
        assert R.observe(b,g,0)==R.observe(h,g,0)
        assert R.observe(b,g,0)!=R.observe(R.hidden_copy(b,g,np.random.default_rng(1),delivered=True),g,0)
        orig=R.client_snapshot
        def leak(b,g,s):
            c,st,gid=orig(b,g,s);j=int(np.searchsorted(b.E.f[:,0],g,side='right')-1)
            c['ctx_inventory_shares']+=b.E.f[j,2]*.01
            return c,st,gid
        R.client_snapshot=leak
        try: assert R.observe(b,g,0)!=R.observe(h,g,0)
        finally:R.client_snapshot=orig
        result.append(dict(decision_ms=g,clean_equal=True,delivered_fires=True,previous_planted_leak_caught=True))
    # Never execute a real partition. Refinalize copies of saved artifacts only.
    a=DATA/'census_20261023_gate2A'; outcomes={}
    with tempfile.TemporaryDirectory(prefix='pf-review-runner-') as tmp:
        def cp(n):
            d=Path(tmp)/n/a.name;d.parent.mkdir();shutil.copytree(a,d);return d
        def fin(d,mod=None):
            with contextlib.redirect_stdout(io.StringIO()):return N.finalize(d,mod)[2]
        d=cp('clean'); oldm,oldr=R.materialize,R.read_parts
        def no_read(*a,**k):raise AssertionError('all-part materialization')
        R.materialize=R.read_parts=no_read
        try: good=fin(d)
        finally:R.materialize,R.read_parts=oldm,oldr
        assert good['state']=='COMPLETE';outcomes['clean_sentinel']=good['state']
        d=cp('resource'); rp=d/'parts/part_0000.json'; r=js(rp);r['elapsed_s']=901.;r['breach']['seconds']=True;N.atomic_json(rp,r)
        plan=js(d/'plan.json');N.atomic_json(d/'run_status.json',dict(state='STOP_RESOURCE',latch=dict(part=0,seconds=True)))
        # All parts already exist: run verifies their hashes, no raw input load.
        src=types.SimpleNamespace(offset=lambda _:plan['inputs']['offset'])
        with contextlib.redirect_stdout(io.StringIO()):
            st=N.run(d,src=src,loaded=({},None,plan['inputs']['hours'],plan['inputs']['tracked']))
        rr=fin(d);assert st=='PARTS_DONE' and rr['state']=='COMPLETE' and rr['partitions']['breaches']==[0]
        outcomes['resource_latch_lost']=dict(run_state=st,final_state=rr['state'],breaches=rr['partitions']['breaches'])
        d=cp('failed');rp=d/'parts/part_0000.json';r=js(rp);r['status']='FAILED';r['recon']['failed']=1;N.atomic_json(rp,r)
        bad=fin(d);assert bad['state']=='FAILED';outcomes['failed_part_control']=bad['state']
        d=cp('different_finalizer'); fake=Path(tmp)/'different_shadow.py';fake.write_text('# alternative source identity\n')
        mod=types.SimpleNamespace(__file__=str(fake),support_merge=R.support_merge)
        assert N.sources(mod)!=js(d/'plan.json')['sources']
        rr=fin(d,mod);assert rr['state']=='COMPLETE'
        outcomes['finalize_omits_source_preflight']=dict(different_source_hash=True,final_state=rr['state'])
    save('controls.json',dict(visibility=result,finalization=outcomes,source='synthetic_visibility_and_saved_copy_controls_only'))
    print('Visibility clean/leak/delivered controls pass; records-only finalize passes. Resource latch and finalize source-preflight gaps reproduced.')

if __name__=='__main__': {'artifacts':artifacts,'controls':controls}[sys.argv[1]]()
