"""Bounded pinned R2-B review; archived outputs only, no tape replay or fitting.
Run modes meta, sample or logs through PF_SLOT=codex run_light.sh.
The 20-part sample is fixed by position before inspecting its outcomes.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast, collections, gzip, hashlib, importlib.util, json, math, re, sys
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-294fd172')
OUT = ROOT/'predictfun/reviews/automatic/20261120_294fd172'
RC = PIN/'predictfun/reviews/r2b_full_day_20261120'
DATA = ROOT/'data/predict_fun/research/r2_shadow_20261119'
D = DATA/'census_20261023_full'
TARGET = '294fd172d7712bdfeb4c776a33ba63eaf1b7fc0a'
SAMPLE = [0,1,2,3,116,117,118,119,236,237,238,239,356,357,358,359,473,474,475,476]
sys.path.insert(0, str(PIN/'predictfun/research'))
import pf_r2_runner as RN
R = RN.R2

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def js(p): return json.loads(Path(p).read_text())
def save(name, d):
    d.update(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
        runner_sha256=sha(RN.__file__), shadow_sha256=sha(R.__file__))
    (OUT/name).write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')

def rec(i): return js(D/'parts'/f'part_{i:04d}.json')

def meta():
    p=js(D/'plan.json'); j=js(DATA/'census_20261023_full.json'); sup=js(DATA/'support_20261023_full.json')
    psha=sha(D/'plan.json'); assert psha=='249161127c39d5a51cc7c374262073af4f38c9245ef08fb505f0ec3e646dd56b'
    for ln in (RC/'checkpoint_gz.sha256').read_text().splitlines():
        h,rel,_=ln.split(); b=gzip.decompress((RC/rel).read_bytes()); assert hashlib.sha256(b).hexdigest()==h
        actual=D/'plan.json' if rel.startswith('full_checkpoint/') else DATA/'census_20261023_full.json'
        assert b==actual.read_bytes()
    assert sha(DATA/'support_20261023_full.json')==sha(RC/'support_20261023_full.json')
    for ln in (RC/'sources_at_launch.sha256').read_text().splitlines():
        h,fn=ln.split(); assert sha(PIN/'predictfun/research'/Path(fn).name)==h
    assert RN.sources(R)==p['sources']==j['sources']
    assert p['schema']['inputs']==R.INPUTS and p['schema']['ctx']==R.CTX and p['schema']['dtype']=='float32'
    assert p['engine_settings']==R.BE.SETTINGS and p['costs']==R.COSTS and p['causal']==dict(on=True,every=None,sample='r2_full_causal_v1',draws=8)
    pre=js(RC/'preflight_full.json'); assert pre['equal'] and pre['plan_sha256']==psha and pre['actual_sources']==pre['plan_sources']==RN.sources(R)
    assert pre['asof']<j['asof']
    kp=Path(p['inputs']['kp_dir'])/'20261023'; kr=js(kp/'receipt.json')
    assert sha(kp/'receipt.json')==p['inputs']['kp_receipt'] and kr['outputs']==p['inputs']['kp_outputs']
    for f,h in kr['outputs'].items(): assert sha(kp/f)==h
    o=pd.read_parquet(kp/'orders.parquet').sort_values(['m','oid'],kind='stable')
    assert len(o)==30466 and o.side.value_counts().to_dict()=={0:15233,1:15233}
    ids=sorted([int(a),int(b)] for a,b in zip(o.side,o.oid)); assert ids==p['population']['selected_ids']
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==p['population']['selected_ids_sha256']
    assert p['population']['by_status']==o.status.value_counts().to_dict()
    pp=[sorted([int(a),int(b)] for a,b in zip(o.side.iloc[k:k+64],o.oid.iloc[k:k+64])) for k in range(0,len(o),64)]
    assert pp==[x['members'] for x in p['partitions']] and len(pp)==477
    ot={(int(x.side),int(x.oid)):x for x in o.itertuples()}
    rh={rel:h for h,rel in (l.split() for l in (RC/'records_full.sha256').read_text().splitlines())}
    assert len(rh)==477
    records=[]; counts=collections.Counter(); statuses=collections.Counter(); refs=collections.Counter(); cs=collections.Counter(); cells={}
    bytes_checked=0; retry=0
    for i in range(477):
        rf=D/'parts'/f'part_{i:04d}.json'; assert sha(rf)==rh[str(rf.relative_to(DATA))]
        r=js(rf); f=D/'parts'/r['file']; assert sha(f)==r['sha256'] and r['plan_sha256']==psha and r['part']==i and r['members']==pp[i]
        assert j['outputs'][str(f.relative_to(DATA))]==r['sha256']
        assert r['status']=='PASS' and not any(r['breach'].values()) and r['elapsed_s']<=900 and r['hwm_mib']<=1536
        assert r['recon']['checked']==r['sampled_ok'] and r['recon']['failed']==0
        for k in ('rows','sampled_ok','weight_groups','weight_groups_one'): counts[k]+=r[k]
        counts['no_fill']+=r['recon']['no_fill']; statuses.update(r['status_counts']); refs.update(r['refusals'])
        c=r['causal']; au=c['audit']; assert len(au)==c['checked'] and not c['failed']
        for k in ('stored_vs_rebuild','stored_vs_perturbed','stored_vs_visibility','firing','not_evaluable'): assert c.get(k,0)==0; cs[k]+=c.get(k,0)
        assert len(set(a['key'] for a in au))==len(au)
        retried=0
        for a in au:
            k=json.loads(a['key']); original=ot[(k[4],k[3])]; decision=original.t+k[9]
            assert k[0]=='r2_full_causal_v1' and k[1]=='20261023' and k[2]==original.m and [k[4],k[3]] in pp[i]
            assert a['seed_market']==int(decision)%100003 and a['seed_hidden']==int(decision)%100043
            assert a['rebuild'] and a['perturbed'] and a['visibility'] and a['fires']
            draws=a['draws']; assert 1<=len(draws)<=8 and draws[-1][1]=='eligible'
            assert [z[0] for z in draws]==list(range(int(decision)%100019,int(decision)%100019+len(draws)))
            retried+=len(draws)>1
        assert retried==c['retried']; retry+=retried; counts['audits']+=len(au)
        for cell in (r['support'] or {}).get('cells',[]):
            k=f"{cell['sh']} {cell['hbin']}"; dst=cells.setdefault(k,collections.Counter())
            dst.update({z:cell[z] for z in ('rows','originals','nonzero_originals')})
        records.append(r); bytes_checked+=f.stat().st_size+rf.stat().st_size
    rd=j['readings']; assert j['state']=='COMPLETE'
    assert counts['rows']==rd['census_rows']==25463127 and counts['sampled_ok']==rd['sampled_ok_rows']==5384015
    assert counts['weight_groups']==counts['weight_groups_one']==rd['weights']['groups']==rd['weights']['groups_summing_to_one']==60176
    assert counts['no_fill']==rd['labels']['no_fill']==4036595 and statuses==rd['by_source_head_status'] and refs==rd['refusals_by_source']
    assert counts['audits']==rd['causality']['checked']==rd['causality']['audited']==191620 and retry==rd['causality']['retried']==83
    assert cs==rd['causality']['components'] and not rd['partitions']['breaches']
    assert R.support_merge([r['support'] for r in records])==sup and len(sup['cells'])==28
    for k,v in cells.items():
        x=sup['cells'][k]; assert (x['rows'],x['episodes'],x['nonzero_episodes'])==(v['rows'],v['originals'],v['nonzero_originals'])
        assert x['support']=='SUPPORTED' and x['episodes']>=200 and x['nonzero_episodes']>=20 and x['fallback'] is None
    shards=[]
    for k in range(4):
        f=D/f'run_status_{k}of4.json'; assert sha(f)==sha(RC/'full_checkpoint'/f.name)
        st=js(f); assert st['state']=='PARTS_DONE' and st['latch'] is None and st['pending']==[] and st['shard']==[k,4]
        assert st['plan_sha256']==psha and st['done']==list(range(k,477,4)); shards.append(st['hwm_mib'])
    # Frozen 256: inspect real part bytes, not only hash strings in records.
    a=DATA/'census_20261023_gate2A'; b=DATA/'census_20261023_shard4'
    pa,pb=js(a/'plan.json'),js(b/'plan.json')
    assert [k for k in pa if pa[k]!=pb[k]]==['sources'] and [k for k in pa['sources'] if pa['sources'][k]!=pb['sources'][k]]==['pf_r2_runner.py']
    drop={'elapsed_s','part_peak_rss_mib','hwm_mib','asof','breach','plan_sha256'}
    for i in range(4):
        x,y=[js(t/'parts'/f'part_{i:04d}.json') for t in (a,b)]
        assert {k:v for k,v in x.items() if k not in drop}=={k:v for k,v in y.items() if k not in drop}
        assert sha(a/'parts'/x['file'])==sha(b/'parts'/y['file'])==x['sha256']==y['sha256']
        assert sha(b/'parts'/f'part_{i:04d}.json')==sha(RC/'shard4_checkpoint'/f'part_{i:04d}.json')
        st=js(b/f'run_status_{i}of4.json'); assert st['state']=='PARTS_DONE' and st['done']==[i]
    assert js(DATA/'support_20261023_shard4.json')==js(DATA/'support_20261023_gate2A.json')
    bycells=pd.read_csv(RC/'support_cross_cells_full.csv')
    save('metadata.json',dict(all_checks_pass=True,census_asof=j['asof'],acceptance_asof=js(RC/'accept_full.json')['asof'],plan_sha256=psha,
        counts=dict(counts),status_counts=dict(statuses),refusals=dict(refs),audits_retried=retry,all_records_and_parts_hashed=477,
        hashed_bytes=bytes_checked,coarse_cells_supported=28,min_coarse_originals=min(x['episodes'] for x in sup['cells'].values()),
        min_coarse_nonzero=min(x['nonzero_episodes'] for x in sup['cells'].values()),shard_hwm_mib=shards,
        max_part_seconds=max(r['elapsed_s'] for r in records),fine_cells=len(bycells),fine_cells_below_200=int((bycells.originals<200).sum()),
        fine_cells_below_20_nonzero=int((bycells.nonzero_originals<20).sum()),gate3_parts_byte_identical=True,
        sampled_parts_declared=SAMPLE))
    print('All 477 records and parquet hashes, population, source preflight, aggregates, audit seeds, support and gate3 identities pass.')

def row_checks(d,r):
    good=(d.status=='ok')&d.sampled.astype(bool); q=d[good]; sg=np.where(q.side==0,1.,-1.)
    st={f'{a} {b} {c}':int(n) for (a,b,c),n in d.groupby(['source','head','status']).size().items()}
    groups=q.groupby(['side','original_episode_id','head']).weight.sum()
    orig=d[d.source=='original']; ep=d[d.source!='original']; status_members=set(zip(orig.side,orig.original_episode_id)); ep_members=set(zip(ep.side,ep.original_episode_id))
    keys=[]
    for _,g in q.groupby(['side','original_episode_id','source','head']):
        kk=[]
        for x in g.itertuples():
            s=json.dumps(['r2_full_causal_v1',str(x.day),int(x.instrument_id),int(x.original_episode_id),int(x.side),x.source,x.head,
                None if pd.isna(x.offset_ms) else int(x.offset_ms),int(x.generation),int(x.decision_ms)],separators=(',',':'))
            kk.append((hashlib.sha256(s.encode()).hexdigest(),s))
        keys.extend(s for _,s in sorted(kk)[:max(1,math.ceil(len(kk)/64))])
    chk=dict(population=set(zip(d.side,d.original_episode_id))==set(map(tuple,r['members'])),
        multiplicity=not (status_members & ep_members) and orig.groupby(['side','original_episode_id']).size().eq(1).all(),
        identity=not d.duplicated(R.SORT_KEY).any(), rows=len(d)==r['rows'] and len(q)==r['sampled_ok'] and st==r['status_counts'],
        weights=len(groups)==r['weight_groups']==r['weight_groups_one'] and np.allclose(groups,1.,atol=1e-12,rtol=0) and (d.weight[~good]==0).all(),
        labels=(q.q>=0).all() and np.isfinite(q[['q','value','qm','severe']]).all().all() and np.allclose(q.value,sg*(q.end_mark-q.price)*q.q,atol=1e-9,rtol=0)
            and np.allclose(q.qm,sg*(q.end_mark-q.pf_mid_ref)*q.q,atol=1e-9,rtol=0) and np.allclose(q.severe,np.maximum(-q.value-R.KP.SEVERE,0),atol=1e-9,rtol=0),
        clocks=np.allclose(q.end_abs_ms-q.t0,R.END,atol=1e-6,rtol=0) and np.allclose(q.decision_abs_ms-q.t0,q.decision_ms,atol=1e-6,rtol=0)
            and np.allclose(q.t_to_end,R.END-q.decision_ms,atol=1e-6,rtol=0)
            and np.allclose(q.effective_ms,np.where(q['head']=='post',q.activation_ms,q.decision_ms+R.KP.CANCEL),atol=1e-9,rtol=0),
        inputs=all(str(d[c].dtype)=='float32' for c in R.INPUTS+R.CTX) and q[R.CTX].notna().all().all(),
        support=R.support_part(d)==r['support'],audit=sorted(keys)==sorted(x['key'] for x in r['causal']['audit']))
    return {k:bool(v) for k,v in chk.items()}

def sample():
    results=[]; controls={}
    for i in SAMPLE:
        r=rec(i); d=pd.read_parquet(D/'parts'/r['file']); ck=row_checks(d,r); assert all(ck.values()), (i,ck)
        results.append(dict(part=i,rows=len(d),sampled_ok=r['sampled_ok'],originals=len(r['members']),audits=len(r['causal']['audit']),checks=ck))
        if i==0:
            idx=d.index[(d.status=='ok')&d.sampled.astype(bool)][0]
            bad=d.copy(); bad.loc[idx,'value']+=1.; controls['wrong_value_refused']=not row_checks(bad,r)['labels']
            dup=pd.concat([d,d.iloc[[idx]]],ignore_index=True); controls['duplicate_identity_refused']=not row_checks(dup,r)['identity']
            rr=dict(r,causal=dict(r['causal'],audit=r['causal']['audit'][1:])); controls['missing_audit_refused']=not row_checks(d,rr)['audit']
            assert all(controls.values()); del bad,dup
    save('sample.json',dict(all_checks_pass=True,parts=results,controls=controls,rows=sum(x['rows'] for x in results),
        sampled_ok=sum(x['sampled_ok'] for x in results),originals=sum(x['originals'] for x in results),audits=sum(x['audits'] for x in results)))
    print('The predeclared 20-part sample passes row, label, clock, input, audit and support checks; all three falsifiers fire.')

def logs():
    sys.path.insert(0,str(RC)); import accept_logs as AL
    a,b=js(RC/'accept_full_v1.json'),js(RC/'accept_full.json')
    assert [k for k in a['results'] if a['results'][k]!=b['results'][k]]==['6_run_disposition']
    assert all(b['results'].values()) and a['notes']==b['notes'] and not a['failing_parts'] and not b['failing_parts']
    assert all(a['logs'][k]==v for k,v in b['logs'].items())
    assert [k for k,v in a['logs'].items() if not v or any(int(x)!=0 or int(y)>1536 for x,y in v)]==['accept_full_v1.log']
    src1=(RC/'accept_full_v1.py').read_text(); src2=(RC/'accept_full.py').read_text()
    # Verify byte-identical population and data predicates, not just matching verdicts.
    assert src1[src1.index('TAG ='):src1.index('# ---- (6)')]==src2[src2.index('TAG ='):src2.index('# ---- (6)')]
    assert src1[src1.index('sh_ok ='):src1.index("(HERE / f'accept")]==src2[src2.index('sh_ok ='):src2.index("(HERE / f'accept")]
    ld=AL.disposition(RC/'logs',b['logs_declared']); assert ld['ok'] and len(ld['logs'])==17 and not ld['missing'] and not ld['bad']
    assert ld['logs']=={k:[tuple(x) for x in v] for k,v in b['logs'].items()}
    cases=AL.cases(); assert len(cases)==11 and all(w==g for _,w,g in cases)
    # The acceptance logs' own dispositions are checked separately, including v1's expected failure.
    for f,rc in [('accept_full_v1.log',1),('accept_full_v2.log',0),('accept_logs_selftest.log',0),('accept_logs_mutants.log',0)]:
        z=AL.FOOTER.findall((RC/'logs'/f).read_text()); assert len(z)==1 and int(z[0][0])==rc and int(z[0][1])<=1536
    import pf_r2_runner_mutants as RM, pf_r2_shadow_mutants as SM
    for m in (RM,SM):
        for i in range(len(m.MUT)): m.source(i)
    runner=(RC/'logs/runner_mutants.log').read_text().splitlines(); shadow=(RC.parent/'r2a_launch_gate_20261120/logs/shadow_mutants_final.log').read_text().splitlines()
    for lines,n in [(runner,20),(shadow,56)]:
        z=[ln for ln in lines if ' -> ' in ln]; assert len(z)==n+1 and 'rc 0 -> passed' in z[0] and all('rc 1 -> caught:' in x for x in z[1:])
    save('logs.json',dict(all_checks_pass=True,unchanged_data_predicates_1_to_5=True,changed_result_only='6_run_disposition',
        old_failure_only='accept_full_v1.log',build_logs=17,falsifiers=cases,runner_saved_mutants=20,shadow_saved_mutants=56,
        source_mutation_targets_resolve=True,acceptance_asof=b['asof'],logs=ld))
    print('v1-to-v2 data predicates identical; sole old failure is own unfinished log. All 17 build logs, 11 controls and archived suites verified.')

if __name__=='__main__': {'meta':meta,'sample':sample,'logs':logs}[sys.argv[1]]()
