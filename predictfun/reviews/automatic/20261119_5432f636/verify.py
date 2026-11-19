"""Pinned R2-A review; saved census and bounded synthetic boundary checks only."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,sys,subprocess,tempfile,contextlib,io
import numpy as np
import pandas as pd
ROOT=Path('/home/yuqing/ctaNew');PIN=Path('/tmp/pf-review-5432f636')
OUT=ROOT/'predictfun/reviews/automatic/20261119_5432f636'
TARGET='5432f636c63c11e7e04d59be35320589b2cfcf64'
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_r2_shadow as R

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    data.update(target_commit=TARGET,asof=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),source_sha256=sha(R.__file__))
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False,default=str)+'\n')

def artifacts():
    d=ROOT/'data/predict_fun/research/r2_shadow_20261119';f=PIN/'predictfun/reviews/r2a_shadow_census_20261119'
    hashes={}
    for line in (f/'sources.sha256').read_text().splitlines():
        h,rel=line.split(None,1);assert sha((ROOT if rel.startswith('data/') else PIN)/rel)==h,rel;hashes[rel]=h
    j=json.loads((d/'census_20261023_v3.json').read_text());S=json.loads((d/'support_20261023_v3.json').read_text())
    for name in ['census_20261023_v3.json','support_20261023_v3.json']:
        assert sha(d/name)==sha(f/name)
    assert j['source_sha256']==sha(R.__file__) and j['engine_sha256']==sha(R.BE.__file__)
    assert j['inputs_list_sha256']==hashlib.sha256(json.dumps(R.INPUTS+R.CTX).encode()).hexdigest()
    kp=ROOT/'data/predict_fun/research/keep_pull_20261117/20261023'
    assert sha(kp/'receipt.json')==j['inputs']['kp_receipt']
    old=json.loads((kp/'receipt.json').read_text());assert sha(kp/'orders.parquet')==old['outputs']['orders.parquet']
    orig=pd.read_parquet(kp/'orders.parquet',columns=['oid','m','side','status','t'])
    sel=pd.concat([orig[orig.side==s].sort_values('oid').head(128) for s in (0,1)])
    ids=sorted((int(s),int(i)) for s,i in zip(sel.side,sel.oid))
    assert [list(x) for x in ids]==j['readings']['population']['selected_ids']
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==j['readings']['population']['selected_ids_sha256']
    D=pd.read_parquet(d/'census_20261023_v3.parquet');ok=D[(D.status=='ok')&D.sampled.astype(bool)]
    assert len(D)==213349 and len(ok)==44959
    observed=set(zip(D.side.astype(int),D.original_episode_id.astype(int)))
    assert observed==set(ids)
    assert not D.duplicated(R.SORT_KEY).any()
    assert D[R.SORT_KEY].equals(R.canonical(D[R.SORT_KEY]))
    V=pd.read_parquet(d/'census_20261023_v2.parquet');V=R.canonical(V.drop(columns=['weight']));V['weight']=R.weights(V)
    assert V.equals(D);del V
    assert np.allclose(D.weight,R.weights(D),rtol=0,atol=1e-12)
    status={f'{a} {b} {c}':int(n) for (a,b,c),n in D.groupby(['source','head','status']).size().items()}
    assert status==j['readings']['by_source_head_status']
    sign=np.where(ok.side==0,1.,-1.)
    assert np.max(np.abs(ok.value-sign*(ok.end_mark-ok.price)*ok.q))<1e-7
    assert np.max(np.abs(ok.qm-sign*(ok.end_mark-ok.pf_mid_ref)*ok.q))<1e-7
    assert np.max(np.abs(ok.severe-np.maximum(-ok.value-R.KP.SEVERE,0)))<1e-7
    assert (ok.q>=-1e-9).all() and (ok.q<=ok.fp_remaining+1e-6).all()
    assert np.allclose(ok.end_abs_ms-ok.t0,R.END,rtol=0,atol=1e-6)
    assert np.allclose(ok.decision_abs_ms-ok.t0,ok.decision_ms,rtol=0,atol=1e-6)
    assert (ok.t_to_end==R.END-ok.decision_ms).all()
    assert ok[R.CTX].notna().all().all() and len(R.INPUTS)==91 and len(R.CTX)==16
    for side,oid in ids:
        g=D[(D.side==side)&(D.original_episode_id==oid)];m=int(g.instrument_id.iloc[0])
        ph={h:int(hashlib.sha256(f'r2_shadow_sampling_v1|20261023|{m}|{oid}|{side}|{h}'.encode()).hexdigest()[:16],16)%5 for h in ['post','retain']}
        if not (g.source=='original').all():
            assert (g.phase_post==ph['post']).all() and (g.phase_retain==ph['retain']).all()
            es=g[g.source=='entry'];assert sorted(es.offset_ms.tolist())==[0.]+[50.*k for k in range(1,100) if k%5==ph['post']]
            retained=g[(g.source=='retain')&(g.status=='ok')]
            assert (((retained.decision_ms/50).astype(int)%5==ph['retain'])|retained.first_live.fillna(False)).all()
    assert R.support(D)==S
    independent_cells={}
    for (s,h),g in ok.groupby(['side','head']):
        for lo,hi in zip(R.HORIZON_BINS[:-1],R.HORIZON_BINS[1:]):
            q=g[(R.END-g.decision_ms>lo)&(R.END-g.decision_ms<=hi)]
            key=f'{"bid" if s==0 else "ask"} {h} ({lo}, {hi}]'
            if len(q):
                ep=q.original_episode_id.nunique();nz=q[q.q>1e-12].original_episode_id.nunique()
                independent_cells[key]=dict(rows=len(q),episodes=ep,nonzero_episodes=nz,support='SUPPORTED' if ep>=200 and nz>=20 else 'INSUFFICIENT_TRAINING_SUPPORT')
    assert independent_cells==S['cells']
    logs=(f/'logs/pr2_mutants8.log').read_text().splitlines();mut=[x for x in logs if ' -> ' in x]
    assert len(mut)==26 and 'rc 0 -> passed' in mut[0] and all('rc 1 -> caught: R2 SHADOW SELFTEST FAILED' in x for x in mut[1:])
    ranges={f'{s} {h}':{c:[float(g[c].min()),float(g[c].max())] for c in ['ctx_generation_age_ms','ctx_generation_ordinal','ctx_inventory_shares','ctx_bid_reserved_shares','ctx_ask_reserved_shares','t_to_end']} for (s,h),g in ok.groupby(['side','head'])}
    categories={f'{s} {h}':{z:int(g['ctx_other_state_'+z].sum()) for z in ['absent','post_pending','live','cancel_pending','terminal']} for (s,h),g in ok.groupby(['side','head'])}
    save('artifacts.json',dict(all_numeric_checks_pass=True,hashes=hashes,saved_asof=j['asof'],rows=len(D),sampled_ok=len(ok),features=len(R.INPUTS),context_fields=len(R.CTX),
        full_original_population=len(orig),full_by_side={str(k):int(v) for k,v in orig.side.value_counts().items()},selected_originals=len(ids),selected_missing=0,
        status=status,weight_groups=len(ok.groupby(['original_episode_id','side','head'])),support_cells=len(S['cells']),
        other_side_counts=categories,per_side_head_ranges=ranges,saved_ranges=S['ranges'],
        memory_bytes_D=int(D.memory_usage(deep=True).sum()),slice_seconds=j['seconds'],runtime_linear_hours=len(orig)/256*j['seconds']/3600,
        full_memory_linear_gib=float(D.memory_usage(deep=True).sum()*len(orig)/256/2**30),comparisons_exact=True,developer_mutants_checked=25))
    print('Saved v3 hashes, canonical v2 equality, identities, weights, timing, label algebra and coarse support counts verified.')

def synthetic():
    # Use only synthetic public windows; no real tape is loaded.
    src=R.KP.Syn();day=R.DAY
    W=next(src.pairs(src.load(day,[101],src.offset(day))))
    t=float(W[0]['ws'])+30000.
    o=pd.Series(dict(side=0,oid=1,m=101,t=t,phase=1))
    path,why=R.QP.build_path(W,t,t+R.END);assert path is not None,why
    rows=R.probes(path,o,day,[])
    R.add_inputs(rows,W,o,None,None,R.X.syn_features(),lambda k:51.)
    row=next(r for r in rows if r['source']=='entry' and r['status']=='ok' and r['offset_ms']==0)
    def check(row_,feat=None):
        return R.causal_rows([row_],path,o,day,W,None,None,every=1,feat=feat or R.X.syn_features(),perturb=R.X.syn_perturb,fire=R.X.syn_fire)
    baseline=check(row);assert baseline[:2]==(1,0),baseline
    corrupted={}
    for field in [R.FC.MARKET[0],'fp_remaining','fair_edge','t_to_end']:
        q=copy.deepcopy(row);q[field]=12345.+float(path['mark'])
        result=check(q);assert result[:2]==(1,0),(field,result)
        corrupted[field]=dict(stored_value=q[field],checked=result[0],failures=result[1])
    q=copy.deepcopy(row);q['ctx_inventory_shares']+=1
    bad_context=check(q);assert bad_context[1]==1
    bad_builder=check(row,R.X.syn_features(leak=1));assert bad_builder[1]==1
    # POST-only refusal is an outcome in both source families, but the joint
    # row does not carry its outcome flag into the census counter.
    oid=next(i for i in range(100) if R.phase(day,7,i,0,'post')==0)
    op=pd.Series(dict(side=0,oid=oid,m=7,t=0.,phase=1))
    pr=R.BE.tape([R.BE.frame(0,0,50,52,0,0),R.BE.frame(40,40,47,49,0,0)],[],51.)
    rp=R.probes(pr,op,day,[])
    entry=next(x for x in rp if x['source']=='entry' and x['decision_ms']==0)
    joint=[x for x in rp if x['source'].startswith('joint') and x['decision_ms']==0]
    c=R.BE.fork_state(pr,R.BE.POST_KEEP,0,0)
    _,_,gen=R.hold_from(c['state'],0,0)
    assert entry['refused'] and gen['refused'] and len(joint)==2
    assert all(x['status']=='ok' and x['q']==0 and 'refused' not in x for x in joint)
    refusal=dict(entry_flag=True,engine_joint_refused=True,joint_rows=[dict(source=x['source'],q=x['q'],flag_present='refused' in x) for x in joint])

    # Two declared originals, one source window unavailable: keep the other as
    # a real status row so the census can finish and expose the missing identity.
    with tempfile.TemporaryDirectory(prefix='pf-r2-review-pop-') as td:
        td=Path(td);kp=td/'kp';(kp/day).mkdir(parents=True)
        orig=pd.DataFrame([dict(oid=1,m=101,side=0,t=t,status='missing_endpoint'),
                           dict(oid=2,m=106,side=0,t=t,status='missing_endpoint')])
        orig.to_parquet(kp/day/'orders.parquet',index=False)
        (kp/day/'receipt.json').write_text(json.dumps(dict(outputs={'orders.parquet':sha(kp/day/'orders.parquet')})))
        class Stub:
            def __init__(self,both):self.both=both
            def markets(self,day):return [101,106]
            def offset(self,day):return 0.
            def load(self,*args):return None
            def pairs(self,data):
                yield {0:{'m':101}}
                if self.both:yield {0:{'m':106}}
        def census(both,tag):
            with contextlib.redirect_stdout(io.StringIO()):
                return R.census(day,per_side=2,out=td/tag,src=Stub(both),kp=kp,inputs=R.X.syn_inputs,causal=False)
        good,_,gr=census(True,'good');lost,_,lr=census(False,'lost')
        assert set(good.original_episode_id)=={1,2} and len(good)==2
        assert lr['population']['selected']==2 and set(lost.original_episode_id)=={1}
        missing=dict(declared_selected=lr['population']['selected'],represented_originals=len(set(lost.original_episode_id)),
                     lost_ids=[2],positive_all_present=len(good),retained_statuses=lost.status.tolist())
    save('synthetic.json',dict(clean_causal_control=dict(checked=1,failures=0),
        stored_feature_corruptions_escape=corrupted,context_corruption_caught=bad_context,builder_lookahead_caught=bad_builder,
        missing_window_population=missing,joint_refusal_flag_missing=refusal,source='synthetic_only'))
    print('Stored-input corruptions escape causal checks; missing window drops a selected original; joint POST refusal flag is lost. Positive and firing controls pass.')

if __name__=='__main__':
    {'artifacts':artifacts,'synthetic':synthetic,'selftest':R.selftest}[sys.argv[1]]()
