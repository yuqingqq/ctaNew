"""Review-only: pinned audit source, saved labels and synthetic fixtures; no fit or tape."""
import collections as co
import copy
import hashlib
import importlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-review-37300cc9')
OUT=ROOT/'predictfun/reviews/automatic/20261121_37300cc9'
RC=PIN/'predictfun/reviews/entry_target_audit_20261121'
sys.path.insert(0,str(PIN/'predictfun/research'))
import numpy as np
import pandas as pd
import pf_entry_audit as M
A,N,LS=M.A,M.N,M.LS
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(PIN),*a])
def same(a,b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True),b.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-9)
def load(name): return pd.read_csv(M.OUT/M.FILES[name],float_precision='round_trip')
def metadata():
    n=0
    for ln in (RC/'outputs/SHA256SUMS').read_text().splitlines():
        h,p=ln.split(); assert sha(RC/'outputs'/p)==h and sha(M.OUT/p)==h,p;n+=1
    js=read(M.OUT/'audit.json');assert js['source_sha256']==sha(PIN/'predictfun/research/pf_entry_audit.py')
    assert js['fit_sha256']==sha(N.OUT/'fit.json') and js['r4e_predictions_sha256']==sha(PIN/M.R4E_PRED)
    assert git('show','6157b1a0:predictfun/research/pf_entry_audit.py')==(PIN/'predictfun/research/pf_entry_audit.py').read_bytes()
    from pf_mutant_summary import judge,last_record
    from pf_mutant_one import sources
    import pf_entry_audit_mutants as H
    rr={k:last_record(RC/f'logs/mutants_audit/mut_audit_{k}.log') for k in ['control']+list(range(len(H.MUTANTS)))}
    assert judge(len(H.MUTANTS),rr,sources(H),[x[0] for x in H.MUTANTS])[0]
    bad=copy.deepcopy(rr);bad[1]=rr[0];assert not judge(len(H.MUTANTS),bad,sources(H),[x[0] for x in H.MUTANTS])[0]
    M.selftest()
    return dict(output_hashes=n,mutants=len(H.MUTANTS),copied_record_refused=True,synthetic_passed=True,audit_sha=js['source_sha256'])
def diagnostics():
    D=load('rows');stats=[]
    for (kind,day,side),g in D[D.status=='ok'].groupby(['set','day','side']):
        w=g.w.to_numpy();v=g.v_initial.to_numpy();d=g.d_following.to_numpy();y=g.Y.to_numpy()
        opposite=v*d<0;overlap=np.where(opposite,np.minimum(np.abs(v),np.abs(d)),0.)
        z=np.abs(v)<=1e-6; q=g.init_filled.to_numpy()>M.TOLP
        stats.append(dict(set=kind,day=int(day),side=int(side),rows=len(g),initial_fill_rows=int(q.sum()),meaningful_rows=int((np.abs(y)>1e-6).sum()),
            abs_initial=float(w@np.abs(v)),abs_following=float(w@np.abs(d)),abs_y=float(w@np.abs(y)),
            offset_mass=float(w@overlap),offset_share_initial=float(w@overlap/(w@np.abs(v))),
            abs_y_initial_zero=float(w[z]@np.abs(y[z])),abs_initial_positive=float(w@np.maximum(v,0)),abs_initial_negative=float(w@np.maximum(-v,0))))
    # Net absolute mass cannot identify whether the initial component was offset on its own rows.
    fixture=pd.DataFrame(dict(v=[100.,0.],d=[-100.,1000.],y=[0.,1000.],w=[1.,1.]))
    pred=M.predicates(dict(rows=2,share_meaningful=.5,init_no_fill=.5,abs_v_initial=100.,abs_Y=1000.))
    assert pred['cancellation'] is False
    overlap=float(np.where(fixture.v*fixture.d<0,np.minimum(fixture.v.abs(),fixture.d.abs()),0.).sum())
    assert overlap==100.
    return dict(stats=stats,cancellation_counterexample=dict(declared_predicate=pred,initial_absolute_mass=100.,initial_mass_offset=overlap,residual_label_mass=1000.),
                limits='Diagnostic offsets do not identify why orders did not fill; the continuation and END are fixed.')
def independent(res,side):
    g=res['k']*50.;m=res['mark'];aa=res['arms'];f={n:[x for x in a['fill_list'] if x[0]>g] for n,a in aa.items()}
    post=[d for d in aa['POST']['decision_list'] if d[0]==g and d[1]==side and d[2:4]==['absent','POST']]
    assert len(post)==1;gid=post[0][4];assert isinstance(gid,int) and 0<=gid<len(aa['POST']['sizes'])
    own=[x for x in f['POST'] if x[1]==gid];assert all(x[2]==side for x in own)
    value=lambda fs:math.fsum((1. if x[2]==0 else -1.)*x[3]*(m-x[4]) for x in fs)
    v=value(own);df=value([x for x in f['POST'] if x[2]==side and x[1]!=gid])-value([x for x in f['WAIT'] if x[2]==side])
    do=value([x for x in f['POST'] if x[2]!=side])-value([x for x in f['WAIT'] if x[2]!=side])
    assert abs(res['Y']-v-df-do)<=M.TOL
    for name,a in aa.items():
        assert LS._ledger_bad(a) is None
        assert abs(value(a['fill_list'])-a['value'])<=M.TOL and a['mark']==m
        assert all(0<=x[0]<=5000 for x in a['fill_list'])
    return dict(v_initial=v,d_focal=df,d_opposite=do,init_filled=math.fsum(x[3] for x in own),gid0=gid)
def data():
    saved=load('rows'); saved['day']=saved.day.astype(str); allrows=[]; counts=co.Counter(); lifetime=co.Counter(); exdiag=[]
    fz,smooth,rich,sup=N.frozen()
    for kind,ss in M.SETS.items():
        collected=[];plan=read(Path(ss['dir'])/'plan.json')
        for r in A.dataset_rows(ss['dir'],ss['sha']):
            if r['head']!='post': continue
            b=M.base(r,kind); counts[kind+' '+r['status']]+=1
            if r['status']=='ok':
                # Source ledgers are bound by accepted part hashes; recompute component allocation independently.
                ac=M.account(r['result'],r['side']);ind=independent(r['result'],r['side'])
                assert ac['status']=='ok'
                for k,v in ind.items(): assert abs(ac[k]-v)<1e-6,(k,ac[k],v)
                b.update(ac); b['_inputs']=r['fork']['inputs'];b['_price']=r['fork']['price']
                # Decision logs identify policy pull requests, not effective structural removals.
                g=r['result']['k']*50.;gid=ind['gid0'];dl=r['result']['arms']['POST']['decision_list']
                pulls=[d[0]-g for d in dl if d[1]==r['side'] and d[3]=='PULL' and d[4]==gid and d[0]>=g]
                exdiag.append(dict(set=kind,day=r['day'],side=r['side'],w=r['w'],q=ind['init_filled'],remaining=5000-g,pull_delay=min(pulls) if pulls else None))
            collected.append(b)
        M.cats_of([b for b in collected if '_inputs' in b]);M.predictions(collected,kind,smooth,rich,sup)
        allrows.extend({k:v for k,v in b.items() if not k.startswith('_')} for b in collected)
    D=pd.DataFrame(allrows)
    same(pd.read_csv(io.StringIO(D.to_csv(index=False)),float_precision='round_trip'),load('rows'))
    R=pd.DataFrame(M.reached_rows());same(pd.read_csv(io.StringIO(R.to_csv(index=False)),float_precision='round_trip'),load('reached_rows'))
    summary=M.summaries(D,R)
    for k in ('groups','strata','crosstab','statuses','reached_statuses'):
        same(pd.read_csv(io.StringIO(pd.DataFrame(summary[k]).to_csv(index=False)),float_precision='round_trip'),load(k))
    same(pd.read_csv(io.StringIO(pd.DataFrame(summary['reached']).to_csv(index=False)),float_precision='round_trip'),load('reached_summary'))
    assert summary['predicates']==read(M.OUT/'audit.json')['predicates']
    E=pd.DataFrame(exdiag); pulls=[]
    for kind,g in E.groupby('set'):
        w=g.w; has=g.pull_delay.notna();early=has&(g.pull_delay<=100)
        pulls.append(dict(set=kind,rows=len(g),policy_pull_recorded=int(has.sum()),weighted_pull_share=float(w[has].sum()/w.sum()),
            first_100ms_pull=int(early.sum()),weighted_first100_pull_share=float(w[early].sum()/w.sum()),
            no_fill_and_first100_pull=int(((g.q<=M.TOLP)&early).sum()),
            note='Policy request timing only. No inference about unrecorded structural lifetime or queue position.'))
    return dict(rows=len(D),reached=len(R),statuses=dict(counts),all_saved_columns_reproduced=True,all_summary_columns_reproduced=True,
                original_ledgers_and_component_allocation_checked=True,policy_pull_diagnostic=pulls)
def controls():
    sys.path.insert(0,str(RC));import audit_checks as V
    V.RES=PIN/'predictfun/research';V.HERE=OUT
    sys.path.insert(0,str(PIN/'predictfun/research'))
    V.main();V.falsify()
    assert read(OUT/'checks.json')['all_true'] and read(OUT/'checks_falsifier.json')['falsifier_ok']
    # An algebraic identity alone cannot validate allocation to components.
    import tempfile,shutil
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        tmp=Path(tmp);shutil.copy(M.OUT/'rows.csv.gz',tmp/'rows.csv.gz')
        assert V.check_B(tmp,{})
        df=load('rows');i=df.index[(df.status=='ok')&(df.cls=='meaningful')][0]
        df.loc[i,'v_initial']+=1.;df.loc[i,'d_focal']-=1.;df.loc[i,'d_following']-=1.
        df.to_csv(tmp/'rows.csv.gz',index=False)
        allocation_accepted=V.check_B(tmp,{})
        assert allocation_accepted
    return dict(checks=read(OUT/'checks.json'),falsifier=read(OUT/'checks_falsifier.json'),compensating_component_corruption_accepted=allocation_accepted,
                mitigation='data.json reproduces each saved component from hash-bound source fills and the forced-generation decision.')
if __name__=='__main__':
    mode=sys.argv[1];r=globals()[mode]();(OUT/(mode+'.json')).write_text(json.dumps(r,indent=2,sort_keys=True,default=str)+'\n');print(json.dumps(r,sort_keys=True,default=str))
