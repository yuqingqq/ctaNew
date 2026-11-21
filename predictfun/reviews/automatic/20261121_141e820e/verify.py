"""Pinned R4E review. Saved artifacts and synthetic controls only; never fit or replay real tape."""
import collections as co
import copy
import gzip
import hashlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-141e820e')
OUT = ROOT/'predictfun/reviews/automatic/20261121_141e820e'
RC = PIN/'predictfun/reviews/r4e_entry_smooth_20261121'
sys.path.insert(0, str(PIN/'predictfun/research'))
import numpy as np
import pandas as pd
import pf_adv2 as A
import pf_adv2_entry as N
import pf_adv2_entry_eval as E
import pf_adv2_entry_replay as R
import pf_adv2_entry_replay_summary as RS
S = A.LS
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda:f.read(4<<20), b''): h.update(b)
    return h.hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(PIN),*a])
def same(a,b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-10)
def metadata():
    n = 0
    for ln in (RC/'checkpoint/outputs.sha256').read_text().splitlines():
        h,p = ln.split(); assert sha(N.OUT/p)==h,p; n+=1
    A.check_inputs(); fz,heads,rich,sup = N.frozen()
    assert (RC/'fit/fit.json').read_bytes()==(N.OUT/'fit.json').read_bytes()
    assert git('show','eaea1161dc39e8048655ebcd775fbf6004b18014:predictfun/reviews/r4e_entry_smooth_20261121/fit/fit.json')==(N.OUT/'fit.json').read_bytes()
    for name in ('pf_adv2_entry.py','pf_adv2_entry_replay.py'):
        assert git('show',f'602fdfc5c3441e5c58b39dc4ffd01e4b541daf1b:predictfun/research/{name}')==(PIN/'predictfun/research'/name).read_bytes()
    for name in ('pf_adv2.py','pf_adv2_eval.py','pf_adv2_replay.py','pf_r2_labels.py','pf_r2_label_sample.py','pf_branch_engine.py','pf_r2_teacher.py','pf_r2_shadow.py'):
        assert git('show',f'371aab568277dea8ed15c0e386b8c07e7bacd340:predictfun/research/{name}')==(PIN/'predictfun/research'/name).read_bytes()
    old=git('show','602fdfc5:predictfun/research/pf_adv2_entry_eval.py').decode()
    new=(PIN/'predictfun/research/pf_adv2_entry_eval.py').read_text()
    # Remove only the declared round-trip reader amendment, then demand exact original bytes.
    begin=new.index('def read_r3(p):'); end=new.index('def r3_identity',begin)
    restored=(new[:begin]+new[end:]).replace('read_r3(N.REPO / R3_PRED)', 'pd.read_csv(N.REPO / R3_PRED)')
    assert restored==old, 'reader amendment changed more than the reader'
    ck='predictfun/reviews/r4e_entry_smooth_20261121/r4e_checks.py'
    assert git('show','5c2ec2ca:'+ck)==(PIN/ck).read_bytes()
    initial=git('show','602fdfc5:'+ck).decode()
    corrected=(PIN/ck).read_text().replace("NE.read_r3(out / 'eval' / 'predictions.csv.gz')", "pd.read_csv(out / 'eval' / 'predictions.csv.gz')").replace('NE.read_r3(N.REPO / NE.R3_PRED)', 'pd.read_csv(N.REPO / NE.R3_PRED)')
    assert corrected==initial
    train=N.r3_train(); assert len(train)==2
    import pf_mutant_one as MO
    import pf_mutant_summary as MS
    counts={}
    for tag,mod in [('entry','pf_adv2_entry'),('entry_eval','pf_adv2_entry_eval'),('entry_replay','pf_adv2_entry_replay'),('rsum','pf_adv2_entry_replay_summary')]:
        H=importlib.import_module(mod+'_mutants'); rr={}
        for k in ['control']+list(range(len(H.MUTANTS))):
            p=RC/f'logs/mutants_{tag}/mut_{tag}_{k}.log'
            records=[json.loads(x) for x in p.read_text().splitlines() if x.startswith('{') and '"cls"' in x]
            assert len(records)==1; rr[k]=records[0]
        assert MS.judge(len(H.MUTANTS),rr,MO.sources(H),[m[0] for m in H.MUTANTS])[0],tag
        bad=copy.deepcopy(rr);bad[1]=bad[0]
        assert not MS.judge(len(H.MUTANTS),bad,MO.sources(H),[m[0] for m in H.MUTANTS])[0]
        counts[tag]=len(H.MUTANTS)
    return dict(checkpoint_hashes=n, models=len(fz['models']), params=N.PARAMS, rounds=N.ROUNDS, mutation_records=counts, unchanged_engine_teacher_labels_and_R3=True, exact_reader_only_amendment=True, checks_A_I_unchanged=True)
def evaluation():
    ex,st=A.extract(A.dataset_rows(E.AE.EVAL_DIR,E.R3_DATASET),('20261026','20261027'))
    fz,smooth,rich,sup=N.frozen(); df=E.frame({h:ex[h] for h in N.HEADS},smooth,rich,sup,fz)
    saved=pd.read_csv(N.OUT/'eval/predictions.csv.gz',float_precision='round_trip',dtype={'day':str})
    same(df[list(saved.columns)],saved)
    idn=E.r3_identity(df,E.read_r3(PIN/E.R3_PRED))
    assert all(v for k,v in idn.items() if k not in ('rows','r3_rows'))
    result=E.evaluate(df)
    for name,rows in result.items():
        fresh=pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)))
        same(fresh,pd.read_csv(N.OUT/f'eval/{name}.csv'))
    stats=[]
    for (day,side),g in df.groupby(['day','side']):
        y,w=g.Y.to_numpy(),g.w.to_numpy()
        for c in E.CANDS:
            p,d=g[f'pred_{c}'].to_numpy(),g[f'd_{c}'].to_numpy()
            stats.append(dict(day=day,side=int(side),candidate=c,n=len(g),meaningful=int((np.abs(y)>1e-6).sum()),mse_over_zero=float(np.dot(w,(p-y)**2)/np.dot(w,y*y)),admissions=int(d.sum()),local_value=float(np.dot(w,d*y)/w.sum())))
    return dict(identity=idn,statuses=st,tables_recomputed=list(result),matched_draws=500,stats=stats)
def synthetic():
    os.environ['PF_TEACHER_SYN']=str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
    for mod in (E,R,RS): mod.selftest()
    return dict(passed=['eval','replay','replay_summary'],fits=0,real_tape_runs=0)
def replay():
    plan=read(E.AE.EVAL_DIR/'plan.json'); parts=R.AR.population(plan)
    fz,smooth,rich,sup=N.frozen(); worlds=co.Counter(); counters=co.Counter(); seen=set()
    for k,(day,members) in enumerate(parts):
        rec=read(R.OUT/f'parts/part_{k:03d}.json'); b=(R.OUT/f'parts/part_{k:03d}.jsonl.gz').read_bytes(); rows=S.gz_rows(b)
        assert R.part_record_ok(rec,members,b,sha(N.OUT/'fit.json'))
        assert rec['part']==k
        assert [(x['day'],x['m'],x['t_ms']) for x in rows]==[(day,x['m'],x['t_ms']) for x in members]
        r3members,r3worlds=R.r3_reference(k)
        assert rec['members']==r3members
        for r in rows:
            key=(day,r['m'],r['t_ms']); assert key not in seen; seen.add(key); worlds[r['status']]+=1
            ref=r3worlds[(int(r['m']),int(r['t_ms']))]; assert ref['status']==r['status']
            if r['status']!='ok': continue
            assert set(r['episodes'])==set(N.POLICIES)
            marks=[]
            for name,e in r['episodes'].items():
                tr=e['trace']; counters['episode_policies']+=1
                assert S._ledger_bad(tr) is None and S._arm_calls_bad(tr,-1.,0,0) is None
                assert tr['end']==5000 and tr['settings']==plan['settings']
                marks.append(tr['mark']); assert math.isfinite(tr['mark'])
                assert all(0<=f[0]<=5000 and f[2] in (0,1) and f[3]>0 and math.isfinite(f[4]) for f in tr['fill_list'])
                cash=-sum((1 if f[2]==0 else -1)*f[3]*f[4] for f in tr['fill_list'])
                pos=sum((1 if f[2]==0 else -1)*f[3] for f in tr['fill_list'])
                assert abs(cash-tr['cash'])<1e-6 and abs(pos-tr['position'])<1e-9
                assert abs(cash+pos*tr['mark']-e['value'])<1e-6 and e['value']==tr['value']
                assert abs(e['value']-e['value_bid']-e['value_ask'])<1e-6
                assert e['ledger_ok'] and e['cap_ok'] and len(tr['sizes'])==e['generations']
                assert len(e['churn'])==e['reviewed_generations'] and sum(c['action']=='PULL' for c in e['churn'])==e['first_review_pulls']
                ca=tr['causal']; t=tr['teacher']
                assert ca['mismatch']==ca['parity_mismatch']==t['binding_bad']==0
                assert ca['checked']==math.ceil(t['calls']/35)
                for field in ('checked','parity_checked'): counters['episode_'+field]+=ca[field]
                counters['episode_callbacks']+=t['calls']; counters['fills']+=len(tr['fill_list'])
                if name in R.CONTROLS:
                    assert R.r3_match(e,ref['episodes'][name]); counters['R3_control_matches']+=1
            assert len(set(marks))==1
            assert sorted(x['side'] for x in r['reached'])==[0,1]
            for x in r['reached']:
                assert x['status']=='ok',x['status']; counters['reached_ok']+=1
                assert not S._pair_bad(plan,day,r,x)
                assert x['inputs_equal_recorded'] and x['eligible']>0
                assert x['rank']==R.rank_of(x['key'])
                xx=np.asarray([x['fork']['inputs']],np.float32); side=x['side']
                assert float(N.predict(smooth[(side,'post')],xx)[0])==x['pred'][N.NAME]
                assert float(A.predict(rich['RICH'][(side,'post')],'RICH',xx)[0])==x['pred']['RICH']
                for ca in x['causal']:
                    counters['reached_checked']+=ca['checked'];counters['reached_parity_checked']+=ca['parity_checked']
                counters['reached_meaningful']+=abs(x['result']['Y'])>1e-6
    eps,reached,churn,exposure,st=R.frames(R.OUT,parts)
    for name,df in zip(RS.CSVS,(eps,reached,churn,exposure)):
        assert df.to_csv(index=False).encode()==(R.OUT/f'{name}.csv').read_bytes()
    js=RS.summary(eps,reached,st,RS.bindings(R.OUT,parts)); saved=read(R.OUT/'replay.json')
    for k,v in js.items(): assert v==saved[k],k
    return dict(worlds=dict(worlds),counters=dict(counters),diffs=js['diffs'],totals=js['totals'],csv_byte_identity=True,summary_exact=True)
def receipt_checks():
    sys.path.insert(0,str(RC))
    import r4e_checks as V
    V.RES=PIN/'predictfun/research'; V.REPO=PIN
    sys.path.insert(0,str(PIN/'predictfun/research'))
    evidence={}; flags={}
    for c in 'ABCDEFGHI':
        fn=getattr(V,'check_'+c)
        flags[c]=fn(evidence) if c in 'HI' else fn(N.OUT,evidence)
    assert all(flags.values()),flags
    import r4e_checks_falsify_v2 as F
    F.HERE=OUT
    F.main()
    # Isolate the label falsifier from CSV round-off: the unchanged round trip must pass.
    import shutil
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        dest=Path(tmp); shutil.copytree(N.OUT/'eval',dest/'eval')
        df=E.read_r3(dest/'eval/predictions.csv.gz'); df.to_csv(dest/'eval/predictions.csv.gz',index=False)
        assert V.check_C(dest,{})
        df.loc[0,'Y']+=1; df.to_csv(dest/'eval/predictions.csv.gz',index=False)
        assert not V.check_C(dest,{})
    # Characterize the episode check's scope. Current traces are checked more fully in replay().
    original=V.part_rows
    def changed(k,rout=None):
        rec,b,rows=original(k,rout)
        if k==0:
            r=next(r for r in rows if r['status']=='ok')
            r['episodes']['SMOOTH']['trace']['end']=4999.
            b=S.gz_bytes(rows); rec=dict(rec,sha256=hashlib.sha256(b).hexdigest())
        return rec,b,rows
    try:
        V.part_rows=changed
        accepts_changed_end=V.check_F(N.OUT,{})
    finally: V.part_rows=original
    return dict(flags=flags,details=evidence,real_artifact_falsifier=read(OUT/'checks_falsifier.json'),isolated_roundtrip_label_falsifier_passed=True,episode_check_accepts_changed_END=accepts_changed_end)
if __name__=='__main__':
    mode=sys.argv[1]; result=globals()[mode]()
    (OUT/(mode+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2,default=str)+'\n')
    print(json.dumps(result,sort_keys=True,default=str))
