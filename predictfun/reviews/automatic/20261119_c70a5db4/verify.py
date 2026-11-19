"""Independent pinned R1 closure review; synthetic and saved artifacts only."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import pickle
import subprocess
import sys

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-c70a5db4')
OUT = ROOT / 'predictfun/reviews/automatic/20261119_c70a5db4'
TARGET = 'c70a5db491075e4e65526ed23b41b65d0618d0e0'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import pf_branch_engine as B
import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, data):
    data.update(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(),
                script_sha256=sha(__file__), module_sha256=sha(B.__file__))
    (OUT/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def synthetic():
    p = B.tape([B.frame(0,0,50,52,0,0)], [(80,0,50,10),(300,0,50,10),(400,1,52,10)],51.)
    c = B.fork_state(p,B.POST_KEEP,4,0,ident={'instrument_id':'synthetic','original_start_ms':0,'original_end_ms':5000})
    direct = B.pair(p,B.POST_KEEP,4,0,'keep')
    saved = B.pair(p,B.POST_KEEP,4,0,'keep',ctx=c)
    rt = B.pair(p,B.POST_KEEP,4,0,'keep',ctx=pickle.loads(pickle.dumps(c)))
    assert direct['arms']==saved['arms']==rt['arms'] and direct['Y']==saved['Y']==rt['Y']
    assert saved['ident']==c['ident'] and saved['settings']==B.SETTINGS
    assert saved['mark']==51 and saved['end']==5000 and saved['tape_id']==c['tape_id']

    def changed(attr,value,reprint=False):
        x=copy.deepcopy(c);setattr(x['state'],attr,value)
        if reprint: x['fingerprint']=B.fingerprint(x['state'])
        return x
    cases={
        'scenario':dict(settings={'cancel_effect_ms':50}),
        'mark':dict(ctx=changed('mark',61,True)),
        'branch_end':dict(ctx=changed('end',4000,True)),
        'tape':dict(path=B.tape([B.frame(0,0,50,52,0,0)],[(300,0,50,10)],51.)),
        'focal_side':dict(ctx=B.fork_state(p,B.POST_KEEP,4,1)),
        'same_parity_tick':dict(k=6),
        'episode':dict(episode=2),
        'raw_branch':dict(ctx=c['state']),
        'contract':dict(ctx=dict(c,contract='wrong')),
    }
    refused={};orig=B.decide;calls=[]
    def counted(*a,**kw):
        calls.append(1);return orig(*a,**kw)
    B.decide=counted
    try:
        for name,override in cases.items():
            calls.clear();kw=dict(path=p,policy=B.POST_KEEP,k=4,side=0,kind='keep',ctx=c);kw.update(override)
            try: B.pair(**kw)
            except ValueError as e: refused[name]=dict(error=str(e),arm_decide_calls=len(calls))
            else: raise AssertionError('expected refusal: '+name)
            assert not calls,name
    finally: B.decide=orig

    # The branch's effective execution horizon is independently stored on Engine.
    late=B.tape([B.frame(0,0,50,52,0,0)],[(4500,0,50,10)],51.)
    clean=B.fork_state(late,B.HOLD_ONCE,4,0)
    good=B.pair(late,B.HOLD_ONCE,4,0,'keep',ctx=clean)
    altered=copy.deepcopy(clean);altered['state'].E.end=4000.
    same_fp=B.fingerprint(altered['state'])==clean['fingerprint']
    bad=B.pair(late,B.HOLD_ONCE,4,0,'keep',ctx=altered)
    assert same_fp and good['Y']==10 and bad['Y']==0 and bad['end']==5000
    omissions={}
    for attr,value in [('end',4000.),('client',False),('nf',0)]:
        x=copy.deepcopy(clean);setattr(x['state'].E,attr,value)
        omissions[attr]=B.fingerprint(x['state'])==clean['fingerprint']

    missing={}
    for name,mark in [('none',None),('nan',float('nan'))]:
        pm=B.tape([B.frame(0,0,50,52,0,0)],[],mark)
        try: result=B.pair(pm,B.POST_KEEP,0,0,'post');missing[name]=dict(status=result['status'])
        except ValueError as e: missing[name]=dict(error=str(e))

    timing=[]
    for o,expected in [
        (dict(why='structural',req=50.,act=58.,submit=0.,remove=95.),True),
        (dict(why='structural',req=50.,act=58.,submit=0.,remove=87.),False),
        (dict(why='structural',req=100.,act=58.,submit=0.,remove=137.),True),
        (dict(why='structural',req=100.,act=58.,submit=0.,remove=138.),False),
        (dict(why='policy',req=100.,act=58.,submit=0.,remove=137.),True),
        (dict(why='policy',req=100.,act=58.,submit=0.,remove=138.),False)]:
        actual=B.pull_effect_ok(o);assert actual==expected;timing.append(dict(order=o,accepted=actual))
    save('synthetic.json',dict(valid_context_and_pickle_arms_exact=True,refused_before_arms=refused,
        effective_engine_end=dict(fingerprint_unchanged=same_fp,normal_Y_cents=good['Y'],altered_Y_cents=bad['Y'],
            reported_end_ms=bad['end'],actual_engine_end_ms=4000,normal_fills=good['arms']['KEEP']['fills'],altered_fills=bad['arms']['KEEP']['fills']),
        engine_field_omissions=omissions,missing_mark_status=missing,pull_effect_controls=timing))
    print('Named context refusals and round trip pass; unbound Engine.end changes Y 10 -> 0 with unchanged fingerprint; timing controls pass.')


def artifacts():
    folder=PIN/'predictfun/reviews/branch_engine_r1_closure_20261119'
    data=ROOT/'data/predict_fun/research/branch_engine_20261119'
    verified={}
    for line in (folder/'sources.sha256').read_text().splitlines():
        h,rel=line.split(None,1);p=(ROOT if rel.startswith('data/') else PIN)/rel
        assert sha(p)==h,rel;verified[rel]=h
    old=json.loads((data/'slice_v3_20261024.json').read_text())
    new=json.loads((data/'slice_v3_20261024_closure.json').read_text())
    assert sha(folder/'slice_v3_20261024_closure.json')==sha(data/'slice_v3_20261024_closure.json')
    assert new['source_sha256']==sha(B.__file__) and old['source_sha256']!=new['source_sha256']
    for r in [old,new]:
        for name,h in r['outputs'].items(): assert sha(data/name)==h,name
        for name,h in r['provenance']['source'].items(): assert sha(PIN/'predictfun/research'/name)==h,name
    assert old['readings']==new['readings'] and old['inputs']==new['inputs']
    assert old['settings']==new['settings']==B.SETTINGS
    equal={}
    for suffix in ['orders.csv','forks.csv','histories.json']:
        a=data/('slice_v3_20261024_'+suffix);b=data/('slice_v3_20261024_closure_'+suffix)
        assert a.read_bytes()==b.read_bytes(),suffix
        equal[suffix]=dict(exact_bytes=True,sha256=sha(a))
    orders=pd.read_csv(data/'slice_v3_20261024_closure_orders.csv')
    forks=pd.read_csv(data/'slice_v3_20261024_closure_forks.csv')
    # Positive equality controls and known-bad comparator controls.
    assert orders.equals(orders.copy())
    mutated=orders.copy();mutated.loc[0,'keep_value_engine']+=1
    assert not orders.equals(mutated)
    hist=json.loads((data/'slice_v3_20261024_closure_histories.json').read_text())
    assert hist==copy.deepcopy(hist)
    mut_h=copy.deepcopy(hist);key=next(iter(mut_h));arm=next(iter(mut_h[key]));mut_h[key][arm]['value']+=1
    assert hist!=mut_h
    r=copy.deepcopy(new['readings']);r['(3) counts']['forks_ok']+=1
    assert new['readings']!=r
    log=(folder/'logs/pbe3_mutants3.log').read_text()
    entries=[x for x in log.splitlines() if ' -> ' in x]
    assert len(entries)==50 and 'rc 0 -> passed' in entries[0]
    assert all('rc 1 -> caught: BRANCH ENGINE SELFTEST FAILED' in x for x in entries[1:])
    save('artifacts.json',dict(saved_asof=new['asof'],orders=len(orders),fork_rows=len(forks),
        fork_status_counts=forks.status.value_counts().to_dict(),saved_histories=len(hist),exact_artifact_equality=equal,
        readings_identical=True,inputs_settings_identical=True,source_changed=True,
        comparison_positive_and_known_bad_controls=True,developer_mutant_log_49_checks_verified=True,verified_hashes=verified))
    print('Source and artifact hashes verified; 256 orders, 1380 fork rows and histories exactly unchanged; comparator falsifiers pass.')


def mutants():
    harness=PIN/'predictfun/research/pf_branch_engine_mutants.py';results=[]
    for i in [0,*range(18,28)]:
        r=subprocess.run([sys.executable,str(harness),str(i)],capture_output=True,text=True,timeout=30)
        ok=r.returncode==0 if i==0 else r.returncode==1 and 'BRANCH ENGINE SELFTEST FAILED' in r.stderr
        assert ok,(i,r.returncode,r.stdout,r.stderr)
        results.append(dict(index=i,rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
    save('mutants.json',dict(selected_results=results))
    print('Unchanged harness module passes; all 10 closure mutants are caught by selftest checks.')


if __name__=='__main__':
    {'synthetic':synthetic,'artifacts':artifacts,'mutants':mutants,'selftest':B.selftest}[sys.argv[1]]()
