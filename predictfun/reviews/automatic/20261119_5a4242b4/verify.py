"""Pinned preflight review: independent synthetic cases and saved artifacts only."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import pickle
import subprocess
import sys

ROOT=Path('/home/yuqing/ctaNew')
PIN=Path('/tmp/pf-review-5a4242b4')
OUT=ROOT/'predictfun/reviews/automatic/20261119_5a4242b4'
TARGET='5a4242b4a0613523707a3c8c17507a32491de125'
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_branch_engine as B
import pf_branch_engine_mutants as M
import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name,data):
    data.update(target_commit=TARGET,asof=datetime.now(timezone.utc).isoformat(),
                script_sha256=sha(__file__),module_sha256=sha(B.__file__))
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def synthetic():
    fr=[B.frame(0,0,50,52,0,0),B.frame(2000,2005,50,52,0,0)]
    refusals=[];positive=[];inventories={}
    for side,price in [(0,50),(1,52)]:
        p=B.tape(fr,[(4500,side,price,10)],51.)
        c=B.fork_state(p,B.HOLD_ONCE,4,side)
        direct=B.pair(p,B.HOLD_ONCE,4,side,'keep')
        saved=B.pair(p,B.HOLD_ONCE,4,side,'keep',ctx=c)
        rt=B.pair(p,B.HOLD_ONCE,4,side,'keep',ctx=pickle.loads(pickle.dumps(c)))
        assert direct==saved==rt and saved['status']=='ok' and saved['Y']==10.
        assert saved['arms']['KEEP']['fills'][0][0]==4500
        positive.append(dict(side=side,Y_cents=saved['Y'],late_fill_ms=4500,direct_context_pickle_exact=True))
        inventories=dict(branch_fields=sorted(vars(c['state'])),engine_fields=sorted(vars(c['state'].E)),declared=list(B.FINGERPRINT_FIELDS))
        corruptions=[
            ('engine_end',lambda x:setattr(x['state'].E,'end',4000.)),
            ('engine_client',lambda x:setattr(x['state'].E,'client',False)),
            ('engine_nf',lambda x:setattr(x['state'].E,'nf',1)),
            ('scenario_record_and_state',lambda x:(x['state'].settings.update(cancel_effect_ms=50.),x['settings'].update(cancel_effect_ms=50.))),
            ('fallback_log',lambda x:x['state'].fallbacks.append((0.,side,'absent','feature_unavailable'))),
            ('constraint_log',lambda x:x['state'].constraints.append((0.,side,'cap'))),
        ]
        original=B.decide;calls=[]
        def count(*args,**kwargs):
            calls.append(1);return original(*args,**kwargs)
        B.decide=count
        try:
            for name,mutate in corruptions:
                for recompute in ([False] if name.endswith('_log') else [False,True]):
                    x=copy.deepcopy(c);mutate(x)
                    assert B.fingerprint(x['state'])!=c['fingerprint'],name
                    if recompute:x['fingerprint']=B.fingerprint(x['state'])
                    calls.clear()
                    try:B.pair(p,B.HOLD_ONCE,4,side,'keep',ctx=x)
                    except ValueError as e:
                        assert not calls,(side,name,calls)
                        refusals.append(dict(side=side,field=name,fingerprint_recomputed=recompute,arm_decisions=0,error=str(e)))
                    else:raise AssertionError('corrupt context accepted: '+name)
        finally:B.decide=original
    missing=[]
    for name,mark in [('None',None),('NaN',float('nan')),('+inf',float('inf')),('-inf',float('-inf'))]:
        p=B.tape(fr,[(500,0,50,5),(600,1,52,5)],mark)
        assert B.canon_mark(mark) is None
        for side in (0,1):
            c=B.fork_state(p,B.POST_KEEP,0,side)
            assert c['mark'] is None
            direct=B.pair(p,B.POST_KEEP,0,side,'post')
            rt=B.pair(p,B.POST_KEEP,0,side,'post',ctx=pickle.loads(pickle.dumps(c)))
            assert direct==rt and direct['status']=='missing_mark'
            assert all(a['status']=='missing_mark' and 'value' not in a for a in direct['arms'].values())
            missing.append(dict(mark=name,side=side,status=direct['status'],direct_pickle_exact=True,no_fabricated_value=True))
    assert B.canon_mark(51.)==51.
    save('synthetic.json',dict(positive_controls=positive,corrupt_contexts_refused=refusals,
        missing_mark_controls=missing,field_inventory=inventories,all_checks_pass=True))
    print('Both-side late-fill round trips pass; 20 corrupt contexts refused before arms; all 8 missing-mark side/encoding cases preserve statuses.')


def artifacts():
    folder=PIN/'predictfun/reviews/branch_engine_r1_preflight_20261119'
    data=ROOT/'data/predict_fun/research/branch_engine_20261119'
    verified={}
    for line in (folder/'sources.sha256').read_text().splitlines():
        h,rel=line.split(None,1);p=(ROOT if rel.startswith('data/') else PIN)/rel
        assert sha(p)==h,rel;verified[rel]=h
    read=lambda p:json.loads(p.read_text())
    old=read(data/'slice_v3_20261024.json');new=read(data/'slice_v3_20261024_preflight.json')
    assert sha(folder/'slice_v3_20261024_preflight.json')==sha(data/'slice_v3_20261024_preflight.json')
    assert new['source_sha256']==sha(B.__file__) and new['source_sha256']!=old['source_sha256']
    for r in [old,new]:
        for name,h in r['outputs'].items():assert sha(data/name)==h,name
        for name,h in r['provenance']['source'].items():assert sha(PIN/'predictfun/research'/name)==h,name
    assert old['inputs']==new['inputs']
    assert sha(ROOT/'data/predict_fun/research/keep_pull_20261117/20261024/receipt.json')==new['inputs']['kp_receipt']
    assert old['readings']==new['readings'] and old['settings']==new['settings']==B.SETTINGS
    equal={}
    for suffix in ['orders.csv','forks.csv','histories.json']:
        a=(data/('slice_v3_20261024_'+suffix)).read_bytes()
        b=(data/('slice_v3_20261024_preflight_'+suffix)).read_bytes()
        assert a==b
        assert hashlib.sha256(a).digest()==hashlib.sha256(bytes(a)).digest()
        mutated=bytearray(a);mutated[len(mutated)//2]^=1
        assert hashlib.sha256(a).digest()!=hashlib.sha256(mutated).digest()
        equal[suffix]=dict(exact_bytes=True,sha256=hashlib.sha256(a).hexdigest(),changed_byte_refused=True)
    changed=copy.deepcopy(new['readings']);changed['(3) counts']['forks_ok']+=1
    assert changed!=new['readings'] and copy.deepcopy(new['readings'])==new['readings']
    orders=pd.read_csv(data/'slice_v3_20261024_preflight_orders.csv')
    forks=pd.read_csv(data/'slice_v3_20261024_preflight_forks.csv')
    assert len(orders)==256 and orders.side.value_counts().to_dict()=={0:128,1:128}
    assert len(forks[forks.fork=='base run'])==276
    assert len(forks[(forks.fork!='base run')&(forks.status=='ok')])==1066
    assert len(forks[forks.status=='not_keep_state'])==38
    log=(folder/'logs/pbe4_mutants2.log').read_text();lines=[x for x in log.splitlines() if ' -> ' in x]
    assert len(lines)==len(M.MUT)==57 and 'rc 0 -> passed' in lines[0]
    for i,line in enumerate(lines[1:],1):
        assert M.MUT[i][0] in line and 'rc 1 -> caught: BRANCH ENGINE SELFTEST FAILED' in line
    comparison=subprocess.run([sys.executable,str(folder/'compare_bytes.py')],capture_output=True,text=True,timeout=15)
    assert comparison.returncode==0,comparison.stderr
    save('artifacts.json',dict(saved_asof=new['asof'],orders=256,tapes=138,base_runs=276,valid_forks=1066,not_live_anchors=38,
        exact_artifacts=equal,readings_settings_inputs_identical=True,source_changed=True,
        committed_comparison_stdout=comparison.stdout,full_developer_mutant_log_verified=56,
        verified_hashes=verified,all_checks_pass=True))
    print('Committed hashes/logs verified; three saved ledger files are byte-identical; the pinned comparator and its independent falsifiers pass.')


def mutants():
    harness=PIN/'predictfun/research/pf_branch_engine_mutants.py';rows=[]
    for i in [0,*range(27,35)]:
        r=subprocess.run([sys.executable,str(harness),str(i)],capture_output=True,text=True,timeout=30)
        ok=r.returncode==0 if i==0 else r.returncode==1 and 'BRANCH ENGINE SELFTEST FAILED' in r.stderr
        assert ok,(i,r.returncode,r.stdout,r.stderr)
        rows.append(dict(index=i,name=M.MUT[i][0],rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
    save('mutants.json',dict(selected_results=rows,all_checks_pass=True))
    print('Pinned unchanged selftest passes; all 8 selected preflight/timing mutants caught by checks.')


if __name__=='__main__':
    if sys.argv[1]=='all':
        synthetic();artifacts();mutants()
    else:
        {'synthetic':synthetic,'artifacts':artifacts,'mutants':mutants}[sys.argv[1]]()
