"""Pinned R1 review: synthetic controls and saved artifacts only; no tape or fit."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-fefa5aca')
OUT = ROOT / 'predictfun/reviews/automatic/20261119_fefa5aca'
TARGET = 'fefa5aca63445e7b9d328b50b6f5f35d7702a2e0'
DATA = ROOT / 'data/predict_fun/research'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import pf_branch_engine as B


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''):
            h.update(x)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(name, obj):
    obj.update(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(),
               script_sha256=sha(__file__), module_sha256=sha(B.__file__))
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def synthetic():
    hidden = {}
    for side,jump,price in ((0,(55,57),50),(1,(45,47),52)):
        fr=[B.frame(0,0,50,52,0,0),B.frame(90,95,*jump,0,0)]
        pol=B.Policy(lambda v: (('POST' if v['t']==0 else 'WAIT') if v['side']==side
                  else ('POST' if v['other']=='cancel_pending' else 'WAIT'))
                  if v['state']=='absent' else 'KEEP', 'pending_sensitive')
        traces=[B.views_until(B.tape(fr,[(80,side,price,q)] if q else [],51),pol,4) for q in (0,10,30)]
        assert traces[0]==traces[1]==traces[2]
        hidden[str(side)]={'before_notice_equal':True,'request_ms':100}
        assert any(e[0]==100 and e[1]=='structural_request' for e in traces[0][-2][3])
    p=B.tape([B.frame(0,0,50,52,0,0)],[(80,0,50,10),(300,0,50,10),(400,1,52,10)],51.)
    state=B.fork_state(p,B.POST_KEEP,4,0)
    clean=B.pair(p,B.POST_KEEP,4,0,'keep',state=state)
    assert clean['status']=='ok'
    # Direct creation rejects a noncontract scenario; saved-state entry bypasses it.
    bad_settings={'cancel_effect_ms':50.}
    try:
        B.pair(p,B.POST_KEEP,4,0,'keep',settings=bad_settings)
    except ValueError:
        direct_refused=True
    else:
        direct_refused=False
    altered=B.pair(p,B.POST_KEEP,4,0,'keep',state=state,settings=bad_settings)
    assert direct_refused and altered['status']=='ok'
    assert altered['settings']['cancel_effect_ms']==50.
    req=[r for r in altered['arms']['PULL']['requests'] if r[0]==200 and r[1]=='policy_request']
    assert req and req[0][3]['effect']==237.
    # The fingerprint omits mark/end/scenario. Different END marks can look identical.
    wrong_mark=copy.deepcopy(state); wrong_mark.mark=61.
    assert B.fingerprint(state)==B.fingerprint(wrong_mark)
    marked=B.pair(p,B.POST_KEEP,4,0,'keep',state=wrong_mark)
    assert marked['status']=='ok'
    assert abs(marked['Y']-clean['Y'])>1.
    # A wrong-side fork can execute the other side twice at one decision tick.
    s=B.fork_state(p,B.POST_KEEP,4,1)
    wrong_side=B.pair(p,B.POST_KEEP,4,0,'keep',state=s)
    assert wrong_side['status']=='ok'
    tick=[d for d in wrong_side['arms']['KEEP']['decisions'] if d[0]==200]
    assert sum(d[1]==0 for d in tick)==2, tick
    # Existing wrong-tick guard is a positive control that still works.
    try:
        B.pair(p,B.POST_KEEP,5,0,'keep',state=state)
    except ValueError:
        wrong_tick_refused=True
    else:
        wrong_tick_refused=False
    assert wrong_tick_refused
    save('synthetic.json',dict(prior_hidden_fill_fix=hidden,
          saved_state_boundary=dict(direct_wrong_scenario_refused=direct_refused,
            saved_wrong_scenario_accepted=altered['status'],declared_cancel_ms=50.,
            executed_cancel_ms=req[0][3]['effect']-200.,wrong_mark_same_fingerprint=True,
            clean_value=clean['Y'],wrong_mark_value=marked['Y'],wrong_side_tick_commands=tick,
            wrong_tick_refused=wrong_tick_refused)))
    print('Prior hidden-fill fix confirmed; saved-state scenario/mark/side boundary counterexamples reproduced.')


def artifacts():
    folder=PIN/'predictfun/reviews/branch_engine_r1_v2_20261119'
    dat=DATA/'branch_engine_20261119'; day='20261024'
    checks={}
    for line in (folder/'sources.sha256').read_text().splitlines():
        expected,rel=line.split(None,1);rel=rel.strip();p=(ROOT if rel.startswith('data/') else PIN)/rel
        assert sha(p)==expected,rel;checks[rel]=expected
    j=read(dat/f'slice_v3_{day}.json')
    assert sha(dat/f'slice_v3_{day}.json')==sha(folder/f'slice_v3_{day}.json')
    for name,h in j['outputs'].items():
        assert sha(dat/name)==h,name
        if name.endswith('.json'):
            assert hashlib.sha256(gzip.decompress((folder/(name+'.gz')).read_bytes())).hexdigest()==h
        else:
            assert sha(folder/name)==h
    assert j['source_sha256']==sha(B.__file__)
    assert j['settings']==B.SETTINGS
    for name,h in j['provenance']['source'].items():
        assert sha(PIN/'predictfun/research'/name)==h,name
    kp=DATA/'keep_pull_20261117'/day; kr=read(kp/'receipt.json')
    assert sha(kp/'receipt.json')==j['inputs']['kp_receipt']
    for name in ('orders.parquet','fills.parquet'):
        assert sha(kp/name)==kr['outputs'][name]
    orders=pd.read_csv(dat/f'slice_v3_{day}_orders.csv')
    forks=pd.read_csv(dat/f'slice_v3_{day}_forks.csv')
    hist=read(dat/f'slice_v3_{day}_histories.json')
    archive=pd.read_parquet(kp/'orders.parquet')
    selection=pd.concat([archive[(archive.status=='eligible') & (archive.side==s)].sort_values('oid').head(128) for s in (0,1)])
    assert set(selection.oid)==set(orders.oid) and len(orders)==256
    assert orders.path_status.eq('eligible').all()
    for c in j['readings']['(1) single generation: false counts']:
        assert orders[c].astype(bool).all(),c
    assert orders.c_parity.all()
    original=archive.set_index('oid')
    for r in orders.itertuples():
        o=original.loc[r.oid]
        assert abs(o.v-r.keep_value_engine)<1e-7
        assert abs(r.keep_value_archive-r.keep_value_engine)<1e-7
        assert abs(r.pull_value_archive-r.pull_value_engine)<1e-7
    fills=pd.read_parquet(kp/'fills.parquet',columns=['oid','clock','t','q'])
    fills=fills[(fills.clock==50.) & fills.oid.isin(selection.oid)]
    for r in orders.itertuples():
        o=original.loc[r.oid]; f=fills[fills.oid==r.oid]; sg=1 if o.side==0 else -1
        for mode,g in [('keep',f),('pull',f[f.t<137])]:
            v=sg*float(g.q.sum())*(o.mark-o.p)
            assert abs(v-getattr(r,mode+'_value_engine'))<1e-7
    base=forks[forks.fork=='base run']; ok=forks[(forks.fork!='base run') & (forks.status=='ok')]
    assert len(base)==276 and len(ok)==1066 and len(hist)==1066
    assert base.p2.all() and base.causality.all()
    assert ok.prefork_identical.all() and ok.value_identity.all() and ok.p2_arms.all()
    assert not forks.status.str.startswith('refused').any()
    assert len(forks)==138*2*5
    coverage={}; counters={}; nonzero_effects=[]
    for (pol,kind,s),g in forks[forks.fork!='base run'].groupby(['policy','fork','side']):
        key=f'{pol} {kind} {"bid" if s==0 else "ask"}'
        coverage[key]=dict(statuses=g.status.value_counts().to_dict(),
            later_post=int(g.later_post.fillna(False).astype(bool).sum()),re_entry=int(g.re_entry.fillna(False).astype(bool).sum()))
    assert coverage==j['readings']['(4) coverage: fork statuses, later posts (POST / WAIT forks), re-entries (KEEP / PULL forks), per policy and side']
    for pol,g in base.groupby('policy'):
        counters[pol]={c:int(g[c].fillna(0).sum()) for c in base if c.startswith('count ')}
    for r in ok.itertuples():
        key=f'{r.m}|{r.t:.0f}|{r.policy}|{r.fork}|{int(r.side)}'; arms=hist[key]
        mark=float(selection[(selection.m==r.m)&(selection.t==r.t)].mark.iloc[0])
        names=list(arms)
        for name,a in arms.items():
            v=sum((1 if f[2]==0 else -1)*f[3]*(mark-f[4]) for f in a['fills'])
            assert abs(v-a['value'])<1e-7,(key,name)
        assert abs(arms[names[0]]['value']-arms[names[1]]['value']-r.Y)<1e-7,key
        g=0 if r.fork=='post' else 200
        pre=lambda a:[f for f in a['fills'] if f[0]<=g]
        assert pre(arms[names[0]])==pre(arms[names[1]]),key
        # Every PULL request obeys the declared delay, including no-op pulls.
        for a in arms.values():
            births={c[4]:c[0] for c in a['commands'] if c[3]=='POST'}
            for t,why,gid,payload in a['requests']:
                d=births[gid]; expected=max(t+37,d+95) if t<d+58 else t+37
                assert abs(payload['effect']-expected)<1e-9,(key,t,expected)
    save('artifacts.json',dict(saved_asof=j['asof'],orders=len(orders),tapes=138,base_runs=len(base),
          forks_ok=len(ok),forks_not_live=int(forks.status.eq('not_keep_state').sum()),
          coverage=coverage,counters=counters,all_saved_ledger_values_and_request_delays_reconciled=True,
          selection_exact=True,verified_hashes=checks))
    print('Saved slice selection, hashes, accounting, fork coverage, counter totals and request delays reconciled.')


if __name__=='__main__':
    {'synthetic':synthetic,'artifacts':artifacts,'selftest':B.selftest}[sys.argv[1]]()
