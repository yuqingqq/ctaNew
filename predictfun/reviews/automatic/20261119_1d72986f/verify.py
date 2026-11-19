"""Bounded independent review: pinned v2 screen/timing, saved receipts, R1 design.

No fit, raw tape, label build or untouched date. WIP design evidence is a separate
hash-pinned snapshot, never treated as part of the reviewed commit.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-1d72986f')
OUT = ROOT / 'predictfun/reviews/automatic/20261119_1d72986f'
TARGET = '1d72986ffa9cee8285bcef838fe2d054b16b2486'
DATA = ROOT / 'data/predict_fun/research'
sys.path.insert(0, str(PIN / 'predictfun/research'))
import pf_pipeline_paired_review as RV


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(name, obj):
    obj.update(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(),
               script_sha256=sha(__file__))
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def synthetic():
    ids = np.r_[np.repeat(np.arange(100), 2), np.arange(100, 200)]
    lab = np.r_[np.ones(200), np.full(100, -2.)]
    cases = {}
    cases['unequal_rows'] = RV.screen(ids, np.full(300, -.5), lab, False)[0]
    cases['one_row_per_episode'] = RV.screen(np.arange(200), np.full(200, -.5),
        np.r_[np.ones(100), np.full(100, -2.)], False)[0]
    cases['wrong_sign'] = RV.screen(ids, np.full(300, -.5), lab, True)[0]
    cases['wrong_amount'] = RV.screen(ids, np.full(300, -5.), lab, False)[0]
    cases['all_zero'] = RV.screen(ids, np.zeros(300), np.zeros(300), False)[0]
    # Weighted mean zero while raw row sum is nonzero: the reverse counterexample.
    cases['weighted_zero_raw_nonzero'] = RV.screen(ids, np.zeros(300),
        np.r_[np.ones(200), np.full(100, -1.)], False)[0]
    cases['insufficient'] = RV.screen(np.arange(199), -np.ones(199), -np.ones(199), False)[0]
    for which in ('pred', 'lab'):
        for nonfinite in (float('nan'), float('inf'), -float('inf')):
            p, y = np.full(300, -.5), lab.copy()
            (p if which == 'pred' else y)[0] = nonfinite
            got = RV.screen(ids, p, y, False)[0]
            assert got == 'NOT_EVALUABLE', (which, nonfinite, got)
    assert cases == dict(unequal_rows='PASS', one_row_per_episode='PASS', wrong_sign='FAIL',
                        wrong_amount='FAIL', all_zero='NOT_EVALUABLE',
                        weighted_zero_raw_nonzero='NOT_EVALUABLE', insufficient='INSUFFICIENT_SUPPORT')
    timing = []
    labs = pd.DataFrame(dict(kind=['retain'], j=[2], tstar=[np.nan]))
    for origin in (0., 1_790_000_000_000.):
        ex = pd.DataFrame(dict(kind=['retain'], j=[2], t=[origin], decision_ms=[origin+100.],
                               effective_ms=[origin+137.], endpoint_ms=[origin+5000.]))
        assert RV.v3_timing(ex, labs, (np.array([80.]), np.array([80.]))) == (0,0,0)
        for field in ('decision_ms','effective_ms','endpoint_ms'):
            for delta, expected in ((.0005, 0), (.002, 1), (.01, 1), (-.01, 1)):
                mutant = ex.copy(); mutant[field] += delta
                got = RV.v3_timing(mutant, labs, (np.array([80.]), np.array([80.])))[0]
                assert got == expected, (origin,field,delta,got)
                timing.append(dict(origin=origin,field=field,delta_ms=delta,refused=bool(got)))
    save('synthetic.json', dict(screen=cases, nonfinite_refusals=6, timing=timing,
                               module_sha256=sha(RV.__file__)))
    print('Pinned v2 fixes: weighted-zero/sign/amount/support and nonfinite cases pass; absolute timing checks pass.')


def artifacts():
    folder = PIN / 'predictfun/reviews/pipeline_paired_review_v2_20261119'
    verified = {}
    for line in (folder/'sources.sha256').read_text().splitlines():
        expected, rel = line.split(None, 1); rel=rel.strip()
        p = (ROOT if rel.startswith('data/') else PIN) / rel
        assert sha(p) == expected, rel
        verified[rel] = expected
    skip = {'asof','source_sha256','outputs','seconds','provenance'}
    summary = {}
    for day in ('20261026','20261027'):
        olddir=DATA/'pipeline_paired_review_20261119'; newdir=DATA/'pipeline_paired_review_v2_20261119'
        a,b=read(olddir/f'review_{day}.json'),read(newdir/f'review_{day}.json')
        assert {k:v for k,v in a.items() if k not in skip} == {k:v for k,v in b.items() if k not in skip}
        assert b['source_sha256'] == sha(RV.__file__) != a['source_sha256']
        assert sha(newdir/f'review_{day}.json') == sha(folder/f'review_{day}.json')
        for name,h in b['outputs'].items():
            assert sha(newdir/name) == h, name
            if (folder/name).exists():
                assert sha(folder/name)==h, name
        # Unchanged ledgers and null draws confirm no numerical replay change.
        for name in (f'ledger_{day}.parquet', f'review_null_draws_{day}.csv'):
            assert sha(olddir/name)==sha(newdir/name), name
        for key,p in [('paired',DATA/'pipeline_paired_v2_20261119/paired.json'),
                      ('evaluation',DATA/f'pipeline_paired_eval_v2_20261119/eval_{day}.json'),
                      ('export_receipt',DATA/f'pipeline_export_v2_20261119/{day}/receipt.json')]:
            assert sha(p)==b['inputs'][key], (day,key)
        for name,h in b['inputs']['parts'].items():
            assert sha(DATA/'pipeline_paired_v2_20261119'/name)==h, name
        summary[day]=dict(asof=b['asof'], populations=b['readings']['V4 POPULATIONS'],
                          verdicts_and_numbers_unchanged=True, ledger_identical=True,
                          null_draws_identical=True, saved_prediction_groups=sum(len(v) for v in b['prediction'].values()))
    save('artifacts.json', dict(days=summary, verified_manifest=verified,
                               scope='saved consumed Oct26/27, no raw tape or experiment'))
    print('Pinned source, receipt/input/output hashes, committed copies and all saved numerical readings verified.')


def wip_design():
    meta=read(OUT/'wip_snapshot.json')['sources']['pf_branch_engine.py']
    path=OUT/'wip_engine_snapshot.py.txt'
    assert sha(path)==meta['sha256']
    # Load captured bytes; resolve all dependencies against the pinned source.
    # __file__ is pinned-dir synthetic name so the WIP import path cannot pull dirty dependencies.
    import types
    B=types.ModuleType('review_wip_engine')
    B.__file__=str(PIN/'predictfun/research/pf_branch_engine_review_snapshot.py')
    exec(compile(path.read_text(),str(path),'exec'), B.__dict__)
    frames=[B.frame(0,0,50,52,0,0),B.frame(90,95,55,57,0,0)]
    # At 100ms, an 80ms match has not yet produced its 180ms private notice.
    # Both paths expose the identical received book and zero acknowledged fills.
    pol=B.side_policy(0,'keep')
    runs={}
    for name,trades in [('unnoticed_full',[(80,0,50,30)]),('unfilled',[]),('unnoticed_partial',[(80,0,50,10)])]:
        p=B.tape(frames,trades,51.)
        branch=B.run(p,pol,until=3)
        runs[name]=dict(view=branch.view(100.,0),decisions=branch.decisions,
                        requests=[e for e in branch.E.events if 'request' in e[1]],
                        engine_fills=branch.E.fills)
    assert runs['unnoticed_full']['view']['known_filled']==runs['unfilled']['view']['known_filled']==0
    assert runs['unnoticed_full']['view']['position']==runs['unfilled']['view']['position']==0
    assert runs['unnoticed_full']['view']['state']=='live'
    assert runs['unfilled']['view']['state']=='cancel_pending'
    assert runs['unnoticed_partial']['view']['state']=='cancel_pending'
    # Control: no structural danger yields the same client view despite the hidden full fill.
    safe=[B.frame(0,0,50,52,0,0)]
    bs=[B.run(B.tape(safe,t,51.),pol,until=3) for t in ([(80,0,50,30)],[])]
    assert bs[0].view(100.,0)==bs[1].view(100.,0)
    # Once notice arrives, a position / state difference is legitimate.
    bs2=[B.run(B.tape(safe,t,51.),pol,until=5) for t in ([(80,0,50,30)],[])]
    assert bs2[0].view(200.,0)['position']==30 and bs2[1].view(200.,0)['position']==0
    save('wip_design.json',dict(wip_source_sha256=meta['sha256'],
         scope='synthetic WIP design evidence, not a finding in target commit',
         unnoticed_full_fill_leaks_through_structural_guard=runs,
         no_structural_danger_views_equal=True, after_notice_difference_legitimate=True))
    print('R1 WIP design: hidden full fill changes client-visible structural-cancel state before its notice; controls pass.')


if __name__=='__main__':
    {'synthetic':synthetic,'artifacts':artifacts,'wip_design':wip_design,
     'selftest':RV.selftest}[sys.argv[1]]()
