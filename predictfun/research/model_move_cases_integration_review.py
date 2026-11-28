"""Reconstruct MOVE-CASES while testing whether its totals are portfolio values.

Exact report/posthoc calculations over published consumed-day tables, plus
exact source counterexamples for repeated event attribution and unobserved
path endpoints. No raw replay, fitted model, policy choice, new interval,
protected-day input or live change.
"""
import ast
import bisect
import copy
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import pf_model_upgrade_spec as M
from . import pf_recovery_experiment_spec as R
from . import model_move_down_integration_review as Previous
from . import model_recovery_stage_review as Identity
from . import model_learn_pull_integration_review as Learn

PRODUCER = '81ae9c60'
PREFIX = 'move_cases_integration_review_v2'
STUDY = M.ROOT/'data/predict_fun/research/move_cases_20261113'
TABLES = ('events', 'segs', 'ours')
CASES = ('1 tick, holds','1 tick, reverts','2 ticks','big, trends','big, partial','big, reverts')
MSEG = ('into_old','into_runover','into_peak','into_after','away_chase','away_old','away_after')
TSEG = ('take_with','take_after','take_against')


def controls(tree):
    ns = dict(np=np, bisect=bisect, MOVE_MS=2000., AFTER_MS=3000., PRE=1000.,
              NAN=float('nan'), MUT_PATH_SIGN=0, MUT_RUNOVER_PEAK=0, MUT_CHASE_REV=0,
              day='20261023')
    Learn.compile_functions(tree, ['move_path','case_of','seg_into','seg_away','selftest','mutation_check'], ns)
    assert ns['selftest']() == []
    mutations = ns['mutation_check'](); assert len(mutations) == 3 and all(mutations.values())
    event_loops = [n for n in ast.walk(Previous.function(tree,'day_run')) if isinstance(n,ast.For)
                   and isinstance(n.target,ast.Tuple) and any(isinstance(t,ast.Name) and t.id=='eid' for t in n.target.elts)]
    if len(event_loops) != 1: raise ValueError('event accumulation loop differs')
    body = event_loops[0].body
    def assignment_index(name):
        hits = [i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        if len(hits) != 1: raise ValueError('accumulator identity differs: '+name)
        return hits[0]
    first,last = assignment_index('acc'),assignment_index('oa')
    assert first < last
    wrapper = ast.parse('def aggregate(pair,x,po,M,tb,t_pk,a0,kev,dr,cs):\n seg_rows=[]\n xa=1-x\n return seg_rows\n').body[0]
    wrapper.body[2:2] = copy.deepcopy(body[first:last])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),
                 '<exact-event-fill-accumulation>','exec'),ns)
    # Two distinct level transitions can share a later peak and a fill.
    # The same physical bid/ask fills enter each event's accumulation.
    pair = {1:{'L':[(1500.,54,5.,'top 5','field',-2.,-2.)]},
            0:{'L':[(1500.,50,5.,'top 5','field',3.,3.)]}}
    a = ns['aggregate'](pair,1,50,4,1000.,1400.,49,'adverse','UP','big, trends')
    b = ns['aggregate'](pair,1,51,3,1100.,1400.,49,'adverse','UP','big, trends')
    only_makers = lambda rows:[r for r in rows if r[5]=='maker']
    value = lambda rows:sum(r[8] for r in only_makers(rows))
    quantity = lambda rows:sum(r[7] for r in only_makers(rows))
    assert a == b and value(a)==5. and quantity(a)==10.
    assert value(a+b)==10. and quantity(a+b)==20.
    disjoint = ns['aggregate'](pair,1,51,3,10000.,10400.,49,'adverse','UP','big, trends')
    assert disjoint == [] and value(a+disjoint)==5.

    # Source endpoint selection accepts the last valid frame even when the
    # requested three-second endpoint is not covered by any later observation.
    short = ns['move_path']([1000.,1200.,1500.],np.array([51,54,54]),np.ones(3,bool),1,50,1000.)
    assert short==(4,1200.,4) and ns['case_of'](short[0],short[2])=='big, trends'
    complete = ns['move_path']([1000.,1200.,1500.,4200.],np.array([51,54,54,50]),np.ones(4,bool),1,50,1000.)
    assert complete==(4,1200.,0) and ns['case_of'](complete[0],complete[2])=='big, reverts'
    return dict(original_positive=True, original_mutations=mutations,
        repeated_event_accumulation=dict(physical_maker_fills=2,physical_shares=10.,physical_value_cents=5.,
             overlapping_event_maker_rows=len(only_makers(a+b)),reported_shares=quantity(a+b),
             reported_value_cents=value(a+b),distinct_events_have_identical_saved_rows=a==b),
        disjoint_event_negative_control=True,
        endpoint_without_future_coverage=dict(last_frame_ms=1500.,requested_endpoint_ms=4200.,
             output=short,assigned_case=ns['case_of'](short[0],short[2])),
        complete_endpoint_positive=dict(output=complete,assigned_case=ns['case_of'](complete[0],complete[2])),
        historical_repeated_fill_count=None,historical_uncovered_endpoint_count=None)


def readonly_function(tree, name, namespace):
    node = copy.deepcopy(Previous.function(tree,name))
    n = 4 if name=='report' else 2
    tail = node.body[-n:]
    calls = [ast.unparse(t.value.func) for t in tail if isinstance(t,ast.Expr) and isinstance(t.value,ast.Call)]
    expected = ['os.makedirs','json.dump'] if name=='report' else []
    if calls[:len(expected)]!=expected or len(calls)!=n or not calls[-2].endswith('.write') or calls[-1]!='print':
        raise ValueError('publication block differs: '+name)
    node.body = node.body[:-n]+ast.parse('return locals()').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 '<readonly-'+name+'>','exec'),namespace)
    return namespace[name]


def metadata(record):
    st=record['stat']
    if (record['days']!=1. or st['windows']!=288 or st['window_sides']!=576
            or any(st[k] for k in ('events_differ','ldg_differ','unpaired_windows'))
            or set(st['planted_caught'])!={'MUT_PATH_SIGN','MUT_RUNOVER_PEAK','MUT_CHASE_REV'}
            or not all(st['planted_caught'].values())):
        raise ValueError('producer metadata differs')


def run():
    R.check()
    own=Path(__file__).resolve();rel=str(own.relative_to(M.WORK))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.WORK,text=True).strip()
    if subprocess.check_output(['git','show',commit+':'+rel],cwd=M.WORK)!=own.read_bytes():
        raise ValueError('commit review before execution')
    producer=subprocess.check_output(['git','rev-parse',PRODUCER],cwd=M.ROOT,text=True).strip()
    sources={}
    for name in ('pf_move_cases.py','run_move_cases.sh','pf_kp_s4.py','pf_pull_depth.py',
                 'pf_robust_check.py','pf_qr_queue_race.py','pf_timing_split.py'):
        p=M.ROOT/'predictfun/research'/name
        if subprocess.check_output(['git','show',producer+':'+str(p.relative_to(M.ROOT))],cwd=M.ROOT)!=p.read_bytes():
            raise ValueError('producer source changed: '+name)
        sources[str(p)]=M.sha(p)
    tree=ast.parse((M.ROOT/'predictfun/research/pf_move_cases.py').read_text())
    paths=[STUDY/'move_cases.txt',STUDY/'move_cases.json',STUDY/'posthoc.txt',
           M.ROOT/'predictfun/handoffs/20261113_move_cases_results.md']
    for day in M.DAYS:paths += [STUDY/('day_'+day+'.json'),*(STUDY/(t+'_'+day+'.parquet') for t in TABLES)]
    inputs={str(p):M.sha(p) for p in paths}
    tested=controls(tree)
    original=(STUDY/'move_cases.txt').read_text();posthoc=(STUDY/'posthoc.txt').read_text()
    match=re.search(r'as of ([0-9TZ:-]+); controls passed',original)
    if not match:raise ValueError('published as-of header absent')
    first=json.loads((STUDY/('day_'+M.DAYS[0]+'.json')).read_text());metadata(first)
    bad=copy.deepcopy(first);bad['stat']['ldg_differ']=1
    Identity.refused(lambda:metadata(bad))
    Identity.refused(lambda:Identity.bind(STUDY/'move_cases.json','0'*64))
    R.write(PREFIX+'_declaration.json',dict(as_of=M.now(),source_commit=commit,
        source_sha={str(p.relative_to(M.WORK)):M.sha(p) for p in
                    (own,Path(Previous.__file__),Path(Identity.__file__),Path(Learn.__file__))},
        producer_commit=producer,producer_source_sha=sources,input_sha=inputs,
        scope=__doc__,controls=tested,days=M.DAYS,
        prior_failed_attempt=dict(source_commit='6c442ecb',
            log=str(R.OUT/'logs/move_cases_integration_review_v1.log'),
            log_sha=M.sha(R.OUT/'logs/move_cases_integration_review_v1.log'),
            stage='synthetic controls before declaration or historical table reads',
            reason='Audit wrapper omitted the producer xa=1-x binding; supplied unchanged binding in v2.',
            prior_declaration_written=False,prior_result_written=False),
        validation=False,live_changed=False))
    ns=dict(np=np,pd=pd,json=json,DAYS9=M.DAYS,OUTD=str(STUDY),MIN_N=30,NAN=float('nan'),
            CASES=CASES,MSEG=MSEG,TSEG=TSEG,
            time=SimpleNamespace(strftime=lambda *a:match[1],gmtime=lambda:None))
    result=readonly_function(tree,'report',ns)()
    post=readonly_function(tree,'posthoc',ns)()
    if '\n'.join(result['L'])+'\n'!=original or '\n'.join(post['L'])+'\n'!=posthoc:
        raise ValueError('published text differs from reconstruction')
    if M.native(result['R'])!=json.loads((STUDY/'move_cases.json').read_text()):
        raise ValueError('published readings differ')
    for record in result['stats'].values():metadata(record)
    frames=[result[n] for n in ('EV','SG','OU')]
    for frame in frames:
        if not frame.day.isin([int(d) for d in M.DAYS]).all():raise ValueError('undeclared table population')
        if set(frame.columns)&{'eid','fill_id','order_hash','signer'}:raise ValueError('saved identity schema changed')
    counts={day:{name:int(frame.day.eq(int(day)).sum()) for name,frame in zip(TABLES,frames)} for day in M.DAYS}
    summaries=[]
    mk=result['SG'];mk=mk[(mk.role=='maker')&(mk.kind=='adverse')&(mk.grp=='top 5')]
    for case in ('big, trends','big, reverts','ALL'):
        x=mk if case=='ALL' else mk[mk.case==case]
        into=x[x.seg.str.startswith('into')];away=x[x.seg.str.startswith('away')]
        summaries.append(dict(case=case,aggregate_rows=len(x),event_weighted_shares=float(x.q.sum()),
            into_mark_usd_per_day=float(into.v5q.sum()/100./9.),
            away_mark_usd_per_day=float(away.v5q.sum()/100./9.),
            total_mark_usd_per_day=float(x.v5q.sum()/100./9.),unique_portfolio_value=None))
    for path,h in {**sources,**inputs}.items():Identity.bind(path,h)
    receipt=dict(as_of=M.now(),status='EXACT_REPORT_RECONSTRUCTION_WITH_PORTFOLIO_LIMITS',
        producer_as_of=match[1],declaration_sha=M.sha(R.OUT/(PREFIX+'_declaration.json')),
        days=M.DAYS,counts=counts,report_and_posthoc_exact=True,producer_readings=result['R'],
        producer_daily_metadata=result['stats'],statuses=result['st_'],controls=tested,
        original_event_weighted_summaries=summaries,unique_fill_identity_available=False,
        historical_duplicate_amount=None,portfolio_value_verified=False,model_changed=False,
        raw_tape_replayed=False,validation=False,live_changed=False,
        limits=['A fill may enter multiple overlapping event windows; summing event marks is not a unique portfolio total.',
                'Saved rows omit event, fill, order and signer identities, preventing exact deduplication or maker-level hedge attribution.',
                'Realized move case and peak are future-conditioned descriptions, not decision-time signals.',
                'Endpoint selection falls back to the last valid frame without requiring observed endpoint coverage.',
                'Group-side marks do not establish that each maker offset its inventory or reduced portfolio risk.',
                'The simulated comparison independently replays six rungs and does not establish joint reserved-capacity or inventory constraints.',
                'Five-second fill marks omit the portfolio path, terminal inventory, later exits and live fill calibration.'])
    R.write(PREFIX+'.json',receipt)
    print(json.dumps(dict(as_of=receipt['as_of'],producer_as_of=match[1],
        receipt_sha=M.sha(R.OUT/(PREFIX+'.json')),counts=counts,summaries=summaries,
        report_and_posthoc_exact=True,statuses=result['st_'])))


if __name__=='__main__':run()
