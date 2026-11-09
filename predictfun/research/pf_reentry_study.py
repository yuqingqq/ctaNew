"""Paired pull/re-entry workload on the consumed Nov8 intent artifact only."""
import collections
import datetime as dt
import json
import math
import subprocess
import sys
import pandas as pd
from .pf_reentry import ARMS,ReentryGate
from .pf_quote_capture import QuotePlanner
from .pf_quote_coverage import coverage
from . import pf_quote_incremental_study as previous

a=previous.s.a
OLD=previous.DEST
DEST=a.ROOT/'data/predict_fun/research/reentry_churn_20261109'
REVIEW=a.HERE.parent/'reviews/reentry_churn_20261109'
SOURCES=['pf_reentry.py','pf_reentry_study.py','tests/test_reentry.py','tests/test_reentry_study.py',
    'pf_quote_capture.py','pf_quote_coverage.py']
AUDIT=['share_ms','age_share_ms2','quoted_lane_ms','base_lane_ms','oldest_lane_age_ms2',
    'reserved_share_ms','desired_lane_ms','places','cancels','base_resize_cancels',
    'kept_base_size_switches','refused_capacity','refused_inflight','refused_minimum',
    'peak_reserved_side','decisions']
SUMS=['places','cancels','desired_lane_ms','reference_desired_lane_ms','aligned_lane_ms',
    'elsewhere_lane_ms','quoted_lane_ms','share_ms','reserved_share_ms','oldest_lane_age_ms2',
    'refused_capacity','refused_inflight','actions','admitted_actions','withheld_actions',
    'delayed_actions','rapid_actions','withheld_rapid_actions','reference_unquoted_lane_ms',
    'reference_quoted_lost_ms','reference_unquoted_gained_ms']


def declare():
    if DEST.exists():raise ValueError('existing reentry study')
    (DEST/'logs').mkdir(parents=True)
    old=json.loads((OLD/'receipt.json').read_text())
    scope=json.loads((a.RP/'scope.json').read_text())
    if scope['days_consumed']!=['20261108'] or len(scope['windows'])!=220:raise ValueError('scope differs')
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        population='220 previously consumed Nov8 windows; first60 seconds; all279059 recorded decisions, four lanes; no fill selection',
        estimand='deterministic no-fill order workload conditional on the same factual desired path; no PnL, predictive skill, queue-rank or inventory-variance claim',
        arms=ARMS,primary='cancel_ack',rapid_pull_gap_ms=100,
        policy='first/new-price entries immediate except all_entry50 control; same-price reopen within100ms waits for outstanding same-lane/tick cancel acknowledgement, or continuous eligibility25/50/100ms. Every original pull/reprice cancels immediately; size changes do not reset the gate.',
        cancellation_clock='ack known by now only; no foresight into remaining latency; reference model40/22/95ms and1s reservation grace as before',
        order_model=old['declaration']['order_model'],sizing='incremental for every arm; no size-policy comparison',
        reference='prior incremental policy reproduced exactly on16 metrics per window; raw desired price/time is the denominator for all arms',
        measures='requests, aligned quote coverage against original target, coverage lost/gained relative to reference, off-target quote time, age, reservations; episodes retained as admitted/withheld with censoring',
        attribution='exact pull reasons absent; no reconstructed score used by the policy; no claim all pulls are prediction-driven',
        null='deterministic mechanical comparison, no statistical hypothesis or economic inference; no permutation test or confidence interval',
        multiplicity=dict(current_arms=6,new_variants=5,previous_capture_arms=7,cumulative_distinct_policies=12),
        comparisons='fewer requests with unchanged-or-better aligned time is computed, never an automatic trading/deployment decision',
        controls='zero-delay parity, immediate pulls, same-price eligibility, censoring, cancellation acknowledgement versus reservation grace, malformed inputs, causal prefixes, exact artifact identities and corrupted-count refusal',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCES},
        inputs={str(p):a.sha(p) for p in [OLD/'decisions.parquet',OLD/'receipt.json',OLD/'window_metrics.parquet',OLD/'coverage.parquet',a.RP/'scope.json']},
        no_live_change=True,no_fill_clock_bypass=True,no_protected_dates_read=True)
    a.write(DEST/'declaration.json',d);a.write(REVIEW/'declaration.json',d)
    print('DECLARED six reentry policies, fixed sizing, same consumed artifact')


def check_inputs():
    d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('source changed')
    for p,h in d['inputs'].items():
        if a.sha(p)!=h:raise ValueError('input changed')
    return d


def aligned_intervals(p,start,end,raw):
    result=[]
    for lane,(tick,size) in enumerate(raw):
        parts=[]
        if tick is not None and size>0:
            for o in p.orders:
                if o.lane!=lane or o.tick!=tick or o.rem<=1e-9:continue
                lo=max(start,o.accepted);hi=min(end,o.cancel_at if o.cancel_at is not None else end)
                if hi>lo:parts.append((lo,hi))
        merged=[]
        for lo,hi in sorted(parts):
            if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
            else:merged.append((lo,hi))
        result.append(merged)
    return result


def difference_time(left,right):
    total=0.
    for aa,bb in zip(left,right):
        for lo,hi in aa:
            overlap=sum(max(0.,min(hi,y)-max(lo,x)) for x,y in bb)
            total+=hi-lo-overlap
    return total


def run_window(g,start,old,oldcov):
    end=start+60000;last=start;raw=[(None,0.)]*4
    states={arm:dict(planner=QuotePlanner(incremental=True),gate=ReentryGate(*cfg),want=raw,
        total=collections.Counter(),peak=0.,subset=True) for arm,cfg in ARMS.items()}
    def integrate(now):
        refparts=aligned_intervals(states['reference']['planner'],last,now,raw)
        for arm,s in states.items():
            p=s['planner'];v=s['total'];v.update(p.integrate(last,now))
            v['desired_lane_ms']+=(now-last)*sum(t is not None and z>0 for t,z in s['want'])
            v['reference_desired_lane_ms']+=(now-last)*sum(t is not None and z>0 for t,z in raw)
            on,off=coverage(p,last,now,raw);v['aligned_lane_ms']+=on;v['elsewhere_lane_ms']+=off
            parts=aligned_intervals(p,last,now,raw)
            v['reference_quoted_lost_ms']+=difference_time(refparts,parts)
            v['reference_unquoted_gained_ms']+=difference_time(parts,refparts)
    for row in g[['t']+[v for lane in range(4) for v in (f'tick{lane}',f'size{lane}')]].to_numpy():
        now=row[0];integrate(now);last=now
        raw=[(None if row[1+2*i]<0 else int(row[1+2*i]),float(row[2+2*i])) for i in range(4)]
        for arm,s in states.items():
            p=s['planner'];want=s['gate'].apply(now,raw,p.orders)
            s['subset'] &= all(w==(None,0.) or w==r for w,r in zip(want,raw))
            if arm=='reference' and want!=raw:raise ValueError('reference target differs')
            p.plan(now,want);s['want']=want
            s['peak']=max(s['peak'],p.reserved(0),p.reserved(1))
    integrate(end);metrics=[];actions=[]
    for arm,s in states.items():
        p=s['planner'];v=s['total'];aa=s['gate'].finish(end)
        if not s['subset'] or s['peak']>150+1e-8:raise ValueError('target/budget invariant fails')
        if any(c['kind']=='place' and (c['tick'] if c['lane']%2==0 else 100-c['tick'])*c['size']/100<1-1e-9 for c in p.commands):raise ValueError('minimum notional fails')
        v.update(actions=len(aa),admitted_actions=sum(x['admitted_at'] is not None for x in aa),
            withheld_actions=sum(x['admitted_at'] is None for x in aa),
            delayed_actions=sum(x['admitted_at'] is not None and x['admitted_at']>x['start'] for x in aa),
            rapid_actions=sum(x['rapid'] for x in aa),withheld_rapid_actions=sum(x['rapid'] and x['admitted_at'] is None for x in aa))
        v['reference_unquoted_lane_ms']=v['reference_desired_lane_ms']-v['aligned_lane_ms']
        record=dict(arm=arm,decisions=len(g),**dict(v),**p.stats,
            **{'refused_'+k:z for k,z in p.refused.items()},peak_reserved_side=s['peak'],target_subset=s['subset'])
        if arm=='reference':
            for k in AUDIT:
                if record[k]!=old[k]:raise ValueError(f'old reference differs: {k}')
            for k in ['aligned_lane_ms','elsewhere_lane_ms']:
                if record[k]!=oldcov[k]:raise ValueError(f'old coverage differs: {k}')
        metrics.append(record);actions.extend(dict(arm=arm,**x) for x in aa)
    return metrics,actions


def summarize(m):
    summary={}
    for arm,g in m.groupby('arm',sort=False):
        q={k:float(g[k].sum()) for k in SUMS}
        q.update(windows=len(g),requests=int(q['places']+q['cancels']),
            aligned_pct_of_reference_intent=100*q['aligned_lane_ms']/q['reference_desired_lane_ms'],
            mean_oldest_quote_ms=q['oldest_lane_age_ms2']/q['quoted_lane_ms'],
            mean_reserved_shares_both_sides=q['reserved_share_ms']/(60000*len(g)))
        summary[arm]=q
    b=summary['reference']
    for arm,q in summary.items():
        q.update(request_change_pct=100*(q['requests']/b['requests']-1),
            aligned_change_pct=100*(q['aligned_lane_ms']/b['aligned_lane_ms']-1),
            desired_time_retained_pct=100*q['desired_lane_ms']/b['desired_lane_ms'],
            reference_quoted_time_preserved_pct=100*(1-q['reference_quoted_lost_ms']/b['aligned_lane_ms']),
            off_target_change_pct=100*(q['elsewhere_lane_ms']/b['elsewhere_lane_ms']-1),
            fewer_requests_and_no_aligned_time_loss=q['requests']<b['requests'] and q['aligned_lane_ms']>=b['aligned_lane_ms'])
    return summary


def run():
    d=check_inputs()
    if (DEST/'receipt.json').exists() or (DEST/'window_metrics.parquet').exists():raise ValueError('refuse overwrite')
    controls=(DEST/'logs/controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing passing controls')
    D=pd.read_parquet(OLD/'decisions.parquet');scope=json.loads((a.RP/'scope.json').read_text())
    prior=pd.read_parquet(OLD/'window_metrics.parquet');prior=prior[prior.arm=='incremental'].set_index(['run','mid'])
    cov=pd.read_parquet(OLD/'coverage.parquet');cov=cov[cov.arm=='incremental'].set_index(['run','mid'])
    by={(r,m):g for (r,m),g in D.groupby(['run','mid'],sort=False)}
    results=[];actions=[];statuses=[]
    for i,w in enumerate(scope['windows']):
        key=(w['run'],w['market']);g=by.get(key)
        if g is None:statuses.append(dict(run=key[0],mid=key[1],status='missing_decisions'));continue
        if not g.t.is_monotonic_increasing or not ((g.t>=w['ws']*1000)&(g.t<w['ws']*1000+60000)).all():raise ValueError('decision clock outside scope')
        r,aa=run_window(g,w['ws']*1000,prior.loc[key],cov.loc[key])
        results.extend(dict(run=key[0],mid=key[1],**x) for x in r)
        actions.extend(dict(run=key[0],mid=key[1],**x) for x in aa)
        statuses.append(dict(run=key[0],mid=key[1],status='present',decisions=len(g)))
        if (i+1)%20==0:print('DONE',i+1,'windows',flush=True)
    m=pd.DataFrame(results);act=pd.DataFrame(actions);stat=pd.DataFrame(statuses)
    m.to_parquet(DEST/'window_metrics.parquet',index=False);act.to_parquet(DEST/'actions.parquet',index=False);stat.to_parquet(DEST/'window_status.parquet',index=False)
    summary=summarize(m)
    ref=act[act.arm=='reference'];complete=ref[ref.end_reason!='window_censored'];rapid=complete[complete.rapid]
    diagnostics=dict(actions=len(ref),complete_actions=len(complete),window_censored_actions=int((ref.end_reason=='window_censored').sum()),
        rapid_actions=int(ref.rapid.sum()),rapid_complete_actions=len(rapid),
        complete_shorter_than40ms=int(((complete.end-complete.start)<40).sum()),
        rapid_complete_shorter_than40ms=int(((rapid.end-rapid.start)<40).sum()),
        complete_median_ms=float((complete.end-complete.start).median()),rapid_complete_median_ms=float((rapid.end-rapid.start).median()),
        status_counts=ref.end_reason.value_counts().to_dict(),attribution='recorded lane-intent episodes; exact pull reason unavailable')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,source_controls=controls,
        population=dict(day='20261108',decision_rows=len(D),declared_windows=220,window_status=collections.Counter(stat.status),windows_per_arm=m.arm.value_counts().to_dict()),
        summary=summary,reference_episode_diagnostics=diagnostics,
        baseline_parity=dict(windows=int((m.arm=='reference').sum()),metrics=AUDIT+['aligned_lane_ms','elsewhere_lane_ms']),
        artifacts={p.name:a.sha(p) for p in DEST.glob('*.parquet')},
        limitations=['one consumed partial day, no independent validation or confidence interval',
            'fixed factual desired path; no counterfactual inventory, ladder-price, fill, throttle or post-only-rejection feedback',
            'modeled cancellation acknowledgements; extra residence is not better queue rank or profitable capture',
            'no exact prediction-only attribution; hard and prediction guards are represented by the same upstream lane pulls'])
    a.write(DEST/'receipt.json',out);a.write(REVIEW/'receipt.json',out)
    print(json.dumps(dict(population=out['population'],summary=summary,diagnostics=diagnostics),indent=2))


if __name__=='__main__':{'declare':declare,'run':run}[sys.argv[1]]()
