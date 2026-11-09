"""Exploratory follow-up: process equal-clock updates before a50ms grace timeout."""
import collections
import datetime as dt
import json
import math
import subprocess
import sys
import pandas as pd
from . import pf_inventory_grace_study as parent
from .pf_inventory_grace import InventoryGrace,exposure
from .pf_deadline_grace import UpdateFirstGrace
from .pf_quote_capture import QuotePlanner
from .pf_soft_pull import lanes,OFF
from .pf_quote_coverage import coverage
from .pf_reentry_study import aligned_intervals,difference_time
from .pf_soft_pull_verify import compare_summary

a=parent.a
old=parent.old
RISK=parent.RISK
LATENCIES=parent.LATENCIES
ARMS=['reference','update_first50']
DEST=parent.DEST/'deadline_order'
REVIEW=parent.REVIEW/'deadline_order'
SOURCES=parent.SOURCES+['pf_deadline_grace.py','pf_deadline_grace_study.py','tests/test_deadline_grace.py']


def gate(arm):
    return InventoryGrace(0.) if arm=='reference' else UpdateFirstGrace()


def run_window(actual,probes,inventory,start,arms,latency):
    end=start+60000;last=start;raw=list(OFF);net=None
    states={arm:dict(p=QuotePlanner(incremental=True,**LATENCIES[latency]),gate=gate(arm),total=collections.Counter(),peak=0.) for arm in arms}
    events=pd.concat([actual.assign(kind=0),probes.assign(kind=1),inventory.assign(kind=2)],ignore_index=True)
    events=events[events.t<end].sort_values(['t','input_index','kind','ordinal'],kind='stable')
    def integrate(now):
        for s in states.values():s['gate'].expire(now,s['p'])
        ref=aligned_intervals(states['reference']['p'],last,now,raw)
        for s in states.values():
            p=s['p'];v=s['total'];v.update(p.integrate(last,now))
            v['desired_lane_ms']+=(now-last)*sum(t is not None and z>0 for t,z in raw)
            on,off=coverage(p,last,now,raw);v['aligned_lane_ms']+=on;v['elsewhere_lane_ms']+=off
            here=aligned_intervals(p,last,now,raw)
            v['reference_quoted_lost_ms']+=difference_time(ref,here);v['reference_unquoted_gained_ms']+=difference_time(here,ref)
            if net is None:v['inventory_unobserved_ms']+=now-last
            else:v['inventory_known_ms']+=now-last;v.update(exposure(p,last,now,raw,net))
    def finish_time(t):
        for state in states.values():
            if hasattr(state['gate'],'finish_timestamp'):state['gate'].finish_timestamp(t,state['p'])
    for row in events.to_dict('records'):
        if row['t']!=last:finish_time(last)
        if not math.isfinite(row['net']):raise ValueError('invalid recorded net')
        if row['kind'] and net==row['net'] and not any(s['gate'].holds for s in states.values()):continue
        now=row['t'];integrate(now);last=now;net=row['net']
        if row['kind']:
            for s in states.values():
                if row['kind']==2 and s['gate'].mode=='all':continue
                before=s['p'].stats['cancels'];s['gate'].observe(now,row,s['p']);s['total']['probe_cancels']+=s['p'].stats['cancels']-before
        else:
            raw=lanes(row)
            for s in states.values():
                s['gate'].apply(now,row,s['p']);s['peak']=max(s['peak'],s['p'].reserved(0),s['p'].reserved(1))
    finish_time(last);integrate(end);finish_time(end);metrics=[];actions=[];checks=[]
    for arm,s in states.items():
        p=s['p'];aa=s['gate'].finish(end,p);v=s['total']
        v.update(actions=len(aa),retained_actions=sum(x['status']=='retained' for x in aa),same_price_reopens=sum(x['end_reason']=='same_price_reopen' for x in aa),retained_ms=sum(x['retained_ms'] for x in aa))
        rec=dict(arm=arm,latency=latency,decisions=len(actual),**dict(v),**p.stats,**{'refused_'+k:z for k,z in p.refused.items()},peak_reserved_side=s['peak'])
        for k in old.SUMS+RISK:rec.setdefault(k,0.)
        if s['peak']>150+1e-8:raise ValueError('side cap exceeded')
        metrics.append(rec);actions.extend(dict(arm=arm,latency=latency,**x) for x in aa);checks.extend(dict(arm=arm,latency=latency,**x) for x in s['gate'].checks)
    return metrics,actions,checks


def declare():
    if DEST.exists():raise ValueError('deadline round exists')
    (DEST/'logs').mkdir(parents=True)
    prior=json.loads((parent.DEST/'duration_latency/declaration.json').read_text())
    paths=list(prior['inputs'])+[str(parent.DEST/'duration_latency'/p) for p in ['receipt.json','verification.json','window_metrics.parquet','actions.parquet']]
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCES},inputs={p:a.sha(p) for p in paths},
        population=prior['population'],arms=ARMS,latencies=LATENCIES,delta_c=.01,max_hold_ms=50,
        design='same accepted-small-touch gate; process all native events stamped exactly at deadline before expiry. Eligible quote-on can keep order normally; still-off order cancels at exact deadline. If next event is later, cancel at deadline inside interval. Native timestamps never shifted; no future timestamp is used.',
        known_result='previous timer-first50ms has zero same-price bridges in all three latency scenarios, while100ms has bridges; expiry order could interact with50ms guard cadence',
        estimand='exploratory event-order follow-up on the same consumed day, not fresh validation; compare reference and prior timer-first50/100 artifacts on request/coverage/exposure only',
        null='deterministic mechanical comparison, no PnL/skill/statistical test or confidence interval',
        controls='equal-clock reopen positive; still-off and hard guard cancel at deadline; later event cannot undo expiry; all equal-clock events processed; old baseline parity and corrupted-count refusal',
        multiplicity=dict(new_variants=1,cumulative_distinct_policies=21,latency_scenarios=3),
        limitations=prior['limitations']+['policy chosen to investigate a result seen in round2; exploratory only'],
        no_live_change=True,no_protected_dates_read=True,no_fill_clock_bypass=True)
    a.write(DEST/'declaration.json',d);a.write(REVIEW/'declaration.json',d)


def check():
    d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('source differs')
    for p,h in d['inputs'].items():
        if a.sha(p)!=h:raise ValueError('input differs')
    return d


def reference_parity(m,act):
    prior=parent.DEST/'duration_latency'
    pm=pd.read_parquet(prior/'window_metrics.parquet');pa=pd.read_parquet(prior/'actions.parquet')
    keys=['run','mid','latency','arm'];aa=['run','mid','latency','arm','action']
    for left,right,sort in [(m,pm,keys),(act,pa,aa)]:
        left=left[left.arm=='reference'].sort_values(sort).reset_index(drop=True)
        right=right[right.arm=='reference'].sort_values(sort).reset_index(drop=True)
        pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=1e-11,atol=1e-4)


def run():
    d=check()
    if (DEST/'window_metrics.parquet').exists():raise ValueError('refuse overwrite')
    controls=(DEST/'logs/controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('controls missing')
    scope=json.loads((a.RP/'scope.json').read_text());metrics=[];actions=[];checks=[];statuses=[]
    for run in old.n.RUNS:
        frames=[]
        for path in [old.n.DEST/f'states_{run}.parquet',old.grid.DEST/f'probes_{run}.parquet',parent.native_inventory.DEST/f'inventory_{run}.parquet']:
            x=pd.read_parquet(path);frames.append(x[x.rel_s<60])
        actual,probes,inventory=frames;ai=actual.groupby('mid').indices;pi=probes.groupby('mid').indices;ii=inventory.groupby('mid').indices
        for w in [w for w in scope['windows'] if w['run']==run]:
            mid=w['market']
            for latency in LATENCIES:
                mm,aa,cc=run_window(actual.iloc[ai[mid]],probes.iloc[pi[mid]],inventory.iloc[ii.get(mid,[])],w['ws']*1000,ARMS,latency)
                for target,rr in [(metrics,mm),(actions,aa),(checks,cc)]:target.extend(dict(run=run,mid=mid,**x) for x in rr)
            statuses.append(dict(run=run,mid=mid,status='present',decisions=len(ai[mid])))
            if len(statuses)%20==0:print('DONE',len(statuses),'windows',flush=True)
    m=pd.DataFrame(metrics);act=pd.DataFrame(actions);reference_parity(m,act)
    for name,rr in [('window_metrics',metrics),('actions',actions),('checks',checks),('window_status',statuses)]:pd.DataFrame(rr).to_parquet(DEST/f'{name}.parquet',index=False)
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,controls=controls,
        population=dict(windows=len(statuses),decisions=sum(x['decisions'] for x in statuses),neutral_actions=int(((act.arm=='reference')&(act.latency=='base')).sum()),excluded_windows=0),
        summary=parent.summarize(m),by_run={r:parent.summarize(g) for r,g in m.groupby('run')},
        equal_clock_reopens={l:int(((act.latency==l)&(act.arm=='update_first50')&(act.end_reason=='same_price_reopen')&(act.retained_ms==50)).sum()) for l in LATENCIES},
        statuses={f'{l}/{arm}':dict(n=len(g),status=g.status.value_counts().to_dict(),end=g.end_reason.value_counts().to_dict()) for (l,arm),g in act.groupby(['latency','arm'])},
        artifacts={p.name:a.sha(p) for p in DEST.glob('*.parquet')},limitations=d['limitations'])
    a.write(DEST/'receipt.json',out);a.write(REVIEW/'receipt.json',out);print(json.dumps(out['summary'],indent=2))


def verify():
    check()
    if (DEST/'verification.json').exists():raise ValueError('verification exists')
    r=json.loads((DEST/'receipt.json').read_text())
    for p,h in r['artifacts'].items():
        if a.sha(DEST/p)!=h:raise ValueError('artifact differs')
    m=pd.read_parquet(DEST/'window_metrics.parquet');act=pd.read_parquet(DEST/'actions.parquet')
    if len(m)!=1320 or len(act)!=13077*6 or m.duplicated(['run','mid','latency','arm']).any() or act.duplicated(['run','mid','latency','arm','action']).any():raise ValueError('population differs')
    reference_parity(m,act)
    identity=['run','mid','action','lane','start','old_tick','v','midpoint','input_index','spread','start_net']
    for latency in LATENCIES:
        mm=m[m.latency==latency];compare_summary(mm,r['summary'][latency]);aa=act[act.latency==latency]
        left=aa[aa.arm=='reference'].sort_values(['run','mid','action'])[identity].reset_index(drop=True)
        right=aa[aa.arm=='update_first50'].sort_values(['run','mid','action'])[identity].reset_index(drop=True)
        pd.testing.assert_frame_equal(left,right)
        rr=aa[(aa.arm=='update_first50')&(aa.status=='retained')]
        if not ((rr.retained_ms>=0)&(rr.retained_ms<=50)&((rr.end-rr.start)==rr.retained_ms)).all():raise ValueError('maximum grace exceeded')
        if int(((rr.end_reason=='same_price_reopen')&(rr.retained_ms==50)).sum())!=r['equal_clock_reopens'][latency]:raise ValueError('equal-clock count differs')
        base=mm[mm.arm=='reference'].set_index(['run','mid'])
        for z in mm[mm.arm=='update_first50'].to_dict('records'):
            b=base.loc[(z['run'],z['mid'])]
            if not math.isclose(z['aligned_lane_ms']-b.aligned_lane_ms,z['reference_unquoted_gained_ms']-z['reference_quoted_lost_ms'],abs_tol=1e-4,rel_tol=1e-11):raise ValueError('coverage identity differs')
    if parent.summarize(m)!=r['summary']:raise ValueError('exposure summary differs')
    for run,g in m.groupby('run'):
        if parent.summarize(g)!=r['by_run'][run]:raise ValueError('run summary differs')
    bad=json.loads(json.dumps(r['summary']['base']));bad['reference']['requests']+=1
    try:compare_summary(m[m.latency=='base'],bad)
    except ValueError:refused=True
    else:raise ValueError('corrupt count accepted')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),receipt_sha256=a.sha(DEST/'receipt.json'),source_sha256=a.sha(a.HERE/'pf_deadline_grace_study.py'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),rows=dict(window_arm=len(m),action_arm=len(act)),
        original_reference_reconciled_all_latencies=True,maximum_grace_ms=50,equal_clock_reopens=r['equal_clock_reopens'],
        controls=dict(valid_artifacts_accepted=True,corrupt_count_refused=refused),limitations=r['limitations'])
    a.write(DEST/'verification.json',out);a.write(REVIEW/'verification.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':{'declare':declare,'run':run,'verify':verify}[sys.argv[1]]()

