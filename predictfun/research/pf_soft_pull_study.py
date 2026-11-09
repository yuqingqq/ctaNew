"""Bounded soft-exit experiment; immutable, no-fill, factual-state conditioning."""
import collections
import datetime as dt
import gzip
import json
import math
import subprocess
import sys
import numpy as np
import pandas as pd
from . import pf_native_guard as n
from . import pf_native_grid as grid
from .pf_soft_pull import ARMS,SoftPull,lanes,OFF
from .pf_quote_capture import QuotePlanner
from .pf_quote_coverage import coverage
from .pf_reentry_study import aligned_intervals,difference_time,AUDIT

a=n.a
DEST=a.ROOT/'data/predict_fun/research/soft_pull_20261109'
REVIEW=a.HERE.parent/'reviews/soft_pull_20261109'
OLD=a.ROOT/'data/predict_fun/research/quote_incremental_20261109'
SOURCES=['pf_soft_pull.py','pf_soft_pull_study.py','tests/test_soft_pull.py','pf_quote_capture.py','pf_quote_coverage.py','pf_reentry_study.py']
SUMS=['places','cancels','aligned_lane_ms','elsewhere_lane_ms','share_ms','reserved_share_ms',
    'quoted_lane_ms','oldest_lane_age_ms2','desired_lane_ms','reference_quoted_lost_ms','reference_unquoted_gained_ms',
    'actions','retained_actions','same_price_reopens','retained_ms','probe_cancels']


def declare():
    if DEST.exists():raise ValueError('study exists')
    (DEST/'logs').mkdir(parents=True)
    paths=[a.RP/'scope.json',n.DEST/'verification.json',grid.DEST/'verification.json',OLD/'window_metrics.parquet',OLD/'coverage.parquet']
    for run in n.RUNS:paths.extend([n.DEST/f'states_{run}.parquet',grid.DEST/f'probes_{run}.parquet',n.DEST/run/'evidence/books.jsonl.gz'])
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCES},inputs={str(p):a.sha(p) for p in paths},
        population='220 consumed Nov8 windows, all first60s decisions; actions are every original touch on-to-off transition, never selected on fills; zero unrecorded exclusions',
        arms=ARMS,primary='exit01',max_hold_ms=100,minute0_line=.0492,
        policy='retain accepted uncancelled small base<=5 at original touch price only on original on-to-off event, line-delta<v<=line, native relaxed quote same price and enough size, no original rung duplicate; no fresh order while guard off; all hard predicates conditional on recorded state preserved; expire100ms or earlier on lower-line/hard/price/size change; no rung retention',
        timing='normal quoting at original decisions; only reevaluate/cancel existing retained orders at native due-event probes; exact100ms timer cancellation before integrating spans',
        reference='fixed incremental sizing, old40/22/95ms acknowledgement and1000ms reservation grace; reproduce every old count exactly and integrals atol1e-4/rtol1e-11 per window',
        estimand='conditional no-fill request workload and original-price coverage; separately report extra off-guard exposure, reservations and adverse future midpoint moves; no fill, PnL, inventory variance or deployment decision',
        price_diagnostic='one row per reference touchpull; signed midpoint change excluding initial spread at1s primary and5s secondary; last known native Frame at or before horizon, age<=250ms at start/end; invalid/missing/stale/mismatched-start statuses retained; unweighted action means, never trade returns',
        diagnostic_controls='same selected action count within run,UTC hour,side,guard-depth floor((.0492-v)/.005),10-tick current midpoint bucket,spread bucket1/2/3+; sample with replacement500 times seed20261109 from all reference pull actions with valid same-horizon labels; selected actions remain in pool; no p-value/CI or skill claim; this proxy cannot select a trading policy',
        null='mechanical policy comparison deterministic; descriptive price diagnostic uses500 matched draws declared here, not an independent economic null; one consumed partial day, no confidence interval',
        multiplicity=dict(new_variants=4,current_arms=5,cumulative_distinct_policies=16),
        controls='retention positive; lower-line, hard guard, deadline, new-order, pending-create, duplication refusals; zero-delta parity; price-clock controls; original baseline parity; artifact count/summary corruptions refused',
        no_live_change=True,no_fill_clock_bypass=True,no_protected_dates_read=True)
    a.write(DEST/'declaration.json',d);a.write(REVIEW/'declaration.json',d)


def check():
    d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('source changed')
    for p,h in d['inputs'].items():
        if a.sha(p)!=h:raise ValueError('input changed')
    return d


def run_window(actual,probes,start,old,oldcov):
    end=start+60000;last=start;raw=list(OFF)
    states={arm:dict(p=QuotePlanner(incremental=True),gate=SoftPull(delta),total=collections.Counter(),peak=0.) for arm,delta in ARMS.items()}
    events=pd.concat([actual.assign(kind=0),probes.assign(kind=1)],ignore_index=True)
    events=events[events.t<end].sort_values(['t','input_index','kind','ordinal'],kind='stable')
    def integrate(now):
        for s in states.values():s['gate'].expire(now,s['p'])
        ref=aligned_intervals(states['reference']['p'],last,now,raw)
        for s in states.values():
            p=s['p'];v=s['total'];v.update(p.integrate(last,now))
            v['desired_lane_ms']+=(now-last)*sum(t is not None and z>0 for t,z in raw)
            on,off=coverage(p,last,now,raw);v['aligned_lane_ms']+=on;v['elsewhere_lane_ms']+=off
            here=aligned_intervals(p,last,now,raw)
            v['reference_quoted_lost_ms']+=difference_time(ref,here)
            v['reference_unquoted_gained_ms']+=difference_time(here,ref)
    for row in events.to_dict('records'):
        if row['kind'] and not any(s['gate'].holds for s in states.values()):continue
        now=row['t'];integrate(now);last=now
        if row['kind']:
            for s in states.values():
                before=s['p'].stats['cancels'];s['gate'].observe(now,row,s['p'])
                s['total']['probe_cancels']+=s['p'].stats['cancels']-before
        else:
            raw=lanes(row)
            for s in states.values():
                s['gate'].apply(now,row,s['p']);s['peak']=max(s['peak'],s['p'].reserved(0),s['p'].reserved(1))
    integrate(end);metrics=[];actions=[]
    for arm,s in states.items():
        p=s['p'];aa=s['gate'].finish(end,p);v=s['total']
        v.update(actions=len(aa),retained_actions=sum(x['status']=='retained' for x in aa),
            same_price_reopens=sum(x['end_reason']=='same_price_reopen' for x in aa),retained_ms=sum(x['retained_ms'] for x in aa))
        rec=dict(arm=arm,decisions=len(actual),**dict(v),**p.stats,**{'refused_'+k:z for k,z in p.refused.items()},peak_reserved_side=s['peak'])
        for k in SUMS:rec.setdefault(k,0.)
        if arm=='reference':
            for k in AUDIT:
                if not math.isclose(rec[k],float(old[k]),abs_tol=1e-4,rel_tol=1e-11):raise ValueError(f'baseline differs {k}: {rec[k]} != {old[k]}')
            for k in ['aligned_lane_ms','elsewhere_lane_ms']:
                if not math.isclose(rec[k],float(oldcov[k]),abs_tol=1e-4,rel_tol=1e-11):raise ValueError('baseline coverage differs')
        if s['peak']>150+1e-8:raise ValueError('reservation budget exceeded')
        metrics.append(rec);actions.extend(dict(arm=arm,**x) for x in aa)
    return metrics,actions


def load_books(run):
    rows=collections.defaultdict(list)
    with gzip.open(n.DEST/run/'evidence/books.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line);bid=d['bids'][0][0];ask=d['asks'][0][0]
            rows[int(d['market'])].append((d['t'],(bid+ask)/2 if 0<bid<ask<100 else math.nan))
    result={}
    for key,rr in rows.items():
        arr=np.array(rr)
        if not (np.diff(arr[:,0])>=0).all():raise ValueError('book clock regressed')
        result[key]=arr
    return result


def price_label(row,book,horizon):
    now=row['start'];t=book[:,0]
    ii=np.searchsorted(t,[now,now+horizon],side='right')-1
    if (ii<0).any():return 'missing_frame',math.nan
    if (np.array([now,now+horizon])-t[ii]>250).any():return 'stale_frame',math.nan
    mm=book[ii,1]
    if not np.isfinite(mm).all() or not math.isfinite(row['midpoint']):return 'invalid_book',math.nan
    if mm[0]!=row['midpoint']:return 'start_identity_ambiguous',math.nan
    return 'valid',float((1 if row['lane']==0 else -1)*(mm[1]-mm[0]))


def summarize(m):
    out={}
    for arm,g in m.groupby('arm',sort=False):
        q={k:float(g[k].sum()) for k in SUMS};q.update(windows=len(g),requests=int(q['places']+q['cancels']),
            mean_oldest_quote_ms=q['oldest_lane_age_ms2']/q['quoted_lane_ms'])
        out[arm]=q
    b=out['reference']
    for q in out.values():
        for name,k in [('request','requests'),('aligned','aligned_lane_ms'),('off_target','elsewhere_lane_ms'),('reserved','reserved_share_ms'),('active_shares','share_ms')]:q[f'{name}_change_pct']=100*(q[k]/b[k]-1)
        q['fewer_requests_and_no_aligned_time_loss']=q['requests']<b['requests'] and q['aligned_lane_ms']>=b['aligned_lane_ms']
    return out


def diagnose(actions):
    ref=actions[actions.arm=='reference'].copy();ref['hour']=(ref.start//3600000).astype(int)
    ref['depth']=np.floor((.0492-ref.v.fillna(-999))/.005).astype(int)
    ref['price_bin']=np.floor(ref.midpoint.fillna(-999)/10).astype(int);ref['spread_bin']=ref.spread.clip(upper=3)
    keys=['run','hour','lane','depth','price_bin','spread_bin'];ids=['run','mid','action']
    out={};rng=np.random.default_rng(20261109)
    for arm in ARMS:
        g=actions[actions.arm==arm];selected=g[g.status=='retained'][ids]
        q=dict(actions=len(g),retained=len(selected),statuses=g.status.value_counts().to_dict(),ends=g.end_reason.value_counts().to_dict(),horizons={})
        for horizon in [1000,5000]:
            tag=f'h{horizon}';s=ref.merge(selected,on=ids,how='inner',validate='one_to_one')
            statuses=s[f'{tag}_status'].value_counts().to_dict();sv=s[s[f'{tag}_status']=='valid']
            rv=ref[ref[f'{tag}_status']=='valid'];draws=np.zeros(500);total=0
            pool={k:grp[f'{tag}_move_c'].to_numpy() for k,grp in rv.groupby(keys)}
            for k,gg in sv.groupby(keys):
                vv=pool[k];draws+=rng.choice(vv,size=(500,len(gg)),replace=True).sum(axis=1);total+=len(gg)
            z=dict(label_statuses=statuses,n_valid=len(sv),mean_move_c=float(sv[f'{tag}_move_c'].mean()) if len(sv) else None,
                adverse_fraction=float((sv[f'{tag}_move_c']<0).mean()) if len(sv) else None,
                matched_draws=500,matched_control_mean_c=float((draws/total).mean()) if total else None,
                interpretation='unweighted future market drift per action; not a fill markout, realized loss, statistical significance, or policy ranking')
            q['horizons'][str(horizon)]=z
        out[arm]=q
    return out


def run():
    d=check()
    if (DEST/'window_metrics.parquet').exists():raise ValueError('refuse overwrite')
    controls=(DEST/'logs/controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('controls missing')
    scope=json.loads((a.RP/'scope.json').read_text());prior=pd.read_parquet(OLD/'window_metrics.parquet');prior=prior[prior.arm=='incremental'].set_index(['run','mid'])
    cov=pd.read_parquet(OLD/'coverage.parquet');cov=cov[cov.arm=='incremental'].set_index(['run','mid'])
    metrics=[];actions=[];statuses=[]
    for run in n.RUNS:
        actual=pd.read_parquet(n.DEST/f'states_{run}.parquet');actual=actual[actual.rel_s<60]
        probes=pd.read_parquet(grid.DEST/f'probes_{run}.parquet');probes=probes[probes.rel_s<60]
        ai=actual.groupby('mid').indices;pi=probes.groupby('mid').indices;books=load_books(run)
        for w in [w for w in scope['windows'] if w['run']==run]:
            mid=w['market'];key=(run,mid)
            if mid not in ai or mid not in pi or mid not in books:raise ValueError(f'missing window {key}')
            mm,aa=run_window(actual.iloc[ai[mid]],probes.iloc[pi[mid]],w['ws']*1000,prior.loc[key],cov.loc[key])
            for x in aa:
                for horizon in [1000,5000]:x[f'h{horizon}_status'],x[f'h{horizon}_move_c']=price_label(x,books[mid],horizon)
            metrics.extend(dict(run=run,mid=mid,**x) for x in mm);actions.extend(dict(run=run,mid=mid,**x) for x in aa)
            statuses.append(dict(run=run,mid=mid,status='present',decisions=len(ai[mid]),probes=len(pi[mid])))
            if len(statuses)%20==0:print('DONE',len(statuses),'windows',flush=True)
        del actual,probes,books
    for name,rr in [('window_metrics',metrics),('actions',actions),('window_status',statuses)]:pd.DataFrame(rr).to_parquet(DEST/f'{name}.parquet',index=False)
    m=pd.read_parquet(DEST/'window_metrics.parquet');act=pd.read_parquet(DEST/'actions.parquet')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,source_controls=controls,
        population=dict(windows=len(statuses),decisions=sum(x['decisions'] for x in statuses),actions=int((act.arm=='reference').sum()),excluded_windows=0),
        summary=summarize(m),by_run={r:summarize(g) for r,g in m.groupby('run')},diagnostics=diagnose(act),
        artifacts={p.name:a.sha(p) for p in DEST.glob('*.parquet')},
        limitations=['one consumed partial day; no confidence interval or independent validation','factual inventory/rung/order-state conditioning; hypothetical fills and feedback absent','fixed latency assumptions; no venue throttle, post-only rejection or true queue-rank model','midpoint diagnostic does not measure net trading value or settlement; fill-clock gate remains closed'])
    a.write(DEST/'receipt.json',out);a.write(REVIEW/'receipt.json',out);print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':{'declare':declare,'run':run}[sys.argv[1]]()
