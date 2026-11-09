"""Paired order-workload benchmark on consumed Nov8 decisions, no fill simulation.

Inputs are the recorded desired quote trace, held fixed across arms. Thus the
inventory-dependent policy feedback remains factual, not a counterfactual PnL
path. Every arm executes a no-fill order workload with identical latency/caps.
"""
import collections
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_quote_capture import QuotePlanner,EntryGate
from . import pf_fill_clock as a

DEST=a.ROOT/'data/predict_fun/research/quote_capture_20261109'
REVIEW=a.HERE.parent/'reviews/quote_capture_20261109'
CAP=a.ROOT.parent/'pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109'
RUNS={'v6':'1794108465172','basis':'1794147344737'}
ARMS={'single':(False,0.),'layered':(True,0.),**{f'layered_entry_{d:g}':(True,d) for d in [.005,.01,.02,.04]}}
DEC=re.compile(rb'^\{"Decision":\{"market":([0-9]+),"t":([0-9.]+),')
TIME=re.compile(rb'"t":([0-9]+(?:\.[0-9]+)?)')
SOURCE=['pf_quote_capture.py','pf_quote_capture_study.py','tests/test_quote_capture.py','tests/test_quote_capture_study.py']


def decision_rows(path,scope,stats):
    h=hashlib.sha256();last={};count=0
    with gzip.open(path,'rb') as f:
        for line in f:
            stamp=TIME.search(line[:512])
            if stamp and float(stamp[1])>=a.END_MS:
                stats['stopped_before_future_payload']=True;break
            m=DEC.match(line)
            if line.startswith(b'{"Decision"') and not m:raise ValueError('unrecognized decision timestamp envelope')
            h.update(line);count+=1
            if not m:continue
            mid=int(m[1]);now=float(m[2])
            if mid not in scope:continue
            if now<last.get(mid,-math.inf):raise ValueError('decision time regressed')
            last[mid]=now;ws=scope[mid]*1000
            if not ws<=now<ws+60000:continue
            row=json.loads(line)['Decision']
            if row['market']!=mid or row['t']!=now or not isinstance(row['lanes'],list) or len(row['lanes'])!=4:
                raise ValueError('decision identity/shape differs')
            out=dict(mid=mid,t=now,sequence=count)
            for lane,item in enumerate(row['lanes']):
                if not isinstance(item,list) or len(item)!=2:raise ValueError('bad lane intent')
                tick,size=item
                if not math.isfinite(size) or size<0 or (tick is None and size!=0) or (tick is not None and (int(tick)!=tick or not 1<=tick<=99)):
                    raise ValueError('bad recorded price/size')
                out[f'tick{lane}']=-1 if tick is None else int(tick);out[f'size{lane}']=float(size)
            yield out
    stats.update(prefix_sha256=h.hexdigest(),lines=count,cutoff_ms=a.END_MS)


def declare():
    if DEST.exists():raise ValueError('existing benchmark')
    (DEST/'logs').mkdir(parents=True)
    paths=[a.RP/'scope.json',a.RP/'research/step5/20261108/preds_t60k.parquet',a.ROOT/'data/predict_fun/research/step4/minute_lines_t60k.json']
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        population='220 consumed Nov8 windows, first60 seconds, four lanes; all recorded desired decisions, never fill-selected',
        estimand='paired fixed-intent no-fill engineering workload; order requests, oldest active quote age, quoting time and pending exposure; no PnL or fill-rate estimate',
        baseline='single-order cancel/repost versus base5+topup; same recorded target prices/sizes/hard-guard pulls',
        arms=ARMS,primary='layered without buffer; layered_entry_0.01 is a predeclared shadow candidate, not an economic winner',
        gate='entry only: v>original minute0 PM90 line+delta; keep only while v>original line and score age<=100ms; reset on hard pull/reprice; original desired target is never enlarged',
        order_model=dict(place_ms=40,cancel_ms=22,pending_cancel_ms=95,cancel_reserve_grace_ms=1000,side_cap_shares=150,creates_inflight_per_side=3,minimum_notional_usd=1),
        sizing='do not refill partial components; fall back to one order when either split ticket is below minimum; global pending reservations and reduce-only room shared across lanes',
        signal_limitation='buffer uses precomputed reconstructed research scores; these are not captured exact live guard values',
        null='deterministic mechanical workload test; no statistical hypothesis, confidence interval, predictive-skill claim or profit-based selection',
        multiplicity=6,validation='positive/refusal controls, zero-buffer exact target parity, all-arm target subset, action/budget checks; fresh-day strategy validation not attempted',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCE},inputs={str(p):a.sha(p) for p in paths},
        raw_inputs={run:str(CAP/run/f'run-{rid}/outputs.jsonl.gz') for run,rid in RUNS.items()},
        raw_boundary='hash/read only decompressed prefix before Nov8 23:02; future payload not decoded',
        no_live_change=True,no_fill_clock_bypass=True,protected_dates_read=False)
    a.write(DEST/'declaration.json',d);a.write(REVIEW/'declaration.json',d)
    print('DECLARED paired quote-workload benchmark')


def verify_inputs():
    d=json.loads((DEST/('declaration_v2.json' if (DEST/'declaration_v2.json').exists() else 'declaration.json')).read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('source changed')
    for p,h in d['inputs'].items():
        if a.sha(p)!=h:raise ValueError('input changed')
    return d


def extract():
    if (DEST/'decisions.parquet').exists():raise ValueError('refuse overwrite')
    d=verify_inputs();scope=json.loads((a.RP/'scope.json').read_text())
    if scope['days_consumed']!=['20261108'] or len(scope['windows'])!=220:raise ValueError('scope differs')
    records=[];prefix={}
    for run,path in d['raw_inputs'].items():
        windows={w['market']:w['ws'] for w in scope['windows'] if w['run']==run};stats={}
        records.extend(dict(x,run=run) for x in decision_rows(path,windows,stats));prefix[run]=stats
    D=pd.DataFrame(records).sort_values(['run','mid','t','sequence'],kind='stable').reset_index(drop=True)
    preds=pd.read_parquet(a.RP/'research/step5/20261108/preds_t60k.parquet',columns=['mid','tau','v_bid','v_ask'])
    by={m:g.sort_values('tau',kind='stable') for m,g in preds.groupby('mid')}
    D['score_at']=np.nan;D['v_bid']=np.nan;D['v_ask']=np.nan
    for mid,ix in D.groupby('mid').indices.items():
        p=by.get(mid)
        if p is None:continue
        if p.tau.duplicated().any():raise ValueError('duplicate score timestamp')
        j=np.searchsorted(p.tau.to_numpy(),D.loc[ix,'t'].to_numpy(),side='right')-1
        ok=j>=0;rows=ix[ok];pick=p.iloc[j[ok]]
        D.loc[rows,['score_at','v_bid','v_ask']]=pick[['tau','v_bid','v_ask']].to_numpy()
    D.to_parquet(DEST/'decisions.parquet',index=False);a.write(DEST/'prefixes.json',prefix)
    a.write(DEST/'extraction.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),decisions=len(D),markets=int(D.mid.nunique()),
        decisions_sha256=a.sha(DEST/'decisions.parquet'),prefixes=prefix))
    print('EXTRACTED',len(D),'decisions,',D.mid.nunique(),'markets',flush=True)


def run():
    d=verify_inputs();suffix='_v2' if d.get('version')==2 else ''
    if (DEST/f'receipt{suffix}.json').exists() or (DEST/f'window_metrics{suffix}.parquet').exists():raise ValueError('refuse overwrite')
    controls=(DEST/f'logs/controls{suffix}.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing passing controls')
    extracted=json.loads((DEST/'extraction.json').read_text())
    if a.sha(DEST/'decisions.parquet')!=extracted['decisions_sha256']:raise ValueError('decision artifact changed')
    D=pd.read_parquet(DEST/'decisions.parquet');scope=json.loads((a.RP/'scope.json').read_text())
    by={(r,m):g for (r,m),g in D.groupby(['run','mid'],sort=False)}
    line=json.loads((a.ROOT/'data/predict_fun/research/step4/minute_lines_t60k.json').read_text())['lines']['90'][0]
    results=[];statuses=[]
    for wi,w in enumerate(scope['windows']):
        key=(w['run'],w['market']);g=by.get(key);start=w['ws']*1000.;end=start+60000
        if g is None:statuses.append(dict(run=key[0],mid=key[1],status='missing_decisions'));continue
        statuses.append(dict(run=key[0],mid=key[1],status='present',n=len(g)))
        arrays=g[['t','score_at','v_bid','v_ask']+[v for lane in range(4) for v in (f'tick{lane}',f'size{lane}')]].to_numpy()
        for arm,config in ARMS.items():
            layered,delta=config[:2];incremental=bool(config[2]) if len(config)>2 else False
            p=QuotePlanner(layered=layered,incremental=incremental);gate=EntryGate(delta);last=start
            total=collections.Counter();last_want=OFF=[(None,0.)]*4;peak=0.;min_step=True
            score_bad=score_conflict=0;parity=True;subset=True
            for row in arrays:
                now,score_at,vb,va=row[:4];lanes=[(None if row[4+2*i]<0 else int(row[4+2*i]),float(row[5+2*i])) for i in range(4)]
                total.update(p.integrate(last,now))
                total['desired_lane_ms']+=(now-last)*sum(q[0] is not None and q[1]>0 for q in last_want)
                last=now
                want=gate.apply(lanes,[vb,va],line,now,score_at)
                parity &= want==lanes if delta==0 else True
                subset &= all(q[0] is None or q==ref for q,ref in zip(want,lanes))
                fresh=math.isfinite(score_at) and 0<=now-score_at<=100
                for lane,q in enumerate(lanes):
                    if q[0] is not None:
                        v=vb if lane%2==0 else va
                        score_bad+=int(not fresh or not math.isfinite(v))
                        score_conflict+=int(fresh and math.isfinite(v) and v<=line)
                p.plan(now,want);last_want=want
                peak=max(peak,p.reserved(0),p.reserved(1))
                if peak>150+1e-8:raise ValueError('side cap exceeded')
            total.update(p.integrate(last,end))
            total['desired_lane_ms']+=(end-last)*sum(q[0] is not None and q[1]>0 for q in last_want)
            if not parity or not subset:raise ValueError('baseline/subset target identity failed')
            if any(c['kind']=='place' and (c['tick'] if c['lane']%2==0 else 100-c['tick'])*c['size']/100<1-1e-9 for c in p.commands):
                raise ValueError('minimum notional violated')
            results.append(dict(run=key[0],mid=key[1],arm=arm,decisions=len(g),**dict(total),**p.stats,
                **{'refused_'+k:v for k,v in p.refused.items()},peak_reserved_side=peak,
                reconstructed_score_missing=score_bad,reconstructed_score_vs_live_conflict=score_conflict,
                zero_delta_parity=parity,target_subset=subset))
        if (wi+1)%20==0:print('DONE',wi+1,'windows',flush=True)
    R=pd.DataFrame(results);R.to_parquet(DEST/f'window_metrics{suffix}.parquet',index=False)
    pd.DataFrame(statuses).to_parquet(DEST/f'window_status{suffix}.parquet',index=False)
    summary={}
    for arm,q in R.groupby('arm',sort=False):
        totals={c:float(q[c].sum()) for c in ['places','cancels','base_resize_cancels','kept_base_size_switches','share_ms','quoted_lane_ms','desired_lane_ms','oldest_lane_age_ms2','reserved_share_ms','refused_capacity','refused_inflight']}
        totals.update(windows=len(q),requests=int(totals['places']+totals['cancels']),
            mean_oldest_active_quote_ms=totals['oldest_lane_age_ms2']/totals['quoted_lane_ms'] if totals['quoted_lane_ms'] else None,
            mean_reserved_shares_both_sides=totals['reserved_share_ms']/(len(q)*60000),peak_reserved_side=float(q.peak_reserved_side.max()))
        summary[arm]=totals
    baseline=summary['single'];same=summary['layered']
    primary=summary[d.get('primary_arm','layered')]
    for arm,q in summary.items():
        q['requests_change_vs_single_pct']=100*(q['requests']/baseline['requests']-1)
        q['desired_time_retained_vs_single_pct']=100*q['desired_lane_ms']/baseline['desired_lane_ms']
        q['quoted_time_retained_vs_single_pct']=100*q['quoted_lane_ms']/baseline['quoted_lane_ms']
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,source_controls=controls,line_cents=line,
        population=dict(day='20261108',declared_markets=220,present_markets=len(by),decision_rows=len(D),
            window_status=collections.Counter(x['status'] for x in statuses),markets_per_arm=R.groupby('arm').size().to_dict()),
        summary=summary,source_score_diagnostics=R[R.arm=='single'][['reconstructed_score_missing','reconstructed_score_vs_live_conflict']].sum().to_dict(),
        predicates=dict(zero_buffer_targets_exact=bool(R.zero_delta_parity.all()),all_targets_subset=bool(R.target_subset.all()),
            all_side_caps_hold=bool((R.peak_reserved_side<=150+1e-8).all()),
            layered_fewer_requests=same['requests']<baseline['requests'],layered_equal_desired_time=same['desired_lane_ms']==baseline['desired_lane_ms'],
            primary_fewer_requests=primary['requests']<baseline['requests'],primary_equal_desired_time=primary['desired_lane_ms']==baseline['desired_lane_ms']),
        artifacts={p.name:a.sha(p) for p in DEST.glob('*.parquet')},extraction=extracted,
        limitations=['one consumed partial day; fixed factual desired path with no simulated fills; no counterfactual inventory feedback',
            'first60 seconds only; no ladder price adaptation to counterfactual order residency',
            'no fill/PnL/variance estimate, queue rank proof, confidence interval or deployment claim',
            'modelled acknowledgements and1s reservation grace; no venue throttling/backoff/post-only rejection model',
            'buffer results depend on reconstructed scores; missing/conflicting scores reported separately'])
    if suffix:out.update(supersedes_receipt_sha256=a.sha(DEST/'receipt.json'),correction=d['correction'],prior_receipt_valid=False)
    a.write(DEST/f'receipt{suffix}.json',out);a.write(REVIEW/f'receipt{suffix}.json',out)
    print(json.dumps(dict(population=out['population'],summary=summary,predicates=out['predicates']),indent=2))


def amend():
    old=json.loads((DEST/'declaration.json').read_text())
    d=dict(old,version=2,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        supersedes_declaration_sha256=a.sha(DEST/'declaration.json'),prior_receipt_sha256=a.sha(DEST/'receipt.json'),
        correction='single-order reference must cancel/repost on target-size increases and decreases; prior generic rem<=target check accidentally retained the smaller order on increases. Layered partial retention applies only when nominal size decreases. No arm/threshold/latency/data change.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCE},inherited_decisions_sha256=a.sha(DEST/'decisions.parquet'))
    a.write(DEST/'declaration_v2.json',d);a.write(REVIEW/'declaration_v2.json',d)
    print('DECLARED v2 reference resize correction; prior result invalid')


if __name__=='__main__':{'declare':declare,'extract':extract,'run':run,'amend':amend}[sys.argv[1]]()
