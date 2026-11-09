"""Two predeclared rounds of the quote-capture/inventory optimization loop."""
import collections
import datetime as dt
import json
import math
import subprocess
import sys
import pandas as pd
from . import pf_soft_pull_study as old
from . import pf_native_inventory_events as native_inventory
from .pf_inventory_grace import InventoryGrace,exposure
from .pf_quote_capture import QuotePlanner
from .pf_soft_pull import lanes,OFF
from .pf_quote_coverage import coverage
from .pf_reentry_study import aligned_intervals,difference_time

a=old.a
DEST=a.ROOT/'data/predict_fun/research/optimization_loop_20261109'
REVIEW=a.HERE.parent/'reviews/optimization_loop_20261109'
ARMS=dict(reference=(0.,'all',100.),all100=(.01,'all',100.),individual100=(.01,'individual',100.),reserved100=(.01,'reserved',100.),
    all50=(.01,'all',50.),all25=(.01,'all',25.))
LATENCIES=dict(base=dict(place_ms=40.,cancel_ms=22.,pending_cancel_ms=95.),
    faster=dict(place_ms=20.,cancel_ms=10.,pending_cancel_ms=50.),
    slower=dict(place_ms=80.,cancel_ms=50.,pending_cancel_ms=160.))
ROUNDS=dict(inventory=dict(arms=['reference','all100','individual100','reserved100'],latencies=['base'],new_policies=2,cumulative_policies=18),
    duration_latency=dict(arms=list(ARMS),latencies=list(LATENCIES),new_policies=2,cumulative_policies=20))
RISK=['off_reducing_share_ms','off_other_share_ms','active_range_extra_share_ms','reserved_range_extra_share_ms','inventory_known_ms','inventory_unobserved_ms']
SOURCES=['pf_inventory_grace.py','pf_inventory_grace_study.py','pf_inventory_grace_verify.py','tests/test_inventory_grace.py',
    'pf_soft_pull.py','pf_soft_pull_study.py','pf_quote_capture.py','pf_quote_coverage.py','pf_reentry_study.py']


def declare(round_name):
    spec=ROUNDS[round_name];dest=DEST/round_name
    if dest.exists():raise ValueError('round exists')
    (dest/'logs').mkdir(parents=True)
    parent=json.loads((old.DEST/'receipt.json').read_text())
    paths=[old.DEST/'receipt.json',old.DEST/'verification.json',old.DEST/'window_metrics.parquet',old.DEST/'actions.parquet',a.RP/'scope.json']
    paths.extend(old.n.DEST/f'states_{r}.parquet' for r in old.n.RUNS)
    paths.extend(old.grid.DEST/f'probes_{r}.parquet' for r in old.n.RUNS)
    paths.extend(native_inventory.DEST/f'inventory_{r}.parquet' for r in old.n.RUNS)
    paths.extend(native_inventory.DEST/r/'verification.json' for r in old.n.RUNS)
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),round=round_name,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCES},inputs={str(p):a.sha(p) for p in paths},
        population=parent['population'],scope='same220 consumed Nov8 windows, first60s; all original decisions and touch-pull actions, never fill selected',
        arms={arm:ARMS[arm] for arm in spec['arms']},latencies={k:LATENCIES[k] for k in spec['latencies']},
        primary='reserved100' if round_name=='inventory' else 'all50',
        policy='original .01-cent soft exit and accepted-small-touch conditions unchanged; individual requires reducing side and remaining<=abs(recorded net); reserved additionally requires total same-side planner reservations<=abs(net). Pending creates/cancels and confirmation+1s grace count. Reevaluate during grace and after planning other lanes; refuse at flat. No original quote-on gate or size change.',
        duration='all50/all25 change only maximum grace; all100/individual100/reserved100 remain100ms. Native lower-line/hard changes can cancel sooner.',
        inventory='recorded native net, not simulated inventory; no fills fed to hypothetical orders; standard planner capacity still uses its original baseline assumptions. This tests conditional selection/exposure, not realized risk or no-flip safety.',
        reservation='total own hypothetical same-side remaining shares with sent<=now and release_at>now, including retained ticket once; opposite orders never net against it; expiry filtered before planner cleanup',
        measures='old request and original desired-price coverage metrics; off-target share-time split by current recorded inventory direction; worst-case active/reserved inventory interval expansion integrated exactly at order lifecycle events; never equate these with PnL or variance',
        range='[recorded net - same-side down amount, recorded net + up amount]; metric max(abs(endpoints))-abs(net), floored0; no cross-side netting. First native observation starts the inventory clock; earlier interval has an explicit unknown status.',
        probe_clock='consume native net changes at exact post-Filled knowledge events and every due probe, even without held orders; inventory-aware arms also cancel immediately at these notices; unrestricted reference arms keep their prior decision/grid clock for parity. Exact hold deadline inside event gaps.',
        null='deterministic paired mechanics only, no statistical null/skill/economic claim and no confidence interval; latency grid is assumed stress scenarios, not estimated venue quantiles',
        comparisons='each latency compared with its own reference; base reference and all100 reproduce prior saved artifacts; duration round repeats all inventory base rows; report lower reservation use separately from active exposure',
        multiplicity=dict(new_variants=spec['new_policies'],cumulative_distinct_policies=spec['cumulative_policies'],latency_scenarios=len(spec['latencies'])),
        controls='reducing positive; flat/wrong-side/overshoot/pending-cancel/new-create refusal; released reservation boundary; inventory reversal during grace; post-plan budget recheck; exact exposure intervals and no opposite-side netting; old-arm parity and corrupt-artifact refusal',
        no_live_change=True,no_protected_dates_read=True,no_fill_clock_bypass=True,
        limitations=parent['limitations']+['Rust A2 paired-order engine not integrated; B4 remains separate','inventory restriction deliberately prevents some otherwise profitable round trips; no value ranking possible without fills'])
    a.write(dest/'declaration.json',d);a.write(REVIEW/round_name/'declaration.json',d)


def check(round_name):
    d=json.loads((DEST/round_name/'declaration.json').read_text())
    if d['round']!=round_name:raise ValueError('round identity differs')
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError(f'source changed {p}')
    for p,h in d['inputs'].items():
        if a.sha(p)!=h:raise ValueError(f'input changed {p}')
    return d


def run_window(actual,probes,inventory,start,arms,latency):
    end=start+60000;last=start;raw=list(OFF);net=None
    states={arm:dict(p=QuotePlanner(incremental=True,**LATENCIES[latency]),gate=InventoryGrace(*ARMS[arm]),total=collections.Counter(),peak=0.) for arm in arms}
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
    for row in events.to_dict('records'):
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
    integrate(end);metrics=[];actions=[];checks=[]
    for arm,s in states.items():
        p=s['p'];aa=s['gate'].finish(end,p);v=s['total']
        v.update(actions=len(aa),retained_actions=sum(x['status']=='retained' for x in aa),same_price_reopens=sum(x['end_reason']=='same_price_reopen' for x in aa),retained_ms=sum(x['retained_ms'] for x in aa))
        rec=dict(arm=arm,latency=latency,decisions=len(actual),**dict(v),**p.stats,**{'refused_'+k:z for k,z in p.refused.items()},peak_reserved_side=s['peak'])
        for k in old.SUMS+RISK:rec.setdefault(k,0.)
        if s['peak']>150+1e-8:raise ValueError('side cap exceeded')
        metrics.append(rec);actions.extend(dict(arm=arm,latency=latency,**x) for x in aa);checks.extend(dict(arm=arm,latency=latency,**x) for x in s['gate'].checks)
    return metrics,actions,checks


def summarize(m):
    out={}
    for latency,lm in m.groupby('latency',sort=False):
        by=old.summarize(lm);base=lm[lm.arm=='reference']
        for arm,am in lm.groupby('arm',sort=False):
            q=by[arm]
            for k in RISK:q[k]=float(am[k].sum())
            for k in RISK[:4]:
                b=float(base[k].sum());q[k+'_change']=q[k]-b;q[k+'_change_pct']=100*(q[k]/b-1) if b else None
            q['aligned_gain_ms']=q['aligned_lane_ms']-by['reference']['aligned_lane_ms']
            q['extra_off_target_share_ms']=sum(q[k] for k in RISK[:2])-sum(by['reference'].get(k,float(base[k].sum())) for k in RISK[:2])
            q['quote_gain_per_extra_off_share_ms']=q['aligned_gain_ms']/q['extra_off_target_share_ms'] if q['extra_off_target_share_ms']>0 else None
        out[latency]=by
    return out


def baseline_parity(m,actions):
    prior=pd.read_parquet(old.DEST/'window_metrics.parquet').set_index(['run','mid','arm'])
    aa=pd.read_parquet(old.DEST/'actions.parquet');identity=['run','mid','action','lane','start','old_tick','v','midpoint','input_index','spread','status','end','end_reason','retained_ms']
    for arm,oldarm in [('reference','reference'),('all100','exit01')]:
        subset=m[(m.latency=='base')&(m.arm==arm)]
        for row in subset.to_dict('records'):
            prev=prior.loc[(row['run'],row['mid'],oldarm)]
            for k in old.SUMS+['decisions','peak_reserved_side']:
                if not math.isclose(row[k],prev[k],abs_tol=1e-4,rel_tol=1e-11):raise ValueError(f'old arm differs {arm}/{k}')
        left=actions[(actions.latency=='base')&(actions.arm==arm)].sort_values(['run','mid','action'])[identity].reset_index(drop=True)
        right=aa[aa.arm==oldarm].sort_values(['run','mid','action'])[identity].reset_index(drop=True)
        pd.testing.assert_frame_equal(left,right,check_dtype=False)


def run(round_name):
    d=check(round_name);dest=DEST/round_name;spec=ROUNDS[round_name]
    if (dest/'window_metrics.parquet').exists():raise ValueError('refuse overwrite')
    controls=(dest/'logs/controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('controls missing')
    scope=json.loads((a.RP/'scope.json').read_text());metrics=[];actions=[];checks=[];statuses=[]
    for run in old.n.RUNS:
        actual=pd.read_parquet(old.n.DEST/f'states_{run}.parquet');actual=actual[actual.rel_s<60]
        probes=pd.read_parquet(old.grid.DEST/f'probes_{run}.parquet');probes=probes[probes.rel_s<60]
        inventory=pd.read_parquet(native_inventory.DEST/f'inventory_{run}.parquet');inventory=inventory[inventory.rel_s<60]
        ai=actual.groupby('mid').indices;pi=probes.groupby('mid').indices
        ii=inventory.groupby('mid').indices
        for w in [w for w in scope['windows'] if w['run']==run]:
            mid=w['market']
            if mid not in ai or mid not in pi:raise ValueError('missing window')
            for latency in spec['latencies']:
                mm,aa,cc=run_window(actual.iloc[ai[mid]],probes.iloc[pi[mid]],inventory.iloc[ii.get(mid,[])],w['ws']*1000,spec['arms'],latency)
                for target,rr in [(metrics,mm),(actions,aa),(checks,cc)]:target.extend(dict(run=run,mid=mid,**x) for x in rr)
            statuses.append(dict(run=run,mid=mid,status='present',decisions=len(ai[mid]),probes=len(pi[mid]),inventory_notices=len(ii.get(mid,[]))))
            if len(statuses)%20==0:print('DONE',round_name,len(statuses),'windows',flush=True)
        del actual,probes,inventory
    m=pd.DataFrame(metrics);act=pd.DataFrame(actions);baseline_parity(m,act)
    for name,rr in [('window_metrics',metrics),('actions',actions),('checks',checks),('window_status',statuses)]:pd.DataFrame(rr).to_parquet(dest/f'{name}.parquet',index=False)
    m=pd.read_parquet(dest/'window_metrics.parquet');act=pd.read_parquet(dest/'actions.parquet')
    counts={}
    for (latency,arm),g in act.groupby(['latency','arm']):counts[f'{latency}/{arm}']=dict(n=len(g),statuses=g.status.value_counts().to_dict(),ends=g.end_reason.value_counts().to_dict())
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,controls=controls,
        population=dict(windows=len(statuses),decisions=sum(x['decisions'] for x in statuses),neutral_actions=int(((act.latency=='base')&(act.arm=='reference')).sum()),excluded_windows=0),
        summary=summarize(m),by_run={r:summarize(g) for r,g in m.groupby('run')},action_statuses=counts,
        artifacts={p.name:a.sha(p) for p in dest.glob('*.parquet')},limitations=d['limitations'])
    a.write(dest/'receipt.json',out);a.write(REVIEW/round_name/'receipt.json',out)
    print(json.dumps({l:{arm:{k:q[k] for k in ['requests','request_change_pct','aligned_change_pct','off_target_change_pct','active_shares_change_pct','reserved_change_pct','retained_actions','off_other_share_ms_change_pct','active_range_extra_share_ms_change_pct']} for arm,q in v.items()} for l,v in out['summary'].items()},indent=2))


if __name__=='__main__':{'declare':declare,'run':run}[sys.argv[1]](sys.argv[2])
