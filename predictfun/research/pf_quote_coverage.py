"""Price-aligned occupancy audit of the already declared no-fill sizing policies.

Separates time at the current desired price from stale/cancel-pending occupancy.
No new arms, thresholds, timestamps or performance claims.
"""
import datetime as dt
import json
import subprocess
import sys
import unittest
import pandas as pd
from . import pf_quote_incremental_study as follow
from .pf_quote_capture import QuotePlanner

a=follow.s.a
DEST=follow.DEST
REVIEW=follow.REVIEW


def coverage(planner,start,end,want):
    if end<start or len(want)!=4:raise ValueError('bad coverage span/target')
    aligned=elsewhere=0.
    for lane,(tick,size) in enumerate(want):
        spans=[[],[]]
        for o in planner.orders:
            if o.lane!=lane or o.rem<=1e-9:continue
            lo=max(start,o.accepted);hi=min(end,o.cancel_at if o.cancel_at is not None else end)
            if hi>lo:spans[int(not (size>0 and tick==o.tick))].append((lo,hi))
        def union(parts):
            total=0.;right=-float('inf')
            for lo,hi in sorted(parts):
                total+=max(0.,hi-max(lo,right));right=max(right,hi)
            return total
        aligned+=union(spans[0]);elsewhere+=union(spans[1])
    return aligned,elsewhere


class CoverageControls(unittest.TestCase):
    def test_price_identity_and_pull(self):
        p=QuotePlanner(layered=False);w=[(50,5)]+[(None,0)]*3
        p.plan(0,w);self.assertEqual(coverage(p,0,100,w),(60,0))
        w[0]=(51,5);p.plan(100,w)
        self.assertEqual(coverage(p,100,200,w),(60,22))
        off=[(None,0)]*4;p.plan(200,off)
        self.assertEqual(coverage(p,200,300,off),(0,22))
    def test_refusal_and_no_double_count(self):
        p=QuotePlanner();w=[(50,25)]+[(None,0)]*3;p.plan(0,w)
        self.assertEqual(coverage(p,0,100,w),(60,0))
        with self.assertRaises(ValueError):coverage(p,100,99,w)
        with self.assertRaises(ValueError):coverage(p,0,100,w[:3])


def declare():
    if (DEST/'coverage_declaration.json').exists():raise ValueError('already declared')
    r=json.loads((DEST/'receipt.json').read_text())
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in ['pf_quote_coverage.py','pf_quote_capture.py','pf_quote_incremental_study.py']},
        receipt_sha256=a.sha(DEST/'receipt.json'),decisions_sha256=a.sha(DEST/'decisions.parquet'),
        arms=follow.ARMS,population=r['population'],multiplicity=7,
        design='same three sizing policies; union of active intervals at the exact current desired price, versus off-target/pulled occupancy; no new arms or selection threshold',
        known_result='incremental has4.52% fewer requests and3.45% more total quote time; audit tests whether that extra time is aligned with desired prices',
        null='deterministic engineering diagnostic only; no profit/fill model, statistical test or interval')
    a.write(DEST/'coverage_declaration.json',out);a.write(REVIEW/'coverage_declaration.json',out)


def run():
    if (DEST/'coverage_receipt.json').exists():raise ValueError('refuse overwrite')
    d=json.loads((DEST/'coverage_declaration.json').read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('source changed')
    for p,k in [('receipt.json','receipt_sha256'),('decisions.parquet','decisions_sha256')]:
        if a.sha(DEST/p)!=d[k]:raise ValueError('input changed')
    controls=(DEST/'logs/coverage_controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing controls')
    ref=pd.read_parquet(DEST/'window_metrics.parquet').set_index(['run','mid','arm'])
    D=pd.read_parquet(DEST/'decisions.parquet');scope=json.loads((a.RP/'scope.json').read_text())
    parent=json.loads((DEST/'receipt.json').read_text())
    if a.sha(DEST/'window_metrics.parquet')!=parent['artifacts']['window_metrics.parquet']:raise ValueError('reference artifact changed')
    if a.sha(a.RP/'scope.json')!=parent['declaration']['inputs'][str(a.RP/'scope.json')]:raise ValueError('scope changed')
    starts={(w['run'],w['market']):w['ws']*1000 for w in scope['windows']}
    result=[]
    for (run,mid),g in D.groupby(['run','mid'],sort=False):
        for arm,(layered,_,incremental) in follow.ARMS.items():
            p=QuotePlanner(layered=layered,incremental=incremental);last=starts[run,mid];end=last+60000
            want=[(None,0.)]*4;on=off=desired=0.
            for row in g[['t']+[v for lane in range(4) for v in (f'tick{lane}',f'size{lane}')]].to_numpy():
                now=row[0];x,y=coverage(p,last,now,want);on+=x;off+=y
                desired+=(now-last)*sum(q[0] is not None and q[1]>0 for q in want)
                want=[(None if row[1+2*i]<0 else int(row[1+2*i]),float(row[2+2*i])) for i in range(4)]
                p.plan(now,want);last=now
            x,y=coverage(p,last,end,want);on+=x;off+=y
            desired+=(end-last)*sum(q[0] is not None and q[1]>0 for q in want)
            old=ref.loc[(run,mid,arm)]
            if desired!=old.desired_lane_ms or any(p.stats[k]!=old[k] for k in p.stats):raise ValueError('original trace diverged')
            if on>desired+1e-6:raise ValueError('aligned time exceeds target time')
            result.append(dict(run=run,mid=mid,arm=arm,aligned_lane_ms=on,elsewhere_lane_ms=off,desired_lane_ms=desired))
    m=pd.DataFrame(result)
    if len(m)!=660:raise ValueError('population differs')
    m.to_parquet(DEST/'coverage.parquet',index=False)
    # Summarize the saved artifact, not just the producer's in-memory totals.
    m=pd.read_parquet(DEST/'coverage.parquet');summary={}
    for arm,g in m.groupby('arm'):
        q={k:float(g[k].sum()) for k in ['aligned_lane_ms','elsewhere_lane_ms','desired_lane_ms']}
        q['aligned_share_of_desired_pct']=100*q['aligned_lane_ms']/q['desired_lane_ms'];q['windows']=len(g)
        summary[arm]=q
    for arm,q in summary.items():
        q['aligned_change_vs_single_pct']=100*(q['aligned_lane_ms']/summary['single']['aligned_lane_ms']-1)
    r=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,source_controls=controls,
        summary=summary,artifact_sha256=a.sha(DEST/'coverage.parquet'),
        predicates=dict(original_order_counts_exact=True,original_desired_time_exact=True,
            incremental_more_aligned=summary['incremental']['aligned_lane_ms']>summary['single']['aligned_lane_ms']),
        limitation='time-weighted occupancy only; not queue rank, fills, profits or inventory risk. On/off-target occupancy can overlap during repricing.')
    a.write(DEST/'coverage_receipt.json',r);a.write(REVIEW/'coverage_receipt.json',r)
    print(json.dumps(r['summary'],indent=2))


if __name__=='__main__':{'declare':declare,'run':run}[sys.argv[1]]()
