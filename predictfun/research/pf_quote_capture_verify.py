"""Verify quote-workload receipts against artifacts and independent mechanism controls."""
import copy
import datetime as dt
import json
import math
import pandas as pd
from . import pf_quote_capture_study as s
from .pf_quote_capture import QuotePlanner


def verify_sums(metrics,summary):
    for arm,g in metrics.groupby('arm',sort=False):
        r=summary[arm]
        if len(g)!=r['windows']:raise ValueError('window population differs')
        for k in ['places','cancels','base_resize_cancels','kept_base_size_switches','share_ms','quoted_lane_ms',
                  'desired_lane_ms','oldest_lane_age_ms2','reserved_share_ms','refused_capacity','refused_inflight']:
            if not math.isclose(float(g[k].sum()),r[k],rel_tol=1e-12,abs_tol=1e-6):raise ValueError('metric differs: '+k)
        if r['requests']!=int(g.places.sum()+g.cancels.sum()):raise ValueError('request count differs')


def main():
    if (s.DEST/'verification.json').exists():raise ValueError('verification exists')
    p=QuotePlanner(layered=False,grace_ms=0);q=QuotePlanner(layered=True,grace_ms=0)
    for at,size in [(0,5),(100,25),(200,5)]:
        want=[(50,size),(None,0.),(None,0.),(None,0.)];p.plan(at,want);q.plan(at,want)
    positive=len(p.commands)==5 and len(q.commands)==3
    if not positive:raise ValueError('resize mechanism control failed')
    r=json.loads((s.DEST/'receipt_v2.json').read_text())
    for path,h in r['declaration']['sources'].items():
        if s.a.sha(s.a.HERE/path)!=h:raise ValueError('declared source changed')
    for path,h in r['artifacts'].items():
        if s.a.sha(s.DEST/path)!=h:raise ValueError('declared artifact changed')
    if s.a.sha(s.DEST/'receipt_v2.json')!=s.a.sha(s.REVIEW/'receipt_v2.json'):raise ValueError('review copy changed')
    if r['supersedes_receipt_sha256']!=s.a.sha(s.DEST/'receipt.json') or r['prior_receipt_valid'] is not False:
        raise ValueError('correction provenance missing')
    m=pd.read_parquet(s.DEST/'window_metrics_v2.parquet');verify_sums(m,r['summary'])
    bad=copy.deepcopy(r['summary']);bad['single']['requests']+=1;refused=False
    try:verify_sums(m,bad)
    except ValueError:refused=True
    if not refused:raise ValueError('corrupted count accepted')
    if m.duplicated(['run','mid','arm']).any() or len(m)!=220*6:raise ValueError('paired population differs')
    if (m.peak_reserved_side>150+1e-9).any() or not m.target_subset.all():raise ValueError('budget/subset mismatch')
    wanted=m.pivot(index=['run','mid'],columns='arm',values='desired_lane_ms')
    if not (wanted.layered==wanted.single).all():raise ValueError('layered target coverage differs')
    if (wanted.drop(columns='single').sub(wanted.single,axis=0)>1e-6).any().any():raise ValueError('buffer adds target exposure')
    D=pd.read_parquet(s.DEST/'decisions.parquet')
    n_wanted=sum(int(((D[f'tick{i}']>=0)&(D[f'size{i}']>0)).sum()) for i in range(4))
    diag=r['source_score_diagnostics']
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),verifier_sha256=s.a.sha(s.a.HERE/'pf_quote_capture_verify.py'),
        receipt_sha256=s.a.sha(s.DEST/'receipt_v2.json'),verified_windows=220,verified_arms=6,verified_metrics_rows=len(m),
        controls=dict(single_resize_and_layered_retention=positive,corrupted_count_refused=refused),
        source_score_population=dict(recorded_wanted_lane_updates=n_wanted,**diag,
            missing_pct=100*diag['reconstructed_score_missing']/n_wanted,
            conflicts_with_original_pull_line_pct=100*diag['reconstructed_score_vs_live_conflict']/n_wanted),
        limitations='engineering workload only; quote residence/coverage cannot establish fills or profit')
    s.a.write(s.DEST/'verification.json',out);s.a.write(s.REVIEW/'verification.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
