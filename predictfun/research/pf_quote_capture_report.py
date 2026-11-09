"""Verify stored capture/coverage artifacts and summarize engineering deltas."""
import copy
import datetime as dt
import json
import math
import subprocess
import pandas as pd
from . import pf_quote_incremental_study as s
from .pf_quote_capture_verify import verify_sums


def check_coverage(frame,summary):
    if len(frame)!=660 or frame.duplicated(['run','mid','arm']).any():raise ValueError('coverage population differs')
    if (frame.aligned_lane_ms>frame.desired_lane_ms+1e-6).any():raise ValueError('coverage exceeds intent')
    for arm,g in frame.groupby('arm'):
        q=summary[arm]
        if len(g)!=q['windows']:raise ValueError('window count differs')
        for k in ['aligned_lane_ms','elsewhere_lane_ms','desired_lane_ms']:
            if float(g[k].sum())!=q[k]:raise ValueError('coverage count differs')
        pct=100*q['aligned_lane_ms']/q['desired_lane_ms']
        if not math.isclose(pct,q['aligned_share_of_desired_pct'],rel_tol=1e-12):raise ValueError('coverage ratio differs')
        change=100*(q['aligned_lane_ms']/summary['single']['aligned_lane_ms']-1)
        if not math.isclose(change,q['aligned_change_vs_single_pct'],abs_tol=1e-10):raise ValueError('coverage change differs')


def main():
    a=s.s.a;dest=s.DEST
    if (dest/'report.json').exists():raise ValueError('report exists')
    r=json.loads((dest/'receipt.json').read_text());c=json.loads((dest/'coverage_receipt.json').read_text())
    for obj in [r,c]:
        for p,h in obj['declaration']['sources'].items():
            if a.sha(a.HERE/p)!=h:raise ValueError('source differs')
    for p,h in r['artifacts'].items():
        if a.sha(dest/p)!=h:raise ValueError('artifact differs')
    if a.sha(dest/'coverage.parquet')!=c['artifact_sha256']:raise ValueError('coverage artifact differs')
    if a.sha(dest/'receipt.json')!=c['declaration']['receipt_sha256']:raise ValueError('reference receipt differs')
    for p in ['receipt.json','coverage_receipt.json']:
        if a.sha(dest/p)!=a.sha(s.REVIEW/p):raise ValueError('review copy differs')
    m=pd.read_parquet(dest/'window_metrics.parquet');cov=pd.read_parquet(dest/'coverage.parquet')
    verify_sums(m,r['summary']);check_coverage(cov,c['summary'])
    keys=['run','mid','arm']
    joined=m.merge(cov,on=keys,validate='one_to_one',suffixes=('','_coverage'))
    if len(joined)!=660 or not (joined.desired_lane_ms==joined.desired_lane_ms_coverage).all():raise ValueError('artifact identity differs')
    bad=copy.deepcopy(c['summary']);bad['incremental']['aligned_lane_ms']+=1;refused=False
    try:check_coverage(cov,bad)
    except ValueError:refused=True
    if not refused:raise ValueError('corrupted coverage accepted')
    def changes(g):
        q=g.groupby('arm').sum(numeric_only=True);b=q.loc['single'];n=q.loc['incremental']
        return dict(windows=len(g)//3,
            requests_pct=100*((n.places+n.cancels)/(b.places+b.cancels)-1),
            aligned_quote_time_pct=100*(n.aligned_lane_ms/b.aligned_lane_ms-1),
            active_share_time_pct=100*(n.share_ms/b.share_ms-1),
            mean_oldest_quote_age_pct=100*((n.oldest_lane_age_ms2/n.quoted_lane_ms)/(b.oldest_lane_age_ms2/b.quoted_lane_ms)-1),
            reserved_share_time_pct=100*(n.reserved_share_ms/b.reserved_share_ms-1),
            off_target_quote_time_pct=100*(n.elsewhere_lane_ms/b.elsewhere_lane_ms-1))
    req=m.assign(requests=m.places+m.cancels).pivot(index=['run','mid'],columns='arm',values='requests')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        report_source_sha256=a.sha(a.HERE/'pf_quote_capture_report.py'),
        receipt_sha256=a.sha(dest/'receipt.json'),coverage_receipt_sha256=a.sha(dest/'coverage_receipt.json'),
        population=r['population'],all_windows=changes(joined),by_run={run:changes(g) for run,g in joined.groupby('run')},
        incremental_request_windows=dict(fewer=int((req.incremental<req.single).sum()),equal=int((req.incremental==req.single).sum()),more=int((req.incremental>req.single).sum())),
        controls=dict(saved_summaries_reconciled=True,coverage_identities_reconciled=True,corrupt_coverage_refused=refused),
        limitation='same consumed partial day, engineering workload only; no economic validation or uncertainty interval')
    a.write(dest/'report.json',out);a.write(s.REVIEW/'report.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
