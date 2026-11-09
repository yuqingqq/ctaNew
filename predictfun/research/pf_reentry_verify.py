"""Independent checks of paired window and reference-action re-entry artifacts."""
import copy
import datetime as dt
import json
import math
import subprocess
import pandas as pd
from . import pf_reentry_study as s
from .pf_reentry import ReentryGate


def near(x,y):
    if not math.isclose(float(x),float(y),rel_tol=1e-12,abs_tol=1e-6):raise ValueError('numeric mismatch')


def check_counts(m,summary):
    if set(m.arm)!=set(summary):raise ValueError('arm population differs')
    b=summary['reference']
    for arm,g in m.groupby('arm'):
        q=summary[arm]
        if len(g)!=q['windows']:raise ValueError('window count differs')
        for k in s.SUMS:near(g[k].sum(),q[k])
        if q['requests']!=g.places.sum()+g.cancels.sum():raise ValueError('request count differs')
        near(q['aligned_pct_of_reference_intent'],100*q['aligned_lane_ms']/q['reference_desired_lane_ms'])
        near(q['mean_oldest_quote_ms'],q['oldest_lane_age_ms2']/q['quoted_lane_ms'])
        near(q['mean_reserved_shares_both_sides'],q['reserved_share_ms']/(60000*len(g)))
        near(q['request_change_pct'],100*(q['requests']/b['requests']-1))
        near(q['aligned_change_pct'],100*(q['aligned_lane_ms']/b['aligned_lane_ms']-1))
        near(q['desired_time_retained_pct'],100*q['desired_lane_ms']/b['desired_lane_ms'])
        near(q['reference_quoted_time_preserved_pct'],100*(1-q['reference_quoted_lost_ms']/b['aligned_lane_ms']))
        near(q['off_target_change_pct'],100*(q['elsewhere_lane_ms']/b['elsewhere_lane_ms']-1))
        computed=q['requests']<b['requests'] and q['aligned_lane_ms']>=b['aligned_lane_ms']
        if q['fewer_requests_and_no_aligned_time_loss']!=computed:raise ValueError('comparison predicate differs')


def main():
    if (s.DEST/'verification.json').exists():raise ValueError('verification exists')
    d=s.check_inputs();r=json.loads((s.DEST/'receipt.json').read_text())
    if d!=r['declaration']:raise ValueError('declaration copy differs')
    for p,h in r['artifacts'].items():
        if s.a.sha(s.DEST/p)!=h:raise ValueError('artifact changed')
    if s.a.sha(s.DEST/'receipt.json')!=s.a.sha(s.REVIEW/'receipt.json'):raise ValueError('review receipt differs')
    m=pd.read_parquet(s.DEST/'window_metrics.parquet');act=pd.read_parquet(s.DEST/'actions.parquet');status=pd.read_parquet(s.DEST/'window_status.parquet')
    if len(m)!=1320 or m.duplicated(['run','mid','arm']).any():raise ValueError('paired windows differ')
    if len(status)!=220 or not (status.status=='present').all() or status.decisions.sum()!=279059:raise ValueError('source population differs')
    if (m.peak_reserved_side>150+1e-9).any() or not m.target_subset.all():raise ValueError('budget/target violation')
    check_counts(m,r['summary'])
    ref=m[m.arm=='reference'].set_index(['run','mid'])
    prior=pd.read_parquet(s.OLD/'window_metrics.parquet');prior=prior[prior.arm=='incremental'].set_index(['run','mid'])
    pcov=pd.read_parquet(s.OLD/'coverage.parquet');pcov=pcov[pcov.arm=='incremental'].set_index(['run','mid'])
    pd.testing.assert_frame_equal(ref[s.AUDIT].sort_index(),prior[s.AUDIT].sort_index(),check_exact=True)
    cols=['aligned_lane_ms','elsewhere_lane_ms']
    pd.testing.assert_frame_equal(ref[cols].sort_index(),pcov[cols].sort_index(),check_exact=True)
    actioncols=['run','mid','action','lane','tick','start','end','end_reason','start_reason','rapid']
    ra=act[act.arm=='reference'].sort_values(['run','mid','action'])[actioncols].reset_index(drop=True)
    for arm,g in act.groupby('arm'):
        ga=g.sort_values(['run','mid','action'])[actioncols].reset_index(drop=True)
        pd.testing.assert_frame_equal(ra,ga,check_exact=True)
        if g.duplicated(['run','mid','action']).any():raise ValueError('duplicate reference action')
        eligible=g.admitted_at.notna()
        if (g.end<g.start).any() or (g.loc[eligible,'admitted_at']<g.loc[eligible,'start']).any() or (g.loc[eligible,'admitted_at']>g.loc[eligible,'end']).any():raise ValueError('noncausal admission')
        for (run,mid),a in g.groupby(['run','mid']):
            q=m[(m.run==run)&(m.mid==mid)&(m.arm==arm)].iloc[0]
            vals=dict(actions=len(a),admitted_actions=int(a.admitted_at.notna().sum()),withheld_actions=int(a.admitted_at.isna().sum()),
                delayed_actions=int((a.admitted_at>a.start).sum()),rapid_actions=int(a.rapid.sum()),
                withheld_rapid_actions=int((a.rapid&a.admitted_at.isna()).sum()))
            if any(q[k]!=v for k,v in vals.items()):raise ValueError('action statuses differ')
    for row in m.itertuples(index=False):
        b=ref.loc[(row.run,row.mid)]
        near(row.reference_desired_lane_ms,b.desired_lane_ms)
        near(row.aligned_lane_ms,b.aligned_lane_ms-row.reference_quoted_lost_ms+row.reference_unquoted_gained_ms)
        if row.desired_lane_ms>b.desired_lane_ms+1e-6 or row.aligned_lane_ms>b.desired_lane_ms+1e-6:raise ValueError('intent time exceeded')
        if min(row.reference_quoted_lost_ms,row.reference_unquoted_gained_ms)<-1e-6:raise ValueError('negative coverage loss/gain')
    complete=ra[ra.end_reason!='window_censored'];rapid=complete[complete.rapid]
    diag=r['reference_episode_diagnostics']
    for k,v in dict(actions=len(ra),complete_actions=len(complete),window_censored_actions=len(ra)-len(complete),
        rapid_actions=int(ra.rapid.sum()),rapid_complete_actions=len(rapid),complete_shorter_than40ms=int(((complete.end-complete.start)<40).sum()),
        rapid_complete_shorter_than40ms=int(((rapid.end-rapid.start)<40).sum())).items():
        if diag[k]!=v:raise ValueError('episode count differs')
    near(diag['complete_median_ms'],(complete.end-complete.start).median());near(diag['rapid_complete_median_ms'],(rapid.end-rapid.start).median())
    if diag['status_counts']!=ra.end_reason.value_counts().to_dict():raise ValueError('censoring counts differ')
    # Positive intervention control and a corrupted report that must be rejected.
    g=ReentryGate('reentry',25);w=[(50,5)]+[(None,0.)]*3
    g.apply(0,w);g.apply(10,[(None,0.)]*4)
    positive=g.apply(20,w)==[(None,0.)]*4 and g.apply(45,w)==w
    if not positive:raise ValueError('confirmation control failed')
    bad=copy.deepcopy(r['summary']);bad['cancel_ack']['requests']+=1;refused=False
    try:check_counts(m,bad)
    except ValueError:refused=True
    if not refused:raise ValueError('corrupted receipt accepted')
    by_run={run:s.summarize(g) for run,g in m.groupby('run')}
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.a.HERE,text=True).strip(),
        verifier_sha256=s.a.sha(s.a.HERE/'pf_reentry_verify.py'),receipt_sha256=s.a.sha(s.DEST/'receipt.json'),
        paired_windows=220,arms=6,window_rows=len(m),reference_actions=len(ra),action_rows=len(act),
        source_decisions=279059,excluded_windows=0,reference_metrics_checked=len(s.AUDIT)+2,
        controls=dict(intervention_positive=positive,corrupt_receipt_refused=refused,exact_reference_parity=True,exact_action_population=True),
        by_run=by_run,limitation='mechanical comparison only; one consumed partial day, no profit or prediction-skill claim')
    s.a.write(s.DEST/'verification.json',out);s.a.write(s.REVIEW/'verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='by_run'},indent=2))


if __name__=='__main__':main()
