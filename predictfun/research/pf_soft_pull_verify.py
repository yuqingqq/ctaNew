"""Check saved populations, original targets, durations, labels and reported ratios."""
import copy
import datetime as dt
import json
import math
import subprocess
import pandas as pd
from . import pf_soft_pull_study as s


def compare_summary(m,reported):
    baseline=m[m.arm=='reference']
    b_requests=int(baseline.places.sum()+baseline.cancels.sum())
    for arm,g in m.groupby('arm'):
        r=reported[arm]
        if r['windows']!=len(g) or r['requests']!=int(g.places.sum()+g.cancels.sum()):raise ValueError('summary population/count differs')
        for k in s.SUMS:
            if not math.isclose(float(g[k].sum()),r[k],rel_tol=1e-12,abs_tol=1e-5):raise ValueError(f'summary total differs: {arm}/{k}')
        ratios=dict(request_change_pct=100*((g.places.sum()+g.cancels.sum())/b_requests-1),
            aligned_change_pct=100*(g.aligned_lane_ms.sum()/baseline.aligned_lane_ms.sum()-1),
            off_target_change_pct=100*(g.elsewhere_lane_ms.sum()/baseline.elsewhere_lane_ms.sum()-1),
            reserved_change_pct=100*(g.reserved_share_ms.sum()/baseline.reserved_share_ms.sum()-1),
            active_shares_change_pct=100*(g.share_ms.sum()/baseline.share_ms.sum()-1),
            mean_oldest_quote_ms=g.oldest_lane_age_ms2.sum()/g.quoted_lane_ms.sum())
        for k,v in ratios.items():
            if not math.isclose(v,r[k],rel_tol=1e-12,abs_tol=1e-8):raise ValueError('summary ratio differs')
        predicate=r['requests']<b_requests and g.aligned_lane_ms.sum()>=baseline.aligned_lane_ms.sum()
        if bool(predicate)!=r['fewer_requests_and_no_aligned_time_loss']:raise ValueError('summary predicate differs')


def main():
    a=s.a;d=s.check()
    if (s.DEST/'verification.json').exists():raise ValueError('verification exists')
    receipt=json.loads((s.DEST/'receipt.json').read_text())
    for p,h in receipt['artifacts'].items():
        if a.sha(s.DEST/p)!=h:raise ValueError('saved artifact changed')
    m=pd.read_parquet(s.DEST/'window_metrics.parquet');act=pd.read_parquet(s.DEST/'actions.parquet');stat=pd.read_parquet(s.DEST/'window_status.parquet')
    scope=json.loads((a.RP/'scope.json').read_text());keys={(w['run'],w['market']) for w in scope['windows']}
    if len(stat)!=220 or set(zip(stat.run,stat.mid))!=keys or not (stat.status=='present').all():raise ValueError('window population differs')
    if int(stat.decisions.sum())!=279059 or len(m)!=220*len(s.ARMS):raise ValueError('decision/arm population differs')
    if m.duplicated(['run','mid','arm']).any() or act.duplicated(['run','mid','arm','action']).any():raise ValueError('duplicate action/window')
    ref=act[act.arm=='reference'].sort_values(['run','mid','action']).reset_index(drop=True)
    identity=['run','mid','action','lane','start','old_tick','v','midpoint','input_index','spread']
    for arm in s.ARMS:
        aa=act[act.arm==arm].sort_values(['run','mid','action']).reset_index(drop=True)
        pd.testing.assert_frame_equal(aa[identity],ref[identity])
        for h in [1000,5000]:
            pd.testing.assert_frame_equal(aa[[f'h{h}_status',f'h{h}_move_c']],ref[[f'h{h}_status',f'h{h}_move_c']])
        if len(aa)!=int(m[m.arm==arm].actions.sum()):raise ValueError('action count differs')
        retained=aa[aa.status=='retained'];delta=s.ARMS[arm]
        if not ((retained.v>.0492-delta)&(retained.v<=.0492)&(retained.accepted<=retained.start)).all():raise ValueError('retention band/acceptance differs')
        if not ((retained.retained_ms>=0)&(retained.retained_ms<=100)&((retained.end-retained.start)==retained.retained_ms)).all():raise ValueError('hold duration differs')
        deadlines=retained[retained.end_reason=='deadline']
        if not (deadlines.retained_ms==100).all():raise ValueError('deadline delayed')
        if len(retained)!=int(m[m.arm==arm].retained_actions.sum()):raise ValueError('retained population differs')
        if int((aa.end_reason=='same_price_reopen').sum())!=int(m[m.arm==arm].same_price_reopens.sum()):raise ValueError('bridge count differs')
    # Regenerate neutral reference pull identities from the native artifact.
    generated=[]
    for run in s.n.RUNS:
        source=pd.read_parquet(s.n.DEST/f'states_{run}.parquet');source=source[source.rel_s<60]
        books=s.load_books(run)
        for mid,g in source.groupby('mid',sort=False):
            before=[False,False];oldtick=[None,None];i=0
            for row in g.to_dict('records'):
                for lane in [0,1]:
                    on=row[f'raw_tick{lane}']>=0 and row[f'raw_size{lane}']>0
                    if before[lane] and not on:
                        generated.append(dict(run=run,mid=mid,action=i,lane=lane,start=row['t'],old_tick=oldtick[lane],input_index=row['input_index']));i+=1
                    before[lane]=on;oldtick[lane]=row[f'raw_tick{lane}']
        for row in ref[ref.run==run].to_dict('records'):
            for h in [1000,5000]:
                status,value=s.price_label(row,books[row['mid']],h)
                if status!=row[f'h{h}_status']:raise ValueError('saved price status differs')
                old=row[f'h{h}_move_c']
                if not ((math.isnan(value) and math.isnan(old)) or value==old):raise ValueError('saved price label differs')
    generated=pd.DataFrame(generated).sort_values(['run','mid','action']).reset_index(drop=True)
    pd.testing.assert_frame_equal(generated,ref[generated.columns],check_dtype=False)
    # Reconcile the original baseline's saved metrics at the named artifact.
    prior=pd.read_parquet(s.OLD/'window_metrics.parquet');prior=prior[prior.arm=='incremental'].set_index(['run','mid'])
    priorcov=pd.read_parquet(s.OLD/'coverage.parquet');priorcov=priorcov[priorcov.arm=='incremental'].set_index(['run','mid'])
    for row in m[m.arm=='reference'].to_dict('records'):
        key=(row['run'],row['mid'])
        for k in s.AUDIT:
            if not math.isclose(row[k],prior.loc[key,k],rel_tol=1e-11,abs_tol=1e-4):raise ValueError('old baseline differs')
        for k in ['aligned_lane_ms','elsewhere_lane_ms']:
            if not math.isclose(row[k],priorcov.loc[key,k],rel_tol=1e-11,abs_tol=1e-4):raise ValueError('old aligned baseline differs')
    compare_summary(m,receipt['summary'])
    for run,g in m.groupby('run'):compare_summary(g,receipt['by_run'][run])
    # More aligned time must equal gained time minus lost time on the same spans.
    by=m.set_index(['run','mid','arm'])
    for run,mid in keys:
        base=by.loc[(run,mid,'reference')]
        for arm in s.ARMS:
            q=by.loc[(run,mid,arm)]
            if not math.isclose(q.aligned_lane_ms-base.aligned_lane_ms,q.reference_unquoted_gained_ms-q.reference_quoted_lost_ms,abs_tol=1e-4,rel_tol=1e-11):raise ValueError('coverage identity differs')
    diagnostic=s.diagnose(act)
    if diagnostic!=receipt['diagnostics']:raise ValueError('diagnostic counts/500 matched draws differ')
    bad=copy.deepcopy(receipt['summary']);bad['reference']['requests']+=1
    try:compare_summary(m,bad)
    except ValueError:refused=True
    else:raise ValueError('corrupt count accepted')
    population=receipt['population']
    if population!=dict(windows=220,decisions=279059,actions=len(ref),excluded_windows=0):raise ValueError('receipt population differs')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        verifier_sha256=a.sha(a.HERE/'pf_soft_pull_verify.py'),receipt_sha256=a.sha(s.DEST/'receipt.json'),
        rows=dict(window_arm=len(m),action_arm=len(act),neutral_pull_actions=len(ref)),
        original_targets_and_baseline_reconciled=True,saved_price_labels_reconciled=True,matched_draws_reconciled=500,
        controls=dict(valid_artifacts_accepted=True,corrupt_request_count_refused=refused),
        limitation='artifact and mechanical checks; not economic validation')
    a.write(s.DEST/'verification.json',out);a.write(s.REVIEW/'verification.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
