"""Independent artifact-level checks for the consumed Nov8 timing audit."""
import copy
import datetime as dt
import json
import math
import numpy as np
import pandas as pd
from . import pf_fill_clock as c


def counts(times, lo, hi, tolerance):
    t=np.asarray(times);l=np.asarray(lo);u=np.asarray(hi)
    if not len(t) or t.shape!=l.shape or t.shape!=u.shape or not np.isfinite([t,l,u]).all() or (l>u).any():
        raise ValueError('invalid population/bounds')
    before=t<l-tolerance;after=t>u+tolerance;bad=before|after
    return {k:int(v) for k,v in [('before_place',before.sum()),('after_terminal_or_notice',after.sum()),
                                ('compatible',(~bad).sum())] if v},bad


def verify_clocks(q,report):
    for name,expected in report['checks'].items():
        clock,tol=name.split('/tol');tol=float(tol)
        times=q[clock] if clock in ['J','X'] else q.Z-float(clock.removeprefix('earlier_').removesuffix('ms'))
        actual,bad=counts(times,q.lower,q.upper,tol)
        if actual!=expected['leg_status']:raise ValueError('clock counts differ from artifact')
        if len(q)!=expected['legs'] or q.order_hash.nunique()!=expected['orders']:raise ValueError('clock population differs')
        if int(q.loc[bad,'order_hash'].nunique())!=expected['violating_orders']:raise ValueError('action count differs')
        if abs(q.loc[bad,'shares'].sum()-expected['violating_shares'])>1e-8:raise ValueError('violating size differs')
        if expected['all_compatible']!=bool(not bad.any()):raise ValueError('predicate differs')


def main():
    c.new_paths('verification.json')
    actual,_=counts([90.,150.,210.],[100.]*3,[200.]*3,0)
    positive=actual==dict(before_place=1,after_terminal_or_notice=1,compatible=1)
    bad_refused=False
    try:counts([float('nan')],[100.],[200.],0)
    except ValueError:bad_refused=True
    if not positive or not bad_refused:raise ValueError('control failed')
    r=json.loads((c.DEST/'report.json').read_text());s=json.loads((c.DEST/'semantics_report.json').read_text())
    for f,h in r['artifacts'].items():
        if c.sha(c.DEST/f)!=h:raise ValueError('original artifact changed')
    if c.sha(c.DEST/'status_clock_cases.parquet')!=s['artifact_sha']:raise ValueError('status artifact changed')
    for filename in ['receipt.json','semantics_receipt.json']:
        src='report.json' if filename=='receipt.json' else 'semantics_report.json'
        if c.sha(c.DEST/src)!=c.sha(c.HERE.parent/'reviews/fill_clock_20261109'/filename):raise ValueError('review copy differs')
    R=pd.read_parquet(c.DEST/'clock_cases.parquet');q=R[R.status=='eligible']
    verify_clocks(q,r)
    corrupted=copy.deepcopy(r);corrupted['checks']['J/tol10']['leg_status']['compatible']+=1
    corruption_refused=False
    try:verify_clocks(q,corrupted)
    except ValueError:corruption_refused=True
    if not corruption_refused:raise ValueError('corrupted result was accepted')
    O=pd.read_parquet(c.DEST/'orders.parquet');M=pd.read_parquet(c.DEST/'own_rest_legs.parquet')
    signed=O.dropna(subset=['order_hash']).set_index('order_hash')
    diff=(signed.core_filled-M.groupby('order_hash').shares.sum().reindex(signed.index,fill_value=0)).abs()
    if int((diff>1e-6).sum())!=r['live_quantity_reconciliation']['orders_differing_over_microshare']:
        raise ValueError('quantity reconciliation differs')
    if len(R)!=99 or len(q)!=93 or len(O)!=42064:raise ValueError('declared cohort changed')
    E=pd.read_parquet(c.DEST/'fill_events.parquet');S=pd.read_parquet(c.DEST/'status_clock_cases.parquet')
    matched=E[E.status=='Matched'].groupby(['order_hash','trade_id']).reported_match_ms.agg(list)
    confirmed=E[E.status=='Confirmed'].groupby(['order_hash','trade_id']).reported_match_ms.agg(list)
    for x in S.itertuples():
        k=(x.order_hash,x.trade_id);ct=confirmed.loc[k];mt=matched.loc[k] if k in matched.index else []
        if x.rest_equals_any_confirmed_second!=(x.exec_s*1000 in ct):raise ValueError('confirmation timestamp identity differs')
        if x.rest_equals_any_matched_second!=(x.exec_s*1000 in mt):raise ValueError('matched timestamp identity differs')
        if len(set(mt))==1 and x.rest_minus_matched_ms!=x.exec_s*1000-mt[0]:raise ValueError('timestamp drift differs')
    for run,expected in s['constant_shift_by_run'].items():
        g=q[q.run==run];lo=float((g.Z-g.upper).max());hi=float((g.Z-g.lower).min())
        if lo!=expected['lower_ms'] or hi!=expected['upper_ms'] or (lo-10<=hi+10)!=expected['exists_at10ms']:
            raise ValueError('per-run clock interval differs')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        verifier_sha=c.sha(c.HERE/'pf_fill_clock_verify.py'),report_sha=c.sha(c.DEST/'report.json'),
        semantics_sha=c.sha(c.DEST/'semantics_report.json'),artifact_hashes_verified=len(r['artifacts'])+1,
        clocks_verified=len(r['checks']),order_actions_verified=len(O),exact_status_histories_verified=len(S),
        controls=dict(before_inside_after=positive,nonfinite_refusal=bad_refused,corrupted_report_refusal=corruption_refused))
    c.write(c.DEST/'verification.json',out)
    c.write(c.HERE.parent/'reviews/fill_clock_20261109/verification.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
