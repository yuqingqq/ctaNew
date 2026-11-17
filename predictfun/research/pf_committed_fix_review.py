"""Review b6eaa7c2 with a synthetic multi-step risk-set exhaustion case.

No market tape or performance claim. Five synthetic days x twenty markets x
twelve candidate orders, all same side/phase and same value per filled share.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import pf_keep_pull as KP
import pf_queue_place_ledger as QL

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'predictfun/reviews/committed_fix_review_20261117'


def matched_gate(actual, null, unmatched):
    if any(u != 0 for u in unmatched) or any(not np.array_equal(x['n'], actual['n']) for x in null):
        raise ValueError('unmatched participation cannot support a matched claim')


def main():
    assert not QL._l5_cases()
    day = np.repeat(np.arange(5), 20*12)
    market = np.tile(np.repeat(np.arange(20), 12), 5)
    cell = day*20 + market
    short = np.tile(np.r_[np.ones(6,bool), np.zeros(6,bool)], 100)
    n = len(day)
    E = pd.DataFrame(dict(day=day.astype(str), oid=np.arange(n), side=np.zeros(n,int),
        mark=np.full(n,49.), p=np.full(n,50.), v=np.full(n,-1.), q=np.ones(n),
        g_struct=np.where(short,150.,np.nan), land_struct=np.where(short,187.,np.nan),
        t_full=np.full(n,np.nan)))
    Fl = pd.DataFrame(dict(day=day.astype(str), oid=np.arange(n), clock=np.full(n,50.),
        t=np.where(short,170.,300.), q=np.ones(n)))
    V = KP.variant(E,Fl,50.,False,37.)
    # A causal score's declared schedule: short-lived orders at the first live
    # decision, others at the second. No future outcome is read by the null.
    P = np.where(short,2,3)
    res = KP.outcome(V,P,day,5)
    picks = KP.null_picks(V['endj'],P,cell,KP.SEED,500)
    null = [KP.outcome(V,p,day,5) for p,u in picks]
    counts = np.array([x['n'].sum() for x in null]); unmatched = np.array([u for p,u in picks])
    assert res['n'].sum()==n and np.array_equal(counts+unmatched,np.full(500,n))
    assert np.allclose([x['X'].sum()/x['R'].sum() for x in null],-1.)
    assert res['X'].sum()/res['R'].sum()==-1.
    flag2, stats2 = KP.k2(res,null); flag3, stats3 = KP.k3(res,null)
    assert flag2 and not flag3 and (unmatched>0).all()
    try:
        matched_gate(res,null,unmatched)
    except ValueError:
        rejected_unmatched = True
    else:
        raise AssertionError('unmatched negative control accepted')

    # Positive control: one common lifetime and a late fill leaves all orders
    # available at both steps; every draw matches counts and K2 does not pass.
    E2 = E.assign(g_struct=np.nan,land_struct=np.nan)
    Fl2 = Fl.assign(t=300.)
    V2 = KP.variant(E2,Fl2,50.,False,37.)
    r2 = KP.outcome(V2,P,day,5)
    p2 = KP.null_picks(V2['endj'],P,cell,KP.SEED,200)
    n2 = [KP.outcome(V2,p,day,5) for p,u in p2]
    matched_gate(r2,n2,[u for p,u in p2])
    assert not KP.k2(r2,n2)[0] and not KP.k3(r2,n2)[0]
    # Confirm the fixes at the old boundary: no pending step in live decisions;
    # missing data late does not suppress a valid earlier trigger.
    fair=np.full((1,99),51.); fair[0,2:]=49.; fair[0,90]=np.nan
    assert KP.score_steps(fair,np.array([0]),np.array([50.]))['A+0'][0]==2
    fair[0,2]=np.nan
    assert KP.score_steps(fair,np.array([0]),np.array([50.]))['A+0'][0]==3
    assert KP.outcome(V,np.ones(n,int),day,5)['n'].sum()==0
    assert KP.landing(2,0.)==100. and KP.landing(2,37.)==137.

    report = dict(asof=QL.C.stamp(),reviewed_commit='b6eaa7c2',
        review_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='Synthetic only; no tape read, no candidate modified, no market performance estimate.',
        source_hashes={name:QL.C.sha(ROOT/'predictfun/research'/name) for name in
            ('pf_keep_pull.py','pf_queue_place_ledger.py','pf_committed_fix_review.py')},
        fixed_checks=dict(legal_payoff_bounds=True,live_lower_bound=True,per_step_missing_fair=True,live_latency_landing=True),
        multi_step_case=dict(orders=n,synthetic_days=5,strata=100,orders_per_stratum=12,draws=500,
            actual_pulls=int(res['n'].sum()),null_mean_pulls=float(counts.mean()),null_min_pulls=float(counts.min()),
            null_max_pulls=float(counts.max()),unmatched_mean=float(unmatched.mean()),
            draws_with_unmatched=int((unmatched>0).sum()),removed_value_per_share=-1.,
            k2=flag2,k2_stats=stats2,k3=flag3,k3_stats=stats3),
        controls=dict(unmatched_comparison_refused=rejected_unmatched,matched_comparison_accepted=True,
            matched_control_draws=200,matched_control_pulls=n,matched_control_k2=False,matched_control_k3=False),
        conclusion='Risk-set exhaustion is recorded but not excluded from matched acceptance. Gate/redefine comparison before economic interpretation.')
    OUT.mkdir(exist_ok=False)
    QL.C.dump(OUT/'receipt_v1.json',report)
    print(json.dumps(QL.C.clean(report),indent=2))


if __name__=='__main__':main()
