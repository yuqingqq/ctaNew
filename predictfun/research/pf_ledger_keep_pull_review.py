"""Synthetic counterexamples for the Nov17 ledger/keep-pull review.

No market tape, candidate changes, or performance estimates. Tests the exact
public report helpers against independent economic identities and controls.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pf_queue_place_ledger as QL
import pf_keep_pull as KP

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'predictfun/reviews/ledger_keep_pull_review_20261117'


def require_equal(x, y):
    if not np.allclose(x, y, atol=1e-10, rtol=0):
        raise ValueError('counterexample arithmetic mismatch')


def main():
    require_equal([1, 2], [1, 2])
    try:
        require_equal([1, 2], [1, 3])
    except ValueError:
        changed_refused = True
    else:
        raise AssertionError('changed result accepted')

    # A missing order can earn more than the largest observed order, without
    # violating binary-payoff/30-share bounds. Old L5 nevertheless says robust.
    inc = pd.Series([-1., -1.]); need, ext, old_flag = QL.bound(inc, 1)
    missing_value = 3.; missing_gain_bound = 30. * (100. - 50.)
    assert missing_value <= missing_gain_bound and old_flag
    assert inc.sum() < 0 < inc.sum() + missing_value
    # Positive control for a genuine worst-case bound at the same price/size.
    assert np.array([-1000., -1000.]).sum() + missing_gain_bound < 0

    # Five synthetic day cells: every removable fill has identical value/share.
    # Half the orders have an eligible early trigger; the other half already
    # have a structural cancellation. Shuffling raw triggers changes the
    # number of actionable pulls, even in expectation, with no value ranking.
    n_day = 200; n = 5*n_day
    active = np.tile(np.r_[np.ones(n_day//2, bool), np.zeros(n_day//2, bool)], 5)
    day = np.repeat(np.arange(5), n_day)
    T = np.where(active, 1, KP.NSTEP+1)
    # Active orders fill at 4,999 ms (after every candidate removal); inactive
    # orders fill at 70 ms before their structural cancel lands at 95 ms.
    # Even inaccessible branch values therefore correspond to feasible fills.
    V = dict(Vk=np.full(n, -1.), Qk=np.ones(n),
             Mv=np.repeat(np.where(active, 0., -1.)[:,None], KP.NSTEP, axis=1),
             Mq=np.repeat(np.where(active, 0., 1.)[:,None], KP.NSTEP, axis=1),
             endj=np.where(active, KP.NSTEP+1, 1),
             life=np.where(active, 4999., 70.), lat=37.)
    actual = KP.outcome(V, T, day, 5)
    null = KP.nulls(V, T, day, day, 5, 20261117, draws=500)
    k2, k2_stats = KP.k2(actual, null)
    k3, k3_stats = KP.k3(actual, null)
    null_pulls = np.array([r['n'].sum() for r in null])
    require_equal(actual['n'].sum(), 500)
    assert 225 < null_pulls.mean() < 275 and k2 and not k3
    require_equal(actual['X'].sum()/actual['R'].sum(), -1)
    require_equal([r['X'].sum()/r['R'].sum() for r in null], np.full(500, -1.))
    # Positive control: without destination eligibility differences, shuffling
    # raw triggers DOES preserve the number of pulls.
    balanced = dict(V, endj=np.full(n, KP.NSTEP+1), Mv=np.zeros((n,KP.NSTEP)),
                    Mq=np.zeros((n,KP.NSTEP)), life=np.full(n,4999.))
    pos = KP.nulls(balanced, T, day, day, 5, 20261117, draws=200)
    require_equal([r['n'].sum() for r in pos], np.full(200, 500.))

    # Every share of a single-price order has the same endpoint value sign.
    # For a losing order the oracle always prefers its earliest usable pull.
    # A 50-ms pending decision lands at 95 ms at both L=0 and L=37.
    E = pd.DataFrame(dict(day=[str(i) for i in range(5)], oid=np.arange(5),
        side=np.zeros(5), mark=np.full(5,49.), p=np.full(5,50.),
        v=np.full(5,-30.), q=np.full(5,30.), g_struct=np.full(5,np.nan),
        land_struct=np.full(5,np.nan), t_full=np.full(5,250.)))
    fills = pd.DataFrame([dict(day=str(i),oid=i,clock=50.,t=t,q=q)
                         for i in range(5) for t,q in [(70.,20.),(250.,10.)]])
    curves = {}
    for lat in (0.,25.,37.,100.):
        vv = KP.variant(E,fills,50.,False,lat)
        policy = KP.oracle(vv)
        res = KP.outcome(vv,policy,np.arange(5),5)
        require_equal(policy,np.ones(5))
        curves[str(lat)] = dict(first_landing_ms=float(KP.landing(1,lat)),
            oracle_improvement=float(res['imp'].sum()),
            k1=KP.k1(res['imp'],np.full(5,30.)))
    require_equal([curves[str(x)]['first_landing_ms'] for x in (0.,25.,37.)],[95.]*3)
    require_equal([curves[str(x)]['oracle_improvement'] for x in (0.,25.,37.)],[50.]*3)
    assert not curves['0.0']['k1'] and not curves['37.0']['k1']
    assert KP.STEP < KP.PLACE

    result = dict(asof=QL.C.stamp(), review_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), scope='Synthetic counterexamples only; no market tape or empirical P&L.',
        sources={name:QL.C.sha(ROOT/'predictfun/research'/name) for name in
                 ('pf_queue_place_ledger.py','pf_keep_pull.py','pf_ledger_keep_pull_review.py')},
        controls=dict(valid_arithmetic_accepted=True,changed_arithmetic_refused=changed_refused,
            genuine_payoff_bound_control=True,equal_eligibility_preserves_pull_counts=True),
        ledger_l5=dict(n_observed=2,n_missing=1,observed_total=-2.,needed_missing_mean=need,
            observed_extreme=ext,reported_robust=old_flag,feasible_missing_value=missing_value,
            resulting_total=float(inc.sum()+missing_value),legal_missing_gain_bound=missing_gain_bound),
        null=dict(n_orders=n,synthetic_day_cells=5,draws=500,actual_pulls=500,
            null_mean_pulls=float(null_pulls.mean()),null_min_pulls=float(null_pulls.min()),
            null_max_pulls=float(null_pulls.max()),identical_removed_value_per_share=-1.,
            k2_passes_without_value_ranking=k2,k2_stats=k2_stats,k3=k3,k3_stats=k3_stats),
        oracle=dict(n_orders=5,loss_per_order=30.,decision_ms=KP.STEP,activation_ms=KP.PLACE,curves=curves))
    result['supersedes'] = 'counterexamples_v1.json'
    result['revision'] = 'Feasible fill timing also supplied for inaccessible null branches; findings unchanged.'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'counterexamples_v2.json').exists(), 'review output exists'
    QL.C.dump(OUT/'counterexamples_v2.json',result)
    print(json.dumps(QL.C.clean(result),indent=2))


if __name__=='__main__': main()
