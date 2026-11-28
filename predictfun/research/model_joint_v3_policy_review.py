"""Tiny actual-source controls for v3 repair and policy cadence; no historical replay."""
import bisect
import datetime as dt
import hashlib
import json
from types import SimpleNamespace

from model_order_harm_v2_fix_review import ROOT, NP, Vec, extract, require_invariance


def lane(future_opp, cap=30.0, mutant=0):
    W = {}
    for s in (0, 1):
        prices = (50, 49) if s == 0 else (52, 53)
        W[s] = dict(tg=Vec([0.0, 50.0, 100.0, 150.0]), tf=Vec([0.0, 40.0]), te=200.0,
                    anc=Vec(), ptl=[], shl=[], P5=None, S5=None,
                    opp=[52, future_opp] if s == 0 else [50, 49 if future_opp == 50 else 50],
                    best=([50, 49, 49, 49] if future_opp == 50 else [50] * 4) if s == 0
                         else ([52, 50, 50, 50] if future_opp == 50 else [52] * 4),
                    bopp=([52, 50, 50, 50] if future_opp == 50 else [52] * 4) if s == 0
                          else ([50, 49, 49, 49] if future_opp == 50 else [50] * 4), vkl=[0.5] * 4,
                    cache={pt: Vec([100.0, 0.0 if s == 0 and pt == 50 and future_opp == 50 else 100.0]) for pt in prices})
    env = dict(bisect=bisect, np=NP, SIZE=30.0, NR=2, INF=float('inf'), L_NOTE=100.0,
               K=SimpleNamespace(L_PLACE=40.0, L_FILL=100.0, L_CANCEL=22.0,
                                 L_PENDCANCEL=80.0, HOLD_K=3, REST_COOL=250.0),
               VIOL=('leg', 'order', 'live', 'priority', 'collision', 'position', 'cap'),
               MUT_REF_RELEASE=mutant, MUT_REST_TRUE=0, MUT_HELD_TRUE=0,
               MUT_LEG_LEAK=0, MUT_PRIORITY=0, INDEP=0)
    sim = extract(ROOT / 'predictfun/research/pf_joint_ladder_v3.py', 'sim_joint', env)
    r = sim(W, 'M', 0.0, cap, {0: [False] * 4, 1: [False] * 4}, mode='hard', nr=2)
    assert not any(r['stat']['viol'].values())
    return dict(prefix_at_zero=[x for x in r['log'] if x[0] == 0.0],
                second_bid_at_zero=any(x[0] == 40.0 for x in r['orders'][(0, 1)]),
                violations=r['stat']['viol'])


class Matrix:
    def __init__(self, scores):
        self.scores = scores

    def __getitem__(self, key):
        return self.scores[key[0]]


def cadence():
    path = ROOT / 'predictfun/research/pf_order_harm_policy.py'
    env = dict(EVERY=5, MUT_SIGN=0, MUT_HOLD=0, MUT_NOINPUT_KEEP=0)
    extract(path, 'keep_rule', env)
    oh_dec = extract(path, 'oh_dec', env)
    vg = [True] * 6
    scores = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
    wi = dict(vg={0: vg}, onset={0: [0] * 6}, MK={0: dict.fromkeys(range(6), 1)})
    pred = lambda s, d, o, k, t: (scores[k], 1.0, 0.0)
    thr = dict(tau_H=0.25)
    counters = dict(noinput=0, keep=[0, 0], pull=[0, 0])
    oh = oh_dec('OHH', 0, 1, wi, pred, thr, counters)
    fb_env = dict(JS=list(range(1, 7)), MUT_ARRIVE_R0=0, MUT_NO_SHIFT=0, MUT_ORACLE_SIGN=0)
    fb_dec = extract(ROOT / 'predictfun/research/pf_future_best.py', 'fb_dec', fb_env)
    fb = fb_dec(vg, [50] * 6, 0, Matrix(scores), 1, tau=0.25)
    order = dict(pt=49)
    rows = [dict(t_ms=k * 50, score=scores[k], OH_pull=oh(k, k * 50.0, order),
                 FB_pull=fb(k, k * 50.0, order)) for k in (1, 2, 5)]
    assert [(r['OH_pull'], r['FB_pull']) for r in rows] == [(False, False), (False, True), (True, True)]
    # Same elevated score is acted on at the same time for a first-seen order.
    assert oh(2, 100.0, dict(pt=49)) is True
    # Planted cadence change must fail the expected hold-at-100ms assertion.
    env['EVERY'] = 1
    changed = oh_dec('OHH', 0, 1, wi, pred, thr, counters)
    changed(1, 50.0, order)
    caught = False
    try:
        require_invariance(changed(2, 100.0, order), False)
    except ValueError:
        caught = True
    assert caught
    return dict(example=rows, new_order_immediate_pull_control=True, altered_cadence_refused=True,
                interpretation='Declared cadence difference, not an implementation bug or measured economic effect')


def main():
    paths = [ROOT / 'predictfun/research' / f for f in
             ('pf_joint_ladder_v3.py', 'pf_order_harm_policy.py', 'pf_future_best.py')]
    sha = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = sha()
    clean = [lane(f) for f in (52, 50)]
    require_invariance(clean[0]['prefix_at_zero'], clean[1]['prefix_at_zero'])
    assert [x['second_bid_at_zero'] for x in clean] == [False, False]
    bad = [lane(f, mutant=1) for f in (52, 50)]
    caught = False
    try:
        require_invariance(bad[0]['prefix_at_zero'], bad[1]['prefix_at_zero'])
    except ValueError:
        caught = True
    assert caught and [x['second_bid_at_zero'] for x in bad] == [False, True]
    positive = lane(52, cap=60.0)
    assert positive['second_bid_at_zero'] and positive['prefix_at_zero'] != clean[0]['prefix_at_zero']
    timing = cadence()
    assert before == sha(), 'source changed during the controls'
    print(json.dumps(dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),
                          kind='source_synthetic_review_not_historical_policy_performance', source_sha256=before,
                          repaired_future_rejection=clean, planted_old_bug=bad,
                          old_bug_refused=caught, known_cap_change_positive_control=positive,
                          policy_cadence=timing), indent=2))


if __name__ == '__main__':
    main()
