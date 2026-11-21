"""SELFTEST of pf_entry_audit (ENTRY-TARGET-AUDIT-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M (the file, or a mutated
copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic pair results only (hand values; fills (t, gid, side, q, price, it)):
 1. the cases of review d6d5d55e item 5: no fills (with an identical fill at the fork tick, pre-fork); a delayed WAIT entry with the same
    economic fill at another time / gid; an initial gain offset by a later loss; an initial loss offset by a later gain; cross-side effects
    (both arms); plus same cash / inventory with different fills, a partial initial fill and an arithmetic residue. Each has its class,
    V_initial / D_focal / D_opposite, the identity, the initial fill and the WAIT arm's next POST;
 2. refusals: an altered fill side, quantity or mark is an identity_failure; a missing or a second forced generation is
    forced_generation_not_unique; a pre-fork difference is an identity_failure;
 3. the predicates: rare_execution / cancellation / both / not_identified on hand summaries at the declared thresholds;
 4. group_summary: weighted class shares and means, absolute totals, the top-5 share, the WAIT-next-POST bins (hand values);
 5. the prediction join: by the exact key whatever the row order; a missing key is the status prediction_missing, never a drop."""
import sys

import numpy as np

FAIL = 'ENTRY AUDIT SELFTEST FAILED'


def run(M):  # noqa: C901
    bad = []; pd = M.pd; k, side, mark = 10, 0, 42.; g = k * 50.
    sg = lambda s: 1. if s == 0 else -1.
    val = lambda fl: sum(sg(f[2]) * f[3] * (mark - f[4]) for f in fl)

    def res(post_f, wait_f, post_d=None, wait_d=None, sizes=(30., 30., 30.), Y=None, m=mark):
        pd_ = post_d if post_d is not None else [[g, side, 'absent', 'POST', 0], [g, 1, 'absent', 'WAIT', None]]
        wd_ = wait_d if wait_d is not None else [[g, side, 'absent', 'WAIT', None], [g, 1, 'absent', 'WAIT', None]]
        y = Y if Y is not None else val(post_f) - val(wait_f)
        return dict(k=k, mark=m, Y=y, arms=dict(POST=dict(fill_list=[list(f) for f in post_f], decision_list=pd_, sizes=list(sizes)),
                                                WAIT=dict(fill_list=[list(f) for f in wait_f], decision_list=wd_, sizes=list(sizes))))
    pre = [(g, 5, 0, 10., 41., 1)]                                                          # an identical fill at the fork tick (pre-fork)
    cases = {
        'no fills': (res(pre, pre), dict(cls='no_post_fork_fills', v_initial=0., d_focal=0., d_opposite=0., init_fill='no_fill', wait_next_post_ms=None)),
        'delayed WAIT entry, same economic fill': (res([(g + 200, 0, 0, 30., 40., 2)], [(g + 400, 0, 0, 30., 40., 3)],
                                                       wait_d=[[g, 0, 'absent', 'WAIT', None], [g + 150, 0, 'absent', 'POST', 0]]),
                                                   dict(cls='economic_identical', v_initial=60., d_focal=-60., d_opposite=0., init_fill='full', wait_next_post_ms=150., timed_identical=False)),
        'initial gain, later loss': (res([(g + 100, 0, 0, 30., 40., 2), (g + 900, 1, 0, 30., 44., 9)], [], post_d=[[g, 0, 'absent', 'POST', 0], [g + 800, 0, 'absent', 'POST', 1]]),
                                     dict(cls='offset_by_mark', v_initial=60., d_focal=-60., d_opposite=0.)),
        'initial loss, later gain': (res([(g + 100, 0, 0, 30., 43., 2)], [(g + 600, 0, 0, 25., 44., 7)]),
                                     dict(cls='meaningful', v_initial=-30., d_focal=50., d_opposite=0.)),
        'cross-side': (res([(g + 300, 4, 1, 10., 45., 4)], [(g + 700, 2, 1, 5., 43., 8)]),
                       dict(cls='meaningful', v_initial=0., d_focal=0., d_opposite=30. - 5.)),
        'same cash and inventory': (res([(g + 100, 0, 0, 30., 40., 2)], [(g + 100, 0, 0, 15., 40., 2), (g + 200, 0, 0, 15., 40., 3)]),
                                    dict(cls='same_cash_inventory', v_initial=60., d_focal=-60.)),
        'partial initial fill': (res([(g + 100, 0, 0, 12., 40., 2)], []), dict(cls='meaningful', v_initial=24., init_fill='partial')),
        'a residue': (res([(g + 200, 0, 0, 30., 40., 2)], [(g + 400, 0, 0, 30., 40., 3)], Y=1e-13), dict(cls='economic_identical', residue=True, exact_zero=False)),
    }
    for name, (r_, want) in cases.items():
        got = M.account(r_, side)
        if got.get('status') != 'ok':
            bad.append(f'{name}: status {got.get("status")} (gap {got.get("identity_gap")})'); continue
        for f, v in want.items():
            x = got.get(f)
            if not ((x is None and v is None) or (isinstance(v, (int, float)) and not isinstance(v, bool) and x is not None and abs(float(x) - v) < 1e-9) or x == v):
                bad.append(f'{name}: {f} = {x}, not {v}')
        if abs(got['Y'] - (got['v_initial'] + got['d_following'])) > 1e-9 and name != 'a residue':
            bad.append(f'{name}: Y is not V_initial + D_following')
    # ---- 2 refusals
    base_ = cases['initial loss, later gain'][0]
    def alt(fn):
        import copy
        r2 = copy.deepcopy(base_); fn(r2); return M.account(r2, side).get('status')
    for nm, fn, want in (('an altered side', lambda r: r['arms']['POST']['fill_list'][0].__setitem__(2, 1), 'identity_failure'),
                         ('an altered quantity', lambda r: r['arms']['WAIT']['fill_list'][0].__setitem__(3, 20.), 'identity_failure'),
                         ('an altered mark', lambda r: r.__setitem__('mark', 41.), 'identity_failure'),
                         ('no forced generation', lambda r: r['arms']['POST'].__setitem__('decision_list', [[g, 1, 'absent', 'WAIT', None]]), 'forced_generation_not_unique'),
                         ('two forced generations', lambda r: r['arms']['POST']['decision_list'].append([g, side, 'absent', 'POST', 1]), 'forced_generation_not_unique'),
                         ('a pre-fork difference', lambda r: r['arms']['POST']['fill_list'].insert(0, [g - 50, 3, 0, 5., 41., 0]), 'identity_failure')):
        if alt(fn) != want:
            bad.append(f'refusal: {nm} -> {alt(fn)}, not {want}')
    # ---- 3 predicates
    P = lambda **k: M.predicates(dict(rows=10, **k))
    for k_, want in ((dict(share_meaningful=.05, init_no_fill=.7, abs_v_initial=0., abs_Y=1.), 'rare_execution'),
                     (dict(share_meaningful=.3, init_no_fill=.2, abs_v_initial=10., abs_Y=4.), 'cancellation'),
                     (dict(share_meaningful=.05, init_no_fill=.6, abs_v_initial=10., abs_Y=5.), 'both'),
                     (dict(share_meaningful=.3, init_no_fill=.6, abs_v_initial=10., abs_Y=8.), 'not_identified'),
                     (dict(share_meaningful=.05, init_no_fill=.4, abs_v_initial=10., abs_Y=6.), 'not_identified')):
        if P(**k_)['diagnosis'] != want:
            bad.append(f'predicates: {k_} -> {P(**k_)["diagnosis"]}, not {want}')
    # ---- 4 group_summary
    recs = [dict(M.account(cases[n][0], side), day='d', m=i, t_ms=i, side=0, w=w_) for i, (n, w_) in
            enumerate((('no fills', 1.), ('delayed WAIT entry, same economic fill', 2.), ('initial loss, later gain', 3.), ('cross-side', 4.)))]
    if not all(r_.get('status') == 'ok' for r_ in recs):                                     # the fixture must account before it is summarized
        bad.append('group_summary: the fixture cases did not account'); s = {}
    else:
        s = M.group_summary(pd.DataFrame(recs))
    W = 10.
    exp = dict(rows=4, w_sum=W, meaningful=2, share_meaningful=7. / W, share_no_post_fork_fills=.1, wmean_Y=(3. * 20. + 4. * 25.) / W, abs_v_initial=2. * 60. + 3. * 30.,
               init_no_fill=(1. + 4.) / W, wait_no_later_post=(1. + 3. + 4.) / W, wait_next_100_500=2. / W, top5_abs_share=1.)
    for f, v in exp.items():
        if s.get(f) is None or abs(float(s[f]) - v) > 1e-9:
            bad.append(f'group_summary: {f} = {s.get(f)}, not {v}')
    # ---- 5 the prediction join
    P0 = pd.DataFrame(dict(day=['20261026'] * 3, m=[1, 2, 3], t_ms=[10, 20, 30], side=[0, 0, 1], pick=['mandatory'] * 3, pred_SMOOTH=[.1, .2, .3], pred_RICH=[1., 2., 3.],
                           act_SMOOTH=['POST'] * 3, act_RICH=['POST'] * 3, support=['supported'] * 3))
    rows = [dict(day='20261026', m=m_, t_ms=t_, side=s_, pick='mandatory', _inputs=[0.]) for m_, t_, s_ in ((3, 30, 1), (1, 10, 0), (9, 90, 0))]
    sv = M.NE.read_r3
    try:
        M.NE.read_r3 = lambda p: P0.copy(); M.predictions(rows, 'eval', None, None, None)
    finally:
        M.NE.read_r3 = sv
    if not (rows[0].get('pred_RICH') == 3. and rows[1].get('pred_SMOOTH') == .1 and rows[2].get('pred_status') == 'prediction_missing' and 'pred_RICH' not in rows[2]):
        bad.append('predictions: not joined by the exact key, or a missing key was not its status')
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('entry audit selftest passed: 8 accounting cases (classes, V_initial / D_focal / D_opposite, identity, initial fill, WAIT next POST); '
          '6 refusals; the predicates (5 cases); group_summary (hand values); the key join and prediction_missing')
