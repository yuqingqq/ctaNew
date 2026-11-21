"""SELFTEST of pf_adv2_eval (the offline prediction diagnostic and the local action-information control; 2026-11-21, session aea5bfb9).
run(M) checks the module object M (the file, or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic only:
 1. the fixed calibration bins: every edge right-closed, the exact zero its own bin, (-0.5, 0) and (0, 0.5] split at zero, the outer bins;
 2. group readings on a hand-computed group: weighted means, bias, MSE / MAE against the prediction, zero and the training mean, the local
    value, oracle, regret, missed positive and admitted negative value (overall and |Y| > 30 c), tail counts and sums, distinct originals,
    exact and meaningful nonzero originals (a 1e-9 label is exact-nonzero but not meaningful);
 3. the screen: PASS; FAIL on sign (both directions) and on amount; INSUFFICIENT_SUPPORT under 200 originals and under 20 meaningful
    nonzero originals; NOT_EVALUABLE for an empty group and a zero mean label;
 4. the frame: stub heads and stub support -> the controller's actions and d per head, fallbacks for unsupported rows (WAIT / KEEP, d 0 / 1),
    every candidate on the identical rows, the training mean of the side / head;
 5. the control: deterministic under its seed; a perfect selector has a positive lift and no draw above it; a label-blind selector's lift
    is small against the perfect one; singleton and no-variation strata are counted; the denominator is the fixed sum of eligible weights;
    only mandatory POST and keep picks are anchors, fallback anchors excluded;
 6. committed(): a file equal to HEAD's passes, a changed or absent one fails (a temporary repository); cmd_eval refuses before reading
    any label when the fit manifest is not committed;
 7. coverage (review 315979d2 item 1) on a hand-counted frame: rows, distinct originals, exact vs meaningful nonzero originals (a 1e-9
    label is exact-nonzero only), the candidate's groups split by its selected action, every dimension present, only 'all' without a
    candidate; the frame and the training frame carry the six support categories and the horizon bin."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ADV2 EVAL SELFTEST FAILED'


class _Head:
    def __init__(self, f):
        self.f = f

    def predict(self, X, num_threads=1):
        return np.asarray([self.f(i) for i in range(len(X))], float)


def _refused(fn, msg=None):
    try:
        fn()
    except ValueError as e:
        return msg is None or msg in str(e)
    except Exception:                                                                       # noqa: BLE001  (a crash is not the check)
        return False
    return False


def run(M):  # noqa: C901
    bad = []; pd = M.pd; A = M.A
    # ---- 1 bins
    want = {-11.: '(-inf, -10]', -10.: '(-inf, -10]', -7.: '(-10, -5]', -.5: '(-1, -0.5]', -.3: '(-0.5, 0)', 0.: '0 (exact)', .3: '(0, 0.5]', .5: '(0, 0.5]',
            .7: '(0.5, 1]', 10.: '(5, 10]', 11.: '(10, inf]', float('nan'): 'nonfinite'}
    got = {k: M.cbin(k) for k in want}
    if got != want:
        bad.append(f'bins: {[(k, got[k], v) for k, v in want.items() if got[k] != v]}')
    # ---- 2 group readings
    g = pd.DataFrame(dict(day=['d'] * 5, m=[1, 1, 2, 3, 4], t_ms=[10, 10, 20, 30, 40], side=[0] * 5, Y=[40., -5., 0., 1e-9, 2.], w=[1., 2., 1., 1., 1.],
                          pred_RICH=[1., -1., .5, 0., 3.], d_RICH=[1., 0., 1., 0., 1.], train_mean=[.5] * 5))
    st = M.group_stats(g, 'RICH'); W = 6.; y, w, p, d = g.Y.values, g.w.values, g.pred_RICH.values, g.d_RICH.values
    exp = dict(rows=5, originals=4, worlds=4, nonzero=4, nonzero_originals=3, meaningful_originals=2, wmean_y=float(np.sum(w * y) / W), wmean_pred=float(np.sum(w * p) / W),
               wbias=float(np.sum(w * (p - y)) / W), wmse=float(np.sum(w * (p - y) ** 2) / W), wmae=float(np.sum(w * abs(p - y)) / W), wmse_zero=float(np.sum(w * y ** 2) / W),
               wmse_train_mean=float(np.sum(w * (.5 - y) ** 2) / W), value=float(np.sum(w * d * y) / W), oracle=float(np.sum(w * np.maximum(y, 0)) / W),
               regret=float(np.sum(w * (np.maximum(y, 0) - d * y)) / W), missed_positive=float(np.sum(w * (1 - d) * np.maximum(y, 0))),
               admitted_negative=0., missed_positive_tail=0., admitted_negative_tail=0., tail_events=1, tail_sum=40.)
    for k, v in exp.items():
        if st.get(k) is None or abs(float(st[k]) - v) > 1e-9:
            bad.append(f'group readings: {k} = {st.get(k)}, not {v}')
    g2 = g.copy(); g2['d_RICH'] = [0., 1., 1., 0., 1.]; st2 = M.group_stats(g2, 'RICH')
    if abs(st2['missed_positive_tail'] - 40.) > 1e-6 or abs(st2['admitted_negative'] + 10.) > 1e-6 or abs(st2['missed_positive'] - 40.) > 1e-6:
        bad.append(f"group readings: the missed / admitted values are wrong ({st2['missed_positive']}, {st2['missed_positive_tail']}, {st2['admitted_negative']})")
    # ---- 3 the screen
    base = dict(rows=300, originals=250, meaningful_originals=25, wmean_y=2., wmean_pred=2.2)
    cases = [('POST', {}, 'PASS'), ('POST', dict(wmean_y=-2., wmean_pred=-2.), 'FAIL'), ('WAIT', {}, 'FAIL'), ('WAIT', dict(wmean_y=-1., wmean_pred=-1.1), 'PASS'),
             ('KEEP', dict(wmean_pred=2.6), 'FAIL'), ('KEEP', dict(wmean_pred=2.44), 'PASS'), ('PULL', dict(wmean_y=-3., wmean_pred=-2.), 'FAIL'),
             ('POST', dict(originals=199), 'INSUFFICIENT_SUPPORT'), ('POST', dict(meaningful_originals=19), 'INSUFFICIENT_SUPPORT'),
             ('POST', dict(wmean_y=0., wmean_pred=0.), 'NOT_EVALUABLE')]
    for grp, ch, v in cases:
        got_v = M.screen(dict(base, **ch), grp)[0]
        if got_v != v:
            bad.append(f'screen: {grp} {ch} -> {got_v}, not {v}')
    if M.screen(dict(rows=0), 'POST')[0] != 'NOT_EVALUABLE':
        bad.append('screen: an empty group is not NOT_EVALUABLE')
    # ---- 4 the frame
    rng = np.random.default_rng(5); n = 12; X = rng.normal(size=(n, 107)).astype(np.float32)
    meta = [dict(day='20261026', m=i, t_ms=60000 * i, side=s, head=h, oid=i, pick=('mandatory' if h == 'post' else 'keep'), source='neutral', decision_ms=0.,
                 generation=1., price=40., pi=.5, hour=0, row_key={}) for i in range(n) for s, h in [(0, 'post')]][:n]
    E = {h: dict(X=np.zeros((0, 107), np.float32), y=np.zeros(0), w=np.zeros(0), meta=[]) for h in A.HEADS}
    E[(0, 'post')] = dict(X=X, y=np.arange(n, dtype=float) - 5., w=np.ones(n), meta=meta)
    km = [dict(mm, head='keep', pick='keep') for mm in meta]; E[(0, 'keep')] = dict(X=X, y=np.arange(n, dtype=float) - 5., w=np.ones(n), meta=km)
    heads = {c: {h: _Head(lambda i, c=c: (i - 5.5) if c == 'RICH' else (0. if i == 6 else 5.5 - i)) for h in A.HEADS} for c in A.CANDS}   # FAIR / MECH: an exact tie at row 6
    sup = type('S', (), {'check': lambda self, s, h, row: None if row['generation'] == 1. else 'outside_training_range'})()
    fz = dict(models={f'RICH {s} {h}': dict(train_weighted_mean_y=.25 if h == 'post' else -.75) for s, h in A.HEADS})
    E[(0, 'post')]['meta'][0]['generation'] = 2.; E[(0, 'keep')]['meta'][0]['generation'] = 2.            # one unsupported row per head
    D = M.frame(E, heads, sup, fz)
    dp, dk = D[D['head'] == 'post'].reset_index(drop=True), D[D['head'] == 'keep'].reset_index(drop=True)
    if not (list(dp.act_RICH[1:]) == ['WAIT'] * 5 + ['POST'] * 6 and dp.act_RICH[0] == 'fallback' and dp.d_RICH[0] == 0. and list(dp.d_RICH[1:]) == [0.] * 5 + [1.] * 6):
        bad.append(f'frame: the post head actions / d are wrong ({list(dp.act_RICH)})')
    if not (list(dk.act_RICH[1:]) == ['PULL'] * 5 + ['KEEP'] * 6 and dk.act_RICH[0] == 'fallback' and dk.d_RICH[0] == 1. and list(dk.d_RICH[1:]) == [0.] * 5 + [1.] * 6):
        bad.append(f'frame: the keep head actions / d are wrong ({list(dk.act_RICH)})')
    if list(dk.act_FAIR[1:]) != ['KEEP'] * 5 + ['KEEP'] + ['PULL'] * 5:
        bad.append(f'frame: a keep tie or the keep signs are wrong ({list(dk.act_FAIR)})')
    if not (list(dp.act_FAIR[1:]) == ['POST'] * 5 + ['WAIT'] * 6 and len(D) == 2 * n and set(dp.train_mean) == {.25} and set(dk.train_mean) == {-.75}):
        bad.append('frame: the controls do not predict the same rows, or the training mean is not the side / head\'s')
    Tf = M.train_frame(E)
    if not (set(A.R2.DIMS) <= set(D.columns) and set(A.R2.DIMS) <= set(Tf.columns) and len(Tf) == 2 * n and 'hbin' in Tf and set(D.c_gen) == {'1', '2'}):
        bad.append('frame: the support categories / horizon bin are missing from the evaluation or training frame')
    if (M.SEED, M.DRAWS, M.MIN_ORIG, M.MIN_NZ, M.MEANINGFUL, M.TAIL, M.ABS_TOL, M.REL_TOL) != (20261121, 500, 200, 20, 1e-6, 30., .05, .20):
        bad.append('constants: the declared seed / draws / floors / tolerances changed')
    # ---- 5 the control
    rows = []
    for st_ in range(40):
        for j in range(4):
            yv = (3. if j < 2 else -3.) * (1 + st_ % 3)
            rows.append(dict(day='20261026', side=0, head='post', pick='mandatory', source='neutral', m=st_, t_ms=60000 * st_ + j, w=1. + (j % 2), Y=yv,
                             d_RICH=1. if yv > 0 else 0., act_RICH='POST' if yv > 0 else 'WAIT', d_FAIR=float(j % 2), act_FAIR='POST' if j % 2 else 'WAIT',
                             d_MECH=1., act_MECH='POST'))
    rows.append(dict(day='20261026', side=0, head='post', pick='mandatory', source='neutral', m=99, t_ms=1, w=1., Y=5., d_RICH=1., act_RICH='POST', d_FAIR=1.,
                     act_FAIR='POST', d_MECH=1., act_MECH='POST'))                                   # a singleton stratum
    rows.append(dict(day='20261026', side=0, head='post', pick='extra', source='delayed', m=98, t_ms=1, w=1., Y=50., d_RICH=1., act_RICH='POST', d_FAIR=1.,
                     act_FAIR='POST', d_MECH=1., act_MECH='POST'))                                   # not an anchor
    rows.append(dict(day='20261026', side=0, head='post', pick='mandatory', source='neutral', m=97, t_ms=1, w=1., Y=60., d_RICH=0., act_RICH='fallback', d_FAIR=1.,
                     act_FAIR='POST', d_MECH=1., act_MECH='POST'))                                   # a fallback anchor (RICH)
    Dc = pd.DataFrame(rows)
    c1, c2 = M.control(Dc, 'RICH', ['day'], draws=200), M.control(Dc, 'RICH', ['day'], draws=200)
    if c1 != c2:
        bad.append('control: not deterministic under its seed')
    r = c1[0]; Wn = float(sum(x['w'] for x in rows[:-2]))
    real = float(sum(x['w'] * x['d_RICH'] * x['Y'] for x in rows[:-2])) / Wn
    if not (r['anchors'] == 161 and abs(r['w_sum'] - Wn) < 1e-9 and abs(r['realized'] - real) < 1e-9 and r['lift'] > 0 and r['draw_max'] < r['realized'] + 1e-12
            and r['share_draws_ge_realized'] == 0. and r['singleton_strata'] == 1 and r['strata'] == 41):
        bad.append(f'control: the perfect selector reading is wrong ({ {k: r[k] for k in ("anchors", "strata", "singleton_strata", "realized", "lift", "w_sum", "draw_max")} })')
    f1 = M.control(Dc, 'FAIR', ['day'], draws=200)[0]; mc = M.control(Dc, 'MECH', ['day'], draws=200)[0]
    if not (abs(f1['lift']) < .25 * r['lift'] and mc['no_variation_strata'] == 40 and abs(mc['lift']) < 1e-12 and f1['anchors'] == 162):
        bad.append(f"control: the label-blind / always-post selectors ({f1['lift']}, {mc['lift']}, {mc['no_variation_strata']}, {f1['anchors']})")
    # ---- 6 committed()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); run_ = lambda *a: subprocess.run(['git', '-C', str(td), *a], capture_output=True, check=True)
        run_('init', '-q'); (td / 'f.json').write_text('{"a": 1}\n'); run_('add', 'f.json'); run_('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-m', 't')
        saved = M.REPO
        try:
            M.REPO = td; ok1 = M.committed('f.json', td / 'f.json'); (td / 'g.json').write_text('{"a": 1}\n'); ok2 = M.committed('f.json', td / 'g.json')
            (td / 'g.json').write_text('{"a": 2}\n'); ok3 = M.committed('f.json', td / 'g.json'); ok4 = M.committed('h.json', td / 'g.json')
        finally:
            M.REPO = saved
        if not (ok1 and ok2 and not ok3 and not ok4):
            bad.append(f'committed: {ok1, ok2, ok3, ok4}')
        sc = M.committed
        try:
            M.committed = lambda rel, local: False
            if not _refused(lambda: M.cmd_eval(out=td / 'nowhere'), 'not committed'):
                bad.append('cmd_eval: ran without a committed fit manifest')
        finally:
            M.committed = sc
    # ---- 7 coverage
    cv = pd.DataFrame(dict(day=['d'] * 6, side=[0] * 6, head=['post'] * 6, m=[1, 1, 2, 3, 4, 5], t_ms=[10, 10, 20, 30, 40, 50], Y=[2., -1., 1e-9, 0., 3., 0.],
                           w=[1., 1., 2., 1., 1., 1.], act_RICH=['POST', 'POST', 'WAIT', 'WAIT', 'fallback', 'POST'], source=['neutral'] * 5 + ['delayed'],
                           hbin=['a'] * 6, **{k: ['x'] * 6 for k in A.R2.DIMS}))
    cov = pd.DataFrame(M.coverage(cv, 'RICH'))

    def cget(grp, dim, val, col):
        z = cov[(cov.group == grp) & (cov.dim == dim) & (cov.value == val)]
        return int(z[col].iloc[0]) if len(z) == 1 else None
    want_c = [(('all', 'source', 'neutral', 'rows'), 5), (('all', 'source', 'neutral', 'originals'), 4), (('all', 'source', 'neutral', 'nonzero_originals'), 3),
              (('all', 'source', 'neutral', 'meaningful_originals'), 2), (('POST', 'source', 'neutral', 'rows'), 2), (('POST', 'source', 'delayed', 'rows'), 1),
              (('WAIT', 'source', 'neutral', 'meaningful_originals'), 0), (('WAIT', 'source', 'neutral', 'nonzero_originals'), 1), (('fallback', 'hbin', 'a', 'rows'), 1),
              (('all', 'c_queue', 'x', 'rows'), 6)]
    wrong_c = [(k, cget(*k), v) for k, v in want_c if cget(*k) != v]
    want_dims = ('source', 'hbin') + tuple(A.R2.DIMS)                                     # the declared list, not the module's constant
    if wrong_c or set(cov.dim) != set(want_dims) or tuple(M.COV_DIMS) != want_dims or set(cov.group) != {'all', 'POST', 'WAIT', 'fallback'} or abs(cov[(cov.group == 'all') & (cov.dim == 'hbin')].w_sum.sum() - 7.) > 1e-12:
        bad.append(f'coverage: {wrong_c} / dims {sorted(set(cov.dim))} / groups {sorted(set(cov.group))}')
    if set(pd.DataFrame(M.coverage(cv)).group) != {'all'}:
        bad.append('coverage: without a candidate it is not the all group only')
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 eval selftest passed: bins (12 cases), group readings (hand-computed), screens (11 cases), frame (actions, d, fallbacks, same '
          'rows, training means), control (deterministic, perfect / blind / always selectors, singleton / no-variation strata, anchors), '
          'committed() and the eval refusal, coverage (hand-counted; exact vs meaningful; groups; dimensions)')
