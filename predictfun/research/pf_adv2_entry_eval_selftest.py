"""SELFTEST of pf_adv2_entry_eval (the ADV2-ENTRY-SMOOTH-v1 offline entry diagnostic; 2026-11-21, session aea5bfb9). run(M) checks the
module object M (the file, or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic only:
 1. the frame: stub heads -> each candidate's POST iff > 0 (a tie WAITs), fallbacks for unsupported rows (d 0), both candidates on identical
    rows, the training mean of the side;
 2. r3_identity: an equal R3 frame passes every field; an altered label, weight or RICH prediction and a missing row each fail their field;
 3. readings / screens: every day / side / candidate has all / POST / WAIT / fallback, and POST / WAIT verdicts from the declared set;
 4. calibration: each bin's w_sum is its weight total and pooling (w_sum x mean) / sum(w_sum) over days equals the direct weighted mean;
 5. strata: per dimension value the rows, posts and the local value sum(w d Y) / sum(w) of each candidate (hand-computed);
 6. predicates: a positive POST group with a PASS screen above its draws passes every predicate; zero admissions is no_admissions with every
    predicate false; a negative POST group fails post_value_positive;
 7. cmd_eval refuses before reading a label when the fit manifest is not committed."""
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ADV2 ENTRY EVAL SELFTEST FAILED'


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
    bad = []; pd, A, N, AE = M.pd, M.A, M.N, M.AE
    # ---- 1 the frame
    rng = np.random.default_rng(3); n = 12; X = rng.normal(size=(n, 107)).astype(np.float32)
    meta = lambda s: [dict(day='20261026', m=i, t_ms=60000 * i, side=s, head='post', oid=i, pick='mandatory', source='neutral', decision_ms=0., generation=1. + (i % 3),
                           price=40., pi=.5, hour=0, row_key={}) for i in range(n)]
    E = {(s, 'post'): dict(X=X, y=np.arange(n, dtype=float) - 5., w=np.ones(n) + (np.arange(n) % 2), meta=meta(s)) for s in (0, 1)}
    E[(0, 'post')]['meta'][0]['price'] = 999.                                                # one unsupported row (the stub support)
    sm = {(s, 'post'): _Head(lambda i: i - 5.5) for s in (0, 1)}; sm[(1, 'post')] = _Head(lambda i: 0. if i == 6 else i - 5.5)
    rich = {'RICH': {(s, h): _Head(lambda i: 5.5 - i) for s in (0, 1) for h in ('post', 'keep')}}
    sup = type('S', (), {'check': lambda self, s, h, row: None if row['price'] < 100 else 'outside_training_range'})()
    fz = dict(models={f'SMOOTH {s} post': dict(train_weighted_mean_y=.25 + s) for s in (0, 1)})
    D = M.frame(E, sm, rich, sup, fz); d0, d1 = D[D.side == 0].reset_index(drop=True), D[D.side == 1].reset_index(drop=True)
    if not (d0.act_SMOOTH[0] == 'fallback' and d0.d_SMOOTH[0] == 0. and list(d0.act_SMOOTH[1:]) == ['WAIT'] * 5 + ['POST'] * 6 and list(d0.d_SMOOTH[1:]) == [0.] * 5 + [1.] * 6):
        bad.append(f'frame: SMOOTH actions / d ({list(d0.act_SMOOTH)})')
    if d1.act_SMOOTH[6] != 'WAIT' or list(d1.act_RICH[:6]) != ['POST'] * 6:
        bad.append('frame: a tie did not WAIT, or RICH is not on the same rows')
    if not (len(D) == 2 * n and set(d0.train_mean) == {.25} and set(d1.train_mean) == {1.25} and set(A.R2.DIMS) <= set(D.columns) and D.day.map(type).eq(str).all()):
        bad.append('frame: rows / training means / categories / day type')
    # ---- 2 r3_identity
    r3 = D[M.KEY + ['Y', 'w', 'pred_RICH']].assign(head='post', day=D.day.astype(int)); idn = M.r3_identity(D, r3)
    if not all(idn[k] for k in ('same_keys', 'y_equal', 'w_equal', 'rich_pred_equal')):
        bad.append(f'r3_identity: an equal frame failed ({idn})')
    for col, fld in (('Y', 'y_equal'), ('w', 'w_equal'), ('pred_RICH', 'rich_pred_equal')):
        r = r3.copy(); r.loc[3, col] = r.loc[3, col] + 1e-9
        if M.r3_identity(D, r)[fld]:
            bad.append(f'r3_identity: an altered {col} passed')
    if M.r3_identity(D, r3.iloc[1:])['same_keys']:
        bad.append('r3_identity: a missing row passed')
    # ---- 3 readings / screens
    rows, scr = M.readings(D); rd, sc = pd.DataFrame(rows), pd.DataFrame(scr)
    for s in (0, 1):
        for c in M.CANDS:
            if set(rd[(rd.side == s) & (rd.candidate == c)].group) != {'all', 'POST', 'WAIT', 'fallback'} or set(sc[(sc.side == s) & (sc.candidate == c)].group) != {'POST', 'WAIT'}:
                bad.append(f'readings: groups of {s} {c}')
    if not set(sc.verdict) <= {'PASS', 'FAIL', 'NOT_EVALUABLE', 'INSUFFICIENT_SUPPORT'}:
        bad.append('screens: a verdict outside the declared set')
    # ---- 4 calibration
    D2 = pd.concat([D, D.assign(day='20261027', w=D.w * 3.)], ignore_index=True); cal = pd.DataFrame(M.calibration(D2))
    for (s, c, b_), x in cal.groupby(['side', 'candidate', 'bin']):
        g = D2[(D2.side == s) & (D2.support == 'supported')]; g = g[g[f'pred_{c}'].map(AE.cbin) == b_]
        pooled = float((x.w_sum * x.wmean_y).sum() / x.w_sum.sum()); direct = float((g.w * g.Y).sum() / g.w.sum())
        if abs(pooled - direct) > 1e-9 or abs(x.w_sum.sum() - g.w.sum()) > 1e-9:
            bad.append(f'calibration: {s} {c} {b_} pooling by w_sum ({pooled}) is not the weighted mean ({direct})'); break
    # ---- 5 strata
    stt = pd.DataFrame(M.strata(D)); z = stt[(stt.side == 0) & (stt.dim == 'c_gen') & (stt.value == '1')]
    g = d0[d0.c_gen == '1']; want = float((g.w * g.d_SMOOTH * g.Y).sum() / g.w.sum())
    if len(z) != 1 or int(z.rows.iloc[0]) != len(g) or int(z.posts_SMOOTH.iloc[0]) != int(g.d_SMOOTH.sum()) or abs(float(z.value_SMOOTH.iloc[0]) - want) > 1e-9 \
            or set(stt.dim) != set(M.STRATA):
        bad.append('strata: rows / posts / value per stratum, or the declared dimensions')
    # ---- 6 predicates
    rd_ = [dict(day='d', side=0, candidate='SMOOTH', group='POST', rows=30, wmean_y=2.), dict(day='d', side=0, candidate='RICH', group='POST', rows=0, wmean_y=None)]
    rd_ += [dict(day='d', side=0, candidate=c, group='all', rows=60, wmean_y=-.5) for c in M.CANDS]                 # the all group differs from POST
    rd_ += [dict(day='d', side=1, candidate=c, group=gname, rows=30, wmean_y=-1.) for c in M.CANDS for gname in ('all', 'POST')]
    sc_ = [dict(day='d', side=s, candidate=c, group='POST', verdict='PASS' if (s, c) == (0, 'SMOOTH') else 'FAIL') for s in (0, 1) for c in M.CANDS]
    ct_ = [dict(day='d', side=0, candidate=c, realized=1., draw_mean=.5, draw_p95=.8) for c in M.CANDS]
    ct_ += [dict(day='d', side=1, candidate=c, realized=.6, draw_mean=.5, draw_p95=.8) for c in M.CANDS]                # above the mean, not the p95
    pr = {(p_['side'], p_['candidate']): p_ for p_ in M.predicates(rd_, sc_, ct_)}
    a, z0, ng = pr[(0, 'SMOOTH')], pr[(0, 'RICH')], pr[(1, 'SMOOTH')]
    if not (a['post_value_positive'] and a['screen_pass'] and a['above_draw_mean'] and a['above_draw_p95'] and a['status'] == 'admitted'):
        bad.append(f'predicates: a passing candidate failed ({a})')
    if not (z0['status'] == 'no_admissions' and not any(z0[k] for k in ('post_value_positive', 'screen_pass', 'above_draw_mean', 'above_draw_p95'))):
        bad.append(f'predicates: zero admissions was not no_admissions with every predicate false ({z0})')
    if ng['post_value_positive'] or not (ng['above_draw_mean'] and not ng['above_draw_p95']):
        bad.append('predicates: a negative POST group had positive value, or the mean / p95 predicates are not separate')
    # ---- 7 the refusal
    sc0 = M.N.committed_equal
    try:
        M.N.committed_equal = lambda rel, local: False
        with tempfile.TemporaryDirectory() as td:
            if not _refused(lambda: M.cmd_eval(out=Path(td) / 'nowhere'), 'not committed'):
                bad.append('cmd_eval: ran without a committed fit manifest')
    finally:
        M.N.committed_equal = sc0
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 entry eval selftest passed: frame (actions, tie, fallback, same rows, training means); r3_identity (equal; altered Y / w / '
          'prediction / missing row); readings and screens; calibration pooling by w_sum; strata (hand-computed); predicates (pass, no_admissions, '
          'negative value); the eval refusal')
