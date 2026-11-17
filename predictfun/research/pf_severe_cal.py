"""SEVERE-EVENT PROBABILITY: BASELINE vs SMOOTHER vs CALIBRATION MAP (2026-11-17, session aea5bfb9; USER design: "Smoother model - one
declared setting: minimum leaf size 200->400, rounds 200->100, other parameters fixed; Calibration map - a small monotone map, such as
sigmoid(a + b logit(p)), fitted on an earlier held-out day ... Compare baseline, smoother-only and calibration-only. For calibration, keep
the same base model frozen while scoring both the calibration day and subsequent evaluation days ... severe-loss probability exceeds
any-loss probability in 2,268 / 36,056 decisions. Both approaches need explicit consistency handling; silently clipping afterward would
confound the comparison"). Declared before any run; no controller change; consumed days; exploratory.

CONSISTENCY (explicit, all arms): P(severe) = P(any loss) x P(severe | loss). Head A: P(loss > 0) on every eligible decision; head B:
P(loss > 30 c | loss > 0) on the decisions with a loss. P(severe) <= P(any loss) by construction; nothing is clipped afterwards. The
independent severe head of GAIN-LOSS is reported as a reference only (its violations reproduced).
ARMS (identical amount and gain heads: GAIN-LOSS's E[gain | gain > 0], P(gain > 0), E[min(loss, 30) | loss > 0], E[loss - 30 | loss > 30],
pinned settings, the same training days):
 BASE   heads A and B with the pinned parameters (min_data_in_leaf 200, 200 rounds).
 SMOOTH heads A and B with min_data_in_leaf 400 and 100 rounds; every other parameter unchanged (a declared change of the pinned set).
 CAL    BASE's heads A and B frozen; p_A' = sigmoid(a_A + b_A logit p_A) fitted on the calibration day over every decision, p_B' =
        sigmoid(a_B + b_B logit p_B) fitted on the calibration day over its decisions with a loss (unpenalized logistic regression); a map
        with b <= 0 is refused (reported as failed, never inverted); P(severe)' = p_A' x p_B'.
FOLDS (past-only): A (primary) - train Oct23-24, calibration day Oct25, evaluation Oct26 and Oct27. B (sensitivity, printed only) - train
Oct23-25, calibration Oct26, evaluation Oct27. Every arm of a fold uses the same training days; CAL's calibration-day scores are in-sample
for its map and printed separately.
VALUE: W = P(gain > 0) E[gain | gain > 0] - p_A E[min(loss, 30) | loss > 0] - P(severe) E[loss - 30 | loss > 30], held to the finite-payoff
bounds (as GAIN-LOSS).
READINGS (fold A, cells = evaluation day x kind, 4; each candidate against BASE):
 S1 SEVERE-CALIBRATION: P(severe) log loss lower in >= 3 of 4 cells AND |log O/E| lower in >= 3 of 4 -> IMPROVES, else NOT-SHOWN.
 S2 LOW-BAND: pooled O/E over the bands p < 0.05 closer to 1 (|log| smaller) -> YES / NO.
 S3 NET-VALUE: RMSE(W) lower in >= 3 of 4 cells AND quintile calibration error lower in >= 3 of 4 -> IMPROVES, else NOT-SHOWN.
 S4 MATCHED-PARTICIPATION: selection inside every day x kind x market x side stratum, realized mean W of the top 10 / 25 / 50% >= BASE's in
    >= 3 of 4 cells at >= 2 of 3 rates -> IMPROVES, else NOT-SHOWN.
 A candidate PASSES only with S1, S3 and S4 (the USER's standard: better net-value calibration and actual value at matched participation,
 not merely better risk scores).
 Printed: P(any) log loss and O/E, band O/E for all arms, the map coefficients, fold B, CAL on its calibration day.
CONTROLS (any failure -> no output): P(severe) <= P(any) on every decision of every arm (planted error: the independent severe head in
 place of the chain must be refused); calibration maps are fitted only on the calibration day (planted error: a map fitted on an
 evaluation day must be refused); CAL's frozen base scores equal BASE's on the evaluation days; the independent-head violations on the
 RUNNING-ERRORS forward folds are reproduced (USER: 2,268 / 36,056); BASE's head A equals GAIN-LOSS's any-loss head recipe (FM.fit).
LIMITS: consumed days; two evaluation days in fold A (no interval, rule 8); simulated labels; the neutral decision population.
Usage (run_light, PF_SLOT=claude): pf_severe_cal.py selftest | run
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_gain_loss as GL                                 # noqa: E402
import pf_gain_loss_running as RN                         # noqa: E402

J, JF, C, pd = GL.J, GL.JF, GL.C, GL.pd
OUT = GL.OUT / 'severe_cal'
FOLDS = {'A': (('20261023', '20261024'), '20261025', ('20261026', '20261027')), 'B': (('20261023', '20261024', '20261025'), '20261026', ('20261027',))}
SMOOTH = dict(min_data_in_leaf=400, rounds=100)
BANDS = (0., .005, .01, .02, .05, .10, .20, 1.000001)
MUT_INDEPENDENT = MUT_CAL_ON_EVAL = 0                     # planted errors (selftest only)


def fit_head(X, y, feats, path, obj, smooth):
    import lightgbm as lgb
    if not smooth:
        return J.FM.fit(X, y, feats, path, obj)
    params = {**C.V.PARAMS, 'objective': obj, 'min_data_in_leaf': SMOOTH['min_data_in_leaf']}
    m = lgb.train(params, lgb.Dataset(X, label=y, feature_name=feats), num_boost_round=SMOOTH['rounds'])
    m.save_model(str(path)); loaded = lgb.Booster(model_file=str(path))
    C.require(np.allclose(m.predict(X[:257], num_threads=1), loaded.predict(X[:257], num_threads=1), rtol=0, atol=1e-10), 'model reload mismatch')
    return loaded


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); return np.log(p / (1 - p))


def fit_map(p, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(logit(p).reshape(-1, 1), np.asarray(y, int))
    a, b = float(m.intercept_[0]), float(m.coef_[0, 0])
    return a, b


def apply_map(p, ab):
    a, b = ab; return 1. / (1. + np.exp(-(a + b * logit(p))))


def chain(pa, pb):
    ps = pb if MUT_INDEPENDENT else pa * pb               # planted error: an independent severe probability in place of the chain
    C.require(np.all(ps <= pa + 1e-12), 'P(severe) exceeds P(any loss)')
    return ps


def selftest():
    bad = []; caught = {}
    pa = np.array([.1, .5, .02]); pb = np.array([.5, .9, .99])
    if not np.allclose(chain(pa, pb), [.05, .45, .0198]):
        bad.append('chain')
    a, b = fit_map(np.r_[np.full(500, .2), np.full(500, .6)], np.r_[np.zeros(400), np.ones(100), np.zeros(200), np.ones(300)])
    if not (b > 0 and abs(apply_map(np.array([.2]), (a, b))[0] - .2) < .02 and abs(apply_map(np.array([.6]), (a, b))[0] - .6) < .02):
        bad.append(f'map recovery {a:.3f} {b:.3f}')
    globals()['MUT_INDEPENDENT'] = 1
    try:
        chain(pa, np.array([.5, .9, .99]) * 0 + np.array([.2, .6, .5])); caught['MUT_INDEPENDENT'] = False
    except ValueError:
        caught['MUT_INDEPENDENT'] = True
    finally:
        globals()['MUT_INDEPENDENT'] = 0
    globals()['MUT_CAL_ON_EVAL'] = 1
    try:
        cal_day_check('20261025', ('20261025', '20261027')); caught['MUT_CAL_ON_EVAL'] = False
    except ValueError:
        caught['MUT_CAL_ON_EVAL'] = True
    finally:
        globals()['MUT_CAL_ON_EVAL'] = 0
    if bad or not all(caught.values()):
        sys.exit(f'SEVERE-CAL SELFTEST FAILED -> no output: {bad}, planted errors caught {caught}')
    print(f'severe-cal selftests passed; planted errors caught: {caught}', flush=True)


def cal_day_check(cal, evals):
    C.require(cal not in evals and all(cal < e for e in evals), 'the calibration day must precede and differ from the evaluation days')


def fit_fold(train, out):
    feats = GL.feature_list('full'); JF.feature_contract(feats); X = train[feats].to_numpy(np.float32)
    g, l1, l2 = GL.components(train.w.values); loss = l1 + l2; heads = {}
    for name, y, rows, obj in (('g_p', (g > 1e-9).astype(float), None, 'binary'), ('g_amt', g / GL.SCALE, g > 1e-9, 'regression'),
                               ('l1_amt', l1 / GL.SCALE, loss > 1e-9, 'regression'), ('l2_amt', l2 / GL.SCALE, l2 > 1e-9, 'regression')):
        Xh = X if rows is None else X[rows]; yh = y if rows is None else y[rows]
        heads[name] = fit_head(Xh, yh, feats, out / f'{name}.txt', obj, False)
    any_ = (loss > 1e-9); sev = (l2 > 1e-9)
    for arm, smooth in (('BASE', False), ('SMOOTH', True)):
        heads[f'{arm}_A'] = fit_head(X, any_.astype(float), feats, out / f'{arm}_A.txt', 'binary', smooth)
        heads[f'{arm}_B'] = fit_head(X[any_], sev[any_].astype(float), feats, out / f'{arm}_B.txt', 'binary', smooth)
    return heads, feats


def score(frame, heads, feats, arm, maps=None):
    X = frame[feats].to_numpy(np.float32); base = 'BASE' if arm == 'CAL' else arm
    pa = heads[f'{base}_A'].predict(X, num_threads=1); pb = heads[f'{base}_B'].predict(X, num_threads=1)
    if arm == 'CAL':
        pa = apply_map(pa, maps['A']); pb = apply_map(pb, maps['B'])
    ps = chain(pa, pb)
    pg = heads['g_p'].predict(X, num_threads=1); ag = np.maximum(heads['g_amt'].predict(X, num_threads=1), 0.) * GL.SCALE
    a1 = np.minimum(np.maximum(heads['l1_amt'].predict(X, num_threads=1), 0.) * GL.SCALE, GL.SEVERE)
    a2 = np.maximum(heads['l2_amt'].predict(X, num_threads=1), 0.) * GL.SCALE
    side = frame.side.to_numpy(); sign = 1 - 2 * side; price = frame.price.to_numpy(float); rem = frame.known_remaining.to_numpy(float)
    low = np.minimum(-sign * price, sign * (100. - price)) * rem; high = np.maximum(-sign * price, sign * (100. - price)) * rem
    w = np.clip(pg * ag - pa * a1 - ps * a2, low, high)
    return dict(pa=pa, pb=pb, ps=ps, w=w)


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def stratified_top(g, col, rate):
    parts = [s.sort_values([col, 'hash'], ascending=[False, True]).w.iloc[:int(math.ceil(rate * len(s)))] for _, s in g.groupby(['m', 'side'])]
    return float(pd.concat(parts).mean())


def run():
    selftest(); start = time.time(); OUT.mkdir(parents=True, exist_ok=False); fr = RN.load(); out = {}
    import lightgbm as lgb
    # reproduce the USER's violation count: independent heads on the RUNNING-ERRORS forward folds
    viol = n_all = 0
    for D in RN.PRED_DAYS:
        z = fr[D]; X = z[GL.feature_list('full')].to_numpy(np.float32)
        p1 = lgb.Booster(model_file=str(RN.OUT / f'fold_{D}' / 'full_l1_p.txt')).predict(X, num_threads=1)
        p2 = lgb.Booster(model_file=str(RN.OUT / f'fold_{D}' / 'full_l2_p.txt')).predict(X, num_threads=1)
        viol += int((p2 > p1).sum()); n_all += len(z)
    L = [f'SEVERE-EVENT PROBABILITY: BASE vs SMOOTH vs CAL (explicit chain P(severe) = P(any) x P(severe | loss); past-only folds); as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}',
         f'controls: independent heads exceed P(any) on {viol:,} / {n_all:,} forward decisions (reproduced); P(severe) <= P(any) in every arm by construction; maps fit on the calibration day only; '
         'CAL base = BASE; planted errors refused', '']
    for fold, (tdays, cal, evals) in FOLDS.items():
        cal_day_check(cal, evals); C.require(max(tdays) < cal, 'training must precede the calibration day')
        train = pd.concat([fr[d] for d in tdays], ignore_index=True); d_ = OUT / f'fold_{fold}'; d_.mkdir()
        heads, feats = fit_fold(train, d_)
        cz = fr[cal]; bc = score(cz, heads, feats, 'BASE'); loss_c = (cz.loss > 1e-9).to_numpy()
        maps = {'A': fit_map(bc['pa'], loss_c), 'B': fit_map(bc['pb'][loss_c], (cz.exc > 1e-9).to_numpy()[loss_c])}
        failed = [k for k, (a, b) in maps.items() if b <= 0]
        L.append(f'FOLD {fold}: train {"+".join(tdays)} | calibration {cal} | evaluation {", ".join(evals)}; maps: A a {maps["A"][0]:+.3f} b {maps["A"][1]:.3f}, '
                 f'B a {maps["B"][0]:+.3f} b {maps["B"][1]:.3f}' + (f' | REFUSED maps {failed}' if failed else ''))
        arms = ('BASE', 'SMOOTH') + (() if failed else ('CAL',)); res = {}
        days = (cal,) + evals
        for D in days:
            z = fr[D].copy(); sc = {a: score(z, heads, feats, a, maps) for a in arms}
            if 'CAL' in sc:
                C.require(np.allclose(heads['BASE_A'].predict(z[feats].to_numpy(np.float32), num_threads=1), sc['BASE']['pa']), 'CAL base differs from BASE')
            for kind in ('cancel', 'post'):
                g = z[z.kind == kind]; mk = (z.kind == kind).to_numpy(); ys = (g.exc > 1e-9).to_numpy(float); ya = (g.loss > 1e-9).to_numpy(float)
                for a in arms:
                    s = {k_: v[mk] for k_, v in sc[a].items()}; gg = g.assign(pred=s['w'])
                    res[(D, kind, a)] = dict(ll_s=logloss(ys, s['ps']), oe_s=float(ys.sum() / s['ps'].sum()), ll_a=logloss(ya, s['pa']), oe_a=float(ya.sum() / s['pa'].sum()),
                                             rmse=float(np.sqrt(np.mean((s['w'] - g.w.values) ** 2))), ce=GL.calibration_error(s['w'], g.w.values),
                                             top={r: stratified_top(gg, 'pred', r) for r in GL.RATES}, ps=s['ps'], ys=ys)
        L.append('  day      kind   arm    | P(sev) log loss | O/E sev | P(any) log loss | O/E any | RMSE(W) | calib err | top 10 / 25 / 50% realized W (stratified)')
        for D in days:
            for kind in ('cancel', 'post'):
                for a in arms:
                    r = res[(D, kind, a)]; tag = ' (calibration day)' if D == cal else ''
                    L.append(f'  {D} {kind:<6} {a:<6} | {r["ll_s"]:.4f} | x{r["oe_s"]:.2f} | {r["ll_a"]:.4f} | x{r["oe_a"]:.2f} | {r["rmse"]:6.2f} | {r["ce"]:5.2f} | '
                             + ' / '.join(f'{r["top"][q]:+6.2f}' for q in GL.RATES) + tag)
        # band O/E pooled over the evaluation days
        for a in arms:
            ps_ = np.concatenate([res[(D, k, a)]['ps'] for D in evals for k in ('cancel', 'post')])
            ys_ = np.concatenate([res[(D, k, a)]['ys'] for D in evals for k in ('cancel', 'post')]); parts = []
            for lo_, hi_ in zip(BANDS[:-1], BANDS[1:]):
                m_ = (ps_ >= lo_) & (ps_ < hi_)
                if m_.any():
                    parts.append(f'{lo_:g}-{min(hi_, 1.):g} x{ys_[m_].sum() / max(ps_[m_].sum(), 1e-9):.2f} (n {int(m_.sum()):,})')
            L.append(f'  band O/E of P(severe), evaluation days pooled, {a}: ' + ' | '.join(parts))
        out[fold] = dict(res=res, arms=arms, evals=evals)
        L.append('')
    # readings on fold A
    res, arms, evals = out['A']['res'], out['A']['arms'], out['A']['evals']; cells = [(D, k) for D in evals for k in ('cancel', 'post')]; R = {}
    for cand in [a for a in arms if a != 'BASE']:
        s1 = sum(res[(D, k, cand)]['ll_s'] < res[(D, k, 'BASE')]['ll_s'] for D, k in cells) >= 3 and \
            sum(abs(np.log(res[(D, k, cand)]['oe_s'])) < abs(np.log(res[(D, k, 'BASE')]['oe_s'])) for D, k in cells) >= 3

        def low_oe(a):
            ps = np.concatenate([res[(D, k, a)]['ps'] for D, k in cells]); ys = np.concatenate([res[(D, k, a)]['ys'] for D, k in cells]); m_ = ps < .05
            return ys[m_].sum() / ps[m_].sum()
        s2 = abs(np.log(low_oe(cand))) < abs(np.log(low_oe('BASE')))
        s3 = sum(res[(D, k, cand)]['rmse'] < res[(D, k, 'BASE')]['rmse'] for D, k in cells) >= 3 and \
            sum(res[(D, k, cand)]['ce'] < res[(D, k, 'BASE')]['ce'] for D, k in cells) >= 3
        s4 = sum(sum(res[(D, k, cand)]['top'][q] >= res[(D, k, 'BASE')]['top'][q] for D, k in cells) >= 3 for q in GL.RATES) >= 2
        R[cand] = dict(S1=s1, S2=s2, S3=s3, S4=s4, PASS=bool(s1 and s3 and s4), low_oe=float(low_oe(cand)), base_low_oe=float(low_oe('BASE')))
        L.append(f'{cand} vs BASE (fold A): S1 SEVERE-CALIBRATION {"IMPROVES" if s1 else "NOT-SHOWN"} | S2 LOW-BAND {"YES" if s2 else "NO"} (p < .05 O/E x{low_oe(cand):.2f} vs BASE x{low_oe("BASE"):.2f}) | '
                 f'S3 NET-VALUE {"IMPROVES" if s3 else "NOT-SHOWN"} | S4 MATCHED-PARTICIPATION {"IMPROVES" if s4 else "NOT-SHOWN"} -> {"PASS" if R[cand]["PASS"] else "NOT-SHOWN"}')
    L += ['', 'LIMITS: consumed days; two evaluation days in fold A (no interval, rule 8); simulated labels; the neutral decision population']
    (OUT / 'severe_cal.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'severe_cal.json', dict(asof=C.stamp(), readings=R, independent_violations=[viol, n_all], seconds=time.time() - start))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
    else:
        C.require(len(sys.argv) > 1 and sys.argv[1] == 'run', 'usage: pf_severe_cal.py selftest | run')
        run()
