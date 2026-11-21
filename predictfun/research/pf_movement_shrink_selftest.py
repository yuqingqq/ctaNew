"""SELFTEST of pf_movement_shrink (PI0-MOVEMENT-SHRINK-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M; any failed check exits
with FAIL. Hand rows only (no tape, no world):
 1. bounds and the prior: the bid / ask physical bounds, a binding projection, E[q] = 0 -> a = 0;
 2. fit_group: mu = sum(w M) / sum(w z); lambda = the weighted least-squares slope (against an independent numpy solve), clipped to [0, 1] above and
    below; the zero denominator; the support floors (29 rows; 9 fills); a training row of the evaluation day or of another side refused;
 3. predict: lambda 0 reproduces the original E[qm], V and masks bit for bit; the prior and candidate arms; an unsupported group keeps the original;
 4. the tie rules (entry V = 0 WAITs, review V = 0 KEEPs); the chosen statistics, severe sums, concentration and changed counts;
 5. the null: chosen counts per stratum preserved, a weight mismatch detected, free / degenerate strata, the percentile;
 6. the readings: a hypothetical positive exposure PASSes end to end; an adversarial mask and all-WAIT do not; equal masks, thin support and a null
    weight mismatch are INCONCLUSIVE; the prediction booleans;
 7. stage_rows: a reference table in other units is audit_mismatch; populations / check_manifest refuse a changed key;
 8. the writers: score_all on hand rows writes every output; build_manifest / run_fit / run_score refuse before the freeze; write-once fit."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'MOVEMENT SHRINK SELFTEST FAILED'


def frame(M, rows):
    pd = M.pd
    base = dict(stage='entry', day=M.EVAL_DAY, m=0, t_ms=0, side=0, weight=1., cday=M.EVAL_DAY, chour=0, mid=50., status='ok', hbin='(2000,5000]', qbin='(0,1]',
                obin='1', P=.5, Eq=10., Eqm=0., V=1., edge=.1, q=0., value=0., z=0., M=0.)
    out = []
    for i, r in enumerate(rows):
        d = dict(base, key=json.dumps(['d', i]), m=i, t_ms=i); d.update(r); out.append(d)
    return pd.DataFrame(out)


def run(M):  # noqa: C901
    bad = []; pd = M.pd
    # ---- 1 bounds and the prior
    try:
        L0, U0 = M.bounds(np.array([30.]), 0); L1, U1 = M.bounds(np.array([30.]), 1)
        a, b_ = M.prior(np.array([.5, .5, .5, .5]), np.array([10., 10., 0., 1.]), np.array([30., 30., 30., 30.]), 0, -100.)
        a1, b1 = M.prior(np.array([.5]), np.array([1.]), np.array([30.]), 1, 100.)
        if not (L0[0] == -30. and U0[0] == 70. and L1[0] == -70. and U1[0] == 30. and list(a) == [-50., -50., 0., -30.] and list(b_) == [False, False, True, True]
                and a1[0] == 30. and bool(b1[0])):
            bad.append(f'prior / bounds ({list(a)}, {list(b_)}, {a1})')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'prior: stopped ({e!r})'[:200])
    # ---- 2 fit_group
    try:
        rng = np.random.default_rng(1); n = 40
        T = frame(M, [dict(day=M.TRAIN_DAY, P=float(p), Eq=10., mid=50., Eqm=float(e), weight=float(w), z=float(z), q=10. * float(z), M=float(mm) if z else 0.)
                      for p, e, w, z, mm in zip(rng.uniform(.1, .9, n), rng.uniform(-30, 0, n), rng.uniform(1, 3, n), (np.arange(n) % 3 == 0), rng.uniform(-80, 20, n))])
        g = M.fit_group(T, 0); w, Mv, z, b = (T[c].to_numpy(float) for c in ('weight', 'M', 'z', 'Eqm')); mu = (w * Mv).sum() / (w * z).sum()
        a, _ = M.prior(T.P, T.Eq, T.mid, 0, mu); d = a - b; sw = np.sqrt(w); lam_ls = np.linalg.lstsq((sw * d)[:, None], sw * (Mv - b), rcond=None)[0][0]
        if not (g['status'] == 'ok' and abs(g['mu'] - mu) < 1e-9 and abs(g['lam_raw'] - lam_ls) < 1e-9 and g['lam'] == min(1., max(0., lam_ls)) and g['fills'] == 14 and g['n'] == 40):
            bad.append(f"fit_group: mu / lambda against the independent solve ({g}, {mu}, {lam_ls})")
        bq = np.linspace(-30., 0., n); wq = np.linspace(1., 2., n); bbar = float((wq * bq).sum() / wq.sum())
        mk_ = lambda k: frame(M, [dict(day=M.TRAIN_DAY, P=1., Eq=10., mid=50., Eqm=float(bb), weight=float(ww), z=1., q=10., M=float(bb + k * (bbar - bb))) for bb, ww in zip(bq, wq)])
        gh, gl, gm = M.fit_group(mk_(2.), 0), M.fit_group(mk_(-1.), 0), M.fit_group(mk_(.3), 0)
        if not (abs(gh['lam_raw'] - 2.) < 1e-9 and gh['lam'] == 1. and abs(gl['lam_raw'] + 1.) < 1e-9 and gl['lam'] == 0. and abs(gm['lam'] - .3) < 1e-9 and abs(gm['mu'] - bbar) < 1e-9):
            bad.append(f"fit_group: the exact mixture / the clip ({gh['lam_raw']}, {gh['lam']}; {gl['lam_raw']}, {gl['lam']}; {gm['lam']})")
        T0 = T.assign(Eq=0.); g0 = M.fit_group(T0.assign(Eqm=0.), 0)
        if g0['status'] != 'zero_denominator' or g0['lam'] != 0.:
            bad.append(f"fit_group: the zero denominator ({g0['status']})")
        if M.fit_group(T.iloc[:29], 0)['status'] != 'insufficient_fit_support' or M.fit_group(T[(T.z == 0) | (T.index < 27)].iloc[:40], 0)['status'] != 'insufficient_fit_support':
            bad.append('fit_group: a support floor')
        if M.fit_group(T.iloc[:30].assign(z=[1.] * 10 + [0.] * 20, M=[-5.] * 10 + [0.] * 20), 0)['status'] != 'ok':
            bad.append('fit_group: exactly 30 rows and 10 fills refused')
        for bad_T, why in ((T.assign(day=[M.TRAIN_DAY] * 39 + [M.EVAL_DAY]), 'an evaluation-day row'), (T.assign(side=[0] * 39 + [1]), 'a row of another side')):
            try:
                M.fit_group(bad_T, 0); bad.append(f'fit_group: {why} was accepted')
            except Exception:                                                               # noqa: BLE001
                pass
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'fit_group: stopped ({e!r})'[:200])
    # ---- 3 predict, 4 ties and chosen statistics
    try:
        E = frame(M, [dict(side=i % 2, P=.4, Eq=5., mid=40., Eqm=-2. + i, V=-1.5 + .7 * i, value=float(i - 3)) for i in range(8)])
        P0 = {f'entry|{s}': dict(mu=-20., lam=0.) for s in M.SIDES}; P1 = {f'entry|{s}': dict(mu=-20., lam=.25) for s in M.SIDES}
        P2 = {'entry|0': dict(mu=None, lam=0.), 'entry|1': dict(mu=-20., lam=.25)}
        e0, e1, e2 = M.predict(E, P0, 'entry'), M.predict(E, P1, 'entry'), M.predict(E, P2, 'entry')
        if not (np.array_equal(e0.Eqm_candidate.to_numpy(), E.Eqm.to_numpy()) and np.array_equal(e0.V_candidate.to_numpy(), E.V.to_numpy())
                and np.array_equal(M.masks(e0, 'entry')['candidate'], M.masks(e0, 'entry')['original'])):
            bad.append('predict: lambda 0 does not reproduce the original vectors / mask exactly')
        a_exp = np.clip(.4 * -20., 5. * np.where(E.side == 0, -40., -60.), 5. * np.where(E.side == 0, 60., 40.))
        if not (np.allclose(e1.a, a_exp) and np.allclose(e1.Eqm_candidate, E.Eqm + .25 * (a_exp - E.Eqm)) and np.allclose(e1.V_candidate, E.V + .25 * (a_exp - E.Eqm))
                and np.allclose(e1.Eqm_prior, a_exp) and np.allclose(e1.V_prior, E.V + (a_exp - E.Eqm))):
            bad.append('predict: the prior / candidate arms')
        s0 = (e2.side == 0).to_numpy()
        if not (np.array_equal(e2.Eqm_prior.to_numpy()[s0], E.Eqm.to_numpy()[s0]) and np.array_equal(e2.Eqm_candidate.to_numpy()[s0], E.Eqm.to_numpy()[s0])):
            bad.append('predict: an unsupported group does not keep the original forecast')
        Z = frame(M, [dict(V=0.), dict(V=1e-9), dict(V=-1e-9)])
        me, mr = M.masks(M.predict(Z, P0, 'entry'), 'entry'), M.masks(M.predict(Z.assign(stage='first_review'), {f'first_review|{s}': dict(mu=-20., lam=0.) for s in M.SIDES}, 'first_review'), 'first_review')
        if list(me['original']) != [False, True, False] or list(mr['original']) != [True, True, False] or me['none'].any() or not me['all'].all():
            bad.append(f"masks: the tie rules ({list(me['original'])}, {list(mr['original'])})")
        G = frame(M, [dict(value=v, weight=w, m=i // 2, t_ms=0) for i, (v, w) in enumerate([(10., 1.), (-40., 2.), (5., 1.), (-31., 1.), (2., 1.), (0., 1.)])])
        cs = M.chosen_stats(G, np.array([True, True, True, True, False, False]))
        if not (cs['chosen'] == 4 and cs['chosen_weight'] == 5. and abs(cs['J'] - (10. - 80. + 5. - 31.) / 7.) < 1e-12 and abs(cs['mean_chosen'] - (10. - 80. + 5. - 31.) / 5.) < 1e-12
                and cs['severe_n'] == 2 and cs['severe_sum'] == -111. and cs['worlds'] == 2 and abs(cs['ess'] - 25. / 7.) < 1e-12):
            bad.append(f'chosen_stats ({cs})')
        ac, mk = M.actions(e1, 'entry'); r_ = {(x['side'], x['rule']): x for x in ac}
        if not (r_[('all', 'none')]['chosen'] == 0 and r_[('all', 'all')]['chosen'] == 8 and r_[('all', 'candidate')]['changed_vs_original'] == int((mk['candidate'] != mk['original']).sum())
                and r_[('0', 'all')]['n'] == 4):
            bad.append('actions: the rules / slices / changed counts')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'predict / masks / actions: stopped ({e!r})'[:220])
    # ---- 5 the null, 6 the readings end to end
    try:
        rows = []
        for h in range(10):
            for j in range(6):
                pos = j < 3
                rows.append(dict(chour=h, m=100 * h + j, t_ms=h, P=.5, Eq=10., mid=50., V=1., Eqm=-3.5 if pos else 0., value=10. if pos else -10., edge=.45 if pos else .1))
        E = frame(M, rows); prm = {f'entry|{s}': dict(mu=-10., lam=.5, status='ok') for s in M.SIDES}
        man = dict(eval={'entry': sorted(E.key), 'first_review': []}, descriptive={'first_pull': []})
        with tempfile.TemporaryDirectory(dir=str(M.PA.R2.I.K.IO)) as tmp:
            js = M.score_all(E.assign(stage='entry'), man, {**prm, **{f'first_review|{s}': dict(mu=None, lam=0., status='insufficient_fit_support') for s in M.SIDES}}, Path(tmp))
            pe = next(p for p in js['predicates'] if p['head'] == 'entry')
            names = ('scores', 'calibration', 'actions', 'null_draws', 'eval_rows')
            if not all((Path(tmp) / f'{n}.csv').exists() for n in names):
                bad.append('score_all: an output is missing')
            nd = pd.read_csv(Path(tmp) / 'null_draws.csv')
        if not (pe['decision'] == 'PASS' and abs(pe['J']['candidate'] - 5.) < 1e-12 and pe['J']['original'] == 0. and pe['J']['prior'] == 0. and pe['free_strata'] == 10
                and len(nd[nd['head'] == 'entry']) == M.NULL_DRAWS):
            bad.append(f"readings: the hypothetical positive exposure is not selectable ({pe['decision']}, {pe['J']}, {pe['inconclusive_reasons']})")
        nu, J = M.null(E, M.masks(M.predict(E, prm, 'entry'), 'entry')['candidate'])
        if not (nu['chosen'] == 30 and nu['free_strata'] == 10 and nu['free_rows'] == 60 and nu['weight_mismatch_draws'] == 0 and nu['percentile'] == 1. and nu['null_p95'] < 5.):
            bad.append(f'null: the matched draws ({nu})')
        Ew = E.assign(weight=[1. + (i % 6) for i in range(len(E))]); nw, _ = M.null(Ew, M.masks(M.predict(Ew, prm, 'entry'), 'entry')['candidate'])
        if nw['weight_mismatch_draws'] == 0:
            bad.append('null: a within-stratum weight difference was not detected')
        nall, _ = M.null(E, np.ones(len(E), bool))
        if nall['free_strata'] != 0 or nall['percentile'] != 1.:
            bad.append(f'null: a degenerate mask ({nall})')
        adv = E.assign(Eqm=np.where(E.value > 0, 0., -3.5)); pa_ = M.predict(adv, prm, 'entry'); mka = M.masks(pa_, 'entry')
        SCa = M.scores(pa_, 'entry'); ACa, _ = M.actions(pa_, 'entry'); NUa, _ = M.null(pa_, mka['candidate'])
        if M.predicates(prm, SCa, ACa, NUa, mka, 'entry')['decision'] == 'PASS':
            bad.append('readings: an adversarial mask passed')
        wt = E.assign(V=-1.); pw = M.predict(wt, prm, 'entry'); mkw = M.masks(pw, 'entry'); NUw, _ = M.null(pw, mkw['candidate'])
        if M.predicates(prm, M.scores(pw, 'entry'), M.actions(pw, 'entry')[0], NUw, mkw, 'entry')['decision'] == 'PASS':
            bad.append('readings: all-WAIT passed')
        p0 = {f'entry|{s}': dict(mu=-10., lam=0., status='ok') for s in M.SIDES}; e0 = M.predict(E, p0, 'entry'); m0 = M.masks(e0, 'entry'); N0, _ = M.null(e0, m0['candidate'])
        r0 = M.predicates(p0, M.scores(e0, 'entry'), M.actions(e0, 'entry')[0], N0, m0, 'entry')
        if r0['decision'] != 'INCONCLUSIVE' or 'candidate_mask_equals_original' not in r0['inconclusive_reasons']:
            bad.append(f"readings: equal masks are not INCONCLUSIVE ({r0['decision']})")
        Es = E[E.chour < 4]; es = M.predict(Es, prm, 'entry'); ms = M.masks(es, 'entry'); Ns, _ = M.null(es, ms['candidate'])
        rs = M.predicates(prm, M.scores(es, 'entry'), M.actions(es, 'entry')[0], Ns, ms, 'entry')
        if rs['decision'] != 'INCONCLUSIVE' or 'inadequate_movable_support' not in rs['inconclusive_reasons']:
            bad.append(f"readings: thin movable support is not INCONCLUSIVE ({rs['decision']}, {rs['inconclusive_reasons']})")
        ew = M.predict(Ew, prm, 'entry'); mw = M.masks(ew, 'entry'); Nw2, _ = M.null(ew, mw['candidate'])
        rw = M.predicates(prm, M.scores(ew, 'entry'), M.actions(ew, 'entry')[0], Nw2, mw, 'entry')
        if rw['decision'] != 'INCONCLUSIVE' or 'null_weight_mismatch' not in rw['inconclusive_reasons']:
            bad.append('readings: a null weight mismatch is not INCONCLUSIVE')
        pn = {f'entry|{s}': dict(mu=None, lam=0., status='insufficient_fit_support') for s in M.SIDES}; en = M.predict(E, pn, 'entry'); mn = M.masks(en, 'entry'); Nn, _ = M.null(en, mn['candidate'])
        if 'no_fit_support' not in M.predicates(pn, M.scores(en, 'entry'), M.actions(en, 'entry')[0], Nn, mn, 'entry')['inconclusive_reasons']:
            bad.append('readings: no fit support is not flagged')
        TY = dict(A=(1., -3.5), C=(1., 0.), B=(-1., -6.5), N=(-1., -5.))

        def typed(spec):
            rr = []
            for h, items in enumerate(spec):
                for j, (ty, val) in enumerate(items):
                    rr.append(dict(chour=h, m=1000 * h + j, t_ms=h, P=.5, Eq=10., mid=50., V=TY[ty][0], Eqm=TY[ty][1], value=val))
            return frame(M, rr)

        def decide(Et, pr_=prm):
            ep = M.predict(Et, pr_, 'entry'); mk_ = M.masks(ep, 'entry'); nu_, _ = M.null(ep, mk_['candidate'])
            return M.predicates(pr_, M.scores(ep, 'entry'), M.actions(ep, 'entry')[0], nu_, mk_, 'entry')
        AC6 = [('A', 10.)] * 3 + [('C', -10.)] * 3
        cases = [('p95', typed([[('A', 10.)] * 3 + [('N', 10.)] * 3] * 6 + [[('C', -10.)] * 6] * 2)),
                 ('prior', typed([AC6] * 10 + [[('B', 30.)] * 6] * 2)),
                 ('original', typed([[('A', 10.)] * 3 + [('N', -10.)] * 3] * 10 + [[('C', 30.)] * 6] * 2)),
                 ('positive_J', typed([[('A', -1.)] * 3 + [('N', -10.)] * 3] * 10 + [[('C', -20.)] * 6] * 2 + [[('B', -30.)] * 6] * 2))]
        for nm_, Et in cases:
            r_ = decide(Et)
            if r_['decision'] != 'FAIL' or r_['inconclusive_reasons']:
                bad.append(f"readings: the {nm_} case is {r_['decision']} {r_['inconclusive_reasons']} (J {r_['J']}, p95 {r_['null_p95']})")
        r1 = decide(typed([AC6] * 10), {f'entry|{s}': dict(mu=-10., lam=1., status='ok') for s in M.SIDES})
        if r1['decision'] != 'INCONCLUSIVE' or 'candidate_mask_equals_prior' not in r1['inconclusive_reasons']:
            bad.append(f"readings: a candidate equal to the prior is not INCONCLUSIVE ({r1['decision']})")
        if decide(typed([AC6] * 10))['decision'] != 'PASS':
            bad.append('readings: the typed positive exposure is not PASS')
        Wt = frame(M, [dict(weight=1., P=.5, z=1., q=10., M=-20., Eqm=-4., V=1. * .1 * 10. - 4. + 0., edge=.1, Eq=10., value=.1 * 10. - 20.),
                       dict(weight=3., P=.2, z=1., q=5., M=-2., Eqm=-1., V=.15 * 10. - 1., edge=.15, Eq=10., value=.15 * 5. - 2.)])
        sw_ = {(r['side'], r['arm']): r for r in M.scores(M.predict(Wt, P0, 'entry'), 'entry')}[('all', 'original')]; wv = Wt.weight.to_numpy()
        cm_ = (Wt.Eqm / Wt.P - Wt.M).to_numpy()
        if not (abs(sw_['mse_m'] - np.average(((Wt.Eqm - Wt.M) ** 2).to_numpy(), weights=wv)) < 1e-12 and abs(sw_['c_m'] - np.average(cm_, weights=wv)) < 1e-12
                and abs(sw_['c_m_unweighted'] - cm_.mean()) < 1e-12 and abs(sw_['brier'] - np.average(((Wt.P - Wt.z) ** 2).to_numpy(), weights=wv)) < 1e-12
                and abs(sw_['mse_mech'] - np.average(((Wt.edge * Wt.Eq - Wt.value) ** 2).to_numpy(), weights=wv)) < 1e-12 and sw_['weighted_fills'] == 4. and sw_['fills'] == 2):
            bad.append(f'scores: the weighted movement MSE / conditional error / Brier / mechanical baseline ({sw_})')
        try:
            with tempfile.TemporaryDirectory(dir=str(M.PA.R2.I.K.IO)) as tmp:
                Ed = E.assign(stage='entry', day=[M.TRAIN_DAY] + [M.EVAL_DAY] * (len(E) - 1))
                M.score_all(Ed, man, {**prm, **{f'first_review|{s}': dict(mu=None, lam=0., status='insufficient_fit_support') for s in M.SIDES}}, Path(tmp))
            bad.append('score_all: an evaluation row of the training day was accepted')
        except Exception as e:                                                              # noqa: BLE001
            if 'evaluation rows differ' not in str(e):
                bad.append(f'score_all: refused a training-day row for another reason ({e!r})'[:200])
        SCx = [dict(head='entry', side='all', arm='original', mse_m=10., mse_V=8., bias_m=.1), dict(head='entry', side='all', arm='candidate', mse_m=5., mse_V=9., bias_m=2.)]
        ACx = [dict(head='entry', side='all', rule=r_, J=0., chosen_weight=0.) for r_ in M.RULES]; mkx = {r_: np.zeros(3, bool) for r_ in M.RULES}
        px = M.predicates({f'entry|{s}': dict(status='ok') for s in M.SIDES}, SCx, ACx, dict(free_strata=0, free_rows=0, weight_mismatch_draws=0, null_p95=0.), mkx, 'entry')
        if px['prediction']['all'] != dict(movement_mse_improves=True, value_mse_improves=False, movement_bias_reduced=False):
            bad.append(f"readings: the prediction booleans when MSE and bias disagree ({px['prediction']['all']})")
        sc = {(r['side'], r['arm']): r for r in M.scores(M.predict(E, prm, 'entry'), 'entry')}; pr = pe['prediction']['all']
        if pr['movement_mse_improves'] != (sc[('all', 'candidate')]['mse_m'] < sc[('all', 'original')]['mse_m']) or pr['value_mse_improves'] != (sc[('all', 'candidate')]['mse_V'] < sc[('all', 'original')]['mse_V']):
            bad.append('readings: the prediction booleans')
        S1 = sc[('all', 'candidate')]; Ep = M.predict(E, prm, 'entry')
        if not (abs(S1['mse_m'] - float(((Ep.Eqm_candidate - Ep.M) ** 2).mean())) < 1e-9 and abs(S1['bias_V'] - float((Ep.V_candidate - Ep.value).mean())) < 1e-9 and S1['fills'] == 0):
            bad.append('scores: the movement MSE / value bias')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'null / readings: stopped ({e!r})'[:220])
    # ---- 7 stage_rows reconciliation and the manifest populations
    try:
        vec = lambda **kw: [float(kw.get(n_, 0.)) for n_ in M.RT.FEATS]; t0 = 1792976399000

        def arow(m, side=0, q=0., M_=0., fills=(), mark=52., day=None):
            mo = dict(P=.3, Eq=9., Eqm=-3., V=.5 * 9. - 3., edge=.5); mr = dict(P=.3, Eq=9., Eqm=-3., V=1. * 9. - 3., edge=1.)
            ent = dict(status='ok', dec_status='model', action='POST', t=0., inputs=vec(pf_mid=50.5, fp_queue=10., fp_remaining=30., t_to_end=5000., ctx_generation_ordinal=1.),
                       moments=mo, target=dict(q=q, value=.5 * q + M_, qm=M_ if q > 0 else 0.), view=dict(received=1.))
            Mr = q * (mark - 51.)
            rev = dict(ent, t=1500., action='PULL', venue=dict(ahead=500.), effect=1537., moments=mr, inputs=vec(pf_mid=51., fp_queue=10., fp_remaining=30., t_to_end=3500.,
                                                                                                               ctx_generation_ordinal=1.),
                       target=dict(q=q, value=1. * q + Mr, qm=Mr if q > 0 else 0.))
            st = dict(entry=ent, first_review=rev, first_pull=dict(status='no_pull'))
            return dict(day=day or M.TRAIN_DAY, m=m, t_ms=t0 + 1000 * m, side=side, gid=0, decision_ms=0, weight=1., n_cell=1, cell=list(M.PA.cell_of(t0 + 1000 * m, 0, side)),
                        status='ok', held=dict(gid=0, act=58., mark=mark, fills=[list(f) for f in fills]), fills=[], stages=st)
        ar = [arow(0, q=10., M_=15., fills=[(2000., 0, 0, 10., 50., 0)]), arow(1), arow(2, day=M.EVAL_DAY)]
        O1 = M.stage_rows(ar)
        e1 = O1[O1.stage.isin(['entry', 'first_review'])]
        k0_ = json.dumps(list(M.MA.GKEY(ar[0]))); r0e = O1[(O1.key == k0_) & (O1.stage == 'entry')].iloc[0]; r0r = O1[(O1.key == k0_) & (O1.stage == 'first_review')].iloc[0]
        if not ((e1.status == 'ok').all() and r0e.mid == 50.5 and r0r.mid == 51. and int(r0e.chour) == 0 and int(r0r.chour) == 1
                and set(O1[O1.stage == 'first_pull'].status) == {'not_eligible'} and e1.chour.notna().all()):
            bad.append(f"stage_rows: the statuses / mid / cells of consistent rows ({dict(zip(O1.stage, O1.status))})")
        ref = O1.assign(cohort='admitted', acc_status=O1.acc_status); k0 = e1.index[0]; ref.loc[k0, 'V'] = float(ref.loc[k0, 'V']) * 100.
        O2 = M.stage_rows(ar, ref); O3 = M.stage_rows(ar, ref.drop(index=e1.index[1]))
        if not (O2.loc[k0, 'status'] == 'audit_mismatch' and (O2.drop(index=k0)[O2.drop(index=k0).stage != 'first_pull'].status == 'ok').all()
                and O3.loc[e1.index[1], 'status'] == 'audit_mismatch'):
            bad.append('stage_rows: a reference value in other units or a missing reference row was accepted')
        pop = M.populations(O1)
        if not (len(pop['train']['entry']) == 2 and len(pop['eval']['entry']) == 1 and pop['descriptive']['first_pull'] == []):
            bad.append(f"populations: the key lists ({ {k: len(v) for k, v in pop['train'].items()} })")
        man = dict(version=M.VERSION, train=pop['train'], eval=pop['eval'], descriptive=pop['descriptive']); M.check_populations(man, O1)
        for bad_man, why in ((dict(man, train=dict(pop['train'], entry=pop['train']['entry'][:-1])), 'a dropped training key'),
                             (dict(man, eval=dict(pop['eval'], first_review=[])), 'a dropped evaluation key'), (dict(man, version='other'), 'another version')):
            try:
                M.check_populations(bad_man, O1); bad.append(f'check_populations: {why} was accepted')
            except Exception:                                                               # noqa: BLE001
                pass
        fa_ = O1[(O1.stage == 'entry') & (O1.day == M.TRAIN_DAY)]
        if not (abs(float(fa_[fa_.z > 0].M.iloc[0]) - 15.) < 1e-12 and float(fa_[fa_.z == 0].M.iloc[0]) == 0.):
            bad.append('stage_rows: M of the filled / unfilled rows')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'stage_rows / populations: stopped ({e!r})'[:220])
    # ---- 8 the freeze guards
    try:
        real = (M.frozen, M.inputs_bound)
        M.frozen = lambda: False; M.inputs_bound = lambda *a_: False
        with tempfile.TemporaryDirectory(dir=str(M.PA.R2.I.K.IO)) as tmp:
            for f, why in ((lambda: M.run_fit(Path(tmp) / 'fit.json'), 'run_fit'), (lambda: M.run_score(Path(tmp) / 'out'), 'run_score'),
                           (lambda: M.build_manifest(Path(tmp) / 'manifest.json'), 'build_manifest')):
                try:
                    f(); bad.append(f'{why}: ran before the freeze')
                except Exception as e:                                                      # noqa: BLE001
                    if 'committed' not in str(e) and 'frozen' not in str(e):
                        bad.append(f'{why}: refused for another reason ({e!r})'[:200])
            (Path(tmp) / 'fit.json').write_text('{}')
            try:
                M.run_fit(Path(tmp) / 'fit.json'); bad.append('run_fit: overwrote an existing fit')
            except Exception as e:                                                          # noqa: BLE001
                if 'write-once' not in str(e):
                    bad.append(f'run_fit: refused an existing fit for another reason ({e!r})'[:200])
            (Path(tmp) / 'out').mkdir(); (Path(tmp) / 'out' / 'audit.json').write_text('{}')
            try:
                M.run_score(Path(tmp) / 'out', rec=Path(tmp)); bad.append('run_score: overwrote existing scores')
            except Exception as e:                                                          # noqa: BLE001
                if 'immutable' not in str(e):
                    bad.append(f'run_score: refused existing scores for another reason ({e!r})'[:200])
            real_ce = M.committed_equal; M.frozen = lambda: True; M.inputs_bound = lambda *a_: True
            try:
                M.committed_equal = lambda rel, local: False
                try:
                    M.run_score(Path(tmp) / 'out2', rec=Path(tmp)); bad.append('run_score: scored before the fit commit')
                except Exception as e:                                                      # noqa: BLE001
                    if 'fit is not committed' not in str(e):
                        bad.append(f'run_score: refused an uncommitted fit for another reason ({e!r})'[:200])
                M.committed_equal = lambda rel, local: True
                (Path(tmp) / 'manifest.json').write_text('{}'); (Path(tmp) / 'fit.json').write_text(json.dumps(dict(sources={}, manifest_sha256='x')))
                try:
                    M.run_score(Path(tmp) / 'out3', rec=Path(tmp)); bad.append('run_score: a fit of another identity was used')
                except Exception as e:                                                      # noqa: BLE001
                    if 'another identity' not in str(e):
                        bad.append(f'run_score: refused a foreign fit for another reason ({e!r})'[:200])
                M.frozen, M.inputs_bound = real
                M.committed_equal = lambda rel, local: not rel.endswith('manifest.json')
                if M.frozen():
                    bad.append('frozen: true without a committed manifest')
                M.committed_equal = lambda rel, local: not rel.startswith(M.ADM_REC_REL)
                if M.inputs_bound():
                    bad.append('inputs_bound: true without committed admission records')
            finally:
                M.committed_equal = real_ce
        M.frozen, M.inputs_bound = real
        man_i = dict(admission_manifest_sha256='m', audit_opportunities_sha256='o', admission_records_sha256={'0': 'r'}); M.check_inputs(man_i, 'm', {'0': 'r'}, 'o')
        for args, why in ((('x', {'0': 'r'}, 'o'), 'the admission manifest'), (('m', {'0': 'x'}, 'o'), 'a record'), (('m', {'0': 'r'}, 'x'), 'the audit table')):
            try:
                M.check_inputs(man_i, *args); bad.append(f'check_inputs: another {why} was accepted')
            except Exception:                                                               # noqa: BLE001
                pass
        if not M.committed_equal('predictfun/research/pf_movement_audit.py', M.HERE / 'pf_movement_audit.py') or M.committed_equal('predictfun/research/no_such_file.py', M.HERE / 'pf_movement_audit.py'):
            bad.append('committed_equal: a committed file refused or a missing one accepted')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'guards: stopped ({e!r})'[:200])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('movement shrink selftest passed: bounds and the prior; mu / lambda against an independent solve, the clip, the zero denominator, support floors, '
          'chronology and side refusals; lambda 0 exact; prior / candidate arms; unsupported groups; tie rules; chosen statistics; the matched null; PASS on a '
          'positive exposure, not on adversarial / all-WAIT; INCONCLUSIVE on equal masks, thin support, null weight mismatch, no fit support; populations; '
          'the unit refusal; the freeze guards and write-once')
