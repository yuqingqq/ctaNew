"""SELFTEST of pf_ladder_queue_front (QUEUE-PRIORITY-SENSITIVITY-v1; 2026-11-22, session aea5bfb9). main(X) checks the module on synthetic tapes; a failed
check raises SystemExit with FAIL. Groups (the plan 425e9fbc's fixtures):
 1 the treatment on hand tapes: same-price queue consumption (CURRENT waits behind 100 shown shares, FRONT fills at once), a through-price sweep
   fills both, partial fills keep FRONT's priority, the one-time initialisation (a later larger display does not raise ahead; a refused
   post-only order is untouched and uncounted; front=False engaged but never triggered), public-volume conservation for two own orders at one
   price, the delayed fill notice (known only 100 ms after the fill);
 2 the swap: refused after an order exists; CURRENT on random tapes equals the plain frozen branch (fingerprint, command log, guard log), with
   accepted generations counted and none changed;
 3 feedback: on random tapes FRONT starts every accepted generation at ahead 0 (the observer's records), finishes every episode, adds fills on
   at least one tape and then changes later decisions there;
 4 fill_classes: same_price / through with the trade's price and size; a fill against a trade it could not fill refuses;
 5 class_rows, mix and predicates on hand tables: the class sums, eligibility and censoring; the volume + value split; every reading;
 6 parts_of: the first two complete markets per date, the rest, an empty rest;
 7 guards, each isolated with every earlier guard passing and its message asserted: run_day before the freeze, under another treatment, under
   another source, a day outside the declared days, the rest without the projection, an existing output; project refuses an existing projection
   and an unfinished pilot; measure refuses an existing measurement."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'QUEUE FRONT SELFTEST FAILED'


def run(mod):
    main(mod)


def _cache(nt):
    z = lambda: np.zeros(nt, np.float32)
    return dict(x={0: z(), 1: z()}, age={0: np.full(nt, 5., np.float32), 1: np.full(nt, 5., np.float32)}, status={0: np.zeros(nt, np.int8), 1: np.zeros(nt, np.int8)})


def main(X):  # noqa: C901
    P, M, LB, BE, pd = X.P, X.M, X.LB, X.BE, X.pd; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    import pf_ladder_rest_selftest as V1
    fr = V1.fr; E_ = P.frozen_edges()

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw); return False
        except ValueError as e2:
            return word in str(e2)
    # ---- 1 the treatment on hand tapes
    try:
        f0 = fr(0, 0, [(50, 100)], [(52, 100)])
        eng = lambda trades, front, frames=None: X.FrontLadderEngine(BE.tape(frames or [f0], trades, 50.), 10000., front)
        a_, b_ = eng([[100., 0, 50., 60.]], False), eng([[100., 0, 50., 60.]], True)
        for e_ in (a_, b_):
            e_.add_order(0, 50., 0., 30., 0.); e_.advance(1000.)
        ok(f'1 same-price prints consume the queue first ({a_.fills}, {b_.fills}, {a_.o[0]["ahead"]})', a_.fills == [] and a_.o[0]['ahead'] == 40. and [f[3] for f in b_.fills] == [30.])
        a_, b_ = eng([[100., 0, 49., 20.]], False), eng([[100., 0, 49., 20.]], True)
        for e_ in (a_, b_):
            e_.add_order(0, 50., 0., 30., 0.); e_.advance(1000.)
        ok('1 a through-price sweep fills both arms alike', [f[3] for f in a_.fills] == [f[3] for f in b_.fills] == [20.])
        b_ = eng([[100., 0, 50., 10.], [150., 0, 50., 10.]], True); b_.add_order(0, 50., 0., 30., 0.); b_.advance(1000.)
        ok(f'1 partial fills keep the front ({[f[3] for f in b_.fills]}, {b_.o[0]["rem"]})', [f[3] for f in b_.fills] == [10., 10.] and b_.o[0]['rem'] == 10. and b_.o[0]['ahead'] == 0.)
        frs = [f0, fr(120, 120, [(50, 500)], [(52, 100)])]
        b_ = eng([[200., 0, 50., 5.]], True, frs); b_.add_order(0, 50., 0., 30., 0.); b_.add_order(0, 52., 0., 30., 1.); b_.advance(1000.)
        a_ = eng([[200., 0, 50., 5.]], False, frs); a_.add_order(0, 50., 0., 30., 0.); a_.add_order(0, 52., 0., 30., 1.); a_.advance(1000.)
        ok(f'1 one-time initialisation ({b_.o[0]["ahead"]}, {b_.o[1]["refused"]}, {b_.accepted_new}, {b_.front_set}, {a_.accepted_new}, {a_.front_set}, {a_.o[0]["ahead"]})',
           b_.o[0]['ahead'] == 0. and [f[3] for f in b_.fills] == [5.] and b_.o[1]['refused'] and b_.o[1]['rem'] == 0. and b_.accepted_new == 1 and b_.front_set == 1
           and a_.accepted_new == 1 and a_.front_set == 0 and a_.o[0]['ahead'] == 95. and a_.fills == [])
        b_ = eng([[100., 0, 50., 50.]], True); b_.add_order(0, 50., 0., 30., 0.); b_.add_order(0, 50., 0., 30., 1.); b_.advance(1000.)
        ok(f'1 public-volume conservation for two own orders ({[f[3] for f in b_.fills]})', [f[3] for f in b_.fills] == [30., 20.] and sum(f[3] for f in b_.fills) <= 50.)
        Bn = X.branch(BE.tape([f0], [[100., 0, 50., 10.]], 50.), dict(end_rel=3000.), True); g = Bn.E.add_order(0, 50., 0., 30., 0.); Bn.E.advance(150.)
        ok(f'1 the fill notice ({Bn.known_filled(g, 199.)}, {Bn.known_filled(g, 200.)})', Bn.known_filled(g, 199.) == 0 and Bn.known_filled(g, 200.) == 10.)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 treatment: stopped ({e!r})'[:220])
    # ---- 2 the swap and CURRENT's inertness
    try:
        B0 = LB.LadderBranch(BE.tape([f0], [], 50.), LB.LADDER, end=3000.); B0.E.add_order(0, 50., 0., 30., 0.)
        ok('2 the swap refuses after an order exists', refuses('before any order', X.swap_engine, B0, True))
        rng = np.random.default_rng(71); diff = 0; counted = 0
        for i in range(4):
            p = BE.rand_tape(rng, 300, 500, end=30000.); cz = _cache(600)
            B1 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2, ident=dict(end_rel=30000.)); g1 = P.PressureGuard(B1, cz, E_); LB.run_episode(B1, g1); LB.finish(B1)
            B2 = X.branch(p, dict(end_rel=30000.), False); B2.episode = i % 2; g2 = P.PressureGuard(B2, cz, E_); LB.run_episode(B2, g2); LB.finish(B2)
            diff += int((LB.fingerprint(B1), B1.cmd_log, g1.log) != (LB.fingerprint(B2), B2.cmd_log, g2.log)); counted += int(B2.E.accepted_new > 0 and B2.E.front_set == 0)
        ok(f'2 CURRENT equals the frozen branch ({diff} of 4 differ; engaged on {counted})', diff == 0 and counted == 4)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 swap: stopped ({e!r})'[:220])
    # ---- 3 feedback
    try:
        rng = np.random.default_rng(72); more = changed = zero_ok = 0
        for i in range(4):
            p = BE.rand_tape(rng, 300, 500, end=30000.); cz = _cache(600); res = {}
            for front in (False, True):
                B = X.branch(p, dict(end_rel=30000.), front); pol = P.PressureGuard(B, cz, E_); obs = P.QueueObs(B, pol); LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B)
                res[front] = (B, Lg, obs.queue())
            (Bc, _, _), (Bf, Lf, qf) = res[False], res[True]
            zero_ok += int(all(float(v[1]) == 0. for v in qf['ahead0'].values() if float(v[2]) > 0) and Lf['status'] == 'ok')
            qc, qfr = sum(f[3] for f in Bc.E.fills), sum(f[3] for f in Bf.E.fills); more += int(qfr > qc)
            changed += int(qfr > qc and Bc.cmd_log != Bf.cmd_log)
        ok(f'3 FRONT starts every accepted generation at 0 and finishes ({zero_ok} of 4); adds fills ({more}) and changes later decisions ({changed})', zero_ok == 4 and more >= 1 and changed >= 1)
        idn = dict(day='d', instrument_id=1, start_abs_ms=0., end_abs_ms=30000., end_rel=30000., opening_ms=0., mark_source='syn', tape_id='syn')
        ra = {arm: X.run_arm(p, idn, _cache(600), E_, arm)[1] for arm in X.RUN_ARMS}
        ok(f'3 run_arm maps the arms ({[ra[a]["treatment"] for a in X.RUN_ARMS]})', ra['CURRENT_QUEUE']['treatment']['front'] is False and ra['CURRENT_QUEUE']['treatment']['front_set'] == 0
           and ra['FRONT_AT_ACTIVATION']['treatment']['front'] is True and ra['FRONT_AT_ACTIVATION']['treatment']['front_set'] == ra['FRONT_AT_ACTIVATION']['treatment']['accepted_new'] > 0
           and all(len(ra[a]['fill_class']) == len(ra[a]['fills']) for a in X.RUN_ARMS))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 feedback: stopped ({e!r})'[:220])
    # ---- 4 fill_classes
    try:
        b_ = eng([[100., 0, 50., 10.], [150., 0, 49., 5.]], True); b_.add_order(0, 50., 0., 30., 0.); b_.advance(1000.)
        ok(f'4 the classes ({X.fill_classes(b_)})', X.fill_classes(b_) == [['same_price', 50., 10.], ['through', 49., 5.]])
        f_ = b_.fills[0]; b_.fills[0] = (f_[0], f_[1], 0, f_[3], 49., f_[5]); ok('4 a fill against a trade it could not fill refuses', refuses('cannot fill', X.fill_classes, b_))
        b_.fills[0] = (f_[0], f_[1], 1, f_[3], 50., f_[5]); ok('4 a fill on the other side of its trade refuses', refuses('cannot fill', X.fill_classes, b_))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 classes: stopped ({e!r})'[:220])
    # ---- 5 class_rows, mix and predicates
    try:
        mm = lambda m1, m5=None, m30=None: [50., 1., 50., 1., 50., 1., m1, 1., m5, 1., m30, 1.]
        rec = dict(mark=52., fills=[[100., 0, 0, 10., 50., 3], [200., 0, 0, 5., 50., 3], [9500., 1, 1, 4., 49., 7]], fill_mids=[mm(51., 51., None), mm(49., None, None), mm(48., None, None)],
                   fill_class=[['same_price', 50., 20.], ['same_price', 50., 20.], ['through', 48., 4.]])
        K1 = X.class_rows('d1', 1, 'CURRENT_QUEUE', rec, 10000.); k = {r['cls']: r for r in K1}
        ok(f'5 class sums ({k["same_price"]})', k['same_price']['fills'] == 2 and k['same_price']['q'] == 15. and k['same_price']['end'] == 30. and k['same_price']['mo1s'] == 10. - 5.
           and k['same_price']['pos1s'] == 10. and k['same_price']['neg1s'] == -5. and k['same_price']['shared_trade_fills'] == 2 and k['same_price']['cens5'] == 1 and k['same_price']['cens30'] == 2
           and k['through']['el_q'] == 0. and k['through']['end'] == -12.)
        K_ = pd.DataFrame([dict(arm='CURRENT_QUEUE', cls='same_price', el_q=10., mo1s=-10.), dict(arm='CURRENT_QUEUE', cls='through', el_q=10., mo1s=-30.),
                           dict(arm='FRONT_AT_ACTIVATION', cls='same_price', el_q=30., mo1s=-15.), dict(arm='FRONT_AT_ACTIVATION', cls='through', el_q=10., mo1s=-30.)])
        mx = X.mix(K_)
        ok(f'5 the volume + value split ({mx["volume"]}, {mx["value"]}, {mx["total"]})', mx['volume'] == 20. * -1. and mx['value'] == 30. * (-0.5 + 1.) and mx['total'] == -5.
           and mx['classes']['through']['volume'] == 0.)

        def R(vals, qf=20., mo=None):
            return pd.DataFrame([dict(day=d, m=m, arm=a, value=v_, all_id_mo1s=(v_ if mo is None else mo[d][a]), all_cut_mo1s=(v_ if mo is None else mo[d][a]), all_cut_valid_q=10., all_q=(qf if a == 'FRONT_AT_ACTIVATION' else 10.), both_exact=.4,
                                      bid_exact=.5, ask_exact=.5, pulls=1, withholds=1, first_review_pulls=0, generations=5, bid_structural=1, ask_structural=1, mean_abs_inv=1., share_abs_ge_S=0.)
                                 for d, av in vals.items() for m in (1, 2) for a, v_ in av.items()])
        V = lambda f1, f2: {'d1': dict(BASELINE=-10., CURRENT_QUEUE=-6., FRONT_AT_ACTIVATION=f1), 'd2': dict(BASELINE=-12., CURRENT_QUEUE=-8., FRONT_AT_ACTIVATION=f2)}
        rd = {nm: X.predicates(R(V(*v), qf), K_)['reading'] for nm, v, qf in (('ins', (-4., -7.), 20.), ('pos', (1., 2.), 20.), ('no', (-7., -9.), 20.), ('mix', (-2., -9.), 20.),
                                                                              ('none', (-7., -9.), 5.))}
        rd['equal'] = X.predicates(R(V(-7., -9.), 10.), K_)['reading']; rd['end_only'] = X.predicates(R(V(-4., -7.), 20., V(-7., -9.)), K_)['reading']
        ok(f'5 the readings ({rd})', rd == dict(ins='queue_helps_but_insufficient', pos='queue_helps_policy_nonnegative', no='front_priority_does_not_rescue', mix='mixed', none='no_added_fills',
                                                equal='no_added_fills', end_only='mixed'))
        pr = X.predicates(R(V(-4., -7.)), K_)
        ok('5 economic, recovered and the END signs', pr['economic'] == 'IMPROVES' and pr['END']['total'] == 2 * (2. + 1.) and abs(pr['recovered'] - 6. / 28.) < 1e-12
           and pr['end_negative']['total']['FRONT_AT_ACTIVATION'] is True and pr['END']['markets_better'] == 4)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 readings: stopped ({e!r})'[:220])
    # ---- 6 parts_of
    try:
        pt = X.parts_of(['d1', 'd2'], {'d1': [5, 3, 9], 'd2': [1, 2]})
        ok(f'6 parts ({pt})', pt == dict(pilot={'d1': [5, 3], 'd2': [1, 2]}, rest={'d1': [9], 'd2': []}))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 parts: stopped ({e!r})'[:200])
    # ---- 7 guards
    saved = dict(frozen=X.frozen, sources=X.sources, treatment_sha=X.treatment_sha); tmp = Path(tempfile.mkdtemp(dir=str(X.A.DATA)))
    try:
        day = X.M.DAYS[0]; mp = tmp / 'man.json'; pp = tmp / 'proj.json'
        mp.write_text(json.dumps(dict(treatment_sha256=X.treatment_sha(), sources=X.sources(), parts=dict(pilot={day: [1]}, rest={day: [2]}), days=[day], caps=dict(batch_core_hours=.25,
                                                                                                                                                                line_ceiling=2.25))))
        rd_ = lambda **kw: X.run_day(day, kw.pop('part', 'pilot'), out=tmp / 'out', man_path=mp, proj_path=pp, **kw)
        X.frozen = lambda: False; ok('7 run_day refuses before the freeze', refuses('committed', rd_))
        X.frozen = lambda: True; X.treatment_sha = lambda: '0' * 64; ok('7 run_day refuses another treatment', refuses('treatment', rd_)); X.treatment_sha = saved['treatment_sha']
        X.sources = lambda: dict(saved['sources'](), **{'pf_ladder_rest.py': '0' * 64}); ok('7 run_day refuses another source', refuses('source', rd_)); X.sources = saved['sources']
        ok('7 run_day refuses an undeclared day', refuses('declared day', X.run_day, '20991231', 'pilot', out=tmp / 'out', man_path=mp, proj_path=pp))
        ok('7 the rest refuses without the projection', refuses('projection', rd_, part='rest'))
        po = X.day_paths(day, 'pilot', tmp / 'out'); po['rec'].parent.mkdir(parents=True, exist_ok=True); po['rec'].write_text('{}')
        ok('7 run_day refuses an existing output', refuses('write-once', rd_)); pp.write_text('{}')
        ok('7 project refuses an existing projection', refuses('write-once', X.project, out=tmp / 'out', man_path=mp, proj_path=pp)); pp.unlink()
        ok('7 project refuses an unfinished pilot', refuses('did not finish', X.project, out=tmp / 'out', man_path=mp, proj_path=pp))
        (tmp / 'out' / 'measure').mkdir(parents=True, exist_ok=True); (tmp / 'out' / 'measure' / 'audit.json').write_text('{}')
        ok('7 measure refuses an existing measurement', refuses('write-once', X.measure, out=tmp / 'out', man_path=mp))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'7 guards: stopped ({e!r})'[:220])
    finally:
        X.frozen, X.sources, X.treatment_sha = saved['frozen'], saved['sources'], saved['treatment_sha']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('QUEUE FRONT SELFTEST PASSED: treatment, swap and inertness, feedback, classes, readings, parts, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_queue_front as X_
    main(X_)
