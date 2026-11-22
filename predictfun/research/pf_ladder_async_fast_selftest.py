"""SELFTEST of pf_ladder_async_fast (ASYNC-EXECUTION-SENSITIVITY-v1; 2026-11-22, session aea5bfb9). main(X) checks the module on synthetic tapes; a failed
check raises SystemExit with FAIL. Groups (the amendment c3fe1d4a's preflight):
 1 timing: ASYNC_FAST10 activation +10, live cancel +10, a still-pending post's cancel at max(req + 10, submit + 20); BASELINE equal to the frozen engine;
 2 the slot model on a hand tape with a scripted policy: a pulled bid frees its desired slot under ASYNC so a replacement at the new best is sent while
   the cancel is in flight (overlap with headroom), the frozen model waits for the acknowledgement; at the 200-share cap the replacement is refused
   (cap, no early release) and posted after the closure; the retiring order's price stays locked; own-cross still sees a retiring opposite order;
 3 pending fills and removal: a fill on a retiring order is counted once (reserved until its notice, closure after the acknowledgement), a trade
   at the effect does not fill, one just before does; a frame received after the tick is not seen earlier under ASYNC;
 4 parity and invariants: BASELINE through the wrapper equals the frozen branch on random tapes (fingerprint, command and guard logs); ASYNC
   finishes every tape under its invariants, records its effective settings and differs on at least one;
 5 blocks on a hand ledger; 6 predicates on a synthetic table (contrasts, reading); 7 guards, each isolated, its message asserted."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ASYNC FAST SELFTEST FAILED'


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

    class Script(LB.Policy):
        def __init__(self, B, pulls=None, q=50., sides=(0,)):
            super().__init__('script', B.end); self.pulls, self.q, self.sides = dict(pulls or {}), float(q), sides

        def live(self, v, o):
            return ('PULL', 'script') if (o['gid'] in self.pulls and v['t'] >= self.pulls[o['gid']]) else ('KEEP', 'script')

        def slot(self, v, r, price):
            return ('POST', self.q, 'script') if v['side'] in self.sides else ('WAIT', 0., 'script')

    def hand(profile, q=50., pulls=None, trades=(), frames=None, end=3000., sides=(0,)):
        p = BE.tape(frames or [fr(0, 0, [(50, 100), (49, 100)], [(52, 100), (53, 100)]), fr(990, 995, [(51, 100), (50, 100)], [(52, 100), (53, 100)])], list(trades), 50.)
        B = X.ProfileBranch(p, end, dict(end_rel=end), False, profile); pol = Script(B, pulls, q, sides); LB.run_episode(B, pol); LB.finish(B); return B
    # ---- 1 timing
    try:
        tp = BE.tape([fr(0, 0, [(50, 100)], [(52, 100)])], [], 50.)
        ea = X.ProfileEngine(tp, 10000., False, X.PROFILES['ASYNC_FAST10']); g0 = ea.add_order(0, 50., 0., 30., 0.); g1 = ea.add_order(0, 49., 0., 30., 1.)
        ea.request(g0, 0., 'policy'); ea.request(g1, 50., 'policy')
        ok(f'1 ASYNC timing ({ea.o[g0]["act"]}, {ea.o[g0]["remove"]}, {ea.o[g1]["remove"]})', ea.o[g0]['act'] == 10. and ea.o[g0]['remove'] == 20. and ea.o[g1]['remove'] == 60.)
        eb = X.ProfileEngine(tp, 10000., False, X.PROFILES['BASELINE']); ef = LB.LadderEngine(tp, 10000.)
        for e_ in (eb, ef):
            h0 = e_.add_order(0, 50., 0., 30., 0.); h1 = e_.add_order(0, 49., 0., 30., 1.); e_.request(h0, 0., 'policy'); e_.request(h1, 100., 'policy')
        ok(f'1 BASELINE equals the frozen engine ({[(o["act"], o["remove"]) for o in eb.o]})', [(o['act'], o['remove'], o['req']) for o in eb.o] == [(o['act'], o['remove'], o['req']) for o in ef.o]
           and eb.o[0]['act'] == 58. and eb.o[0]['remove'] == 95. and eb.o[1]['remove'] == 137.)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 timing: stopped ({e!r})'[:220])
    # ---- 2 the slot model
    try:
        Ba, Bf = hand('ASYNC_FAST10', 50., {0: 1000.}), hand('BASELINE', 50., {0: 1000.})
        newa = [o for o in Ba.E.o if o['submit'] > 0 and o['side'] == 0]; newf = [o for o in Bf.E.o if o['submit'] > 0 and o['side'] == 0]
        ok(f'2 overlap with headroom ({[(o["submit"], o["price"]) for o in newa][:2]}, {[(o["submit"], o["price"]) for o in newf][:2]})',
           bool(newa) and newa[0]['submit'] == 1000. and newa[0]['price'] == 51. and bool(newf) and newf[0]['submit'] > 1100.)
        ok('2 the retiring price stays locked', all(not (o['submit'] == 1000. and o['price'] == 50.) for o in Ba.E.o))
        Bc = hand('ASYNC_FAST10', 100., {0: 1000.}); cap_at = [c for c in Bc.constraints if c[2] == 'cap']; newc = [o for o in Bc.E.o if o['submit'] > 0 and o['side'] == 0]
        ok(f'2 at the cap the replacement waits for the closure ({cap_at[:2]}, {[o["submit"] for o in newc][:1]})', bool(cap_at) and cap_at[0][0] == 1000. and bool(newc) and newc[0]['submit'] >= 1110.)
        frs = [fr(0, 0, [(50, 100)], [(52, 100), (53, 100)]), fr(990, 995, [(52, 100)], [(54, 100), (55, 100)])]
        Bo = hand('ASYNC_FAST10', 50., {1: 1000.}, frames=frs, sides=(0, 1))
        ok(f'2 own-cross sees a retiring opposite order ({[c for c in Bo.constraints if c[2] == "own_cross_blocked"][:1]})', any(c[2] == 'own_cross_blocked' and c[0] == 1000. for c in Bo.constraints))
        tp2 = BE.tape([fr(0, 0, [(50, 100), (49, 100)], [(52, 100)])], [], 50.)
        Bi = X.ProfileBranch(tp2, 3000., dict(end_rel=3000.), False, 'ASYNC_FAST10'); Bi.unclosed[0] = [Bi.E.add_order(0, 50. - k, 0., 100., 0.) for k in range(3)]
        Bj = X.ProfileBranch(tp2, 3000., dict(end_rel=3000.), False, 'ASYNC_FAST10'); Bj.unclosed[0] = [Bj.E.add_order(0, 50. - k, 0., 50., 0.) for k in range(3)]
        ok('2 the async invariants refuse (reservation over the cap; three desired orders)', refuses('reservation inequality', Bi.check_position, 0.) and refuses('desired generations', Bj.check_position, 0.))
        ne = {}
        for prof in ('ASYNC_FAST10', 'BASELINE'):
            Bn = X.ProfileBranch(tp2, 3000., dict(end_rel=3000.), False, prof); Bn.pre_tick(2950.); Bn.side_commands(2950., 0, Script(Bn)); ne[prof] = (len(Bn.E.o), [c[2] for c in Bn.constraints])
        ok(f'2 near END with the profile\'s activation ({ne})', ne['ASYNC_FAST10'][0] == 2 and ne['BASELINE'][0] == 0 and 'near_end' in ne['BASELINE'][1])
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 slots: stopped ({e!r})'[:220])
    # ---- 3 pending fills, removal and receipt
    try:
        Bp = hand('ASYNC_FAST10', 50., {0: 1000.}, trades=[[1005., 0, 49., 20.], [1010., 0, 49., 20.]])                 # through-price: no queue wait
        f0 = [f for f in Bp.E.fills if f[1] == 0]
        ok(f'3 a fill on a retiring order counted once; none at the effect ({f0}, {Bp.closed.get(0)})', [f[3] for f in f0] == [20.] and f0[0][0] == 1005. and Bp.closed[0][0] == 'removed'
           and Bp.closed[0][1] >= 1110.)
        Bq = hand('BASELINE', 50., {0: 1000.}, trades=[[1036., 0, 49., 20.], [1037., 0, 49., 20.]])
        ok(f'3 BASELINE: a trade 1 ms before the effect fills, at the effect not ({[f[0] for f in Bq.E.fills if f[1] == 0]})', [f[0] for f in Bq.E.fills if f[1] == 0] == [1036.])
        frs2 = [fr(0, 0, [(50, 100), (49, 100)], [(52, 100)]), fr(990, 1030, [(55, 100)], [(57, 100)])]
        seen = {}
        for prof in ('ASYNC_FAST10', 'BASELINE'):
            Bx = X.ProfileBranch(BE.tape(frs2, [], 50.), 3000., dict(end_rel=3000.), False, prof); Bx.pre_tick(1000.); seen[prof] = Bx.side_view(1000., 0)['bid']
        ok(f'3 a frame received after the tick is not seen sooner ({seen})', seen == {'ASYNC_FAST10': 50., 'BASELINE': 50.})
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 fills: stopped ({e!r})'[:220])
    # ---- 4 parity and invariants
    try:
        rng = np.random.default_rng(81); diff = differ = fin = 0
        for i in range(3):
            p = BE.rand_tape(rng, 300, 500, end=30000.); cz = _cache(600)
            B1 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=0, ident=dict(end_rel=30000.)); g1 = P.PressureGuard(B1, cz, E_); LB.run_episode(B1, g1); LB.finish(B1)
            B2 = X.ProfileBranch(p, 30000., dict(end_rel=30000.), False, 'BASELINE'); g2 = P.PressureGuard(B2, cz, E_); LB.run_episode(B2, g2); LB.finish(B2)
            B3 = X.ProfileBranch(p, 30000., dict(end_rel=30000.), False, 'ASYNC_FAST10'); g3 = P.PressureGuard(B3, cz, E_); LB.run_episode(B3, g3); L3 = LB.finish(B3)
            diff += int((LB.fingerprint(B1), B1.cmd_log, g1.log) != (LB.fingerprint(B2), B2.cmd_log, g2.log)); differ += int(B3.cmd_log != B1.cmd_log)
            fin += int(L3['status'] == 'ok' and B3.settings['activation_ms'] == 10. and B3.settings['slot_model'] == 'desired_levels' and B2.settings == B1.settings)
        ok(f'4 BASELINE parity ({diff} of 3 differ); ASYNC finishes with its settings ({fin}) and differs ({differ})', diff == 0 and fin == 3 and differ >= 1)
        idn = dict(day='d', instrument_id=1, start_abs_ms=0., end_abs_ms=30000., end_rel=30000., opening_ms=0., mark_source='syn', tape_id='syn')
        ra = {arm: X.run_arm(p, idn, _cache(600), E_, arm)[1] for arm in X.RUN_ARMS}
        ok(f'4 run_arm maps the arms ({ {a: (r["treatment"]["front"], r["profile"]["name"]) for a, r in ra.items()} })', [(ra[a]['treatment']['front'], ra[a]['profile']['name']) for a in X.RUN_ARMS]
           == [(False, 'BASELINE'), (False, 'ASYNC_FAST10'), (True, 'ASYNC_FAST10')] and ra['FRONT_ASYNC_FAST10']['treatment']['front_set'] == ra['FRONT_ASYNC_FAST10']['treatment']['accepted_new'] > 0)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 parity: stopped ({e!r})'[:220])
    # ---- 5 blocks
    try:
        rec = dict(orders=[dict(gid=0, side=0, submit=0., req=100.), dict(gid=1, side=0, submit=0., req=None), dict(gid=2, side=1, submit=50., req=None), dict(gid=3, side=0, submit=0., req=None)],
                   closed={'0': ['removed', 250.], '2': ['filled', 150.], '3': ['filled', 100.]}, constraints=[[100., 0, 'cap'], [200., 1, 'cap'], [300., 0, 'near_end']])
        bk = X.blocks('d', 1, 'A', rec, 400.)
        ok(f'5 blocks ({bk})', bk['retiring_ticks'] == 3. and bk['cap'] == 2 and bk['cap_while_retiring'] == 1 and bk['near_end'] == 1 and abs(bk['desired_occupancy'] - (2 + 8 + 2) / 32.) < 1e-12)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 blocks: stopped ({e!r})'[:220])
    # ---- 6 predicates
    try:
        def R(vals):
            return pd.DataFrame([dict(day=d, m=1, arm=a, value=v_, all_id_mo1s=v_, all_cut_mo1s=v_, all_cut_valid_q=10., all_q=10.) for d, av in vals.items() for a, v_ in av.items()])
        V = {'d1': dict(BASELINE=-30., CURRENT_BASELINE=-10., FRONT_BASELINE=-8., CURRENT_ASYNC_FAST10=-6., FRONT_ASYNC_FAST10=-1.),
             'd2': dict(BASELINE=-30., CURRENT_BASELINE=-12., FRONT_BASELINE=-11., CURRENT_ASYNC_FAST10=-13., FRONT_ASYNC_FAST10=-9.)}
        pr = X.predicates(R(V)); c = pr['contrasts']
        ok(f'6 contrasts ({ {k: (v["END_total"], v["END_answer"]) for k, v in c.items()} })', c['EXEC_AT_CURRENT']['END_total'] == 3. and c['EXEC_AT_CURRENT']['END_answer'] == 'MIXED'
           and c['EXEC_AT_FRONT']['END_answer'] == 'POSITIVE_EVERY_DAY' and c['EXEC_AT_FRONT']['economic'] == 'IMPROVES' and c['QUEUE_AT_FAST']['END_total'] == 9.
           and c['INTERACTION']['END_total'] == 9. - 3. and pr['reading'] == 'assumptions_do_not_rescue')
        V['d1']['FRONT_ASYNC_FAST10'] = 9.; pr2 = X.predicates(R(V))                                                     # pooled exactly 0
        ok(f'6 a nonnegative arm ({pr2["nonnegative_arms"]}, {pr2["reading"]})', pr2['nonnegative_arms'] == ['FRONT_ASYNC_FAST10'] and pr2['reading'] == 'nonnegative_development_result')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 predicates: stopped ({e!r})'[:220])
    # ---- 7 guards
    saved = dict(frozen=X.frozen, sources=X.sources, profile_sha=X.profile_sha); tmp = Path(tempfile.mkdtemp(dir=str(X.A.DATA)))
    try:
        day = X.M.DAYS[0]; mp = tmp / 'man.json'; pp = tmp / 'proj.json'
        mp.write_text(json.dumps(dict(profile_sha256=X.profile_sha(), treatment_sha256=X.QF.treatment_sha(), sources=X.sources(), parts=dict(pilot={day: [1]}, rest={day: [2]}), days=[day],
                                      caps=dict(batch_core_hours=.25, line_ceiling=2.5))))
        rd_ = lambda **kw: X.run_day(day, kw.pop('part', 'pilot'), out=tmp / 'out', man_path=mp, proj_path=pp, **kw)
        X.frozen = lambda: False; ok('7 run_day refuses before the freeze', refuses('committed', rd_))
        X.frozen = lambda: True; X.profile_sha = lambda: '0' * 64; ok('7 run_day refuses another profile', refuses('profile', rd_)); X.profile_sha = saved['profile_sha']
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
        X.frozen, X.sources, X.profile_sha = saved['frozen'], saved['sources'], saved['profile_sha']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('ASYNC FAST SELFTEST PASSED: timing, slot model, pending fills / removal / receipt, parity and invariants, blocks, predicates, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_async_fast as X_
    main(X_)
