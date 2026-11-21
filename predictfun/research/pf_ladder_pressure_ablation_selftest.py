"""SELFTEST of pf_ladder_pressure_ablation (PRESSURE-ACTION-ATTRIBUTION-v1; 2026-11-21, session aea5bfb9). main(X) checks the module; a failed check
raises SystemExit with FAIL. Each group names only what its code runs.
 1 delegation on 8 random tapes with random caches: BASELINE_00 equals pf_branch_ladder.LadderInventory (fingerprint, fills, value, command
   log, no intervention); FULL_11 equals the frozen PressureGuard (fingerprint, command log, log); PULL_ONLY_10 never withholds and
   WITHHOLD_ONLY_01 never pulls; each differs from BASELINE_00 on at least one tape;
 2 a hand tape (bids ON over ticks 20-39): FULL_11 pulls the bids at 1,000 and re-posts at 2,000; PULL_ONLY_10 pulls at 1,000, re-posts
   during pressure and pulls the re-posted bids at their first live review; WITHHOLD_ONLY_01 keeps the bids posted at 0 and records their
   first review; every arm's cancel effect = request + 37 ms; at the policy level a disabled component cannot act while the other can, and
   the inventory rule keeps precedence and its reasons in both single-component arms;
 3 parts_of: the pilot and the rest are disjoint and cover the complete markets; a pilot outside them is refused;
 4 the readings on a synthetic four-arm table: the four contrasts and the interaction by day and in total, POSITIVE / NEGATIVE / MIXED against
   zero (a zero day is MIXED), better / worse / equal markets, the top-3 share, all_arms_negative, the pooled per share;
 5 guards, each isolated with every earlier guard passing and its refusal message asserted: run_day before the freeze, under another rule,
   under another source, an unknown part, the rest without the projection, an existing output, a parent day record differing from the
   manifest; a part without members writes DAY_EMPTY without loading; with every guard passing run_day reaches the data load (a sentinel);
   measure refuses to overwrite."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ABLATION SELFTEST FAILED'


class _Sentinel(Exception):
    pass


def run(mod):
    main(mod)


def _cache(nt, x0, x1, st0=None, st1=None):
    z = lambda v: np.asarray(v, np.float32)
    return dict(x={0: z(x0), 1: z(x1)}, age={0: np.full(nt, 5., np.float32), 1: np.full(nt, 5., np.float32)},
                status={0: np.zeros(nt, np.int8) if st0 is None else np.asarray(st0, np.int8), 1: np.zeros(nt, np.int8) if st1 is None else np.asarray(st1, np.int8)})


def main(X):  # noqa: C901
    P, M, LB, BE, pd = X.P, X.M, X.LB, X.BE, X.pd; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    import pf_ladder_rest_selftest as V1
    fr = V1.fr; E = P.frozen_edges(); G_ = lambda B, cz, arm: X.AblatedPressureGuard(B, cz, E, *X.ARMS[arm])
    # ---- 1 delegation
    try:
        rng = np.random.default_rng(47); n00 = n11 = d10 = d01 = w10 = p01 = 0
        for i in range(8):
            p = BE.rand_tape(rng, 300, 500, end=30000.); nt = 600; vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32)
            cz = _cache(nt, rng.choice(vals, nt), rng.choice(vals, nt), rng.choice([0, 0, 0, 1, 3], nt), rng.choice([0, 0, 0, 2], nt)); run_ = {}
            for nm, mk in (('inv', lambda B: LB.LadderInventory(B)), ('pg', lambda B: P.PressureGuard(B, cz, E)), *[(a, (lambda a_: lambda B: G_(B, cz, a_))(a)) for a in X.ARMS]):
                B = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); pol = mk(B); LB.run_episode(B, pol); Lg = LB.finish(B); run_[nm] = (B, Lg, pol)
            fp = lambda nm: LB.fingerprint(run_[nm][0]); cl = lambda nm: run_[nm][0].cmd_log; rs = lambda nm: {c[8] for c in cl(nm)}
            n00 += int(not (fp('BASELINE_00') == fp('inv') and run_['BASELINE_00'][1]['fills'] == run_['inv'][1]['fills'] and run_['BASELINE_00'][1]['value'] == run_['inv'][1]['value']
                            and cl('BASELINE_00') == cl('inv') and not run_['BASELINE_00'][2].log))
            n11 += int(not (fp('FULL_11') == fp('pg') and cl('FULL_11') == cl('pg') and run_['FULL_11'][2].log == run_['pg'][2].log))
            d10 += int(fp('PULL_ONLY_10') != fp('BASELINE_00')); d01 += int(fp('WITHHOLD_ONLY_01') != fp('BASELINE_00'))
            w10 += int('pressure_withhold' in rs('PULL_ONLY_10')); p01 += int('pressure_pull' in rs('WITHHOLD_ONLY_01'))
        ok(f'1 BASELINE_00 equals LadderInventory ({n00} of 8 differ)', n00 == 0)
        ok(f'1 FULL_11 equals the frozen PressureGuard ({n11} of 8 differ)', n11 == 0)
        ok(f'1 the single-component arms act ({d10}, {d01} of 8 differ from BASELINE_00) and never use the other component ({w10}, {p01})', d10 >= 1 and d01 >= 1 and w10 == 0 and p01 == 0)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 delegation: stopped ({e!r})'[:220])
    # ---- 2 the hand tape and the policy level
    try:
        p = BE.tape([fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])], [], 50.); nt = 80; xb = np.zeros(nt, np.float32); xb[20:40] = 1.

        def arm(name):
            B = LB.LadderBranch(p, LB.LADDER, end=4000., episode=0); g = G_(B, _cache(nt, xb, np.zeros(nt)), name); ob = P.QueueObs(B, g); LB.run_episode(B, g, observe=ob); return B, g, ob
        b0 = lambda B: [o for o in B.E.o if o['side'] == 0 and o['submit'] == 0.]; later = lambda B: sorted(o['submit'] for o in B.E.o if o['side'] == 0 and o['submit'] > 0.)
        B11, g11, _ = arm('FULL_11'); B10, g10, _ = arm('PULL_ONLY_10'); B01, g01, o01 = arm('WITHHOLD_ONLY_01')
        ok(f'2 FULL_11: pulled at 1,000, re-posted at 2,000 ({[o["req"] for o in b0(B11)]}, {later(B11)[:1]})', all(o['req'] == 1000. for o in b0(B11)) and later(B11)[0] == 2000.)
        rp10 = [o for o in B10.E.o if o['side'] == 0 and 1000. < o['submit'] < 2000.]
        ok(f'2 PULL_ONLY_10: pulled at 1,000, re-posted during pressure and those pulled at their first live review ({[(o["submit"], o["req"]) for o in rp10][:4]})',
           all(o['req'] == 1000. for o in b0(B10)) and rp10 and all(o['req'] == o['submit'] + 100. for o in rp10 if o['submit'] + 100. < 2000.) and not any(z[10] == 'pressure_withhold' for z in g10.log))
        ok(f'2 WITHHOLD_ONLY_01: the bids posted at 0 are kept and their first review recorded ({[o["req"] for o in b0(B01)]})',
           all(o['req'] is None for o in b0(B01)) and not any(z[10] == 'pressure_pull' for z in g01.log) and all(o01.first[o['gid']][0] == 100. for o in b0(B01)))
        eff = [(e[0], e[3]['effect']) for B in (B11, B10) for e in B.E.events if e[1] == 'policy_request']
        ok(f'2 the cancel effect is request + 37 ms in every arm ({eff[:2]})', eff and all(abs(b - a - 37.) < 1e-9 for a, b in eff))
        Bh = LB.LadderBranch(BE.tape([fr(0, 0, [(50, 0)], [(52, 0)])], [], 50.), LB.LADDER, end=3000.); cz = _cache(60, [1.] * 60, [1.] * 60)
        h10, h01 = G_(Bh, cz, 'PULL_ONLY_10'), G_(Bh, cz, 'WITHHOLD_ONLY_01')
        v = lambda t, s, **k: dict(dict(t=float(t), side=s, position=0., B=0., A=0., mode='UNRESTRICTED', avail=200.), **k)
        o = lambda s: dict(gid=3, price=50. if s == 0 else 52., rung=0., size=100., state='live', known_filled=0., act=58., submit=0.)
        ok('2 a disabled component cannot act while the other can', h10.live(v(100, 0), o(0)) == ('PULL', 'pressure_pull') and h10.slot(v(100, 1), 0, 52.)[0] == 'POST'
           and h01.live(v(100, 0), o(0))[0] == 'KEEP' and h01.slot(v(100, 1), 0, 52.) == ('WAIT', None, 'pressure_withhold'))
        ok('2 the inventory rule keeps precedence and its reasons in both single-component arms',
           all(g.live(v(100, 1, mode='SHORT_RESTRICTED', position=-120.), o(1)) == ('PULL', 'inventory_pull') and g.slot(v(100, 1, mode='SHORT_RESTRICTED', position=-120.), 0, 52.)[2] == 'restricted'
               for g in (h10, h01)))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 hand tape: stopped ({e!r})'[:220])
    # ---- 3 parts_of
    try:
        comp = {'20261024': [1, 2, 3, 4], '20261027': [8, 9]}; pt = X.parts_of(comp, {'20261024': [1, 2], '20261027': [8, 9]})
        ok(f'3 parts_of covers and splits ({pt})', pt == dict(pilot={'20261024': [1, 2], '20261027': [8, 9]}, rest={'20261024': [3, 4], '20261027': []}))
        try:
            X.parts_of(comp, {'20261024': [1, 7], '20261027': [8, 9]}); bad.append('3 parts_of accepted a pilot outside the complete markets')
        except ValueError as e:
            ok('3 parts_of refuses a pilot outside the complete markets', 'cover' in str(e))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 parts: stopped ({e!r})'[:200])
    # ---- 4 readings
    try:
        V = {('20261024', 1): dict(BASELINE_00=-10., PULL_ONLY_10=-6., WITHHOLD_ONLY_01=-8., FULL_11=-3.), ('20261024', 2): dict(BASELINE_00=-10., PULL_ONLY_10=-9., WITHHOLD_ONLY_01=-10., FULL_11=-9.),
             ('20261025', 1): dict(BASELINE_00=-20., PULL_ONLY_10=-14., WITHHOLD_ONLY_01=-12., FULL_11=-5.), ('20261025', 2): dict(BASELINE_00=-4., PULL_ONLY_10=-6., WITHHOLD_ONLY_01=-4., FULL_11=-2.)}
        R_ = pd.DataFrame([dict(day=d, m=m, arm=a, value=v_, all_id_mo1s=v_ / 2, all_cut_mo1s=v_ / 2, all_cut_valid_q=10.) for (d, m), av in V.items() for a, v_ in av.items()])
        pr = X.predicates(R_); c = pr['contrasts']
        ok(f'4 the contrasts by day and in total ({c["cancel_alone"]["END"]["by_day"]}, {c["withhold_given_cancel"]["END"]["total"]})',
           c['cancel_alone']['END']['by_day'] == {'20261024': 5., '20261025': 4.} and c['withhold_alone']['END']['by_day'] == {'20261024': 2., '20261025': 8.}
           and c['cancel_given_withhold']['END']['total'] == (-3. - 9. - 5. - 2.) - (-8. - 10. - 12. - 4.) and c['withhold_given_cancel']['END']['total'] == (-19.) - (-35.))
        ok(f'4 the interaction ({pr["interaction"]["END"]["by_day"]})', pr['interaction']['END']['by_day'] == {'20261024': (-12.) - (-15.) - (-18.) + (-20.), '20261025': (-7.) - (-20.) - (-16.) + (-24.)}
           and pr['interaction']['mo1s']['total'] == pr['interaction']['END']['total'] / 2)
        ok(f'4 classification against zero ({c["withhold_alone"]["END"]["answer"]}, {c["cancel_alone"]["END"]["answer"]})', c['cancel_alone']['END']['answer'] == 'POSITIVE_EVERY_DAY'
           and X.classify([1., 0.]) == 'MIXED' and X.classify([-1., -2.]) == 'NEGATIVE_EVERY_DAY' and c['cancel_alone']['per_share']['answer'] == 'POSITIVE_EVERY_DAY')
        ok(f'4 better / worse / equal markets and the top-3 share ({c["withhold_alone"]["markets_better"]}, {c["withhold_alone"]["markets_equal"]}, {c["withhold_alone"]["top3_abs_share"]})',
           c['withhold_alone']['markets_better'] == 2 and c['withhold_alone']['markets_equal'] == 2 and c['cancel_alone']['markets_worse'] == 1
           and abs(c['cancel_alone']['top3_abs_share'] - (6. + 4. + 2.) / (4. + 1. + 6. + 2.)) < 1e-12)
        ok('4 all_arms_negative and the pooled per share', pr['all_arms_negative']['total'] is True and pr['arm_totals']['FULL_11']['per_share'] == (-19. / 2) / 40.)
        R2 = R_.copy(); R2.loc[(R2.arm == 'FULL_11') & (R2.m == 1), 'value'] = 30.; p2 = X.predicates(R2)
        ok('4 one positive arm breaks all_arms_negative', p2['all_arms_negative']['total'] is False)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 readings: stopped ({e!r})'[:220])
    # ---- 5 guards
    saved = dict(frozen=X.frozen, rule_sha=X.rule_sha, sources=X.sources); ld = M.load_day; tmp = Path(tempfile.mkdtemp(dir=str(X.A.DATA)))
    try:
        day = '20261024'
        man = dict(rule_sha256=X.rule_sha(), sources=X.sources(), parts=dict(pilot={day: [3697902], '20261027': []}, rest={day: [3699278], '20261027': []}), days=[day],
                   parent=dict(day_records={day: X.C.sha(P.day_paths(day, False)['rec'])}))
        mp = tmp / 'manifest.json'; mp.write_text(json.dumps(man)); pp = tmp / 'projection.json'

        def refuses(word, man_=None, day_=day, **kw):
            if man_ is not None:
                mp.write_text(json.dumps(man_))
            try:
                X.run_day(day_, man_path=mp, proj_path=pp, out=tmp / 'out', **kw); return False
            except ValueError as e:
                return word in str(e)
            except _Sentinel:
                return False
            finally:
                mp.write_text(json.dumps(man))

        def sentinel(*a, **k):
            raise _Sentinel()
        M.load_day = sentinel
        X.frozen = lambda: False; ok('5 run_day refuses before the freeze', refuses('committed', part='pilot'))
        X.frozen = lambda: True; X.rule_sha = lambda: 'x' * 64; ok('5 run_day refuses another rule', refuses('rule', part='pilot')); X.rule_sha = saved['rule_sha']
        X.sources = lambda: dict(saved['sources'](), **{'pf_stop_model.py': '0' * 64}); ok('5 run_day refuses another source', refuses('source', part='pilot')); X.sources = saved['sources']
        ok('5 run_day refuses an unknown part', refuses('part', part='all'))
        ok('5 the rest refuses without the projection', refuses('projection', part='rest'))
        Pd = X.day_paths(day, 'pilot', tmp / 'out'); Pd['rec'].parent.mkdir(parents=True, exist_ok=True); Pd['rec'].write_text('{}')
        ok('5 run_day refuses an existing output', refuses('write-once', part='pilot')); Pd['rec'].unlink()
        ok('5 run_day refuses a parent day record differing from the manifest', refuses('differs from the manifest', dict(man, parent=dict(day_records={day: '0' * 64})), part='pilot'))
        try:
            st_ = X.run_day('20261027', 'pilot', man_path=mp, proj_path=pp, out=tmp / 'out')
            ok(f'5 a part without members writes DAY_EMPTY without loading ({st_})', st_ == 'DAY_EMPTY' and json.loads(X.day_paths('20261027', 'pilot', tmp / 'out')['rec'].read_text())['members'] == [])
        except Exception as e:                                                             # noqa: BLE001
            bad.append(f'5 the empty part stopped ({e!r})'[:200])
        try:
            X.run_day(day, 'pilot', man_path=mp, proj_path=pp, out=tmp / 'out'); bad.append('5 the all-pass run_day did not reach the load')
        except _Sentinel:
            pass
        except Exception as e:                                                             # noqa: BLE001
            bad.append(f'5 the all-pass run_day stopped before the load ({e!r})'[:200])
        mo = tmp / 'meas'; (mo / 'measure').mkdir(parents=True); (mo / 'measure' / 'audit.json').write_text('{}')
        try:
            X.measure(out=mo, man_path=mp); bad.append('5 measure overwrote')
        except ValueError as e:
            ok('5 measure refuses to overwrite', 'write-once' in str(e))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 guards: stopped ({e!r})'[:200])
    finally:
        X.frozen, X.rule_sha, X.sources = saved['frozen'], saved['rule_sha'], saved['sources']; M.load_day = ld; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('ABLATION SELFTEST PASSED: delegation, the hand tape and the policy level, the parts, the readings, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_pressure_ablation as X_
    main(X_)
