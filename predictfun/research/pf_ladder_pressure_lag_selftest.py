"""SELFTEST of pf_ladder_pressure_lag (PRESSURE-LEAD-SENSITIVITY-v1; 2026-11-21, session aea5bfb9). main(L) checks the module; a failed check
raises SystemExit with FAIL. Each group names only what its code runs.
 1 the zero-lag control: LaggedPressureGuard(lag 0) equals the frozen PressureGuard on 8 random tapes with random caches (fingerprint, fills,
   value, command log; the log's first eleven fields equal the parent's); lag 1 differs on at least one;
 2 a hand tape with synthetic caches (review 909c671e's controls): an ON crossing at tick 20 and an OFF crossing at tick 40 on the bids ->
   lag 0 pulls at 1,000 and re-posts at 2,000, lag 1 pulls at 1,050 and re-posts at 2,050 (no timer); a signal ON only at tick 0 -> lag 0
   withholds at 0 and posts at 50, lag 1 posts at 0 (warm-up, counted) and never acts; a signal ON only at tick 30 -> lag 0 pulls at 1,500,
   lag 1 at 1,550 and never before (no one-tick-future read); asks untouched by a bid-only signal (wrong side refused); lag 1's log carries
   the source time t - 50 and the effective age source age + 50;
 3 lag_join: the cached value at k - 1 per (day, m), side and tick; warm-up at k = 0 and a non-ok status at k - 1 give NaN; off-tick states
   counted; the zero-lag join counts mismatched saved values; run_selection refuses a join that does not reproduce the saved values and
   refuses to overwrite (parent_day and the audit states injected);
 4 measurement helpers: reentry_pairs (both gids / prices / rungs, same-price and same-rung flags - a pair with a different price and rung and
   one with the same price and a different rung -, the activation-clock ahead and the ahead difference next - pull, a pull without a next order); the queue relabelling in arm_row; retained; the reading (survives_50ms_screen only
   with END, per share and the selection screen; otherwise not_demonstrated_at_50ms with each failed component);
 5 guards, each isolated with every earlier guard passing and its refusal message asserted: run_day before the freeze, under another rule,
   under another source, the full run without the projection, an existing output, a parent day record differing from the manifest; with
   every guard passing run_day reaches the data load (a sentinel); measure refuses without the selection control and refuses to overwrite."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'LAG SELFTEST FAILED'


class _Sentinel(Exception):
    pass


def run(mod):
    main(mod)


def _cache(nt, x0, x1, st0=None, st1=None, age=5.):
    z = lambda v: np.asarray(v, np.float32)
    return dict(x={0: z(x0), 1: z(x1)}, age={0: np.full(nt, age, np.float32), 1: np.full(nt, age, np.float32)},
                status={0: np.zeros(nt, np.int8) if st0 is None else np.asarray(st0, np.int8), 1: np.zeros(nt, np.int8) if st1 is None else np.asarray(st1, np.int8)})


def main(L):  # noqa: C901
    P, M, LB, BE, pd = L.P, L.M, L.LB, L.BE, L.pd; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    import pf_ladder_rest_selftest as V1
    fr = V1.fr; E = P.frozen_edges()
    # ---- 1 the zero-lag control
    try:
        rng = np.random.default_rng(43); nb = nd = 0
        for i in range(8):
            p = BE.rand_tape(rng, 300, 500, end=30000.); nt = 600; vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32)
            cz = _cache(nt, rng.choice(vals, nt), rng.choice(vals, nt), rng.choice([0, 0, 0, 1, 3], nt), rng.choice([0, 0, 0, 2], nt))
            B1 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); g1 = P.PressureGuard(B1, cz, E); LB.run_episode(B1, g1); L1 = LB.finish(B1)
            B2 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); g2 = L.LaggedPressureGuard(B2, cz, E, 0); LB.run_episode(B2, g2); L2 = LB.finish(B2)
            B3 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); LB.run_episode(B3, L.LaggedPressureGuard(B3, cz, E, 1)); LB.finish(B3)
            nb += int(not (LB.fingerprint(B1) == LB.fingerprint(B2) and L1['fills'] == L2['fills'] and L1['value'] == L2['value'] and B1.cmd_log == B2.cmd_log
                           and [tuple(e[:11]) for e in g2.log] == [tuple(e) for e in g1.log]))
            nd += int(LB.fingerprint(B1) != LB.fingerprint(B3))
        ok(f'1 lag 0 equals the frozen pressure rule ({nb} of 8 differ)', nb == 0)
        ok(f'1 lag 1 differs from it ({nd} of 8 differ)', nd >= 1)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 zero-lag: stopped ({e!r})'[:200])
    # ---- 2 the hand tape
    try:
        p = BE.tape([fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])], [], 50.); nt = 80

        def arm(xb, lag, episode=0):
            B = LB.LadderBranch(p, LB.LADDER, end=4000., episode=episode); g = L.LaggedPressureGuard(B, _cache(nt, xb, np.zeros(nt)), E, lag); LB.run_episode(B, g); return B, g
        bids = lambda B, t0: [o for o in B.E.o if o['side'] == 0 and o['submit'] == t0]
        nxt = lambda B, after: min([o['submit'] for o in B.E.o if o['side'] == 0 and o['submit'] > after] or [None])
        x1 = np.zeros(nt, np.float32); x1[20:40] = 1.; Ba, ga = arm(x1, 0); Bb, gb = arm(x1, 1)
        ok(f'2 ON crossing: lag 0 pulls at 1,000, lag 1 at 1,050 ({[o["req"] for o in bids(Ba, 0.)]}, {[o["req"] for o in bids(Bb, 0.)]})',
           all(o['req'] == 1000. for o in bids(Ba, 0.)) and all(o['req'] == 1050. for o in bids(Bb, 0.)) and len(bids(Bb, 0.)) == 2)
        ok(f'2 OFF crossing: lag 0 re-posts at 2,000, lag 1 at 2,050 ({nxt(Ba, 0.)}, {nxt(Bb, 0.)})', nxt(Ba, 0.) == 2000. and nxt(Bb, 0.) == 2050.)
        x2 = np.zeros(nt, np.float32); x2[0] = 1.; Bc, gc = arm(x2, 0); Bd, gd = arm(x2, 1)
        ok(f'2 warm-up: lag 0 withholds at 0 and posts at 50, lag 1 posts at 0 and never acts ({nxt(Bc, -1.)}, {nxt(Bd, -1.)}, {len(gd.log)})',
           nxt(Bc, -1.) == 50. and nxt(Bd, -1.) == 0. and len(gd.log) == 0 and gd.missing[(0, 'warmup')] >= 1 and gd.missing[(1, 'warmup')] >= 1)
        x3 = np.zeros(nt, np.float32); x3[30] = 1.; Be, ge = arm(x3, 0); Bf, gf = arm(x3, 1)
        ok(f'2 no one-tick-future read: lag 0 pulls at 1,500, lag 1 at 1,550 and never before ({[o["req"] for o in bids(Be, 0.)]}, {[o["req"] for o in bids(Bf, 0.)]})',
           all(o['req'] == 1500. for o in bids(Be, 0.)) and all(o['req'] == 1550. for o in bids(Bf, 0.)) and min(z[0] for z in gf.log) == 1550.)
        ok('2 a bid-only signal leaves the asks untouched (both lags)', all(o['req'] is None for B in (Ba, Bb, Be, Bf) for o in B.E.o if o['side'] == 1)
           and not any(z[1] == 1 for g in (ga, gb, ge, gf) for z in g.log))
        ok('2 lag 1 logs the source time t - 50 and the effective age source age + 50', gb.log and all(z[11] == z[0] - 50. and z[12] == z[6] + 50. for z in gb.log)
           and all(z[11] == z[0] and z[12] == z[6] for z in ga.log))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 hand tape: stopped ({e!r})'[:220])
    # ---- 3 lag_join and run_selection
    tmp = Path(tempfile.mkdtemp(dir=str(P.A.DATA))); saved = dict(parent_day=L.parent_day, AU_OUT=L.AU.OUT, frozen=L.frozen)
    try:
        nt = 20; xs = np.arange(nt, dtype=np.float32) * np.float32(0.1); st = np.zeros(nt, np.int8); st[6] = 1
        caches = {('20261024', 7): dict(x={0: xs, 1: -xs}, age={0: np.zeros(nt, np.float32), 1: np.zeros(nt, np.float32)}, status={0: st, 1: np.zeros(nt, np.int8)})}
        rows = [dict(day='20261024', m=7, kind='live', side=s, t=t, adv_bn_r250=float((np.float32(0.1) * np.float32(t / 50.)) * (1 if s == 0 else -1)))
                for s, t in ((0, 0.), (0, 250.), (0, 350.), (1, 500.), (0, 125.))]
        rows.append(dict(day='20261024', m=7, kind='entry', side=0, t=100., adv_bn_r250=9.)); S_ = pd.DataFrame(rows); S_.loc[3, 'adv_bn_r250'] = 5.
        J, st_ = L.lag_join(S_, caches)
        ok(f'3 lag_join values ({list(J.adv_bn_r250)}, {st_})', np.isnan(J.adv_bn_r250[0]) and J.adv_bn_r250[1] == float(xs[4]) and np.isnan(J.adv_bn_r250[2])
           and J.adv_bn_r250[3] == float(-xs[9]) and np.isnan(J.adv_bn_r250[4]) and J.adv_bn_r250[5] == 9.)
        ok(f'3 lag_join counts ({st_})', st_ == dict(live_states=5, zero_lag_mismatched=1, off_tick=1, warmup=1, non_ok=1))
        L.parent_day = lambda day, expect=None: (None, {7: None}, {f'7__cache__{k}{s}': caches[('20261024', 7)][k][s] for k in ('x', 'age', 'status') for s in (0, 1)})
        L.AU.OUT = tmp; S_.to_csv(tmp / 'states.csv', index=False); mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(days=['20261024'], parent=dict(day_records={'20261024': 'x'}))))
        try:
            L.run_selection(out=tmp / 'sel', require_frozen=False, man_path=mp); bad.append('3 run_selection accepted a join that does not reproduce the saved values')
        except ValueError as e:
            ok('3 run_selection refuses a join that does not reproduce the saved values', 'join keys' in str(e))
        (tmp / 'sel2').mkdir(); (tmp / 'sel2' / 'selection.json').write_text('{}')
        try:
            L.run_selection(out=tmp / 'sel2', require_frozen=False, man_path=mp); bad.append('3 run_selection overwrote')
        except ValueError as e:
            ok('3 run_selection refuses to overwrite', 'write-once' in str(e))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 lag_join: stopped ({e!r})'[:220])
    finally:
        L.parent_day, L.AU.OUT, L.frozen = saved['parent_day'], saved['AU_OUT'], saved['frozen']
    # ---- 4 measurement helpers
    try:
        orders = [dict(gid=0, side=0, rung=0., price=49., submit=0., act=58., remove=1037., size=100., refused=False, req=1000., why='policy'),
                  dict(gid=1, side=0, rung=1., price=48., submit=1500., act=1558., remove=4000., size=100., refused=False, req=None, why=None),
                  dict(gid=2, side=1, rung=0., price=52., submit=0., act=58., remove=2037., size=100., refused=False, req=2000., why='policy'),
                  dict(gid=3, side=0, rung=0., price=49., submit=2500., act=2558., remove=3037., size=100., refused=False, req=3000., why='policy'),
                  dict(gid=4, side=1, rung=1., price=52., submit=2600., act=2658., remove=4000., size=100., refused=False, req=None, why=None)]
        qv = dict(ahead0={'0': [0., 30., 100.], '1': [1500., 12., 100.], '2': [0., 40., 100.], '3': [2500., 70., 100.], '4': [2600., 15., 100.]},
                  first={'0': [100., 25., 100.], '2': [100., 35., 100.]}, pull={'0': [1000., 20., 100.], '2': [2000., 10., 100.], '3': [3000., 50., 100.]})
        rec = dict(orders=orders, queue_venue=qv, queue_known={'0': dict(first=24.), '2': dict(first=None)}, fills=[], fill_mids=[], mark=50., value=0., cash=0., position=0., end=4000.,
                   guard_log=[], missing={})
        rp = L.reentry_pairs(rec); r0, r2, r3 = ([r for r in rp if r['pull_gid'] == g][0] for g in (0, 2, 3))
        ok(f'4 reentry_pairs ({r0}, {r2["next_gid"]}, {r3["next_gid"]})', r0['next_gid'] == 1 and r0['next_price'] == 48. and r0['next_rung'] == 1. and r0['same_price'] is False
           and r0['same_rung'] is False and r0['next_activation_ahead'] == 12. and r0['ahead_difference'] == 12. - 20. and r0['gap_ms'] == 500.
           and r2['next_gid'] == 4 and r2['same_price'] is True and r2['same_rung'] is False and r2['ahead_difference'] == 15. - 10. and r3['next_gid'] is None and np.isnan(r3['ahead_difference']))
        ps = L.pair_summary(rp); ok(f'4 pair_summary ({ps})', ps['pairs_pulls'] == 3 and ps['pairs_with_next'] == 2 and ps['pairs_same_price'] == 1 and ps['pairs_same_rung'] == 0
                                    and ps['pairs_ahead_difference_mean'] == -1.5 and ps['pairs_same_price_ahead_difference_mean'] == 5.)
        cm = np.array([[100., 0, M.CMD_KINDS.index('live'), 0, 0., 49., np.nan, M.CMD_ACTIONS.index('KEEP'), P.REASONS.index('keep')]], float)
        row = L.arm_row('20261024', 7, 'lag1', rec, dict(commands=cm, nva=np.zeros((80, 2), np.int8)), 4000.)
        ok('4 arm_row carries the relabelled queue fields only', 'q_next_minus_pull_ahead_mean' in row and 'q_surrendered_mean' not in row and 'q_activation_ahead_mean' in row
           and row['q_activation_ahead_mean'] == np.mean([30., 12., 40., 70., 15.]) and row['pairs_with_next'] == 2)
        R_ = pd.DataFrame([dict(day=d, m=m, arm=a, value=v, all_q=q, all_cut_mo1s=c, all_cut_valid_q=10., bid_cut_mo1s=c, bid_cut_valid_q=5., ask_cut_mo1s=c, ask_cut_valid_q=5., bid_end=v / 2,
                                ask_end=v / 2, both_exact=.4, bid_exact=.5, ask_exact=.5, pulls=0, withholds=0, reducing_suppressed=0, mean_abs_inv=1., share_abs_ge_S=0., terminal_inv=0.)
                           for d in ('20261024', '20261025') for m in (1, 2) for a, v, q, c in (('baseline', 0., 100., -10.), ('lag0', 10., 80., -2.), ('lag1', 5., 90., -6.))])
        rt = L.retained(R_); ok(f'4 retained ({rt["END"]["all"]}, {rt["per_share"]["all"]})', abs(rt['END']['all'] - .5) < 1e-12 and abs(rt['per_share']['all'] - .5) < 1e-12
                                                                                         and rt['lag1_minus_lag0_END']['20261024'] == -10.)
        p1 = L.predicates(R_, 'SELECTION_BEYOND_EXPOSURE', 'SELECTION_BEYOND_EXPOSURE'); p2 = L.predicates(R_, 'INCONCLUSIVE', 'SELECTION_BEYOND_EXPOSURE')
        R3 = R_.copy(); R3.loc[(R3.arm == 'lag1') & (R3.day == '20261025'), 'value'] = -3.; R3.loc[R3.arm == 'lag1', 'all_cut_mo1s'] = -12.; p3 = L.predicates(R3, 'SELECTION_BEYOND_EXPOSURE', 'x')
        ok(f'4 the reading ({p1["reading"]}; {p2["failed"]}; {p3["failed"]})', p1['reading'] == 'survives_50ms_screen' and not p1['failed'] and p2['reading'] == 'not_demonstrated_at_50ms'
           and p2['failed'] == ['selection: INCONCLUSIVE'] and p3['failed'] == ['END_paired: MIXED', 'mo1s_per_share: WORSENS_EVERY_DAY'])
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 measurement: stopped ({e!r})'[:220])
    # ---- 5 guards
    saved = dict(frozen=L.frozen, rule_sha=L.rule_sha, sources=L.sources); ld = M.load_day
    try:
        day = '20261024'; pm = L.parent_manifest()
        man = dict(rule_sha256=L.rule_sha(), sources=L.sources(), pilot={day: [3697902]}, complete={day: [3697902]}, days=[day],
                   parent=dict(day_records={day: L.C.sha(P.day_paths(day, False)['rec'])}, rule_sha256=pm['rule_sha256']))
        mp = tmp / 'manifest.json'; mp.write_text(json.dumps(man)); pp = tmp / 'projection.json'

        def refuses(word, man_=None, **kw):
            if man_ is not None:
                mp.write_text(json.dumps(man_))
            try:
                L.run_day(day, man_path=mp, proj_path=pp, out=tmp / 'out', **kw); return False
            except ValueError as e:
                return word in str(e)
            except _Sentinel:
                return False
            finally:
                mp.write_text(json.dumps(man))

        def sentinel(*a, **k):
            raise _Sentinel()
        M.load_day = sentinel
        L.frozen = lambda: False; ok('5 run_day refuses before the freeze', refuses('committed', pilot=True))
        L.frozen = lambda: True; L.rule_sha = lambda: 'x' * 64; ok('5 run_day refuses another rule', refuses('rule', pilot=True)); L.rule_sha = saved['rule_sha']
        L.sources = lambda: dict(saved['sources'](), **{'pf_stop_model.py': '0' * 64}); ok('5 run_day refuses another source', refuses('source', pilot=True)); L.sources = saved['sources']
        ok('5 the full run refuses without the projection', refuses('projection', pilot=False))
        Pd = L.day_paths(day, True, tmp / 'out'); Pd['rec'].parent.mkdir(parents=True, exist_ok=True); Pd['rec'].write_text('{}')
        ok('5 run_day refuses an existing output', refuses('write-once', pilot=True)); Pd['rec'].unlink()
        ok('5 run_day refuses a parent day record differing from the manifest', refuses('differs from the manifest', dict(man, parent=dict(man['parent'], day_records={day: '0' * 64})), pilot=True))
        try:
            L.run_day(day, pilot=True, man_path=mp, proj_path=pp, out=tmp / 'out'); bad.append('5 the all-pass run_day did not reach the load')
        except _Sentinel:
            pass
        except Exception as e:                                                             # noqa: BLE001
            bad.append(f'5 the all-pass run_day stopped before the load ({e!r})'[:200])
        mo = tmp / 'meas'; (mo / 'measure').mkdir(parents=True)
        try:
            L.measure(out=mo, man_path=mp); bad.append('5 measure ran without the selection control')
        except ValueError as e:
            ok('5 measure refuses without the selection control', 'selection' in str(e))
        (mo / 'measure' / 'audit.json').write_text('{}')
        try:
            L.measure(out=mo, man_path=mp); bad.append('5 measure overwrote')
        except ValueError as e:
            ok('5 measure refuses to overwrite', 'write-once' in str(e))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 guards: stopped ({e!r})'[:200])
    finally:
        L.frozen, L.rule_sha, L.sources = saved['frozen'], saved['rule_sha'], saved['sources']; M.load_day = ld; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('LAG SELFTEST PASSED: the zero-lag control, the hand tape (ON / OFF crossings, warm-up, no future read, wrong side, log fields), lag_join and '
          'run_selection, measurement helpers and the reading, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_pressure_lag as L_
    main(L_)
