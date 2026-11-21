"""SELFTEST of pf_ladder_pressure_positive (POSITIVE-PRESSURE-GUARD-v1; 2026-11-21, session aea5bfb9). main(X) checks the module; a failed check raises
SystemExit with FAIL. Each group names only what its code runs.
 1 the signal: the smallest positive float32 ON, 0 / -0 / negative / NaN OFF, a non-ok status OFF and counted, the disabled guard OFF, both
   sides; on 6 random tapes whose cache holds only 0, -0.5 and 1.0 (the same ON set as q5) the challenger equals the frozen PressureGuard
   (fingerprint, command log, log); with 0.24 (positive, below both q5 edges) it differs on at least one;
 2 a hand tape (bids at 0.24 over ticks 20-39): the challenger pulls the bids at 1,000 and re-posts at 2,000; Q5_CONTROL keeps them;
 3 screen_population: only non-fallback KEEP callbacks (an inventory PULL, a fallback KEEP and a state without a callback are excluded, the
   last counted), finite and q5-OFF values; the encoded mask reproduces x > 0 through the frozen selection's q5 test;
 4 the readings on a synthetic three-arm table: IMPROVES / WORSENS / MIXED from the END and eligible +1 s day signs, the screen apart,
   stop_candidate, all_arms_negative, market counts;
 5 guards, each isolated with every earlier guard passing and its refusal message asserted: run_day before the freeze, under another rule,
   under another source, a day outside its part, the rest without the projection, an existing output, a parent day record differing from
   the manifest; with every guard passing run_day reaches the data load (a sentinel); run_screen and measure refuse to overwrite; measure
   refuses without the screen."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'POSITIVE SELFTEST FAILED'


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
    fr = V1.fr; E = P.frozen_edges()
    # ---- 1 the signal and delegation
    try:
        tiny = float(np.nextafter(np.float32(0.), np.float32(1.))); xs = [tiny, 0., -0., -1., np.nan, 0.3, 0.3]; nt = len(xs)
        Bh = LB.LadderBranch(BE.tape([fr(0, 0, [(50, 0)], [(52, 0)])], [], 50.), LB.LADDER, end=3000.)
        g = X.PositivePressureGuard(Bh, _cache(nt, xs, xs, [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 3]), E); gd = X.PositivePressureGuard(Bh, _cache(nt, xs, xs), E, enabled=False)
        v = lambda k, s: dict(t=50. * k, side=s, position=0., B=0., A=0., mode='UNRESTRICTED', avail=200.)
        on = [g.signal(v(k, s))[0] for k in range(nt) for s in (0, 1)]
        ok(f'1 the signal boundary on both sides ({on})', on == [True, True, False, False, False, False, False, False, False, False, True, True, False, False] and g.missing[(0, 'off_grid')] == 1
           and g.missing[(1, 'nan')] == 1 and not gd.signal(v(0, 0))[0])
        rng = np.random.default_rng(53); ne = nd = 0
        for i in range(6):
            p = BE.rand_tape(rng, 300, 500, end=30000.); n6 = 600; same = np.array([0., -0.5, 1.0], np.float32); low = np.array([0., -0.5, 0.24], np.float32)
            for vals, tag in ((same, 'same'), (low, 'low')):
                cz = _cache(n6, rng.choice(vals, n6), rng.choice(vals, n6)); r_ = {}
                for nm, cls in (('q5', P.PressureGuard), ('pos', X.PositivePressureGuard)):
                    B = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); pol = cls(B, cz, E); LB.run_episode(B, pol); LB.finish(B); r_[nm] = (LB.fingerprint(B), B.cmd_log, pol.log)
                if tag == 'same':
                    ne += int(r_['q5'] != r_['pos'])
                else:
                    nd += int(r_['q5'][0] != r_['pos'][0])
        ok(f'1 with the q5 ON set the challenger equals PressureGuard ({ne} of 6 differ); below q5 it acts ({nd} of 6 differ)', ne == 0 and nd >= 1)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 signal: stopped ({e!r})'[:220])
    # ---- 2 the hand tape
    try:
        p = BE.tape([fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])], [], 50.); nt = 80; xb = np.zeros(nt, np.float32); xb[20:40] = 0.24
        res = {}
        for nm, cls in (('q5', P.PressureGuard), ('pos', X.PositivePressureGuard)):
            B = LB.LadderBranch(p, LB.LADDER, end=4000., episode=0); LB.run_episode(B, cls(B, _cache(nt, xb, np.zeros(nt)), E)); res[nm] = B
        b0 = lambda B: [o for o in B.E.o if o['side'] == 0 and o['submit'] == 0.]; nxt = lambda B: min([o['submit'] for o in B.E.o if o['side'] == 0 and o['submit'] > 0.] or [None])
        ok(f'2 the challenger pulls at 1,000 and re-posts at 2,000; Q5 keeps ({[o["req"] for o in b0(res["pos"])]}, {nxt(res["pos"])}, {[o["req"] for o in b0(res["q5"])]})',
           all(o['req'] == 1000. for o in b0(res['pos'])) and nxt(res['pos']) == 2000. and all(o['req'] is None for o in b0(res['q5'])))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 hand tape: stopped ({e!r})'[:220])
    # ---- 3 screen_population
    try:
        rows, cm = [], []
        LIVE, KEEP, PULL = M.CMD_KINDS.index('live'), M.CMD_ACTIONS.index('KEEP'), M.CMD_ACTIONS.index('PULL')
        spec = [(0.24, 'keep', KEEP), (0.0, 'keep', KEEP), (-0.3, 'keep', KEEP), (1.0, 'keep', KEEP), (0.24, 'inventory_pull', PULL), (0.24, 'fallback', KEEP), (np.nan, 'keep', KEEP), (0.24, None, None)]
        for i, (x, why, act) in enumerate(spec):
            rows.append(dict(day='20261024', m=7, kind='live', side=i % 2, gid=i, t=100. + 50 * i, adv_bn_r250=x, win_n=1, win_q=10., win_mo1s=-1., censored_1s=False, ctx_bid_reserved_shares=100.,
                             fp_remaining=50., ctx_generation_age_ms=1000., inv_known=0.))
            if why is not None:
                cm.append([100. + 50 * i, i % 2, LIVE, i, 0., 50., np.nan, act, P.REASONS.index(why)])
        rows.append(dict(rows[0], kind='entry', gid=99)); S_ = pd.DataFrame(rows)
        h, cnt = X.screen_population(S_, {('20261024', 7): np.array(cm, float)}, E)
        ok(f'3 the population ({cnt}, {list(h.gid)})', cnt == dict(live_states=8, without_callback=1, keep_callbacks=5, finite=4, q5_off=3, mask_on=1) and sorted(h.gid) == [0, 1, 2])
        sel = P.selection_table(h, E)
        ok('3 the encoded mask reproduces x > 0 through the frozen q5 test', list(sel.sort_values('gid').selected) == [True, False, False])
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 screen: stopped ({e!r})'[:220])
    # ---- 4 readings
    try:
        V = {'20261024': dict(BASELINE=-10., Q5_CONTROL=-6., POSITIVE=-4.), '20261025': dict(BASELINE=-12., Q5_CONTROL=-8., POSITIVE=-7.)}
        R_ = pd.DataFrame([dict(day=d, m=m, arm=a, value=v_, all_id_mo1s=v_, all_cut_mo1s=v_, all_cut_valid_q=10., all_q=10., both_exact=.4, bid_exact=.5, ask_exact=.5, pulls=1, withholds=1,
                                reducing_suppressed=0, first_review_pulls=0, generations=5, bid_structural=1, ask_structural=1, mean_abs_inv=1., share_abs_ge_S=0.)
                           for d, av in V.items() for m in (1, 2) for a, v_ in av.items()])
        p1 = X.predicates(R_, 'SELECTION_BEYOND_EXPOSURE'); p2 = X.predicates(R_, 'INCONCLUSIVE')
        R3 = R_.copy(); R3.loc[(R3.arm == 'POSITIVE') & (R3.day == '20261025'), 'value'] = -9.; p3 = X.predicates(R3, 'SELECTION_BEYOND_EXPOSURE')
        R4 = R_.copy(); R4.loc[R4.arm == 'POSITIVE', ['value', 'all_id_mo1s']] = -20.; p4 = X.predicates(R4, 'SELECTION_BEYOND_EXPOSURE')
        ok(f'4 the economic reading ({p1["economic"]}, {p3["economic"]}, {p4["economic"]})', p1['economic'] == 'IMPROVES' and p3['economic'] == 'MIXED' and p4['economic'] == 'WORSENS')
        ok('4 stop_candidate unless IMPROVES and the screen passes', not p1['stop_candidate'] and p2['stop_candidate'] and p3['stop_candidate'] and p2['screen'] == 'INCONCLUSIVE')
        ok('4 all_arms_negative and market counts', p1['all_arms_negative']['total'] is True and p1['END']['markets_better'] == 4 and p1['END']['total'] == 2 * (2. + 1.))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 readings: stopped ({e!r})'[:220])
    # ---- 5 guards
    saved = dict(frozen=X.frozen, rule_sha=X.rule_sha, sources=X.sources); ld = M.load_day; tmp = Path(tempfile.mkdtemp(dir=str(X.A.DATA)))
    try:
        day = '20261024'; man = dict(rule_sha256=X.rule_sha(), sources=X.sources(), pilot_day=day, parts=dict(pilot={day: [3697902], '20261025': []}, rest={day: [], '20261025': [3723774]}),
                                     days=[day], parent=dict(day_records={day: X.C.sha(P.day_paths(day, False)['rec'])}))
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
        ok('5 run_day refuses a day outside its part', refuses('belong', part='rest') and refuses('belong', day_='20261025', part='pilot'))
        ok('5 the rest refuses without the projection', refuses('projection', day_='20261025', part='rest'))
        Pd = X.day_paths(day, 'pilot', tmp / 'out'); Pd['rec'].parent.mkdir(parents=True, exist_ok=True); Pd['rec'].write_text('{}')
        ok('5 run_day refuses an existing output', refuses('write-once', part='pilot')); Pd['rec'].unlink()
        ok('5 run_day refuses a parent day record differing from the manifest', refuses('differs from the manifest', dict(man, parent=dict(day_records={day: '0' * 64})), part='pilot'))
        try:
            X.run_day(day, 'pilot', man_path=mp, proj_path=pp, out=tmp / 'out'); bad.append('5 the all-pass run_day did not reach the load')
        except _Sentinel:
            pass
        except Exception as e:                                                             # noqa: BLE001
            bad.append(f'5 the all-pass run_day stopped before the load ({e!r})'[:200])
        so = tmp / 'scr'; so.mkdir(); (so / 'screen.json').write_text('{}')
        try:
            X.run_screen(out=so, require_frozen=False, man_path=mp); bad.append('5 run_screen overwrote')
        except ValueError as e:
            ok('5 run_screen refuses to overwrite', 'write-once' in str(e))
        mo = tmp / 'meas'; (mo / 'measure').mkdir(parents=True)
        try:
            X.measure(out=mo, man_path=mp); bad.append('5 measure ran without the screen')
        except ValueError as e:
            ok('5 measure refuses without the screen', 'screen' in str(e))
        (mo / 'measure' / 'audit.json').write_text('{}')
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
    print('POSITIVE SELFTEST PASSED: the signal and delegation, the hand tape, the screen population, the readings, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_pressure_positive as X_
    main(X_)
