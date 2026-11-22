"""SELFTEST of pf_ladder_retention_discrimination (LONG-VALUE-RETENTION-DISCRIMINATION-v1; 2026-11-22, session aea5bfb9). main(X) checks the module on
synthetic tapes and hand tables only (no data is read); a failed check raises SystemExit with FAIL. Groups:
 1 features: a hand branch view -> every declared feature by its definition (distance, spread, the received level size, inventory toward the
   order, reservations, counts, the receipt-clocked pressure value and its status, time left, the received 1 s mid change, the mode);
 2 run_market on a synthetic market (the plain run standing in for the saved record): every pair ok, base parity, the expected callback
   (KEEP / PULL at the state's time and gid), features recorded once and equal in both arms, the declared columns present;
 3 known-bad controls: features that differ between arms and an unexpected callback each fail their gate (status gate_failed);
 4 the candidate: winsorize / standardize / clip / ridge reproduced by hand; heavy shrinkage pulls predictions to the clipped mean;
 5 evaluate: V_rule, the controls, deterministic matched draws and p, per-day conditions, and the three readings on constructed tables;
 6 guards: run_part / measure refuse before the freeze and under another source, a rest without the projection; manifest / projection
   write-once."""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'RETENTION DISCRIMINATION SELFTEST FAILED'


def run(mod):
    main(mod)


def main(X):  # noqa: C901
    import pandas as pd
    import pf_ladder_long_continuation_selftest as T
    LC, P, M, LB, BE, AF = X.LC, X.P, X.M, X.LB, X.BE, X.AF
    bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    E = P.frozen_edges(); nt = int(np.ceil(T.END / BE.STEP)) + 2

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw); return False
        except ValueError as e2:
            return word in str(e2)
    # ---- 1 features
    try:
        import pf_ladder_rest_selftest as V1
        fr = np.array([V1.fr(0, 0, [(50, 10)], [(53, 10)]), V1.fr(500, 600, [(50, 30), (49, 40)], [(52, 20)])], float)

        class Bk:
            pass
        B_ = Bk(); B_.E = Bk(); B_.E.f = fr; B_.E.known = 1
        cache = T._cache(nt, np.full(nt, 0.47), np.zeros(nt)); pol = Bk(); pol.value = lambda v: (0.47, 'ok', 5.)
        v = dict(t=1599., side=0, bid=50., ask=52., position=-12., B=30., A=5., orders=[1, 2], opp_prices=[52.], end=6000., mode='UNRESTRICTED')
        o = dict(gid=3, price=49., rung=1., size=30., known_filled=10., act=1549., submit=1539.)
        fx = X.features(B_, pol, v, o)
        want = dict(age_ms=50., rung=1., dist_ticks=1., spread_ticks=2., size=30., known_rem=20., shown_at_price=40., position_toward=-12., reservation_side=30., reservation_opp=5.,
                    n_side=2., n_opp=1., pressure_x=0.47, pressure_ok=1., time_left_ms=4401., mid_change_toward_1s=51. - 51.5, restricted=0.)
        ok(f'1 the features by definition ({ {k: (fx[k], want[k]) for k in want if fx[k] != want[k]} })', fx == want and tuple(fx) == X.FEATURES)
        va = dict(v, side=1); oa = dict(o, price=53.); pol.value = lambda v_: (np.nan, 'stale', 5.); fa = X.features(B_, pol, va, oa)
        ok(f'1 an ask, a missing pressure and a restricted mode ({fa["dist_ticks"]}, {fa["position_toward"]}, {fa["pressure_ok"]})', fa['dist_ticks'] == 1. and fa['position_toward'] == 12.
           and fa['pressure_x'] == 0. and fa['pressure_ok'] == 0. and fa['shown_at_price'] == 0. and fa['mid_change_toward_1s'] == -(51. - 51.5)
           and X.features(B_, pol, dict(va, mode='LONG_RESTRICTED'), oa)['restricted'] == 1.)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 features: stopped ({e!r})'[:220])
    # ---- 2 run_market
    saved = X.saved_record
    try:
        rng = np.random.default_rng(7); path = BE.rand_tape(rng, 70, 160, end=T.END); vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32)
        cache = T._cache(nt, rng.choice(vals, nt), rng.choice(vals, nt))
        B = AF.ProfileBranch(path, T.END, dict(end_rel=T.END), False, LC.PROFILE); pol = P.PressureGuard(B, cache, E); obs = P.QueueObs(B, pol); LB.run_episode(B, pol, observe=obs)
        L = LB.finish(B); rec = M.ledger_record(B, L); rec['guard_log'] = [list(x) for x in pol.log]; cm = P.command_arrays(B)['commands']
        lk = cm[(cm[:, 2] == 0) & (cm[:, 7] == 2) & (cm[:, 8] == LC.KEEP_REASON)]; sts = []
        for g in np.unique(lk[:, 3]).astype(int)[:5]:
            x = lk[lk[:, 3] == g]; i = int(np.argmin(x[:, 0]))
            sts.append(dict(day='d', m=1, part='pilot', kind='REVIEW', side=int(x[i, 1]), gid=int(g), rung=float(x[i, 4]), t=float(x[i, 0]), key=f'R{g}', p_incl=0.5))
        X.saved_record = lambda day, m, part: (rec, cm)
        pairs, par, lgs = X.run_market('d', 1, 'pilot', path, dict(end_rel=T.END), cache, E, sts)
        ok(f'2 pairs ok and parity ({[p["status"] for p in pairs]}, {par["all"]})', len(pairs) == len(sts) >= 4 and all(p['status'] == 'ok' for p in pairs) and par['all'] is True)
        ok('2 the expected callback and equal features in both arms', all(p['gate_expected_callback'] and p['gate_features_equal'] for p in pairs))
        ok('2 every declared feature column present', all(all(f'f_{f}' in p for f in X.FEATURES) for p in pairs) and all(p['p_incl'] == 0.5 and p['part'] == 'pilot' for p in pairs))
        ok('2 the ledgers carry the features', all(x['features'] is not None for x in lgs if x['kind'] == 'REVIEW') and sum(x['kind'] == 'BASE' for x in lgs) == 1)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 run_market: stopped ({e!r})'[:240])
    # ---- 3 known-bad controls
    try:
        fg = X.FeatureGuard; cnt = dict(n=0)

        class Odd(fg):
            def live(self, v, o):
                a = super().live(v, o)
                if self.features is not None and self.fork is not None and self.fork['action'] == 'PULL':
                    self.features = dict(self.features, age_ms=self.features['age_ms'] + 1.)
                return a
        X.FeatureGuard = Odd
        try:
            pb, _, _ = X.run_market('d', 1, 'pilot', path, dict(end_rel=T.END), cache, E, sts[:2])
        finally:
            X.FeatureGuard = fg
        ok(f'3 features differing between arms fail their gate ({[p["status"] for p in pb]})', all(not p['gate_features_equal'] and p['status'] == 'gate_failed' for p in pb))
        pc, _, _ = X.run_market('d', 1, 'pilot', path, dict(end_rel=T.END), cache, E, [dict(sts[0], gid=sts[0]['gid'] + 1000)])
        ok(f'3 an unexpected callback fails ({pc[0]["status"]})', not pc[0]['gate_expected_callback'] and pc[0]['status'] != 'ok')
        cache0 = T._cache(nt, np.zeros(nt), np.zeros(nt)); B0 = AF.ProfileBranch(path, T.END, dict(end_rel=T.END), False, LC.PROFILE); p0 = P.PressureGuard(B0, cache0, E)
        o0 = P.QueueObs(B0, p0); LB.run_episode(B0, p0, observe=o0); L0 = LB.finish(B0); rec0 = M.ledger_record(B0, L0); rec0['guard_log'] = [list(x) for x in p0.log]
        cm0 = P.command_arrays(B0)['commands']; X.saved_record = lambda day, m, part: (dict(rec0, fills=rec0['fills'][:-1]), cm0)
        lk0 = cm0[(cm0[:, 2] == 0) & (cm0[:, 7] == 2) & (cm0[:, 8] == LC.KEEP_REASON)]; g0 = int(np.unique(lk0[:, 3])[0]); x0 = lk0[lk0[:, 3] == g0]; i0 = int(np.argmin(x0[:, 0]))
        st0 = dict(day='d', m=1, part='pilot', kind='REVIEW', side=int(x0[i0, 1]), gid=g0, rung=float(x0[i0, 4]), t=float(x0[i0, 0]), key='Q0', p_incl=0.5)
        pq, parq, _ = X.run_market('d', 1, 'pilot', path, dict(end_rel=T.END), cache0, E, [st0])
        ok(f'3 a retain arm unequal to the saved record fails ({pq[0]["gate_unused_exemption"]}, {pq[0]["status"]})', pq[0]['gate_unused_exemption'] is False and pq[0]['status'] == 'gate_failed'
           and parq['all'] is False)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 known-bad: stopped ({e!r})'[:240])
    finally:
        X.saved_record = saved
    # ---- 4 the candidate
    try:
        rng = np.random.default_rng(3); Xa = rng.normal(size=(60, 3)); Xa[0, 0] = 50.; ya = Xa @ np.array([2., -1., 0.]) + rng.normal(size=60); ya[1] = 400.
        m = X.fit(Xa, ya); lo, hi = np.quantile(Xa, .01, axis=0), np.quantile(Xa, .99, axis=0); Xc = np.clip(Xa, lo, hi); mu, sd = Xc.mean(0), Xc.std(0); Z = (Xc - mu) / sd
        yc = np.clip(ya, np.quantile(ya, .05), np.quantile(ya, .95)); w = np.linalg.solve(Z.T @ Z + 60. * np.eye(3), Z.T @ (yc - yc.mean()))
        ok('4 the candidate reproduced by hand', np.allclose(m['w'], w) and abs(m['ym'] - yc.mean()) < 1e-12 and np.allclose(X.predict(m, Xa), yc.mean() + Z @ w) and m['n'] == 60)
        ok('4 heavy shrinkage', float(np.std(X.predict(m, Xa))) < float(np.std(yc)) / 1.5)
        m0 = X.fit(np.column_stack([np.ones(60), Xa[:, :1]]), ya); ok('4 a constant feature keeps sd 1', m0['sd'][0] == 1. and np.isfinite(X.predict(m0, np.column_stack([np.ones(60), Xa[:, :1]]))).all())
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 candidate: stopped ({e!r})'[:220])
    # ---- 5 evaluate
    try:
        def table(ys, preds, days, side=0):
            return pd.DataFrame(dict(Y=ys, pred=preds, day=days, side=side, m=1, weight=2.))
        Tr = table([10.] * 20 + [-5.] * 20, [0.] * 40, ['20261024'] * 40)
        good_y = [50., 40., -30., -20.] * 10; good_p = [1., 1., -1., -1.] * 10; dd = ['20261026'] * 20 + ['20261027'] * 20
        r1 = X.evaluate(Tr, table(good_y, good_p, dd))
        ok(f'5 a discriminating rule ({r1["V_rule"]}, {r1["p_draw"]}, {r1["reading"]})', r1['V_rule'] == 900. and r1['V_always_retain'] == 400. and r1['V_constant'] == 400.
           and r1['constant_retains'] is True and r1['retained'] == 20 and r1['p_draw'] <= 0.05 and r1['reading'] == 'useful_local_discrimination' and r1['V_rule_weighted'] == 1800.)
        r1b = X.evaluate(Tr, table(good_y, good_p, dd)); ok('5 the draws are deterministic', r1b['draws_mean'] == r1['draws_mean'] and r1b['p_draw'] == r1['p_draw'])
        r2 = X.evaluate(Tr, table(good_y, [-p for p in good_p], dd)); ok(f'5 a harmful rule ({r2["reading"]})', r2['reading'] == 'no_useful_discrimination' and r2['V_rule'] == -500.)
        r3 = X.evaluate(Tr, table([1., -1.] * 10, [1., -1.] * 10, ['20261026'] * 10 + ['20261027'] * 10)); ok(f'5 thin support ({r3["reading"]})', r3['reading'] == 'INSUFFICIENT')
        mixed_y = good_y[:20] + [-5., -4., 3., 2.] * 5; r4 = X.evaluate(Tr, table(mixed_y, good_p, dd))
        ok(f'5 one evaluation day failing ({r4["per_day"]["20261027"]["V_rule"]}, {r4["reading"]})', r4['per_day']['20261027']['V_rule'] < 0 and r4['V_rule'] > 0
           and r4['V_rule'] > r4['V_always_retain'] and r4['p_draw'] <= 0.05 and r4['reading'] == 'no_useful_discrimination')
        ok('5 MSE, calibration, concentration fields', r1['mse_zero'] > 0 and len(r1['calibration']) == 4 and 0 < r1['selected_top5_share'] <= 1 and r1['selected_min'] == 40.)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 evaluate: stopped ({e!r})'[:240])
    # ---- 6 guards
    sv = dict(frozen=X.frozen, sources=X.sources, af_records=X.af_records); tmp = Path(tempfile.mkdtemp())
    try:
        mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(sources=X.sources(), parts=dict(prefix=['d'], rest=['e']), markets={}, selected=[], af_records={})))
        X.frozen = lambda: False; ok('6 run_part refuses before the freeze', refuses('committed', X.run_part, 'prefix', 'd', out=tmp / 'o', man_path=mp))
        ok('6 measure refuses before the freeze', refuses('committed', X.measure, out=tmp / 'o', man_path=mp))
        X.frozen = lambda: True; X.sources = lambda: dict(sv['sources'](), **{'pf_ladder_retention_discrimination.py': '0' * 64})
        ok('6 run_part refuses another source', refuses('source', X.run_part, 'prefix', 'd', out=tmp / 'o', man_path=mp)); X.sources = sv['sources']
        X.af_records = lambda gm_af: {}
        ok('6 a rest without the projection is refused', refuses('projection', X.run_part, 'rest', 'e', out=tmp / 'o', man_path=mp, proj_path=tmp / 'absent.json'))
        ok('6 manifest / projection write-once', refuses('write-once', X.build_manifest, mp, tmp / 'c.csv') and refuses('write-once', X.project, mp))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 guards: stopped ({e!r})'[:200])
    finally:
        X.frozen, X.sources, X.af_records = sv['frozen'], sv['sources'], sv['af_records']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('RETENTION DISCRIMINATION SELFTEST PASSED: features, run_market gates, known-bad controls, the candidate, evaluate and readings, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_retention_discrimination as X_
    main(X_)
