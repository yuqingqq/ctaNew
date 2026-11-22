"""SELFTEST of pf_ladder_long_continuation (LONG-CONTINUATION-PAIRED-VALUE-v1; 2026-11-22, session aea5bfb9). main(X) checks the module on synthetic
random tapes only (no saved data is read); a failed check raises SystemExit with FAIL. Groups (contract_v2 section 5 / review 32d102bd):
 1 inert until armed: an unarmed LineageGuard = the frozen PressureGuard (fingerprint, guard log);
 2 run_market on a synthetic market, the plain run standing in for the saved record: base parity; every pair's forced action applied
   once, at a context equal in both arms and to the plain run's own callback context (a second-side callback among them); pre-fork fills
   identical; the unused-exemption control; reconciliation; the queue reset of every lineage replacement;
 3 known-bad controls, each refused by its named check: an exemption missing in one arm (no_lineage_pull), inherited replacement priority
   (queue_reset), a reused arm state (fork_key), a wrong command cursor (callback_not_reached), double-counted carried inventory (reconciles);
 4 future-only perturbation: a tape changed only after T1 leaves both arms' commands and fills up to T1 unchanged;
 5 pure parts: shown_at, fifo, active_both, sequence_axis, pair_result's classes, summarize's reading;
 6 guards: run_part / measure refuse before the freeze and under another source, a rest without the projection; manifest / projection
   write-once."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'LONG CONTINUATION SELFTEST FAILED'
END = 6000.


def run(mod):
    main(mod)


def _cache(nt, x0, x1):
    z = lambda v: np.asarray(v, np.float32)
    return dict(x={0: z(x0), 1: z(x1)}, age={0: np.full(nt, 5., np.float32), 1: np.full(nt, 5., np.float32)}, status={0: np.zeros(nt, np.int8), 1: np.zeros(nt, np.int8)},
                kk={0: np.arange(nt), 1: np.arange(nt)}, nt=nt, grids_equal=True)


def main(X):  # noqa: C901
    import pandas as pd
    P, M, LB, BE, AF = X.P, X.M, X.LB, X.BE, X.AF
    bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    E = P.frozen_edges(); nt = int(np.ceil(END / BE.STEP)) + 2

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw); return False
        except ValueError as e2:
            return word in str(e2)

    class Rec(X.LineageGuard):
        def __init__(self, B, cache, edges):
            super().__init__(B, cache, edges); self.ctx = {}

        def live(self, v, o):
            self.ctx[('live', float(v['t']), int(o['gid']))] = X.ctx_hash(v, o); return super().live(v, o)

        def slot(self, v, r, price):
            self.ctx[('slot', float(v['t']), int(v['side']), float(r))] = X.ctx_hash(v); return super().slot(v, r, price)

    def plain(path, cache, guard=None):
        B = AF.ProfileBranch(path, END, dict(end_rel=END), False, X.PROFILE); pol = (guard or Rec)(B, cache, E); obs = P.QueueObs(B, pol)
        LB.run_episode(B, pol, observe=obs); L = LB.finish(B); rec = M.ledger_record(B, L); rec['guard_log'] = [list(x) for x in pol.log]
        return B, pol, rec, P.command_arrays(B)['commands']

    def states_of(cm, fills=()):
        lk = cm[(cm[:, 2] == 0) & (cm[:, 7] == 2) & (cm[:, 8] == X.KEEP_REASON)]; sp = cm[(cm[:, 2] == 1) & (cm[:, 7] == 0) & (cm[:, 3] >= 0)]; out = []
        firsts = sorted({int(g): float(lk[lk[:, 3] == g, 0].min()) for g in np.unique(lk[:, 3])}.items(), key=lambda x: x[1])
        sec = [g for g, t in firsts if int(lk[(lk[:, 3] == g) & (lk[:, 0] == t), 1][0]) != int(round(t / BE.STEP)) % 2]
        pick = ([sec[0]] if sec else []) + [g for g, t in firsts if g not in sec][:2]; pos = lambda t: sum((1. if f[2] == 0 else -1.) * f[3] for f in fills if f[0] <= t)
        late = [g for g, t in firsts if t >= 1000. and g not in pick and abs(pos(t)) > 0][:1]
        for g in pick[:3] + late:
            t = dict(firsts)[g]; r = lk[(lk[:, 3] == g) & (lk[:, 0] == t)][0]
            out.append(dict(day='d', m=1, kind='REVIEW', side=int(r[1]), gid=int(g), rung=float(r[4]), t=t, key=f'R{g}'))
        for r in sp[1:3]:
            out.append(dict(day='d', m=1, kind='ADMISSION', side=int(r[1]), gid=int(r[3]), rung=float(r[4]), t=float(r[0]), key=f'A{int(r[3])}'))
        return out, bool(sec)

    def pair_manual(path, cache, st, exempt=(True, True), plant=None):
        B = AF.ProfileBranch(path, END, dict(end_rel=END), False, X.PROFILE); pol = X.LineageGuard(B, cache, E); obs = P.QueueObs(B, pol); k = int(round(st['t'] / BE.STEP))
        LB.run_episode(B, pol, until=k, observe=obs); ctx = (B, pol, obs); key0 = X.fork_key(B, pol); out, keys = {}, {}
        for (name, action), ex in zip(zip(('retain', 'alternative'), X.ACTIONS[st['kind']]), exempt):
            Bx, px, ox = X.fork_copy(ctx); px.arm(dict(st, gid=st['gid'] if plant != 'cursor' else st['gid'] + 1000), action, exempt=ex); LB.run_episode(Bx, px, k0=k, observe=ox)
            if plant == 'inherit' and name == 'alternative':
                for g_, v_ in ox.ahead0.items():
                    if g_ != st['gid'] and Bx.E.o[g_]['side'] == st['side'] and Bx.E.o[g_]['rung'] == st['rung'] and Bx.E.o[g_]['submit'] >= st['t'] and not Bx.E.o[g_]['refused']:
                        v_[1] = v_[1] + 7.
            keys[name] = X.fork_key(Bx, px); out[name] = X.arm_result(Bx, px, ox, st, END)
        return X.pair_result(st, out['retain'], out['alternative']), out, key0, keys
    # ---- 1 inert until armed
    try:
        rng = np.random.default_rng(7); path = BE.rand_tape(rng, 70, 160, end=END); vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32)
        cache = _cache(nt, rng.choice(vals, nt), rng.choice(vals, nt))
        B1, p1, r1, c1 = plain(path, cache, guard=X.LineageGuard); B2, p2, r2, c2 = plain(path, cache, guard=P.PressureGuard)
        ok('1 an unarmed LineageGuard = the frozen PressureGuard', r1['fingerprint'] == r2['fingerprint'] and r1['guard_log'] == r2['guard_log'] and np.array_equal(c1, c2, equal_nan=True)
           and len(r1['guard_log']) > 0)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 inert: stopped ({e!r})'[:220])
    # ---- 2 run_market
    saved = X.saved_record
    try:
        Bp, pp, rec, cm = plain(path, cache); sts, has_sec = states_of(cm, rec['fills']); X.saved_record = lambda day, m: (rec, cm)
        pairs, arms, par, reps, lgs = X.run_market('d', 1, path, dict(end_rel=END), cache, E, sts)
        ok(f'2 states ({len(sts)}, second side {has_sec})', len(sts) >= 4 and has_sec and {s['kind'] for s in sts} == {'REVIEW', 'ADMISSION'})
        ok(f'2 base parity ({par})', par['all'] is True)
        ok(f'2 every pair ok ({[p["status"] for p in pairs]})', all(p['status'] == 'ok' for p in pairs) and len(pairs) == len(sts))
        ctx_ok = True
        for st in sts:
            _, out, _, _ = pair_manual(path, cache, st); a = out['retain']['applied'][0]
            want = pp.ctx.get(('live', st['t'], st['gid'])) if st['kind'] == 'REVIEW' else pp.ctx.get(('slot', st['t'], st['side'], st['rung']))
            ctx_ok &= bool(want is not None and a[4] == want == out['alternative']['applied'][0][4])
        ok('2 the forced callback context = the plain run\'s own callback context, in both arms', ctx_ok)
        cache0 = _cache(nt, np.zeros(nt), np.zeros(nt)); B0, p0, rec0, cm0 = plain(path, cache0); sts0, _ = states_of(cm0); X.saved_record = lambda day, m: (rec0, cm0)
        pairs0, _, par0, _, _ = X.run_market('d', 1, path, dict(end_rel=END), cache0, E, sts0)
        ok(f'2 the unused-exemption control ran and held ({[p["gate_unused_exemption"] for p in pairs0]})', len(pairs0) >= 3 and par0['all'] is True
           and all(p['gate_unused_exemption'] is True and p['status'] == 'ok' for p in pairs0) and all(p['gate_unused_exemption'] in (True, None) for p in pairs))
        ok('2 reconciled, pre-fork identical, queue reset in every pair', all(p['gate_reconciled'] and p['gate_pre_fork_identical'] and p['gate_queue_reset'] for p in pairs))
        ok(f'2 lineage replacements were checked ({len(reps)})', len(reps) >= 1 and all(r['refused'] or r['ahead0'] == r['shown'] for r in reps))
        A_ = pd.DataFrame(arms); ok('2 the Y identity: Q_retain - Q_alternative = the post-fork difference', all(abs(p['Y'] - p['Y_post']) <= 1e-6 * max(1., abs(p['Y'])) and
                                                                                                      abs(p['Y_focal'] + p['Y_other'] - p['Y_post']) <= 1e-6 * max(1., abs(p['Y_post'])) for p in pairs))
        ok('2 per-arm reconciliation of every arm', bool(A_.reconciled.all()) and len(A_) == 2 * len(sts))
        lb = [x for x in lgs if x['kind'] == 'BASE']; lp = [x for x in lgs if x['kind'] != 'BASE']; tk = {s_['key']: s_['t'] for s_ in sts}
        ok('2 the carried mark is the venue mid at the fork', all(x[a]['M_fork'] == X.AR.mid_at(path['frames'], tk[x['key']]) for x in lp for a in ('retain', 'alternative')))
        adm = [x for x in lp if x['kind'] == 'ADMISSION']
        ok('2 POST posts the focal order at the fork, WAIT does not', bool(adm) and all(any(r['delay'] == 0. for r in x['retain']['replacements'])
                                                                        and not any(r['delay'] == 0. for r in x['alternative']['replacements']) for x in adm))
        ok(f'2 the ledgers for the checker ({len(lb)}, {len(lp)})', len(lb) == 1 and lb[0]['base']['fingerprint'] == rec['fingerprint'] and len(lp) == len(sts)
           and all(abs(sum((1. if f[2] == 0 else -1.) * f[3] * (x[a]['mark'] - f[4]) for f in x[a]['post_raw']) + x[a]['I_fork'] * (x[a]['mark'] - x[a]['M_fork']) - x[a]['Q']) < 1e-6
                   for x in lp for a in ('retain', 'alternative')))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 run_market: stopped ({e!r})'[:240])
    finally:
        X.saved_record = saved
    # ---- 3 known-bad controls
    try:
        rv = [s for s in sts if s['kind'] == 'REVIEW']; st = rv[0]; k = int(round(st['t'] / BE.STEP)); hot = np.where(np.arange(nt) > k + 4, 1.0, cache['x'][st['side']]).astype(np.float32)
        cache_hot = _cache(nt, hot if st['side'] == 0 else cache['x'][0], hot if st['side'] == 1 else cache['x'][1])
        pr_ok, o_ok, _, _ = pair_manual(path, cache_hot, st); pr_bad, o_bad, _, _ = pair_manual(path, cache_hot, st, exempt=(False, True))
        ok(f'3 an exemption missing in one arm is refused ({o_ok["retain"]["suppressed"]}, {o_bad["retain"]["lineage_pressure_pulls"]})', pr_ok['gate_no_lineage_pull']
           and o_ok['retain']['suppressed'] >= 1 and not pr_bad['gate_no_lineage_pull'] and pr_bad['status'] == 'gate_failed')
        stp = next(s for s in sts if s['kind'] == 'REVIEW'); pr_i, o_i, _, _ = pair_manual(path, cache, stp, plant='inherit')
        has_rep = any(r['gid'] != stp['gid'] and not r['refused'] for r in o_i['alternative']['replacements'])
        ok(f'3 inherited replacement priority is refused (replacement {has_rep})', has_rep and not pr_i['gate_queue_reset'])
        pr_c, _, key0, keys = pair_manual(path, cache, stp, plant='cursor')
        ok('3 a reused arm state differs from the fork context', keys['retain'] != key0 and keys['alternative'] != key0)
        ok(f'3 a wrong command cursor is refused ({pr_c["status"]})', pr_c['status'] == 'callback_not_reached' and not pr_c['gate_applied_once'])
        ok('3 double-counted carried inventory is refused', X.reconciles(10., -50., 2., 30., 0.) and not X.reconciles(10., -50., 2., 30., 0. + 2. * 52.))
        ok('3 a late state carries inventory into the fork', any(abs(x[a]['I_fork']) > 0 for x in lgs if x['kind'] != 'BASE' for a in ('retain', 'alternative')))
        X.saved_record = lambda day, m: (dict(rec0, fills=rec0['fills'][:-1]), cm0)
        try:
            pb, _, parb, _, _ = X.run_market('d', 1, path, dict(end_rel=END), cache0, E, sts0)
        finally:
            X.saved_record = saved
        ok(f'3 a retain arm unequal to the saved record is refused ({[p["status"] for p in pb]})', parb['all'] is False and all(p['gate_unused_exemption'] is False and p['status'] == 'gate_failed' for p in pb))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 known-bad: stopped ({e!r})'[:240])
    # ---- 4 future-only perturbation
    try:
        st = sts[0]; T1 = st['t'] + 1500.; q = dict(path, frames=path['frames'].copy(), trades=path['trades'].copy())
        q['trades'][q['trades'][:, 0] > T1, 3] *= 2.; fm = q['frames'][:, 0] > T1; q['frames'][fm, 7:12] *= 1.5; q['frames'][fm, 17:22] *= 1.5
        _, oa, _, _ = pair_manual(path, cache, st); _, ob, _, _ = pair_manual(q, cache, st); same = True; diff_after = False
        for name in ('retain', 'alternative'):
            la, lb = oa[name]['ledger'], ob[name]['ledger']
            same &= [f for f in la['fills'] if f[0] <= T1] == [f for f in lb['fills'] if f[0] <= T1]
            diff_after |= la['fills'] != lb['fills']
        ok(f'4 a change after T1 leaves everything up to T1 unchanged (and changes something after: {diff_after})', same and diff_after)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 perturbation: stopped ({e!r})'[:200])
    # ---- 5 pure parts
    try:
        import pf_ladder_rest_selftest as V1
        f1 = np.array([V1.fr(0, 0, [(50, 100), (49, 30)], [(52, 100)]), V1.fr(100, 105, [(50, 60)], [(52, 100)])], float)
        f5 = np.array([V1.fr(0, 0, [(50, 10), (49, 10), (48, 10), (47, 10), (46, 10)], [(52, 100)])], float)
        ok(f'5 shown_at ({X.shown_at(f1, 0, 50., 10.)}, {X.shown_at(f1, 0, 50., 100.)})', X.shown_at(f1, 0, 50., 10.) == (100., 0) and X.shown_at(f1, 0, 49., 10.) == (30., 0)
           and X.shown_at(f1, 0, 50., 100.) == (60., 1) and X.shown_at(f1, 1, 53., 10.) == (0., 0) and X.shown_at(f5, 0, 45., 10.) == (0., 0) and X.shown_at(f5, 0, 46., 10.) == (10., 0))
        ok(f'5 fifo ({X.fifo([(100., 0, 10.), (300., 1, 4.), (500., 1, 10.)], 0., 0.)})', X.fifo([(100., 0, 10.), (300., 1, 4.), (500., 1, 10.)], 0., 0.) == (10., 320., -4.)
           and X.fifo([(100., 1, 3.)], 5., 0.) == (3., 100., 2.))
        ab = X.active_both([dict(gid=0, side=0, act=0., remove=1000., size=10., refused=False), dict(gid=1, side=1, act=500., remove=2000., size=10., refused=False)], [], 0., 2000.)
        ok(f'5 active_both ({ab})', abs(ab[0] - .5) < 1e-12 and abs(ab[1] - .75) < 1e-12 and abs(ab['both'] - .25) < 1e-12)
        post = [(100., 7, 0, 10., 51., 1), (200., 7, 0, 10., 47., 2), (300., 8, 0, 10., 47., 3)]
        ok(f'5 sequence_axis ({X.sequence_axis(post, {7, 8}, 7, 49.)})', X.sequence_axis(post, {7, 8}, 7, 49.) == ('early_loss_later_benefit', -20., 40.)
           and X.sequence_axis(post, {7}, 9, 49.)[0] == 'no_focal_fill' and X.sequence_axis(post[1:], {7}, 7, 49.)[0] == 'no_early_loss_no_later_benefit')
        base = dict(applied=[('live', 0., 1, 'KEEP', 'h')], pre_fills=[], reconciled=True, queue_reset_ok=True, lineage_pressure_pulls=0, M_fork=50., focal_c=0., other_c=0.)
        stx = dict(day='d', m=1, kind='REVIEW', side=0, gid=1, rung=0., t=0., key='k')
        rr = X.pair_result(stx, dict(base, Q=5., post_c=5., seq='early_loss_later_benefit'), dict(base, Q=1., post_c=1., seq='no_focal_fill'))
        rt = X.pair_result(stx, dict(base, Q=1. + 5e-7, post_c=1., seq='no_focal_fill'), dict(base, Q=1., post_c=1., seq='no_focal_fill'))
        ra = X.pair_result(stx, dict(base, Q=0., post_c=0., seq='no_early_loss_later_benefit'), dict(base, Q=3., post_c=3., seq='no_focal_fill'))
        ok(f'5 pair classes ({rr["econ"]}, {rr["mechanism"]}, {rt["econ"]}, {ra["econ"]})', rr['econ'] == 'retain_better' and rr['mechanism'] == 'retention_beneficial_with_early_loss'
           and rt['econ'] == 'tie' and ra['econ'] == 'alternative_better' and rr['queue_mechanism'] == 'unidentified' and rr['status'] == 'ok')
        def tab(ys, st_=None):
            Pr = pd.DataFrame([dict(day='d', kind='REVIEW', side=0, key=f'k{i}', status=(st_ or 'ok'), Y=y, Y_post=y, Y_focal=y, Y_other=0., econ='retain_better' if y > 1e-6 else
                                    ('alternative_better' if y < -1e-6 else 'tie'), mechanism='x', seq_retain='no_focal_fill') for i, y in enumerate(ys)])
            Ar = pd.DataFrame([dict(key=f'k{i}', arm=a, **{c: 0. for c in ('Q', 'edge', 'movement', 'buy_q', 'sell_q', 'matched', 'hold_ms', 'tw_abs_I', 'peak_abs_I', 'terminal_I',
                                                                        'both_sides', 'requests', 'reposts', 'reversals', 'pressure_pulls', 'suppressed', 'first_rep_delay', 'common_n')},
                                    **{f'hz{h}_{x}': 0. for h in ('1000', '5000', '30000', '60000') for x in ('native', 'common')}) for i in range(len(ys)) for a in ('retain', 'alternative')])
            return X.summarize(Pr, Ar)[1]
        ok(f'5 the reading ({tab([600., -600., 5.])}, {tab([1., 1., 1.])})', tab([600., -600., 5.])['reading'] == 'reliable_and_varied' and tab([1., 1., 1.])['reading'] == 'reliable_not_varied'
           and tab([600., -600.], st_='gate_failed')['reading'] == 'not_reliable')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 parts: stopped ({e!r})'[:240])
    # ---- 6 guards
    sv = dict(frozen=X.frozen, sources=X.sources, pilot_records=X.pilot_records); tmp = Path(tempfile.mkdtemp())
    try:
        mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(sources=X.sources(), parts=dict(subset=['d'], rest=['e']), pilot_markets={}, selected=[], pilot_records={})))
        X.frozen = lambda: False; ok('6 run_part refuses before the freeze', refuses('committed', X.run_part, 'subset', 'd', out=tmp / 'o', man_path=mp))
        ok('6 measure refuses before the freeze', refuses('committed', X.measure, out=tmp / 'o', man_path=mp))
        X.frozen = lambda: True; X.sources = lambda: dict(sv['sources'](), **{'pf_ladder_long_continuation.py': '0' * 64})
        ok('6 run_part refuses another source', refuses('source', X.run_part, 'subset', 'd', out=tmp / 'o', man_path=mp)); X.sources = sv['sources']
        X.pilot_records = lambda gm_af: {}
        ok('6 a rest without the projection is refused', refuses('projection', X.run_part, 'rest', 'e', out=tmp / 'o', man_path=mp, proj_path=tmp / 'absent.json'))
        ok('6 the manifest and the projection are write-once', refuses('write-once', X.build_manifest, mp, tmp / 'c.csv') and refuses('write-once', X.project, mp))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 guards: stopped ({e!r})'[:200])
    finally:
        X.frozen, X.sources, X.pilot_records = sv['frozen'], sv['sources'], sv['pilot_records']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('LONG CONTINUATION SELFTEST PASSED: inert guard, run_market gates, known-bad controls, future-only perturbation, pure parts, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_long_continuation as X_
    main(X_)
