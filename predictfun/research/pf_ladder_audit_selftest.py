"""SELFTEST of pf_ladder_audit (RECEIPT-LEAD-TIME-AUDIT-v1) and pf_branch_ladder_safe (2026-11-21, session aea5bfb9). main(M) checks the audit
module M; any failed check raises SystemExit with FAIL. Hand fixtures:
 1 the engine-consistent boundaries: eff / boundaries on hand orders; the review's control on the ladder client (a request at 50 on a
   post-pending order lands at 95 and prevents a 100 ms fill, which fills without the signal) equals boundaries()['pending_first'];
 2 sweeps: one public trade filling two own generations is an event; two rows at the same time are only a timestamp group;
 3 stage windows: an entry takes every fill; a live state at d takes the fills at t >= d + 37, the fills in [d, d + 37) are pre-effect; a
   window fill without its +1 s mid censors the +1 s label (None, never 0); lifetime fields apart; cancel-pending window fills;
 4 recovery classification (recovered / reversal / censored) and early requests by reason (churn);
 5 exact durations: the union and intersection of venue-active intervals, a full fill ending its interval;
 6 bins and the lead rule: a consistent q5 - q1 sign in all eight cells is a LEAD, one flipped cell or a thin cell is not, censored rows
   are excluded; the closed round trip (bid 50 / ask 51, mark 80): END legs +3,000 / -2,900 for a +100 joint value;
 7 the order-input causal check: future frames / unnotified fills perturbed leave it unchanged; the frame received at d changed fires it;
 8 pf_branch_ladder_safe: a saved / resumed run equals the continuous one; next_k, the envelope ident, a runtime cap inside the pickle and a
   changed external path / mark with a matching envelope tape id are each refused, while the frozen resume accepts the last two;
 9 guards: run_saved and causal_day refuse before the freeze and refuse an existing output."""
import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'AUDIT SELFTEST FAILED'


def run(mod):
    import importlib
    main(mod if hasattr(mod, 'run_saved') else importlib.import_module('pf_ladder_audit'))


def main(A):  # noqa: C901
    M, LB, BE, pd = A.M, A.LB, A.BE, A.pd; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    import pf_ladder_rest_selftest as V1
    fr = V1.fr
    # ---- 1 boundaries
    try:
        o0 = dict(submit=0., act=58., remove=1e30, req=None); o1 = dict(submit=50., act=108., remove=1e30, req=None)
        ok('1 eff before / after activation', A.eff(50., o0) == 95. and A.eff(200., o0) == 237. and A.eff(100., o1) == 145.)
        ok('1 boundaries', A.boundaries(o0) == dict(pending_first=95., live_first=137., actual=None) and A.boundaries(o1)['pending_first'] == 145.
           and A.boundaries(o1)['live_first'] == 187.)
        res = []
        for sig in (True, False):
            frames = [fr(0, 0, [(50, 0)], [(52, 0)])] + ([fr(40, 40, [(55, 0)], [(57, 0)])] if sig else [])
            p = BE.tape(frames, [(100, 0, 50, 10)], 50.); B = LB.LadderBranch(p, LB.LADDER, end=1000.)

            class OneBid(LB.Policy):
                def live(self, v, o):
                    return 'KEEP', 'keep'

                def slot(self, v, r, price):
                    return ('POST', 100., 'post') if (v['t'] == 0. and v['side'] == 0 and r == 0) else ('WAIT', None, 'test')
            LB.run_episode(B, OneBid('one', B.end)); o = B.E.o[0]; res.append((sum(f[3] for f in B.E.fills), o['req'], o['remove']))
        b = A.boundaries(dict(submit=0., act=58., remove=res[0][2], req=res[0][1]))
        ok(f'1 the review control: a request at 50 lands at 95 and prevents the 100 ms fill ({res})', res[0] == (0., 50., 95.) and res[1][0] == 10.
           and b['pending_first'] == 95. and b['actual'] == 95.)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 boundaries: stopped ({e!r})'[:200])
    # ---- 2 sweeps, 3 windows, 4 churn / recovery, 5 durations (one hand market)
    try:
        mm = lambda m0, m1: [m0, 0., m0, 0., m0, 0., m1, 0., None, None, None, None]
        orders = [dict(gid=0, side=0, rung=0., price=50., size=20., act=58., remove=1e30, submit=0., refused=False, req=None, why=None),
                  dict(gid=1, side=0, rung=1., price=49., size=10., act=58., remove=250., submit=0., refused=False, req=50., why='structural'),
                  dict(gid=2, side=1, rung=0., price=52., size=5., act=558., remove=737., submit=500., refused=False, req=700., why='policy')]
        fills = [[100., 0, 0, 5., 50., 7], [100., 1, 0, 5., 49., 7], [150., 0, 0, 2., 50., 8], [150., 1, 0, 1., 49., 9], [236., 0, 0, 3., 50., 10],
                 [237., 0, 0, 4., 50., 11], [600., 2, 1, 5., 52., 12]]
        fmid = [mm(51., 48.), mm(51., 48.), mm(51., 48.), mm(51., 48.), mm(51., 48.), mm(51., None), mm(51., 53.)]
        L_ = dict(mark=47., end=2000., value=0., fills=fills, fill_mids=fmid, orders=orders, closed={'1': ['removed', 350.]}, requests=[], struct_log=[], mode_log=[
            (100., 100., 'UNRESTRICTED', 'LONG_RESTRICTED'), (600., 40., 'LONG_RESTRICTED', 'UNRESTRICTED'), (700., -100., 'UNRESTRICTED', 'SHORT_RESTRICTED'),
            (900., 100., 'SHORT_RESTRICTED', 'LONG_RESTRICTED')], ahead0={}, queue={})
        r = dict(day='20261024', m=5, status='complete', ws=0., ledger=L_, path_info=dict(t_start=0., end_rel=2000.))
        sw = A.sweep_rows(r); te = [x for x in sw if x['trade_event']]; tg = [x for x in sw if x['time_group']]
        ok(f'2 sweep events by public trade (2 rows) vs timestamp groups (4 rows): {len(te)}, {len(tg)}', len(te) == 2 and len(tg) == 4 and all(x['trade'] == 7 for x in te))
        G = A.gen_v2(r); g0 = [x for x in G if x['gid'] == 0][0]; g1 = [x for x in G if x['gid'] == 1][0]
        ok('2 sweep membership per generation', g0['n_sweep_event_fills'] == 1 and g0['ever_swept'] and g1['n_sweep_event_fills'] == 1)
        ok(f'1 gap fills in [95, 137): gid 0 {g0["n_gap"]}', g0['n_gap'] == 1 and abs(g0['q_gap'] - 5.) < 1e-12 and g0['q_before_pending_first'] == 0.
           and g0['q_from_live_first'] == 9.)
        ok('1 cancel-pending fills of gid 1 (req 50 <= t < 250)', g1['n_cancel_pending'] == 2 and g1['n_before_request'] == 0)
        arrs = {'20261024|5__commands': np.array([[100., 0, 0, 0, 0., 50., np.nan, 2, 1], [150., 0, 0, 0, 0., 50., np.nan, 2, 1], [600., 1, 0, 2, 0., 52., np.nan, 3, 2]], float),
                '20261024|5__mid': 50. + .1 * np.arange(40), '20261024|5__nva': np.zeros((40, 2), np.int8)}
        F = pd.DataFrame([dict(day='20261024', m=5, kind='first', side=0, gid=0, t=200., rung=0., status='ok', n_eligible=3, ctx_inventory_shares=10.),
                          dict(day='20261024', m=5, kind='entry', side=0, gid=0, t=0., rung=0., status='ok', n_eligible=-1, ctx_inventory_shares=0.),
                          dict(day='20261024', m=5, kind='live', side=1, gid=2, t=550., rung=0., status='ok', n_eligible=2, ctx_inventory_shares=0.)])
        S = A.state_rows(F, [r], arrs); fs = S[S.kind == 'first'].iloc[0]; es = S[S.kind == 'entry'].iloc[0]; ls = S[S.kind == 'live'].iloc[0]
        ok(f'3 the live window starts at d + 37 = 237 (q {fs.win_q}), [200, 237) pre-effect (q {fs.pre_q}), censored', fs.effect == 237. and fs.win_n == 1 and fs.win_q == 4.
           and fs.pre_n == 1 and fs.pre_q == 3. and bool(fs.censored_1s) and (fs.win_mo1s is None or (isinstance(fs.win_mo1s, float) and np.isnan(fs.win_mo1s))))
        ok('3 the entry window is every fill of its generation from its activation; lifetime apart', es.win_n == 4 and es.win_q == 14. and bool(es.lifetime_any)
           and es.lifetime_q == 14. and es.effect == 58. and es.pre_n == 0)
        ok('3 a pending (pre-activation) state uses submit + 95', A.eff(550., orders[2]) == 595. and ls.effect == 595. and ls.win_n == 1)
        ok(f'3 the inventory move to d + 1 s: I 10 x (mid(1,200) - mid(200)) = 20 ({fs.inv_move_1s})', abs(fs.inv_move_1s - 20.) < 1e-9)
        rc = A.recovery(r)
        ok(f'4 recovery / reversal / censored ({[x["outcome"] for x in rc]})', [x['outcome'] for x in rc] == ['recovered', 'reversal', 'censored'] and rc[0]['ms'] == 500.)
        ch = A.churn(r, arrs)
        ok(f'4 early requests by reason ({ch})', ch['by_reason'] == {'structural': 1} and ch['requested_by_first_live'] == 1 and ch['reviewed'] == 2 and ch['policy_first_review_pulls'] == 1)
        du = A.durations(r, arrs)
        ok(f'5 exact durations: bid [58, 2000), ask [558, 600) closed by its full fill ({du["both_ms"]}, {du["either_ms"]})', du['both_ms'] == 42. and abs(du['either_ms'] - 1942.) < 1e-9
           and abs(du['both_exact'] - 42. / 2000.) < 1e-12)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2-5 hand market: stopped ({e!r})'[:220])
    # ---- 6 bins, leads, the closed round trip
    try:
        rows = []
        for day in M.DAYS:
            for s in (0, 1):
                for i in range(12):
                    rows.append(dict(day=day, m=1, kind='first', side=s, gid=i, t=0., status='ok', win_n=1, win_q=1., win_any=True, win_end=0., win_mo1s=-5., censored_1s=False,
                                     **{c: 0.1 for c in M.DICT_COLS}))
                    rows.append(dict(day=day, m=1, kind='first', side=s, gid=100 + i, t=0., status='ok', win_n=1, win_q=1., win_any=True, win_end=0., win_mo1s=5., censored_1s=False,
                                     **{c: 0.9 for c in M.DICT_COLS}))
        S_ = pd.DataFrame(rows); edges = {f'first|{s}|{c}': [0.2, 0.4, 0.6, 0.8] for s in (0, 1) for c in M.DICT_COLS}
        BT = A.bin_table(S_, edges); LT, cand = A.leads(BT, S_, edges)
        ok(f'6 a consistent q5 - q1 is a LEAD ({cand})', cand['answer'] == 'LEAD' and cand['kind'] == 'first' and cand['feature'] == M.DICT_COLS[0] and cand['sign'] == 1
           and abs(cand['pooled_q5_minus_q1'] - 10.) < 1e-12)
        S2 = S_.copy(); S2.loc[(S2.day == '20261027') & (S2.side == 1) & (S2.gid >= 100), 'win_mo1s'] = -9.
        ok('6 one flipped cell is no LEAD', A.leads(A.bin_table(S2, edges), S2, edges)[1]['answer'] == 'NO_LEAD')
        S3 = S_[~((S_.day == '20261025') & (S_.side == 0) & (S_.gid < 3))].copy()
        ok('6 a thin cell (< MIN_CELL) is no LEAD', A.leads(A.bin_table(S3, edges), S3, edges)[1]['answer'] == 'NO_LEAD')
        S4 = S_.copy(); S4.loc[S4.gid == 100, 'censored_1s'] = True; S4.loc[S4.gid == 100, 'win_mo1s'] = np.nan; B4 = A.bin_table(S4, edges)
        ok('6 censored rows are excluded from the +1 s means', int(B4.censored.sum()) == 8 * len(M.DICT_COLS) and (B4.uncensored + B4.censored == B4.n).all())
        rt = [(0., 0, 0, 100., 50., [None] * 12), (1., 1, 1, 100., 51., [None] * 12)]; q, v, _ = A.vals(rt, 80.); vb = A.vals(rt[:1], 80.)[1]; va = A.vals(rt[1:], 80.)[1]
        ok('6 the closed round trip: END legs +3,000 / -2,900, joint +100', (vb, va, v) == (3000., -2900., 100.))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 bins / leads: stopped ({e!r})'[:200])
    # ---- 7 the order-input causal check
    try:
        w = dict(kf=np.array([0., 100., 200., 300.]), P5=np.array([[50., 49., -1., -1., -1.]] * 4), S5=np.array([[40., 30., 0., 0., 0.]] * 4))
        same, shown, fired = A.order_causal(w, 0, 0., 50., np.array([10., 150.]), np.array([3., 5.]), 120., 5000., 30., 5000., 0., np.random.default_rng(1))
        ok(f'7 future-only perturbation unchanged, the frame at d fires ({same}, {shown}, {fired})', same and shown and fired)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'7 order causal: stopped ({e!r})'[:200])
    # ---- 8 safe resume
    try:
        import pf_branch_ladder_safe as SF
        p = BE.rand_tape(np.random.default_rng(5), 300, 500, end=30000.); B = LB.LadderBranch(p, LB.LADDER, end=30000.); LB.run_episode(B, LB.LadderInventory(B)); Lc = LB.finish(B)
        Bh = LB.LadderBranch(p, LB.LADDER, end=30000.); LB.run_episode(Bh, LB.LadderInventory(Bh), until=300); sv = SF.save_state_v2(Bh, 300)
        Br, k = SF.resume_v2(sv, LB.LADDER, p, 30000.); LB.run_episode(Br, LB.LadderInventory(Br), k0=k); Lr = LB.finish(Br)
        ok('8 a resume_v2 run equals the continuous one', LB.fingerprint(Br) == LB.fingerprint(B) and Lr['fills'] == Lc['fills'] and Lr['value'] == Lc['value'])
        p2 = dict(p, mark=float(p['mark']) + 1.)
        cases = dict(next_k=dict(sv, next_k=310), ident=dict(sv, ident=dict(sv['ident'], extra=1)),
                     runtime_cap=(lambda x: dict(sv, blob=SF.pickle.dumps(x)))(_set(SF.pickle.loads(sv['blob']), 'cap', 999.)),
                     tape_envelope=dict(sv, tape_id=BE.tape_id(p2['frames'], p2['trades'], p2['mark']), mark=p2['mark']))
        refused = {}
        for nm, svx in cases.items():
            try:
                SF.resume_v2(svx, LB.LADDER, p2 if nm == 'tape_envelope' else p, 30000.); refused[nm] = False
            except ValueError:
                refused[nm] = True
        ok(f'8 resume_v2 refuses the four review falsifiers ({refused})', all(refused.values()))
        try:
            SF.resume_v2(sv, LB.LADDER, p2, 30000.); bad.append('8 resume_v2 accepted another path under an untouched envelope')
        except ValueError:
            pass
        old = LB.save_state(Bh, 300); acc = {}
        for nm, x in (('runtime_cap', dict(old, blob=SF.pickle.dumps(_set(SF.pickle.loads(old['blob']), 'cap', 999.)))),
                      ('tape_envelope', dict(old, tape_id=BE.tape_id(p2['frames'], p2['trades'], p2['mark'])))):
            try:
                LB.resume(x, LB.LADDER, p2 if nm == 'tape_envelope' else p, 30000.); acc[nm] = True
            except ValueError:
                acc[nm] = False
        ok(f'8 the frozen resume accepts them (the defect the falsifiers detect: {acc})', all(acc.values()))
        try:
            SF.save_state_v2(Bh, 301); bad.append('8 a save off the tick boundary was accepted')
        except ValueError:
            pass
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'8 safe resume: stopped ({e!r})'[:200])
    # ---- 9 guards
    try:
        fz = A.frozen; A.frozen = lambda: False
        for fn in (lambda: A.run_saved(out=Path(tempfile.mkdtemp(dir=str(M.KP.I.K.IO)))), lambda: A.causal_day('20261024', out=Path(tempfile.mkdtemp(dir=str(M.KP.I.K.IO))))):
            try:
                fn(); bad.append('9 a stage ran before the freeze')
            except ValueError as e:
                ok(f'9 a stage refuses before the freeze ({e})', 'committed' in str(e))
        A.frozen = lambda: True
        with tempfile.TemporaryDirectory(dir=str(M.KP.I.K.IO)) as tmp:
            (Path(tmp) / 'audit.json').write_text('{}'); (Path(tmp) / 'causal_20261024.json').write_text('{}')
            for fn, nm in ((lambda: A.run_saved(out=Path(tmp)), 'saved'), (lambda: A.causal_day('20261024', out=Path(tmp)), 'causal')):
                try:
                    fn(); bad.append(f'9 {nm} overwrote its output')
                except ValueError as e:
                    ok(f'9 {nm} refuses an existing output ({e})', 'write-once' in str(e))
        A.frozen = fz
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'9 guards: stopped ({e!r})'[:200])
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('AUDIT SELFTEST PASSED: boundaries and the review control, sweeps, windows, churn / recovery, durations, bins / leads, the round trip, '
          'order causality, safe resume, guards')


def _set(o, k, v):
    setattr(o, k, v); return o


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_audit as A_
    main(A_)
