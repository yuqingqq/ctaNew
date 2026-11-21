"""SELFTEST of pf_ladder_pressure_residual (PRESSURE-RESIDUAL-TIMING-AUDIT-v1; 2026-11-21, session aea5bfb9). main(X) checks the module; a failed check
raises SystemExit with FAIL. Each group names only what its code runs (review e40bd01d's synthetic cases).
 1 classify_fill on hand orders and commands: the three removal reasons pending (structural / pressure / inventory); a request at the fill's
   own timestamp is not pending; the fill / effect boundary (a fill 1 ms before the removal is pending, a fill at the removal a mismatch); a
   fill at the first live callback and 36 ms after it are before_feasible_effect, 37 ms after it (signal OFF) unflagged_later; a never-live
   order's fill before_feasible_effect; pressure ON only after the last feasible tick unflagged_later; ON at a feasible evaluable tick a
   mismatch; non-ok input at every feasible tick or fallback callbacks nonevaluable; a non-ok tick beside an evaluable OFF one unflagged_later;
 2 market_rows on a hand ledger: a no-fill generation, a refused one and repeated partial fills (each fill once, its generation kept), the
   values (edge, +50 / +100, eligible +1 s on the END - 1 s cutoff, movement, late END; END identity);
 3 reconcile: the hand ledger reconciles; a duplicated fill and a missing fill each refuse; a sum differing from the arm row refuses;
 4 the table and the reading: status x day x side x rung sums, the generation statuses, the concentration; the fractions of the gross negative
   +1 s; TIMING_LIMITED_DOMINANT / UNFLAGGED_EXPOSURE_DOMINANT / MIXED / THIN / AUDIT_FAILED;
 5 guards: audit refuses before the freeze, under another source, an unknown scope and an existing output (each message asserted)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'RESIDUAL SELFTEST FAILED'


def run(mod):
    main(mod)


def main(X):  # noqa: C901
    P, M, pd = X.P, X.M, X.pd; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    E = P.frozen_edges(); nt = 100

    def cache(on_ticks=(), side=0, nonok=()):
        x = {0: np.zeros(nt, np.float32), 1: np.zeros(nt, np.float32)}; st = {0: np.zeros(nt, np.int8), 1: np.zeros(nt, np.int8)}
        for k in on_ticks:
            x[side][k] = 1.
        for k in nonok:
            st[side][k] = 1
        return dict(x=x, age={0: np.full(nt, 7., np.float32), 1: np.full(nt, 7., np.float32)}, status=st)
    order = lambda **k: dict(dict(gid=0, side=0, rung=0., price=49., size=100., submit=0., act=58., remove=4000., req=None, why=None, refused=False), **k)
    live = lambda ts, reason='keep', action='KEEP': [(float(t), 'live', action, reason) for t in ts]
    # ---- 1 classify_fill
    try:
        c0 = cache(); L100 = live(range(100, 3000, 50))
        st = lambda t_f, o, cmds=L100, cz=c0: X.classify_fill(float(t_f), o, cmds, cz, E[0])[0]
        o_s = order(req=1000., why='structural', remove=1037.); o_p = order(req=1000., why='policy', remove=1037.); o_i = order(req=1000., why='policy', remove=1037.)
        cmd_p = [c for c in L100 if c[0] != 1000.] + [(1000., 'live', 'PULL', 'pressure_pull')]; cmd_i = [c for c in L100 if c[0] != 1000.] + [(1000., 'live', 'PULL', 'inventory_pull')]
        cmd_p.sort(key=lambda c: c[0]); cmd_i.sort(key=lambda c: c[0])
        ok('1 the three removal reasons pending', st(1010, o_s) == 'pending_structural' and st(1010, o_p, cmd_p) == 'pending_pressure' and st(1010, o_i, cmd_i) == 'pending_inventory')
        ok('1 a request at the fill\'s own timestamp is not pending', st(1000, o_s) == 'unflagged_later')
        ok('1 the fill / effect boundary', st(1036, o_s) == 'pending_structural' and st(1037, o_s) == 'event_or_action_mismatch')
        ok(f'1 first-live timing ({st(100, order())}, {st(136, order())}, {st(137, order())})', st(100, order()) == 'before_feasible_effect' and st(136, order()) == 'before_feasible_effect'
           and st(137, order()) == 'unflagged_later')
        ok('1 a never-live order\'s fill is before_feasible_effect', st(2000, order(), []) == 'before_feasible_effect')
        ok('1 pressure ON only after the last feasible tick is unflagged_later', st(1036, order(), L100, cache(on_ticks=(20,))) == 'unflagged_later')
        r_on = X.classify_fill(1037., order(), L100, cache(on_ticks=(20,)), E[0])
        ok(f'1 ON at a feasible evaluable tick is a mismatch ({r_on[1]["why"]})', r_on[0] == 'event_or_action_mismatch' and r_on[1]['n_on'] == 1 and r_on[1]['k_last'] == 1000.)
        ok('1 an ON on the other side is not this order\'s signal', st(1037, order(), L100, cache(on_ticks=(20,), side=1)) == 'unflagged_later')
        ok('1 non-ok input at every feasible tick, or fallback callbacks, is nonevaluable', st(237, order(), L100, cache(nonok=(2, 3, 4))) == 'nonevaluable'
           and st(237, order(), live(range(100, 3000, 50), 'fallback')) == 'nonevaluable' and st(237, order(), L100, cache(nonok=(2,))) == 'unflagged_later')
        d_ = X.classify_fill(437., order(), L100, cache(on_ticks=(9,)), E[0])[1]
        ok(f'1 the diagnostics ({d_["n_feasible"]}, {d_["k_last"]}, {d_["act_last"]})', d_['n_feasible'] == 7 and d_['k_last'] == 400. and d_['act_last'] == 'KEEP/keep' and d_['T1'] == 100.
           and d_['sig_T1'][2] == 7. and d_['sig_submit'][1] == 'ok')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 classify: stopped ({e!r})'[:220])
    # ---- 2 market_rows
    try:
        mm = lambda m0, m1: [m0, 0., m0, 0., m0, 0., m1, 0., None, None, None, None]
        orders = [order(gid=0, remove=1037., req=1000., why='structural'), order(gid=1, side=1, price=52., rung=1.), order(gid=2, refused=True, act=0., remove=0.), order(gid=3, price=48.)]
        L_ = dict(end=4000., mark=50., value=None, orders=orders, fills=[[500., 0, 0, 30., 49., 1], [1010., 0, 0, 20., 49., 2], [3500., 1, 1, 10., 52., 3]],
                  fill_mids=[mm(50., 49.5), mm(50., 48.), mm(51., 50.)])
        cash = -(30 * 49 + 20 * 49) + 10 * 52; pos = 50 - 10; L_['value'] = cash + pos * 50.
        cm = np.array([[float(t), 0, M.CMD_KINDS.index('live'), 0, 0., 49., np.nan, M.CMD_ACTIONS.index('KEEP'), P.REASONS.index('keep')] for t in range(100, 1000, 50)]
                      + [[float(t), 1, M.CMD_KINDS.index('live'), 1, 1., 52., np.nan, M.CMD_ACTIONS.index('KEEP'), P.REASONS.index('keep')] for t in range(100, 3500, 50)], float)
        F_, G_ = X.market_rows('20261024', 7, L_, cm, cache(), E); F = pd.DataFrame(F_); G = pd.DataFrame(G_)
        ok(f'2 statuses and generations ({list(F.status)}, {list(G.gen_status)})', list(F.status) == ['unflagged_later', 'pending_structural', 'unflagged_later']
           and list(G.gen_status) == ['filled', 'filled', 'refused', 'no_fill'] and list(G.fills) == [2, 1, 0, 0])
        ok('2 repeated partial fills each once with their generation', list(F.fill) == [0, 1, 2] and list(F.gid) == [0, 0, 1])
        r2 = F.iloc[2]
        ok(f'2 values on the END - 1 s cutoff ({r2.eligible}, {r2.late_end})', F.iloc[0].mo1s == 30 * (49.5 - 49.) and F.iloc[0].move_end == 30 * (50 - 49.5) and not r2.eligible
           and r2.late_end == -10 * (50 - 52.) and abs(F.end.sum() - L_['value']) < 1e-9 and F.iloc[0].edge == 30 * (50 - 49.))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 market_rows: stopped ({e!r})'[:220])
    # ---- 3 reconcile
    try:
        arm = {('20261024', 7): dict(fills=3, all_q=60., all_end=float(F.end.sum()), all_id_mo1s=float(F.mo1s.fillna(0.).sum()), all_edge=float(F.edge.fillna(0.).sum()))}
        rc = X.reconcile(F, G, arm, {('20261024', 7): L_}); ok(f'3 the hand ledger reconciles ({rc})', rc['fills'] == 3)

        def refuses(word, F2, arm2=arm):
            try:
                X.reconcile(F2, G, arm2, {('20261024', 7): L_}); return False
            except ValueError as e:
                return word in str(e)
        ok('3 a duplicated fill refuses', refuses('exactly once', pd.concat([F, F.iloc[[0]]], ignore_index=True)))
        ok('3 a missing fill refuses', refuses('exactly once', F.iloc[1:]))
        ok('3 a sum differing from the arm row refuses', refuses('reconcile', F, {k: dict(v, all_end=v['all_end'] + 1.) for k, v in arm.items()}))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 reconcile: stopped ({e!r})'[:220])
    # ---- 4 the table and the reading
    try:
        T, gs, conc = X.table(F, G)
        ok(f'4 the table ({len(T)}, {conc})', set(T.status) == {'unflagged_later', 'pending_structural'} and int(T.fills.sum()) == 3 and abs(T.end.sum() - L_['value']) < 1e-9
           and int(gs.generations.sum()) == 4 and conc['unflagged_later']['markets'] == 0)
        rows = []
        for day in ('20261024', '20261025'):
            for i in range(40):
                rows.append(dict(day=day, m=1, status='pending_pressure' if i < 25 else 'unflagged_later', eligible=True, mo1s=-1., q=1.))
        R1 = pd.DataFrame(rows); r1 = X.reading(R1)
        ok(f'4 the timing-limited fraction ({r1["by_day"]["20261024"]["frac_timing"]}, {r1["answer"]})', r1['answer'] == 'TIMING_LIMITED_DOMINANT' and r1['by_day']['all']['frac_timing'] == 25 / 40)
        R2 = R1.assign(status=np.where(R1.status == 'pending_pressure', 'unflagged_later', 'pending_pressure')); ok('4 unflagged dominant', X.reading(R2)['answer'] == 'UNFLAGGED_EXPOSURE_DOMINANT')
        R3 = R1.copy(); R3.loc[(R3.day == '20261025'), 'status'] = R2.loc[(R2.day == '20261025'), 'status'].values; ok('4 mixed days', X.reading(R3)['answer'] == 'MIXED')
        ok('4 a thin day', X.reading(R1[(R1.day == '20261024') | (R1.index % 2 == 0)])['answer'] == 'THIN')
        R5 = R1.copy(); R5.loc[0, 'status'] = 'event_or_action_mismatch'; ok('4 a mismatch fails the audit', X.reading(R5)['answer'] == 'AUDIT_FAILED' and X.reading(R5)['mismatches'] == 1)
        R6 = R1.copy(); R6.loc[R6.index % 3 == 0, 'mo1s'] = 2.; r6 = X.reading(R6)
        ok('4 positive values sit beside, never in, the negative fractions', r6['by_day']['all']['total']['pos'] > 0 and abs(r6['by_day']['all']['frac_timing'] + r6['by_day']['all']['frac_unflagged'] - 1.) < 1e-12)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 table / reading: stopped ({e!r})'[:220])
    # ---- 5 guards
    saved = dict(frozen=X.frozen, sources=X.sources); tmp = Path(tempfile.mkdtemp(dir=str(X.A.DATA)))
    try:
        mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(sources=X.sources())))

        def refuses(word, **kw):
            try:
                X.audit(man_path=mp, out=tmp / 'out', **kw); return False
            except ValueError as e:
                return word in str(e)
        X.frozen = lambda: False; ok('5 audit refuses before the freeze', refuses('committed', scope='pilot'))
        X.frozen = lambda: True; ok('5 audit refuses an unknown scope', refuses('scope', scope='all'))
        X.sources = lambda: dict(saved['sources'](), **{'pf_stop_model.py': '0' * 64}); ok('5 audit refuses another source', refuses('source', scope='pilot')); X.sources = saved['sources']
        (tmp / 'out' / 'pilot').mkdir(parents=True); (tmp / 'out' / 'pilot' / 'audit.json').write_text('{}'); ok('5 audit refuses an existing output', refuses('write-once', scope='pilot'))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 guards: stopped ({e!r})'[:200])
    finally:
        X.frozen, X.sources = saved['frozen'], saved['sources']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('RESIDUAL SELFTEST PASSED: classify_fill, market_rows, reconcile, the table and the reading, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_pressure_residual as X_
    main(X_)
