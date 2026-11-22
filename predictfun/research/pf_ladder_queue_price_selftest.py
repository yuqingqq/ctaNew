"""SELFTEST of pf_ladder_queue_price (QUEUE-PRICE-STATE-AUDIT-v1; 2026-11-22, session aea5bfb9). main(X) checks the module on synthetic rows only (no data
is read); a failed check raises SystemExit with FAIL. Groups:
 1 labels: price_state at -1 / -0.0 / +-1e-9 / 0 / 2 / NaN; display at 0 / 5 / -1 / NaN; the queue bands (zero is front, missing / negative unknown);
 2 join: the exact key joins; a missing row, a 1 ms key shift, a duplicated key and a disagreeing fp_queue each refuse (with the reason); NaN = NaN;
 3 state_table: the principal flag, the four view labels ('outside' kept), the exact-depth label; a saved band that differs refuses;
 4 union and cells on hand rows: all-state counts beside the cohort's valid / valid at-risk / valid filled / valid shares; the valid any-fill rate
   (differs from the all-state rate); net / per share / per opportunity; censored_from_common; the time-to-END median and < 30 s count; the union
   status from the end cause; the sparse flag on the cohort's valued filled states (19 valid of 20 filled stays sparse);
 5 guards: run refuses before the freeze, under another source and over an existing output; build_manifest refuses an existing manifest."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'QUEUE PRICE SELFTEST FAILED'
HS = (0, 50, 100, 1000, 5000, 30000)


def run(mod):
    main(mod)


def iv_row(pd, g, iv, follow, cause='removal', fills=0, shares=0., v=None, ok=None, day='d1'):
    v = v or {}; ok = ok or {}
    r = dict(day=day, m=1, gid=g, interval=iv, cause=cause, follow_up=follow, at_risk_ms=0., fills=fills, shares=shares, filled_out=False, vEND=0.)
    for h in HS:
        r[f'ok{h}'] = ok.get(h, True); r[f'v{h}'] = (v.get(h, 0.) if r[f'ok{h}'] else np.nan); r[f'fpos{h}'] = max(v.get(h, 0.), 0.); r[f'fneg{h}'] = min(v.get(h, 0.), 0.)
    r['common'] = all(r[f'ok{h}'] for h in HS); return r


def main(X):  # noqa: C901
    import pandas as pd
    bad = []
    ok = lambda name, cond: None if cond else bad.append(name)

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw); return False
        except ValueError as e2:
            return word in str(e2)
    # ---- 1 labels
    try:
        ps = [X.price_state(x) for x in (-1., -0., 1e-9, -1e-9, 0., 2., np.nan, None)]
        ok(f'1 price states ({ps})', ps == ['inside_best', 'at_touch', 'behind_best', 'inside_best', 'at_touch', 'behind_best', 'unknown', 'unknown'])
        ds = [X.display(x) for x in (0., 5., -1., np.nan, None, 1e-9)]
        ok(f'1 display ({ds})', ds == ['shown_zero', 'shown_positive', 'shown_invalid', 'shown_unknown', 'shown_unknown', 'shown_positive'])
        qb = [X.queue_band(q, r) for q, r in ((0., 10.), (np.nan, 10.), (-1., 10.), (10., 10.), (10.5, 10.), (5., 0.), (5., np.nan))]
        ok(f'1 queue bands ({qb})', qb == ['front', 'unknown', 'unknown', 'within_R', 'beyond_R', 'invalid_R', 'invalid_R'])
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'1 labels: stopped ({e_!r})'[:200])
    # ---- 2 join
    try:
        S = pd.DataFrame([dict(day='d1', m=1, gid=1, t=1000., side=0, Q=0., R=10., o_depth=0., queue_band='front', bn_band='non_adverse'),
                          dict(day='d1', m=1, gid=2, t=1500., side=1, Q=np.nan, R=10., o_depth=2., queue_band='unknown', bn_band='q5')])
        F = pd.DataFrame([dict(day='d1', m=1, gid=1, t=1000., side=0, fp_queue=0., fp_remaining=10., fp_shown=7., o_depth=0., t_to_end=90000., fp_size=10.),
                          dict(day='d1', m=1, gid=2, t=1500., side=1, fp_queue=np.nan, fp_remaining=10., fp_shown=0., o_depth=2., t_to_end=20000., fp_size=10.),
                          dict(day='d1', m=1, gid=3, t=2000., side=0, fp_queue=1., fp_remaining=5., fp_shown=1., o_depth=0., t_to_end=1., fp_size=5.)])
        J = X.join(S, F); ok(f'2 the exact key joins ({len(J)})', len(J) == 2 and list(J.fp_shown) == [7., 0.] and list(J.t_to_end) == [90000., 20000.])
        ok('2 a missing feature row refuses', refuses('without its feature row', X.join, S, F.iloc[1:]))
        ok('2 a 1 ms key shift refuses', refuses('without its feature row', X.join, S, F.assign(t=F.t + np.where(F.gid == 1, 1., 0.))))
        ok('2 a duplicated feature key refuses', refuses('duplicated', X.join, S, pd.concat([F, F.iloc[[0]]], ignore_index=True)))
        ok('2 a disagreeing fp_queue refuses', refuses('disagrees', X.join, S, F.assign(fp_queue=np.where(F.gid == 1, 1., F.fp_queue))))
        ok('2 a disagreeing depth refuses', refuses('disagrees', X.join, S, F.assign(o_depth=np.where(F.gid == 2, 1., F.o_depth))))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'2 join: stopped ({e_!r})'[:200])
    # ---- 3 state_table
    try:
        St = X.state_table(J)
        ok(f'3 labels ({list(St.price_display_queue)}, {list(St.behind_depth_queue)}, {list(St.principal_pressure)})',
           list(St.principal) == [True, False] and list(St.price_display_queue) == ['at_touch|shown_positive|front', 'behind_best|shown_zero|unknown']
           and list(St.behind_depth_queue) == ['outside', 'outside'] and list(St.principal_pressure) == ['front|non_adverse', 'outside'])
        J2 = J.assign(fp_shown=[7., 3.]); St2 = X.state_table(J2)
        ok(f'3 the exact-depth label ({list(St2.behind_depth_queue)})', list(St2.behind_depth_queue) == ['outside', 'depth=2.0|unknown'])
        ok(f'3 the principal needs both at_touch and shown_positive ({list(St2.principal)})', list(St2.principal) == [True, False] and list(St2.principal_pressure) == ['front|non_adverse', 'outside'])
        ok('3 a saved band that differs refuses', refuses('queue band', X.state_table, J.assign(queue_band=['within_R', 'unknown'])))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'3 state_table: stopped ({e_!r})'[:200])
    # ---- 4 union and cells
    try:
        base = dict(day='d1', m=1, t=1000., R=10., price_state='at_touch', display='shown_positive', bn_band='non_adverse')
        Sx = pd.DataFrame([dict(base, gid=1, side=0, Q=0., o_depth=0., fp_shown=9., t_to_end=100000., queue_band='front'),
                           dict(base, gid=2, side=1, Q=5., o_depth=0., fp_shown=9., t_to_end=20000., queue_band='within_R'),
                           dict(base, gid=3, side=0, Q=20., o_depth=2., fp_shown=3., t_to_end=50000., queue_band='beyond_R', price_state='behind_best', day='d2'),
                           dict(base, gid=4, side=0, Q=6., o_depth=0., fp_shown=9., t_to_end=40000., queue_band='within_R')])
        Sx['principal'] = (Sx.price_state == 'at_touch') & (Sx.display == 'shown_positive'); Sx['price_display'] = Sx.price_state + '|' + Sx.display
        Sx['price_display_queue'] = Sx.price_display + '|' + Sx.queue_band
        Sx['behind_depth_queue'] = np.where((Sx.price_state == 'behind_best') & (Sx.display == 'shown_positive'), 'depth=' + Sx.o_depth.astype(str) + '|' + Sx.queue_band, 'outside')
        Sx['principal_pressure'] = np.where(Sx.principal, Sx.queue_band + '|' + Sx.bn_band, 'outside')
        Ix = pd.DataFrame([iv_row(pd, 1, 'I0', 'at_risk_full', fills=1, shares=10., v={0: 2., 1000: -10., 30000: -12.}), iv_row(pd, 1, 'I1', 'removed_in_interval'),
                           iv_row(pd, 1, 'I2', 'not_at_risk'), iv_row(pd, 1, 'I3', 'not_at_risk'),
                           iv_row(pd, 2, 'I0', 'at_risk_full', fills=1, shares=4., v={1000: 4.}, ok={30000: False}), iv_row(pd, 2, 'I1', 'at_risk_full'),
                           iv_row(pd, 2, 'I2', 'removed_in_interval'), iv_row(pd, 2, 'I3', 'not_at_risk'),
                           iv_row(pd, 3, 'I0', 'not_at_risk', day='d2'), iv_row(pd, 3, 'I1', 'not_at_risk', day='d2'), iv_row(pd, 3, 'I2', 'not_at_risk', day='d2'),
                           iv_row(pd, 3, 'I3', 'not_at_risk', day='d2'),
                           iv_row(pd, 4, 'I0', 'at_risk_full'), iv_row(pd, 4, 'I1', 'removed_in_interval'), iv_row(pd, 4, 'I2', 'not_at_risk'), iv_row(pd, 4, 'I3', 'not_at_risk')])
        U = X.union(Ix.merge(Sx[['day', 'm', 'gid', 'side']], on=['day', 'm', 'gid']))
        ok(f'4 the union status from the end cause ({dict(zip(U.gid, U.follow_up))})', dict(zip(U.gid, U.follow_up)) == {1: 'removed_in_interval', 2: 'removed_in_interval', 3: 'not_at_risk', 4: 'removed_in_interval'}
           and np.isnan(U[U.gid == 2].v30000.iloc[0]) and not bool(U[U.gid == 2].common.iloc[0]))
        T = X.cells(Sx, Ix); sel = lambda view, band, iv, coh, day='all', side='all': T[(T.view == view) & (T.band == band) & (T.interval == iv) & (T.day == day) & (T.side == side)
                                                                                         & (T.cohort == coh)].iloc[0]
        a = sel('price_display_queue', 'at_touch|shown_positive|front', 'I0', 'native')
        ok(f'4 an interval cell ({a.all_opportunities}, {a.h1000_valid_filled}, {a.h1000_per_share}, {a.h1000_sparse})', a.all_opportunities == 1 and a.all_filled == 1 and a.all_shares == 10.
           and a.h1000_valid == 1 and a.h1000_valid_filled == 1 and a.h1000_valid_shares == 10. and a.h1000_net == -10. and a.h1000_per_share == -1. and a.h1000_per_opportunity == -10.
           and a.h1000_any_fill_rate == 1. and bool(a.h1000_sparse) and a.h0_net == 2.)
        u = sel('price_display', 'at_touch|shown_positive', 'union', 'common'); un = sel('price_display', 'at_touch|shown_positive', 'union', 'native')
        ok(f'4 cohort counts beside the all-state counts ({u.all_opportunities}, {u.all_filled}, {u.h1000_valid}, {u.censored_from_common}, {u.h1000_any_fill_rate}, {u.all_any_fill_rate})',
           u.all_opportunities == 3 and u.all_filled == 2 and u.all_at_risk == 3 and u.h1000_valid == 2 and u.h1000_valid_risk == 2 and u.h1000_valid_filled == 1
           and u.censored_from_common == 1 and u.h1000_any_fill_rate == 0.5 and abs(u.all_any_fill_rate - 2. / 3.) < 1e-12 and u.h1000_net == -10. and u.h1000_per_share == -1.
           and u.h1000_per_opportunity == -5.)
        ok(f'4 the native union ({un.h1000_valid}, {un.h1000_net}, {un.h30000_valid})', un.h1000_valid == 3 and un.h1000_net == -6. and abs(un.h1000_per_share - (-6. / 14.)) < 1e-12
           and un.h30000_valid == 2 and un.h30000_net == -12. and un.h1000_fill_positive == 4. and un.h1000_fill_negative == -10. and un.h1000_state_positive == 4.)
        ok(f'4 time to END ({u.all_t_to_end_median}, {u.all_t_to_end_lt30s})', u.all_t_to_end_median == 40000. and u.all_t_to_end_lt30s == 1)
        b = sel('behind_depth_queue', 'depth=2.0|beyond_R', 'union', 'native'); o = sel('behind_depth_queue', 'outside', 'union', 'native')
        ok(f'4 the exact-depth view keeps outside ({b.all_opportunities}, {o.all_opportunities})', b.all_opportunities == 1 and o.all_opportunities == 3 and b.all_n_not_at_risk == 1)
        many = pd.concat([Sx[Sx.gid == 1].assign(gid=g) for g in range(100, 119)] + [Sx[Sx.gid == 2].assign(gid=200, side=0)], ignore_index=True)
        mi = pd.concat([Ix[Ix.gid == 1].assign(gid=g) for g in range(100, 119)] + [Ix[Ix.gid == 2].assign(gid=200)], ignore_index=True)
        many['price_display_queue'] = 'at_touch|shown_positive|front'; T2 = X.cells(many, mi)
        c2 = T2[(T2.view == 'price_display_queue') & (T2.interval == 'union') & (T2.day == 'all') & (T2.side == 'all') & (T2.cohort == 'common')].iloc[0]
        c3 = T2[(T2.view == 'price_display_queue') & (T2.interval == 'union') & (T2.day == 'all') & (T2.side == 'all') & (T2.cohort == 'native')].iloc[0]
        ok(f'4 sparse on the valued filled states ({c2.all_filled}, {c2.h1000_valid_filled}, {c2.h1000_sparse}, {c3.h1000_sparse})', c2.all_filled == 20 and c2.h1000_valid_filled == 19
           and bool(c2.h1000_sparse) and c3.h1000_valid_filled == 20 and not bool(c3.h1000_sparse) and bool(c3.h30000_sparse))
        ok('4 a missing interval row refuses', refuses('interval row', X.cells, Sx, Ix.iloc[1:]))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'4 cells: stopped ({e_!r})'[:220])
    # ---- 5 guards
    saved = dict(frozen=X.frozen, sources=X.sources); tmp = Path(tempfile.mkdtemp())
    try:
        mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(sources=X.sources())))
        X.frozen = lambda: False; ok('5 run refuses before the freeze', refuses('committed', X.run, man_path=mp, out=tmp / 'out'))
        X.frozen = lambda: True; X.sources = lambda: dict(saved['sources'](), **{'pf_ladder_queue_price.py': '0' * 64}); ok('5 run refuses another source', refuses('source', X.run, man_path=mp, out=tmp / 'out'))
        X.sources = saved['sources']; (tmp / 'out').mkdir(); (tmp / 'out' / 'audit.json').write_text('{}'); ok('5 run refuses an existing output', refuses('write-once', X.run, man_path=mp, out=tmp / 'out'))
        ok('5 build_manifest refuses an existing manifest', refuses('write-once', X.build_manifest, mp))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'5 guards: stopped ({e_!r})'[:200])
    finally:
        X.frozen, X.sources = saved['frozen'], saved['sources']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('QUEUE PRICE SELFTEST PASSED: labels, join, state_table, union and cells, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_queue_price as X_
    main(X_)
