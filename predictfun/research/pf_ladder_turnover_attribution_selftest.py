"""SELFTEST of pf_ladder_turnover_attribution (PAIRED-TURNOVER-ATTRIBUTION-v1; 2026-11-22, session aea5bfb9). main(X) checks the module on hand
ledgers with realizable fixed-price generations only (no data is read); a failed check raises SystemExit with FAIL. Groups (review d799cb77 item 5):
 1 FIFO decomposition: profitable buy / sell turnover despite a negative buy END mark; a loss-making round trip; residual long and short
   inventory; short lots closed by buys; pre-fork carried inventory closed later (basis = the fork mark, holding from the actual
   acquisition, a pre-fork partial round trip); partial closes; a corrupted Q refused;
 2 focal round trips: an own-END-negative fill with a profitable opposite close is classed profitable, not an uncompensated loss;
 3 events: common / retain-only / alternative-only by execution event (the same trade with another gid is common), the Y identity;
 4 replacement delays: gid reuse after WAIT, the forked POST excluded, attempts vs accepted;
 5 the summary row (turnover share, top-5 concentration, quantiles) and the computed day signs;
 6 guards: run refuses before the freeze, under another source and over an existing audit; the manifest is write-once."""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

FAIL = 'TURNOVER ATTRIBUTION SELFTEST FAILED'


def run(mod):
    main(mod)


def F(t, gid, side, q, p, idx=None):
    return [float(t), int(gid), int(side), float(q), float(p), int(idx if idx is not None else t)]


def arm(pre, post, M0, M1, lineage=(), I0=None):
    s = {0: 1., 1: -1.}; I0 = sum(s[f[2]] * f[3] for f in pre) if I0 is None else I0
    Q = sum(s[f[2]] * f[3] * (M1 - f[4]) for f in post) + I0 * (M1 - M0)
    return dict(pre_fills=pre, post_raw=post, M_fork=M0, mark=M1, Q=Q, lineage=list(lineage))


def main(X):  # noqa: C901
    import numpy as np
    import pandas as pd
    bad = []
    ok = lambda name, cond: None if cond else bad.append(name)

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw); return False
        except ValueError as e2:
            return word in str(e2)
    # ---- 1 FIFO decomposition
    try:
        a1 = arm([], [F(100, 1, 0, 10, 50), F(200, 1, 0, 10, 50), F(300, 2, 1, 20, 51)], 50., 49., lineage=[1]); d1, c1, l1 = X.decompose(a1)
        ok(f'1 profitable turnover despite a negative buy END mark ({d1["matched"]}, {d1["residual"]}, {a1["Q"]})', a1['Q'] == 20. and d1['matched'] == 20. and d1['residual'] == 0.
           and d1['matched_q'] == 20. and d1['profit_per_share'] == 1. and d1['hold_ms'] == 150.)
        a2 = arm([], [F(100, 1, 0, 10, 50), F(200, 2, 1, 10, 49)], 50., 49., lineage=[1]); d2, _, _ = X.decompose(a2)
        ok(f'1 a loss-making round trip ({d2["matched"]})', d2['matched'] == -10. and d2['residual'] == 0.)
        a3 = arm([], [F(100, 1, 0, 10, 50)], 50., 52.); a3s = arm([], [F(100, 1, 1, 10, 50)], 50., 52.); d3, _, _ = X.decompose(a3); d3s, _, _ = X.decompose(a3s)
        ok(f'1 residual long / short inventory ({d3["residual"]}, {d3s["residual"]})', d3['residual'] == 20. and d3['matched'] == 0. and d3s['residual'] == -20. and d3['remaining_q'] == 10.
           and d3s['remaining_q'] == -10.)
        a4s = arm([], [F(100, 1, 1, 10, 52), F(200, 2, 0, 10, 50)], 51., 49.); d4s, _, _ = X.decompose(a4s)
        ok(f'1 a short lot closed by a buy ({d4s["matched"]})', d4s['matched'] == 20. and d4s['residual'] == 0.)
        a5 = arm([F(50, 7, 0, 10, 48), F(60, 8, 1, 4, 49)], [F(300, 2, 1, 6, 51)], 50., 49.); d5, c5, _ = X.decompose(a5)
        ok(f'1 carried inventory closed later ({d5["matched"]}, {d5["carried_q"]}, {d5["hold_ms"]})', d5['matched'] == 6. and d5['matched_carried'] == 6. and d5['carried_q'] == 6.
           and d5['hold_ms'] == 250. and c5[0]['carried'] and c5[0]['basis'] == 50. and c5[0]['open_price'] == 48.)
        a6 = arm([], [F(100, 1, 0, 10, 50), F(200, 2, 1, 4, 52)], 50., 49.); d6, _, l6 = X.decompose(a6)
        ok(f'1 a partial close ({d6["matched"]}, {d6["residual"]}, {d6["remaining_q"]})', d6['matched'] == 8. and d6['residual'] == -6. and d6['remaining_q'] == 6. and a6['Q'] == 2.)
        ok('1 a corrupted Q is refused', refuses('does not reconcile', X.decompose, dict(a6, Q=a6['Q'] + 1.)))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 fifo: stopped ({e!r})'[:220])
    # ---- 2 focal round trips
    try:
        fc = X.focal_classes(a1, c1, l1)
        ok(f'2 own-END-negative fills with a profitable close ({fc})', fc == {'own_end_negative_round_trip_profitable': dict(n=2, own=-20., round_trip=20.)})
        a7 = arm([], [F(100, 1, 0, 10, 50)], 50., 49., lineage=[1]); d7, c7, l7 = X.decompose(a7)
        ok(f'2 an unmatched focal fill has no round trip ({X.focal_classes(a7, c7, l7)})', X.focal_classes(a7, c7, l7) == {'own_end_negative_round_trip_none': dict(n=1, own=-10., round_trip=0.)})
        a8 = arm([], [F(100, 3, 0, 10, 50), F(200, 1, 1, 10, 51)], 50., 52., lineage=[1]); d8, c8, l8 = X.decompose(a8)
        ok(f'2 a focal close of an earlier lot ({X.focal_classes(a8, c8, l8)})', X.focal_classes(a8, c8, l8) == {'own_end_negative_round_trip_profitable': dict(n=1, own=-10., round_trip=10.)})
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 focal: stopped ({e!r})'[:220])
    # ---- 3 events
    try:
        R = arm([], [F(100, 1, 0, 10, 50, idx=5), F(200, 4, 1, 5, 51, idx=6)], 50., 49.); A = arm([], [F(100, 2, 0, 10, 50, idx=5), F(300, 4, 1, 3, 52, idx=7)], 50., 49.)
        ev = X.event_diff(R, A)
        ok(f'3 events by execution event ({ev})', ev['events_common'] == 1 and ev['events_retain_only'] == 1 and ev['events_alternative_only'] == 1
           and abs(ev['Y_events'] - (R['Q'] - A['Q'])) < 1e-12 and ev['value_retain_only'] == 10. and ev['value_alternative_only'] == 9.)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 events: stopped ({e!r})'[:220])
    # ---- 4 delays
    try:
        rw = [dict(gid=30, submit=1050., refused=False), dict(gid=31, submit=1800., refused=False)]
        rp = [dict(gid=30, submit=1000., refused=False), dict(gid=33, submit=1500., refused=False)]
        rr = [dict(gid=40, submit=1000., refused=True), dict(gid=41, submit=1200., refused=False)]
        ok(f'4 gid reuse after WAIT ({X.first_delays(rw, "ADMISSION", "alternative", 1000.)})', X.first_delays(rw, 'ADMISSION', 'alternative', 1000.) == (50., 50.))
        ok(f'4 the forked POST excluded ({X.first_delays(rp, "ADMISSION", "retain", 1000.)})', X.first_delays(rp, 'ADMISSION', 'retain', 1000.) == (500., 500.))
        ok(f'4 attempts vs accepted ({X.first_delays(rr, "REVIEW", "alternative", 1000.)})', X.first_delays(rr, 'REVIEW', 'alternative', 1000.) == (0., 200.)
           and all(math.isnan(x) for x in X.first_delays([], 'REVIEW', 'retain', 1000.)))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 delays: stopped ({e!r})'[:220])
    # ---- 5 summary and day signs
    try:
        T = pd.DataFrame(dict(kind=['A'] * 4 + ['B'] * 3, day=['d1', 'd2', 'd1', 'd2', 'd1', 'd2', 'd1'], side=[0, 0, 1, 1, 0, 0, 0], Y=[-1., -2., -3., 10., 1., -1., 0.],
                              dMatched=[1., 1., 1., 1., 0., 0., 0.], dResidual=[-2., -3., -4., 9., 1., -1., 0.], dMatched_lineage=0., dMatched_other=0.,
                              Y_events=[-1., -2., -3., 10., 1., -1., 0.]))
        ds = X.day_signs(T)
        ok(f'5 day signs ({ds})', ds['A']['sign'] == 'MIXED' and ds['B']['sign'] == 'MIXED' and X.day_signs(T[T.Y < 0])['A']['sign'] == 'NEGATIVE_EVERY_DAY'
           and X.day_signs(T[T.Y > 0])['A']['sign'] in ('POSITIVE_EVERY_DAY', 'MIXED') and X.day_signs(T.assign(Y=T.Y.abs() + 1))['A']['sign'] == 'POSITIVE_EVERY_DAY')
        r = X.srow(T, {})
        ok(f'5 the summary row ({r["turnover_share_abs"]}, {r["top5_share_abs_Y"]})', r['pairs'] == 7 and r['retain_better'] == 2 and r['alternative_better'] == 4 and r['tie'] == 1
           and abs(r['turnover_share_abs'] - 4. / (4. + 20.)) < 1e-12 and abs(r['top5_share_abs_Y'] - 17. / 18.) < 1e-12 and r['Y_q50'] == -1. and r['Y_sum'] == 4.)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 summary: stopped ({e!r})'[:220])
    # ---- 6 guards
    sv = dict(frozen=X.frozen, sources=X.sources); tmp = Path(tempfile.mkdtemp())
    try:
        mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(sources=X.sources())))
        X.frozen = lambda: False; ok('6 run refuses before the freeze', refuses('committed', X.run, out=tmp / 'o', man_path=mp))
        X.frozen = lambda: True; X.sources = lambda: dict(sv['sources'](), **{'pf_ladder_turnover_attribution.py': '0' * 64})
        ok('6 run refuses another source', refuses('source', X.run, out=tmp / 'o', man_path=mp)); X.sources = sv['sources']
        (tmp / 'o').mkdir(); (tmp / 'o' / 'audit.json').write_text('{}'); ok('6 run refuses an existing audit', refuses('write-once', X.run, out=tmp / 'o', man_path=mp))
        ok('6 the manifest is write-once', refuses('write-once', X.build_manifest, mp))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 guards: stopped ({e!r})'[:200])
    finally:
        X.frozen, X.sources = sv['frozen'], sv['sources']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('TURNOVER ATTRIBUTION SELFTEST PASSED: FIFO decomposition, focal round trips, events, delays, summary and day signs, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_turnover_attribution as X_
    main(X_)
