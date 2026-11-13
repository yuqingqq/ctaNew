"""QUEUE-PLACE post hoc (2026-11-13, session ce68ebe0): WHY does rung 0 join a new best level late? POST HOC, descriptive, no readings:
written after QUEUE-PLACE's report (pf_queue_place.py, 2f7143ea) showed LATE-ENTRY as the largest part of the queue ahead (R481) and a
median 359 ms from a level's birth to our first order there.

Same simulator (pf_queue_place.sim4, control (i) passed there), rung 0, M 50, the same population: at-best fills on exact trades, aged,
decomposed (status ok), POST chains only (our first order at that price went live after the run's birth tb). For each such fill, on the
simulator's clock:
  state at tb   HELD   an order of ours was live at another price at tb (rung 0 keeps its order while the best is <= 3 ticks better)
                PEND   an order was placed but not yet live, or cancelled and not yet gone
                OUT    no order: pulled by the guard / in its 250 ms clear wait, or just filled
  then          prev_end - tb   how long after the birth the order before our first one at this price ended (cancel decision or fill)
                why             why it ended (trig = guard pull, drift, filled, cross, nobook)
                ta(c0) - tb     our first order's go-live after the birth (= QUEUE-PLACE's entry delay)
Printed: share of shares by state, the why split, median / p25 / p75 of the gaps; c/sh and STOP share by state.
Usage (run_light, PF_SLOT=claude): pf_queue_place_why_late.py day <day> | report
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as QR                             # noqa: E402
import pf_queue_place as QP                               # noqa: E402

OUTW = f'{QP.OUTQ}/why_late'
INF = float('inf'); NAN = float('nan')


def state_at(O, tb, c0):
    """Lane state at tb from the orders before c0 (one order per lane at a time, cancels may overlap the next order)."""
    prev = [o for o in O[:c0] if not o['ref']]
    live = [o for o in prev if o['ta'] <= tb < min(o['tr'], o['tfill'])]
    if live:
        return 'HELD' if all(o['pt'] != O[c0]['pt'] for o in live) else 'SAME', live[-1]
    pend = [o for o in O[:c0] if o['ts'] <= tb < o['ta'] or (o['tc'] <= tb < o['tr'])]
    if pend:
        return 'PEND', pend[-1]
    return 'OUT', (prev[-1] if prev else None)


def day_run(day):
    import pf_gap_top_makers as GP
    import pf_depth_timing as DT
    QP.all_selftests()
    T0 = time.time(); D = K.load_day(day); st = dict(window_sides=0, rows=0, sim_differ=0)
    full = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet', columns=['mid', 'ptick'] + QP.OLDKEY[1:] + ['how', 't_ms'])
    inw = full.mid.isin(list(D['W0'])).values
    MP = pd.read_parquet(f'{QP.ORD}/map_{day}.parquet').set_index('row')
    F = full[inw].join(MP, how='left')
    F['comp'] = F.groupby(QP.OLDKEY, sort=False).tko_ok.transform('all').values
    wsT = np.where((F.how.values > 0) & F.tko_ok.values, F.t_ms.values.astype(float), np.nan)
    F['tws'] = pd.Series(wsT, index=F.index).groupby(F.tko.values).transform('min').values
    T = F[F.comp.values & np.isfinite(F.tws.values)]; TP = T.groupby('tko').ptick.agg(list).to_dict(); exact = set(T.index.tolist())
    rows = []
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg = w['m'], w['side'], w['sg']
        if (m, side) not in D['LI']:
            continue
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)
        l = D['LT'].iloc[D['LI'][(m, side)]]
        o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist(); cache = w.setdefault('cache', {})
        tt = w['anc'] - 50.0; ttl = tt.tolist()
        Fo, Ot, O, fx = QP.sim4(side, tf, opp, P5, S5, cache, w['tg'], w['vkl'], -INF, w['best'], w['bopp'], tt, w['ptl'], w['shl'], 'M', w['te'], vg, 0, QP.SIZE)
        Fk, Ok, _ = K.sim_arm(w, 'M', 50.0, 'rest', aux=vg, offset=0, size=QP.SIZE)
        st['sim_differ'] += int(Fk != Fo or Ok != Ot)
        for f, x in zip(Fo, fx):
            i, pt, qq, thr, age = f[0], int(f[1]), f[2], f[3], f[4]; o = O[x[0]]; r_ = int(lrow[i])
            if age < 100.0 or not (r_ in exact and F.tko.at[r_] in TP):
                continue
            t = float(F.tws.at[r_])
            if not (w['ws'] <= t < w['ws'] + 60000.0) or GP.depth_of(tfl, P5, opp, side, t, pt) != 0:
                continue
            ru = QP.run_at(tfl, b0, valid, rs, kind, ttl[i], pt)
            if ru is None or ru[1] == 'censored':
                continue
            tb = ru[0]; c0, lk = QP.chain(O, o['id'], tb)
            if O[c0]['ta'] < tb:
                continue                                   # PRE chains are not late
            s_, p_ = state_at(O, tb, c0)
            pe = min(p_['tc'], p_['tfill']) if p_ is not None else NAN
            why = '' if p_ is None else ('filled' if p_['tfill'] <= p_['tc'] else p_['why'])
            dt = ((pt - p_['pt']) if side == 0 else (p_['pt'] - pt)) if (p_ is not None and s_ == 'HELD') else 0
            typ = 'THROUGH' if thr else DT.trade_type(side, pt, TP[F.tko.at[r_]])
            rows.append((int(day), s_, why, int(dt), float(pe - tb), float(O[c0]['ta'] - tb), ru[1], float(qq), sg * (mff[max(int(np.searchsorted(tf, t + 1000.0, side='right')) - 1, 0)] - pt), typ))
        w.pop('cache', None)
    if st['sim_differ']:
        sys.exit(f'WHY-LATE CONTROL FAILED -> no output: {st}')
    st['rows'] = len(rows)
    os.makedirs(OUTW, exist_ok=True)
    pd.DataFrame(rows, columns=['day', 'state', 'why', 'ticks', 'end_after_tb', 'live_after_tb', 'kind', 'q', 'v1', 'type']).to_parquet(f'{OUTW}/rows_{day}.parquet', index=False)
    json.dump(st, open(f'{OUTW}/day_{day}.json', 'w'))
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


def report():
    A = pd.concat([pd.read_parquet(f'{OUTW}/rows_{d}.parquet') for d in QP.DAYS9], ignore_index=True); A = A[np.isfinite(A.v1)]
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if x.q.sum() else NAN
    pq_ = lambda v: f'{np.percentile(v, 50):.0f} ({np.percentile(v, 25):.0f} / {np.percentile(v, 75):.0f})' if len(v) else 'n/a'
    L = [f'QUEUE-PLACE post hoc WHY-LATE: rung 0, M 50, exact, aged, at-best fills from POST chains; {len(A)} fills, {A.q.sum():.0f} sh; '
         f'as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; sim4 = sim_arm on every window-side', '',
         'state of our rung-0 lane at the level\'s birth: share of shares | c/sh | STOP | our first order live after the birth, ms median (p25 / p75) |'
         ' the earlier order ended after the birth, ms']
    for s_ in ('HELD', 'PEND', 'OUT', 'SAME'):
        x = A[A.state == s_]
        if len(x):
            L.append(f'  {s_:<5} {x.q.sum() / A.q.sum():5.1%} | {cps(x):+.3f} | {x.q[x.type == "STOP"].sum() / x.q.sum():4.0%} | {pq_(x.live_after_tb.values)} |'
                     f' {pq_(x.end_after_tb.values[np.isfinite(x.end_after_tb.values)])}')
    L.append('')
    for s_ in ('HELD', 'OUT', 'PEND'):
        x = A[A.state == s_]
        if len(x):
            L.append(f'  {s_}: why the earlier order ended (share of shares): ' + ', '.join(f'{wy or "none"} {g.q.sum() / x.q.sum():.0%}' for wy, g in x.groupby('why')))
    h = A[A.state == 'HELD']
    if len(h):
        L.append('  HELD: the held order was this many ticks behind the new best (share of shares): ' +
                 ', '.join(f'{k_} {g.q.sum() / h.q.sum():.0%}' for k_, g in h.groupby('ticks')))
    L.append('  by the run\'s birth kind (share of shares): ' + ', '.join(f'{k_} {g.q.sum() / A.q.sum():.0%} (HELD {g.q[g.state == "HELD"].sum() / g.q.sum():.0%})'
                                                                     for k_, g in A.groupby('kind')))
    open(f'{OUTW}/why_late.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
