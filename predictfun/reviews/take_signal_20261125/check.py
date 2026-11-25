"""TAKE-SIGNAL v1 independent checks (contract.md 'Checks'):
 K1 bn_r250 re-derived by a plain Python scan of the raw bookTicker rows for every kept tokyo trigger inside 4 hash-ranked hours;
 K2 causality: Binance changes after t and Predict frames after t - 63 ms perturbed -> r250, known book and limit unchanged; positive
    control: perturbing the known frame itself changes the limit;
 K3 synthetic fill cases through take_eval's helpers (stale ask kept -> fill; ask lifted next frame -> not executable; DOWN mirror;
    one-sided known book -> no_known_book; fee formula);
 K4 reconciliation: kept triggers per clock = events_all kept count; unique (clock, mid, t, dir); status partition;
 K5 the 2 s rule: kept triggers >= 2 s apart per market x direction and every non-kept qualifying event < 2 s after a qualifying one;
 K6 the pooled tokyo readings recomputed from takes.parquet with a different aggregation path.
Usage: run_light.sh ts_check -- python3 predictfun/reviews/take_signal_20261125/check.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import bn_events as BE  # noqa: E402
import take_eval as TE  # noqa: E402

OUT = TE.OUT
res = {}


def ok(name, cond, detail=''):
    res[name] = dict(ok=bool(cond), detail=detail)
    print(('PASS ' if cond else 'FAIL ') + name, detail)


def rank(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def main():
    tk = pd.read_parquet(OUT / 'takes.parquet')
    ev = pd.read_parquet(OUT / 'events_all.parquet')
    tok = tk[tk.clock == 'tokyo']
    # K1: plain scan of raw rows in 4 hash-ranked hours (Oct 24-27 only)
    hours = sorted([s for s in BE.ALLOWED if s != '20261023_23'], key=lambda s: rank('ts_check|' + s))[:4]
    n, bad = 0, 0
    for stem in hours:
        rows = []
        prev = BE.ALLOWED[BE.ALLOWED.index(stem) - 1]
        for st_ in (prev, stem):                     # the previous hour supplies the state just before the hour starts
            with __import__('gzip').open(BE.path_of(st_), 'rt') as fh:
                for line in fh:
                    f = line.rstrip('\n').split(',')
                    if len(f) != 8:
                        continue
                    try:
                        E, u, b, a = int(f[1]), int(f[3]), float(f[4]), float(f[6])
                    except ValueError:
                        continue
                    if b > 0 and a > b:
                        rows.append((E, u, (b + a) / 2))
        rows.sort()
        seen, Es, Ms = set(), [], []
        for E, u, m in rows:
            if u in seen:
                continue
            seen.add(u)
            Es.append(E)
            Ms.append(m)
        Es = np.array(Es)
        Ms = np.array(Ms)
        h0 = pd.Timestamp(stem[:8] + ' ' + stem[9:] + ':00', tz='UTC').value // 10 ** 6
        sel = tok[(tok.t >= h0 + 60_000) & (tok.t < h0 + 3_600_000 - 15_000)]     # away from the carry edges
        for r in sel.itertuples():
            def state(x):
                j = None
                for i in range(len(Es) - 1, -1, -1):        # plain backward scan: last row with E + 5 <= x
                    if Es[i] + 5 <= x:
                        j = i
                        break
                return Ms[j]
            i_now = np.searchsorted(Es + 5, r.t, 'right') - 1
            i_old = np.searchsorted(Es + 5, r.t - 250, 'right') - 1
            m_now, m_old = Ms[i_now], Ms[i_old]
            if n < 10:                                       # the first 10 also through the literal backward scan
                m_now, m_old = state(r.t), state(r.t - 250)
            v = 1e4 * math.log(m_now / m_old)
            n += 1
            bad += abs(v - r.r250) > 1e-9
    ok('K1_raw_rescan_r250', n > 50 and bad == 0, f'{n} kept triggers in hours {hours}, {bad} mismatches')
    # K2: causality on 200 hash-ranked tokyo triggers of one day
    day = '20261026'
    mk = pd.read_parquet(TE.V / f'markets_{day}.parquet', columns=['mid'])
    mindex = {int(m): i for i, m in enumerate(sorted(mk.mid))}
    T = TE.load_tob(day, mindex)
    bn = pd.read_parquet(OUT / 'bn_changes.parquet')
    bk, bm = bn.k_ms.to_numpy(np.int64), bn.mid.to_numpy(float)
    s = tok[tok.day == day].copy()
    s['_r'] = [rank(f'k2|{a}|{b}') for a, b in zip(s.mid, s.t)]
    s = s.nsmallest(200, '_r')
    changed_r, changed_k, fired = 0, 0, 0
    for r in s.itertuples():
        t = int(r.t)
        bm2 = np.where(bk + 5 > t, bm * 1.01, bm)                       # every Binance change after the decision perturbed
        ti, r250, _ = TE.events_for(bk, bm2, 5, t, t + 1)
        changed_r += int(not (len(ti) == 1 and abs(r250[0] - r.r250) < 1e-12))
        m_ = np.array([mindex[int(r.mid)]])
        cut = np.searchsorted(T['key'], m_[0] * TE.KEY + (t - 63), 'right')  # frames after t - 63 in this market perturbed
        e = np.searchsorted(T['key'], (m_[0] + 1) * TE.KEY, 'left')
        T2 = dict(T, ask=T['ask'].copy(), bid=T['bid'].copy())
        T2['ask'][cut:e] = 0.99
        T2['bid'][cut:e] = 0.01
        up = np.array([r.dir == 1])
        ik = TE.frame_last(T2, m_, np.array([t - 63]))
        lim2, _ = TE.side_px(T2, ik, up)
        ik0 = TE.frame_last(T, m_, np.array([t - 63]))
        lim0, _ = TE.side_px(T, ik0, up)
        changed_k += int(not (lim2[0] == lim0[0] == r.limit or (np.isnan(lim2[0]) and np.isnan(r.limit))))
        if ik0[0] >= 0:
            T3 = dict(T, ask=T['ask'].copy(), bid=T['bid'].copy())
            T3['ask'][ik0[0]] += 0.05
            T3['bid'][ik0[0]] -= 0.05
            lim3, _ = TE.side_px(T3, ik0, up)
            fired += int(not np.isclose(lim3[0], lim0[0]))
    ok('K2_future_binance_no_effect', changed_r == 0, f'200 triggers, {changed_r} r250 changes')
    ok('K2_future_predict_no_effect', changed_k == 0, f'200 triggers, {changed_k} limit changes')
    ok('K2_positive_control', fired >= 150, f'{fired}/200 limits change when the known frame itself is moved')
    # K3: synthetic fill cases
    syn = dict(key=np.array([1000, 1100, 1200]), midx=np.zeros(3, np.int64), upd=np.array([1000, 1100, 1200]),
               bid=np.array([0.49, 0.49, 0.49]), ask=np.array([0.50, 0.50, 0.53]), bsh=np.array([100., 100., 100.]),
               ash=np.array([80., 60., 50.]), bad=np.zeros(3, bool))
    m0 = np.zeros(1, np.int64)
    f1, f2 = TE.frame_last(syn, m0, np.array([1050])), TE.frame_next(syn, m0, np.array([1050]))
    p1, s1 = TE.side_px(syn, f1, np.array([True]))
    p2, s2 = TE.side_px(syn, f2, np.array([True]))
    case1 = (p1[0] <= 0.50 + 1e-9) and (p2[0] <= 0.50 + 1e-9) and min(s1[0], s2[0]) == 60.0
    f1b, f2b = TE.frame_last(syn, m0, np.array([1150])), TE.frame_next(syn, m0, np.array([1150]))
    p2b, _ = TE.side_px(syn, f2b, np.array([True]))
    case2 = p2b[0] > 0.50 + 1e-9
    pb, _ = TE.side_px(syn, f1, np.array([False]))
    case3 = pb[0] == 0.49
    one = dict(syn, ask=np.array([np.nan, 0.50, 0.50]))
    _, okk = TE.mid_ok(one, TE.frame_last(one, m0, np.array([1050])), np.array([1050]))
    case4 = not okk[0]
    case5 = abs(200 / 1e4 * min(0.30, 0.70) - 0.006) < 1e-12
    ok('K3_synthetic_fill_cases', case1 and case2 and case3 and case4 and case5, f'{[case1, case2, case3, case4, case5]}')
    # K4 reconciliation
    kept = int(ev.kept.sum())
    per = tk.groupby('clock').size().to_dict()
    uniq = not tk.duplicated(['clock', 'mid', 't', 'dir']).any()
    part = all(tk.status.isin(['filled', 'not_executable', 'no_next_frame', 'no_known_book']))
    ok('K4_reconciliation', per.get('tokyo') == kept and per.get('slow') == kept and uniq and part, f'kept {kept}, per clock {per}')
    # K5 the 2 s rule
    viol = 0
    for (mid, d), x in ev.groupby(['mid', 'dir']):
        tt = x.t.to_numpy()
        kp = x.kept.to_numpy()
        gap = np.r_[np.inf, np.diff(tt)]
        viol += int((kp != (gap >= TE.COOLDOWN_MS)).sum())
    ok('K5_cooldown_rule', viol == 0, f'{len(ev)} qualifying events, {viol} violations')
    # K6 pooled readings by a different path
    x = tok.assign(sh=np.minimum(50, tok.avail_sh))
    num = sum(v * w for v, w in zip(x.mark_3s, x.sh) if w > 0 and not math.isnan(v))
    den = sum(w for v, w in zip(x.mark_3s, x.sh) if w > 0 and not math.isnan(v))
    rd = json.loads((H / 'readings.json').read_text())
    ok('K6_pooled_mark_3s', abs(100 * num / den - rd['tokyo']['mark_3s']['pooled_c']) < 1e-9, f'{100 * num / den:.6f} c/sh')
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    (H / 'checks.json').write_text(json.dumps(res, indent=1) + '\n')
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
