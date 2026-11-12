"""CS post hoc 2 (2026-11-12, session ce68ebe0), after the simulator review: how much of CS's gain is fills while two of our own orders
are live at once? The replay serves our own orders in time order (not price priority as the frozen engine) and lets two of them take
shares from one trade leg (review finding 2). Overlap fills = fills on a trade leg that also filled another of our orders, and fills
on an order while another of ours was live (time overlap). Per arm: $/day at 1 s without young fills (CS's metric), all and
excluding overlap fills, at M 0 / M 50, on the 9 days. POST HOC, no reading. Control: the arms' totals reproduce CS's day files.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pf_kp_s4 as K, pf_qr_queue_race as Q, pf_causal_speed as C    # noqa: E402,E401
ARMS = ['J-v3', 'E(cur,40)', 'E(cur,20)', 'E(cur,10)', 'E(30,10)', 'E(15.2,40)', 'E(15.2,10)']
FP = {a: fp for a, fp in zip(C.ARMS, [(f, p) for f in C.FS for p in C.PS])}
CFG = [('M', 0.0), ('M', 50.0)]; OUT = f'{K.IO}/cs_20261112/posthoc'


def main():
    os.makedirs(OUT, exist_ok=True); R = {}; bad = {}
    for day in C.DAYS9:
        D = K.load_day(day); cs = json.load(open(f'{K.IO}/cs_20261112/day_{day}.json'))['acc']
        for w in K.window_sides(D):
            side = w['side']; tf = w['tf']; cache = w.setdefault('cache', {})
            b0, valid, rs, kind = Q.runs(tf, w['P5'], w['opp'], side); bi = Q.fb_births(tf, b0, valid, w['ws'], w['te'], side)
            for q_, shf in CFG:
                tt = w['anc'] - shf; key = f'{q_}|{shf:.0f}'
                a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stv3'], w['P90'], w['best'], w['bopp'], tt, w['ptl'], w['shl'], q_, w['te'])
                for arm in ARMS:
                    F, O = C.sim5(*a) if arm == 'J-v3' else C.sim5(*a, sched=C.schedule(tf, w['kf'], b0, bi, *FP[arm]))
                    r = K.vals(w, F, tt); legs = {}
                    for f in F:
                        legs[f[0]] = legs.get(f[0], 0) + 1
                    iv = [(o['ta'], min(o['tr'], o['tfill'])) for o in O if not o['ref']]
                    e = R.setdefault(f'{arm}|{key}', {}).setdefault(day, dict(q=0.0, v1=0.0, wo=0.0, wo_noov=0.0, ov_leg_q=0.0, ov_time_q=0.0))
                    for f, qq, m1 in zip(F, r['q'], r['m1']):
                        vv = float(qq * m1) if np.isfinite(m1) else 0.0; t = tt[f[0]]
                        n_live = sum(1 for s_, e_ in iv if s_ <= t < e_)
                        ov = legs[f[0]] > 1 or n_live > 1
                        e['q'] += float(qq); e['v1'] += vv; e['ov_leg_q'] += float(qq) * (legs[f[0]] > 1); e['ov_time_q'] += float(qq) * (n_live > 1)
                        if f[4] >= 100.0:
                            e['wo'] += vv; e['wo_noov'] += vv * (not ov)
            w.pop('cache', None)
        for arm in ARMS:
            for q_, shf in CFG:
                key = f'{q_}|{shf:.0f}'; ref = cs[f'{arm}|{key}']; e = R[f'{arm}|{key}'][day]
                if abs(ref['v1'] - e['v1']) > 1e-6:
                    bad[f'{day} {arm}|{key}'] = (ref['v1'], e['v1'])
        print(day, 'done', flush=True)
        if bad:
            sys.exit(f'CONTROL FAILED (CS not reproduced) -> no table: {bad}')
    nd = 9.0; L = [f'CS post hoc 2: fills while two of our orders are live; as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; CS reproduced on every day']
    for q_, shf in CFG:
        key = f'{q_}|{shf:.0f}'; jv = np.array([R[f'J-v3|{key}'][d]['wo_noov'] for d in C.DAYS9]) / 100
        L.append(f'{key}: arm | w/o-young $/day | same, excluding overlap fills | days above J-v3 (excl.) | overlap share of shares: same leg / time')
        for arm in ARMS:
            E = R[f'{arm}|{key}']; s = lambda k: sum(E[d][k] for d in C.DAYS9)
            ex = np.array([E[d]['wo_noov'] for d in C.DAYS9]) / 100
            L.append(f'  {arm:<11} {s("wo") / 100 / nd:>6.2f} | {s("wo_noov") / 100 / nd:>6.2f} | {int(np.sum(ex > jv))} of 9 | {s("ov_leg_q") / s("q"):.1%} / {s("ov_time_q") / s("q"):.1%}')
    json.dump(R, open(f'{OUT}/cs_overlap.json', 'w'), default=float)
    open(f'{OUT}/cs_overlap.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
