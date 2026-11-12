"""CS post hoc (2026-11-12, session ce68ebe0), after CS's tables: is the faster-placement gain an artifact of the pending-cancel window?

CS (pf_causal_speed.py, 7e16d03e) shortened the pending-cancel window of fast-path birth orders to 55 + p ms (95 ms at p = 40).
Arms with a 10-20 ms order path gained over v3 (without young fills); a faster feed alone did not. This re-runs the fast-path arms
with the window held at 95 ms for every order (CS's sim5 with that one expression replaced; the replacement is asserted to match
exactly once), on the same 9 days at M 0 / M 50 / M 250, and reports CS's metric (w/o-young $/day at 1 s) beside CS's own numbers.
POST HOC: no reading; a diagnostic of CS's own assumption. Control: with an identity replacement the copied simulator equals CS's
sim5 fill for fill on the first 40 window-sides of day 1 for every arm (else no table).
Usage: run_light.sh cs_posthoc -- python3 predictfun/research/pf_cs_posthoc.py
"""
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as Q                              # noqa: E402
import pf_causal_speed as C                               # noqa: E402

SRC = open(os.path.join(HERE, 'pf_causal_speed.py')).read()
OLD = "pend=L_PENDCANCEL - L_PLACE + lp)"
if SRC.count(OLD) != 1:
    sys.exit('REFUSED: the pending-window expression is not found exactly once')


def variant(expr):
    m = types.ModuleType('cs_variant'); m.__file__ = os.path.join(HERE, 'pf_causal_speed.py')
    exec(compile(SRC.replace(OLD, expr), 'cs_variant', 'exec'), m.__dict__)
    return m


IDENT = variant(OLD); FIXED = variant("pend=L_PENDCANCEL)")
ARMS = ['E(cur,20)', 'E(cur,10)', 'E(30,20)', 'E(30,10)', 'E(15.2,20)', 'E(15.2,10)']
FP = {a: fp for a, fp in zip(C.ARMS, [(f, p) for f in C.FS for p in C.PS])}
CFG = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
OUT = f'{K.IO}/cs_20261112/posthoc'


def main():
    os.makedirs(OUT, exist_ok=True); T0 = time.time(); R = {}; ctl = dict(checked=0, differ=0)
    for day in C.DAYS9:
        D = K.load_day(day); n = 0
        for w in K.window_sides(D):
            n += 1; side = w['side']; tf = w['tf']; cache = w.setdefault('cache', {})
            b0, valid, rs, kind = Q.runs(tf, w['P5'], w['opp'], side); bi = Q.fb_births(tf, b0, valid, w['ws'], w['te'], side)
            for q_, shf in CFG:
                tt = w['anc'] - shf; key = f'{q_}|{shf:.0f}'
                a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stv3'], w['P90'], w['best'], w['bopp'], tt, w['ptl'], w['shl'], q_, w['te'])
                for arm in ARMS:
                    f, p = FP[arm]; sch = C.schedule(tf, w['kf'], b0, bi, f, p)
                    if day == C.DAYS9[0] and n <= 40 and key == 'M|50':
                        F1, O1 = C.sim5(*a, sched=sch); F2, O2 = IDENT.sim5(*a, sched=sch)
                        ctl['checked'] += 1; ctl['differ'] += int(F1 != F2 or Q.otuples(O1, w['te']) != Q.otuples(O2, w['te']))
                    F, O = FIXED.sim5(*a, sched=sch); r = K.vals(w, F, tt)
                    e = R.setdefault(f'{arm}|{key}', {})
                    dd = e.setdefault(day, dict(v1_old=0.0, v1=0.0, q=0.0))
                    for ff, qq, m1 in zip(F, r['q'], r['m1']):
                        vv = float(qq * m1) if np.isfinite(m1) else 0.0; dd['v1'] += vv; dd['q'] += float(qq)
                        if ff[4] >= 100.0:
                            dd['v1_old'] += vv
            w.pop('cache', None)
        print(f'{day}: {n} window-sides, {time.time() - T0:.0f} s', flush=True)
        if ctl['differ']:
            sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    CS = {d: json.load(open(f'{K.IO}/cs_20261112/day_{d}.json')) for d in C.DAYS9}
    nd = sum(CS[d]['days'] for d in C.DAYS9); L = [f'CS post hoc: pending-cancel window fixed at 95 ms; control {ctl}; as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}']
    wo = lambda e: e['v1|100-500 ms'] + e['v1|>= 500 ms']
    for q_, shf in CFG:
        key = f'{q_}|{shf:.0f}'; jv = np.array([wo(CS[d]['acc'][f'J-v3|{key}']) / 100 / CS[d]['days'] for d in C.DAYS9])
        L.append(f'{key}: w/o-young $/day - J-v3 {jv.sum() * 0 + sum(wo(CS[d]["acc"][f"J-v3|{key}"]) for d in C.DAYS9) / 100 / nd:.2f}')
        for arm in ARMS:
            cs = sum(wo(CS[d]['acc'][f'{arm}|{key}']) for d in C.DAYS9) / 100 / nd
            fx = np.array([R[f'{arm}|{key}'][d]['v1_old'] / 100 / CS[d]['days'] for d in C.DAYS9])
            L.append(f'  {arm:<11} CS (55 + p window) {cs:>6.2f} | window 95 ms {fx.sum() * 0 + sum(R[f"{arm}|{key}"][d]["v1_old"] for d in C.DAYS9) / 100 / nd:>6.2f} '
                     f'| days above J-v3 {int(np.sum(fx > jv))} of 9')
    json.dump(dict(control=ctl, res=R), open(f'{OUT}/cs_posthoc.json', 'w'), default=float)
    open(f'{OUT}/cs_posthoc.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
