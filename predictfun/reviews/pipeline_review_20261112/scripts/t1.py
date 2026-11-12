import sys; sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import numpy as np
import pf_kp_s4 as K
INF = float('inf')
def mk(fr):
    tf = np.array([f[0] for f in fr], float); n = len(fr); PB = np.full((n, 5), -1); SB = np.zeros((n, 5))
    for i, f in enumerate(fr):
        for j, (p, z) in enumerate(f[1][:5]): PB[i, j] = p; SB[i, j] = z
    return tf, PB, SB, np.array([f[2] for f in fr])
# T1: rest lane, our order alone at 50 (shown 0), full fill at t=60, known at 160. A callback that pulls at step 3 (t=150).
fr = [(-100, [(50, 0.0)], 52)]
tf, PB, SB, ba = mk(fr)
tau = np.arange(0.0, 1000.0, 50.0); n = len(tau)
tr = [(60, 50, 5), (300, 50, 5)]
tt = np.array([x[0] for x in tr], float); tp = [x[1] for x in tr]; tq = [float(x[2]) for x in tr]
calls = []
def cb(k, t, cur):
    calls.append((k, t, cur['rem'])); return k == 3
K.KP_DECIDE = cb
F, O = K.sim2('rest', 0, tf, ba, PB, SB, {}, tau, [1.0]*n, [True]*n, -INF, [50]*n, [52]*n, tt, tp, tq, 'M', 1000.0, aux=[False]*n)
K.KP_DECIDE = None
print('T1 callback consulted at steps:', [c[:2] for c in calls][:6])
print('T1 fills:', [(tt[f[0]], f[2]) for f in F], 'orders (live, gone):', [(o[0], o[1]) for o in O][:3])
# what live would do: the same callback fires at k=3 (fill not yet known) -> cancel + 250 ms cooldown -> no order at t=300
aux = [False]*n; aux[3] = True
F2, O2 = K.sim2('rest', 0, tf, ba, PB, SB, {}, tau, [1.0]*n, [True]*n, -INF, [50]*n, [52]*n, tt, tp, tq, 'M', 1000.0, aux=aux)
print('T1 with the pull applied at k=3 regardless of the true fill state:', [(tt[f[0]], f[2]) for f in F2], [(o[0], o[1]) for o in O2][:3])

# T2: activation frame stamped exactly at ta (= trade's anchor at M 0)
fr = [(-100, [(50, 100.0)], 52), (40, [(50, 60.0)], 52)]   # frame at 40 already shows the 40-sh trade at 40
tf, PB, SB, ba = mk(fr)
tau = np.arange(0.0, 200.0, 50.0); n = len(tau)
tr = [(40, 50, 40), (45, 50, 30)]
tt = np.array([x[0] for x in tr], float); tp = [x[1] for x in tr]; tq = [float(x[2]) for x in tr]
F, O = K.sim2('join', 0, tf, ba, PB, SB, {}, tau, [1.0]*n, [True]*n, 0.0, [50]*n, [52]*n, tt, tp, tq, 'M', 200.0)
print('T2 frame at ta: fills', [(tt[f[0]], f[2]) for f in F], 'a0', [o[2] for o in O])
fr = [(-100, [(50, 100.0)], 52), (41, [(50, 60.0)], 52)]   # same, frame stamped 1 ms later
tf, PB, SB, ba = mk(fr)
tr = [(41, 50, 40), (45, 50, 30)]
tt = np.array([x[0] for x in tr], float); tp = [x[1] for x in tr]; tq = [float(x[2]) for x in tr]
F, O = K.sim2('join', 0, tf, ba, PB, SB, {}, tau, [1.0]*n, [True]*n, 0.0, [50]*n, [52]*n, tt, tp, tq, 'M', 200.0)
print('T2 frame at ta+1: fills', [(tt[f[0]], f[2]) for f in F], 'a0', [o[2] for o in O])
