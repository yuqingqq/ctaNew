"""STOP replay lane with activation-time queue initialization.

Versioned from pf_stop_model.sim_joint_g at e315abe8. Only queue initialization
and per-order provenance differ; the original module remains available for
its ongoing frozen experiment. Future STOP-EV/replay code should import this
sim_joint_g explicitly and write a new output namespace. Existing movement
predictions are reusable; policy calibration and replay must be rerun.
"""
import bisect
import numpy as np
import pf_stop_model as M
from pf_queue_activation import initial_queue, CONTRACT as QUEUE_CONTRACT

def sim_joint_g(W, q, shf, cap, aux, decide=None, gate=None, idle_only=0, size=None, nr=None, mode='soft', state='client', cooldown=None):
    """STOP lane API with causal queue initialization and unchanged gate semantics.

    M.QUNK remains the caller's explicit unknown-depth setting. Comparison with
    historical CL.sim_joint is valid only where initialization is unchanged.
    """
    CL = M.CL
    size = CL.SIZE if size is None else size; nr = CL.NR if nr is None else nr
    K_, L_NOTE, INF_ = CL.K, CL.L_NOTE, CL.INF
    offsets = tuple(range(nr))
    if mode == 'none':
        cap = None
    client = state == 'client'
    cool = K_.REST_COOL if cooldown is None else float(cooldown)
    taul = W[0]['tg'].tolist(); tfl = W[0]['tf'].tolist(); nf = len(tfl); t_end = W[0]['te']
    TL = []
    for s in (0, 1):
        tt = (W[s]['anc'] - shf).tolist()
        TL += [(tt[i], s, i) for i in range(len(tt))]
    TL.sort(key=lambda x: (x[0], x[1], x[2]))
    ST = {}
    for s in (0, 1):
        w = W[s]
        ST[s] = dict(P5=w['P5'], S5=w['S5'], opp=w['opp'], best=w['best'], bopp=w['bopp'], v=w['vkl'], ptl=w['ptl'], shl=w['shl'],
                     cache=w.setdefault('cache', {}), live=[], cl=[], cur=[None] * nr, lastpull=[-INF_] * nr, fills=[[] for _ in range(nr)], fq=[[] for _ in range(nr)],
                     orders=[[] for _ in range(nr)], aux=aux[s], dec=[(decide.get((s, d)) if decide else None) for d in range(nr)],
                     gate=[((gate.get((s, d)) if gate else None) if not M.MUT_LANE_NOGATE else None) for d in range(nr)])
    stat = dict(pulls=[0, 0], cap_pulls=[0, 0], cap_steps=0, skip=[0, 0], inter=[0, 0], maxpos=0.0, hard_block=[0, 0], gated=[0, 0],
                viol={v_: 0 for v_ in CL.VIOL})
    pos = [0.0]; KN = []; fl = []; log = []

    def kfill(o, t):
        return sum(q_ for (tq_, q_) in o['fl'] if tq_ + L_NOTE <= t)

    def live_c(o, t):
        if o['ts'] > t:
            return False
        if o['ref'] and o['ta'] <= t:
            return False
        if kfill(o, t) >= size - 1e-9:
            return False
        if o['tc'] < INF_ and o['tr'] + L_NOTE <= t:
            return False
        return True

    def lv(s, pt):
        c = ST[s]['cache']
        if pt not in c:
            P5, S5 = ST[s]['P5'], ST[s]['S5']; last = P5[:, 4]
            c[pt] = np.where((last >= 0) & ((pt < last) if s == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return c[pt]

    def place(s, d, pt, t):
        ta = t + K_.L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1; opp = ST[s]['opp']
        o = dict(pt=pt, ts=t, ta=ta, tr=INF_, tc=INF_, rem=size, ahead=0.0, ci=fi, tfill=INF_, a0=np.nan, ref=False, rung=d, fl=[], side=s)
        if fi < 0 or ((0 <= opp[fi] <= pt) if s == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta
        else:
            a, ahead, unk = initial_queue(lv(s, pt)[fi], q, M.QUNK)
            o['a0'] = a; o['ahead'] = ahead; o['qunk'] = bool(unk)
            o['queue_snapshot_time'] = float(tfl[fi])
            o['queue_initialization'] = QUEUE_CONTRACT
            if any(x['pt'] == pt and x['rem'] > 1e-12 and x['tr'] > ta for x in ST[s]['live']):
                stat['viol']['collision'] += 1
            ST[s]['live'].append(o)
        ST[s]['orders'][d].append(o); ST[s]['cl'].append(o)
        log.append((t, s, d, 'place', pt))
        return o

    def cancel(o, t, why):
        if o['tc'] == INF_:
            log.append((t, o['side'], o['rung'], f'cancel:{why}', o['pt']))
        o['tc'] = min(o['tc'], t)
        o['tr'] = min(o['tr'], max(t + K_.L_CANCEL, o['ts'] + K_.L_PENDCANCEL) if t < o['ta'] else t + K_.L_CANCEL)

    def exposure(s, t, kpos):
        e = kpos if s == 0 else -kpos
        for o in ST[s]['cl']:
            if live_c(o, t):
                e += size - kfill(o, t)
        return e

    def apply(t, s, i):
        S_ = ST[s]; p = S_['ptl'][i]; left = S_['shl'][i]; full = left; got = 0.0
        el = [o for o in S_['live'] if o['rem'] > 1e-12 and o['ta'] <= t < o['tr']]
        el.sort(key=lambda o: ((-o['pt']) if s == 0 else o['pt'], o['ta']))
        reach = [o for o in el if o['pt'] == p or ((p < o['pt']) if s == 0 else (p > o['pt']))]
        if len(reach) >= 2:
            stat['inter'][s] += 1
        filled = []
        for o in el:
            if q == 'M':
                j = bisect.bisect_left(tfl, t) - 1
                if j > o['ci']:
                    seg = lv(s, o['pt'])[o['ci'] + 1:j + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg) and seg.min() < o['ahead']:
                        o['ahead'] = float(seg.min())
                    o['ci'] = j
            thr = (p < o['pt']) if s == 0 else (p > o['pt'])
            if p == o['pt']:
                use = min(left, o['ahead']); o['ahead'] -= use; qq = min(left - use, o['rem'])
            elif thr:
                qq = min(left, o['rem']); o['ahead'] = 0.0
            else:
                qq = 0.0
            if qq > 1e-12:
                if not (o['ta'] <= t < o['tr']):
                    stat['viol']['live'] += 1
                left -= qq; o['rem'] -= qq; got += qq; filled.append(o)
                g = bisect.bisect_right(taul, t) - 1
                ph = 'pend' if t >= o['tc'] else ('above' if g >= 0 and S_['v'][g] > -INF_ else 'hold')
                S_['fills'][o['rung']].append((i, o['pt'], qq, thr, t - o['ta'], ph, o['a0'])); S_['fq'][o['rung']].append(o.get('qunk', False))
                o['fl'].append((t, qq))
                sq = qq if s == 0 else -qq
                pos[0] += sq; KN.append((t + L_NOTE, sq)); fl.append((s, o['rung'], t, o['pt'], qq))
                stat['maxpos'] = max(stat['maxpos'], abs(pos[0]))
                if cap is not None and mode == 'hard' and abs(pos[0]) > cap + 1e-9:
                    stat['viol']['cap'] += 1
                if o['rem'] <= 1e-12:
                    o['tfill'] = t
        if got > full + 1e-9:
            stat['viol']['leg'] += 1
        for o in filled:
            for a in reach:
                if a is not o and ((a['pt'] > o['pt']) if s == 0 else (a['pt'] < o['pt'])) and a['rem'] > 1e-12:
                    stat['viol']['priority'] += 1

    it = 0; ik = 0; kpos = 0.0
    for k, t in enumerate(taul):
        while it < len(TL) and TL[it][0] <= t:
            apply(*TL[it]); it += 1
        while ik < len(KN) and KN[ik][0] <= t:
            kpos += KN[ik][1]; ik += 1
        capside = -1
        if mode == 'soft' and cap is not None and abs(kpos) >= cap - 1e-9:
            capside = 0 if kpos > 0 else 1
            stat['cap_steps'] += 1
        for s in (0, 1):
            S_ = ST[s]; b = S_['best'][k]
            for d in range(nr):
                offset = offsets[d]
                cur = S_['cur'][d]
                if cur is not None:
                    if client:
                        done = (cur['ref'] and cur['ta'] <= t) or kfill(cur, t) >= size - 1e-9
                    else:
                        done = (cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + K_.L_FILL <= t)
                    if done:
                        cur = None
                trig = bool(S_['aux'][k])
                if client:
                    rest_ = cur is not None and live_c(cur, t) and cur['ta'] <= t and cur['tc'] > t
                else:
                    rest_ = cur is not None and not cur['ref'] and cur['ta'] <= t and cur['rem'] > 1e-12
                if idle_only and rest_:
                    trig = False
                if S_['dec'][d] is not None and rest_ and not trig:
                    trig = bool(S_['dec'][d](k, t, cur))
                if capside == s:
                    if rest_ and not trig:
                        stat['cap_pulls'][s] += 1
                    trig = True
                if trig and rest_:
                    stat['pulls'][s] += 1
                if cur is not None:
                    why = ''
                    if trig:
                        why = 'trig'
                    elif b < 0:
                        why = 'nobook'
                    else:
                        beh = (b - cur['pt']) if s == 0 else (cur['pt'] - b)
                        if beh > K_.HOLD_K + offset:
                            why = 'drift'
                        elif not ((cur['pt'] < S_['bopp'][k]) if s == 0 else (cur['pt'] > S_['bopp'][k])):
                            why = 'cross'
                    if why:
                        cancel(cur, t, why); cur = None
                if trig:
                    S_['lastpull'][d] = t
                if cur is None and b >= 0 and not trig and t - S_['lastpull'][d] >= cool:
                    if S_['gate'][d] is not None and S_['gate'][d](k):
                        stat['gated'][s] += 1
                    else:
                        tgt = int(b) if offset == 0 else ((int(b) - offset) if s == 0 else (int(b) + offset))
                        if offset == 0 or 1 <= tgt <= 99:
                            if client:
                                held = any(o['pt'] == tgt and live_c(o, t) for o in S_['cl'])
                            else:
                                held = any(o['pt'] == tgt and o['rem'] > 1e-12 and o['tr'] > t for o in S_['live'])
                            if held:
                                stat['skip'][s] += 1; log.append((t, s, d, 'held', tgt))
                            elif mode == 'hard' and cap is not None and exposure(s, t, kpos) + size > cap + 1e-9:
                                stat['hard_block'][s] += 1; log.append((t, s, d, 'cap', tgt))
                            else:
                                cur = place(s, d, tgt, t)
                S_['cur'][d] = cur
            lv_ = S_['live']
            if lv_ and (lv_[0]['tr'] <= t or lv_[0]['rem'] <= 1e-12):
                S_['live'] = [o for o in lv_ if o['rem'] > 1e-12 and o['tr'] > t]
            if len(S_['cl']) > 24:
                S_['cl'] = [o for o in S_['cl'] if live_c(o, t)]
    while it < len(TL) and TL[it][0] <= t_end:
        apply(*TL[it]); it += 1
    for s in (0, 1):
        for d in range(nr):
            if ST[s]['cur'][d] is not None:
                cancel(ST[s]['cur'][d], t_end, 'end')
    while it < len(TL) and TL[it][0] <= t_end + CL.TAIL_MS:
        apply(*TL[it]); it += 1
    for s in (0, 1):
        for d in range(nr):
            for o in ST[s]['orders'][d]:
                tot = sum(qq for (_, qq) in o['fl'])
                if tot > size + 1e-9 or abs((size - tot) - o['rem']) > 1e-9:
                    stat['viol']['order'] += 1
    if abs(pos[0] - sum((x[4] if x[0] == 0 else -x[4]) for x in fl)) > 1e-6:
        stat['viol']['position'] += 1
    return dict(fills={(s, d): ST[s]['fills'][d] for s in (0, 1) for d in range(nr)}, fill_qunk={(s, d): ST[s]['fq'][d] for s in (0, 1) for d in range(nr)},
                orders={(s, d): [(o['ta'], min(o['tr'], o['tfill'], t_end + CL.TAIL_MS), o['a0'], o['ref']) for o in ST[s]['orders'][d]]
                        for s in (0, 1) for d in range(nr)},
                cancels={(s, d): [o['tc'] for o in ST[s]['orders'][d]] for s in (0, 1) for d in range(nr)},
                log=log, stat=stat, endpos=pos[0], fl=fl, terminal_time=t_end + CL.TAIL_MS,
                order_states={(s, d): ST[s]['orders'][d] for s in (0, 1) for d in range(nr)})
