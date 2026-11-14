"""Shared client-state lane for repaired ORDER-HARM labels and policy (2026-11-14).

Engine derived from JOINT-LADDER v3 at eb735795. Its default execution and
client-decision rules are unchanged. Explicit rung offsets / active sides allow
a single reference rung to use this same engine; raw order states support label
recording. The completed cancellation tail defines the portfolio terminal.
The frozen v3 source and prior results remain unchanged.
"""
import bisect
import numpy as np
import pf_joint_ladder_v3 as J3

K = J3.K
SIZE, NR, L_NOTE, VIOL, INF = J3.SIZE, J3.NR, J3.L_NOTE, J3.VIOL, J3.INF
MUT_REF_RELEASE = MUT_REST_TRUE = MUT_HELD_TRUE = MUT_LEG_LEAK = MUT_PRIORITY = INDEP = 0
CONTRACT = 'order-harm-client-50ms-terminal-tail-v2'
DECISION_MS = 50
REPOST_MS = 0.0      # ORDER-HARM labels and replay: an empty rung re-posts at the first 50 ms step after the guard clears, no timer (USER 2026-11-14:
                     # the fixed 250 ms re-post timer does not match the current system); the engine default stays K.REST_COOL (frozen-v3 parity)
TAIL_MS = 200.0

def sim_joint(W, q, shf, cap, aux, decide=None, idle_only=0, size=SIZE, nr=NR, mode='soft', state='client', rung_offsets=None, active_sides=(0, 1),
              cooldown=None):
    """Both sides of one window. mode: 'soft' / 'hard' / 'none' (cap); state: 'client' (decisions on client-known state, the header) or
    'compat' (v1's venue-state decisions; regression control only). Returns per (s, d) fills / orders (sim2 tuple form), the decision log,
    stats incl. conservation violations."""
    offsets = tuple(range(nr)) if rung_offsets is None else tuple(rung_offsets)
    if len(offsets) != nr or any(int(d) != d or d < 0 for d in offsets):
        raise ValueError('invalid rung offsets')
    if not active_sides or any(s not in (0, 1) for s in active_sides):
        raise ValueError('invalid active sides')
    if mode == 'none':
        cap = None
    client = state == 'client'
    cool = K.REST_COOL if cooldown is None else float(cooldown)   # re-post wait after the last guard / pull step
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
                     cache=w.setdefault('cache', {}), live=[], cl=[], cur=[None] * nr, lastpull=[-INF] * nr, fills=[[] for _ in range(nr)],
                     orders=[[] for _ in range(nr)], aux=aux[s], dec=[(decide.get((s, d)) if decide else None) for d in range(nr)])
    stat = dict(pulls=[0, 0], cap_pulls=[0, 0], cap_steps=0, skip=[0, 0], inter=[0, 0], maxpos=0.0, hard_block=[0, 0],
                viol={v_: 0 for v_ in VIOL})
    pos = [0.0]; KN = []; fl = []; log = []

    def kfill(o, t):
        return sum(q_ for (tq_, q_) in o['fl'] if tq_ + L_NOTE <= t)

    def live_c(o, t):                                     # LIVE to the client at t (header)
        if o['ts'] > t:
            return False
        if o['ref'] and o['ta'] <= t:
            return False
        if o['ref'] and MUT_REF_RELEASE:
            return False
        if kfill(o, t) >= size - 1e-9:
            return False
        if o['tc'] < INF and o['tr'] + L_NOTE <= t:
            return False
        return True

    def lv(s, pt):
        c = ST[s]['cache']
        if pt not in c:
            P5, S5 = ST[s]['P5'], ST[s]['S5']; last = P5[:, 4]
            c[pt] = np.where((last >= 0) & ((pt < last) if s == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return c[pt]

    def place(s, d, pt, t):
        ta = t + K.L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1; opp = ST[s]['opp']
        o = dict(pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=size, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, ref=False, rung=d, fl=[], side=s)
        if fi < 0 or ((0 <= opp[fi] <= pt) if s == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta                 # the venue's refusal at ta (the client learns it at ta)
        else:
            a = lv(s, pt)[fi]; a = float(a) if np.isfinite(a) else 0.0
            if fi + 1 < nf and tfl[fi + 1] - ta <= 100.0:
                d2 = lv(s, pt)[fi + 1]
                if np.isfinite(d2) and d2 > a:
                    a = float(d2)
            o['a0'] = a; o['ahead'] = 0.0 if q == 'U' else a
            if not INDEP and any(x['pt'] == pt and x['rem'] > 1e-12 and x['tr'] > ta for x in ST[s]['live']):
                stat['viol']['collision'] += 1            # venue: two of our orders resting at one price at once
            ST[s]['live'].append(o)
        ST[s]['orders'][d].append(o); ST[s]['cl'].append(o)
        log.append((t, s, d, 'place', pt))
        return o

    def cancel(o, t, why):
        if o['tc'] == INF:
            log.append((t, o['side'], o['rung'], f'cancel:{why}', o['pt']))
        o['tc'] = min(o['tc'], t)
        o['tr'] = min(o['tr'], max(t + K.L_CANCEL, o['ts'] + K.L_PENDCANCEL) if t < o['ta'] else t + K.L_CANCEL)

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
        if MUT_PRIORITY:
            el = el[::-1]
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
            lft = full if (INDEP or MUT_LEG_LEAK) else left
            if p == o['pt']:
                use = min(lft, o['ahead']); o['ahead'] -= use; qq = min(lft - use, o['rem'])
            elif thr:
                qq = min(lft, o['rem']); o['ahead'] = 0.0
            else:
                qq = 0.0
            if qq > 1e-12:
                if not (o['ta'] <= t < o['tr']):
                    stat['viol']['live'] += 1
                left -= qq; o['rem'] -= qq; got += qq; filled.append(o)
                g = bisect.bisect_right(taul, t) - 1
                ph = 'pend' if t >= o['tc'] else ('above' if g >= 0 and S_['v'][g] > -INF else 'hold')
                S_['fills'][o['rung']].append((i, o['pt'], qq, thr, t - o['ta'], ph, o['a0']))
                o['fl'].append((t, qq))
                sq = qq if s == 0 else -qq
                pos[0] += sq; KN.append((t + L_NOTE, sq)); fl.append((s, o['rung'], t, o['pt'], qq))
                stat['maxpos'] = max(stat['maxpos'], abs(pos[0]))
                if cap is not None and mode == 'hard' and abs(pos[0]) > cap + 1e-9:
                    stat['viol']['cap'] += 1
                if o['rem'] <= 1e-12:
                    o['tfill'] = t
        if not INDEP and got > full + 1e-9:
            stat['viol']['leg'] += 1
        if not INDEP:
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
        for s in active_sides:
            S_ = ST[s]; b = S_['best'][k]
            for d in range(nr):
                offset = offsets[d]
                cur = S_['cur'][d]
                if cur is not None:
                    if client:
                        done = (cur['ref'] and cur['ta'] <= t) or kfill(cur, t) >= size - 1e-9
                    else:
                        done = (cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + K.L_FILL <= t)
                    if done:
                        cur = None
                trig = bool(S_['aux'][k])
                if client and not MUT_REST_TRUE:
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
                        if beh > K.HOLD_K + offset:
                            why = 'drift'
                        elif not ((cur['pt'] < S_['bopp'][k]) if s == 0 else (cur['pt'] > S_['bopp'][k])):
                            why = 'cross'
                    if why:
                        cancel(cur, t, why); cur = None
                if trig:
                    S_['lastpull'][d] = t
                if cur is None and b >= 0 and not trig and t - S_['lastpull'][d] >= cool:
                    tgt = int(b) if offset == 0 else ((int(b) - offset) if s == 0 else (int(b) + offset))
                    if offset == 0 or 1 <= tgt <= 99:
                        if client and not MUT_HELD_TRUE:
                            held = (not INDEP) and any(o['pt'] == tgt and live_c(o, t) for o in S_['cl'])
                        else:
                            held = (not INDEP) and any(o['pt'] == tgt and o['rem'] > 1e-12 and o['tr'] > t for o in S_['live'])
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
    while it < len(TL) and TL[it][0] <= t_end + TAIL_MS:
        apply(*TL[it]); it += 1
    for s in (0, 1):
        for d in range(nr):
            for o in ST[s]['orders'][d]:
                tot = sum(qq for (_, qq) in o['fl'])
                if tot > size + 1e-9 or abs((size - tot) - o['rem']) > 1e-9:
                    stat['viol']['order'] += 1
    if abs(pos[0] - sum((x[4] if x[0] == 0 else -x[4]) for x in fl)) > 1e-6:
        stat['viol']['position'] += 1
    return dict(fills={(s, d): ST[s]['fills'][d] for s in (0, 1) for d in range(nr)},
                orders={(s, d): [(o['ta'], min(o['tr'], o['tfill'], t_end + TAIL_MS), o['a0'], o['ref']) for o in ST[s]['orders'][d]]
                        for s in (0, 1) for d in range(nr)},
                cancels={(s, d): [o['tc'] for o in ST[s]['orders'][d]] for s in (0, 1) for d in range(nr)},
                log=log, stat=stat, endpos=pos[0], fl=fl, terminal_time=t_end + TAIL_MS,
                order_states={(s, d): ST[s]['orders'][d] for s in (0, 1) for d in range(nr)})


def reference_lane(w, q, shf, rung, guard, size=SIZE, start_k=0, end_time=None,
                   decide=None, state='client', cooldown=None):
    """One reference rung, using the policy engine and its client-known state.

    A fork starts without an order at start_k. Both KEEP and PULL use this
    function; guard-on blocks entry. cooldown: the re-post wait after the last guard / pull step (None = the engine's 250 ms;
    ORDER-HARM passes REPOST_MS = 0: re-post at the first step after the guard clears).
    """
    end = float(w['te'] if end_time is None else end_time)
    stop = int(np.searchsorted(w['tg'], end, side='left'))
    if not (0 <= start_k < stop):
        raise ValueError('empty or reversed reference interval')
    own = dict(w)
    for key in ('tg', 'vkl', 'best', 'bopp'):
        own[key] = w[key][start_k:stop]
    own['te'] = end
    # The inactive side cannot place or fill; its frame/grid clock is shared.
    other = dict(own, anc=np.zeros(0), ptl=[], shl=[], cache={})
    side = int(w['side']); W = {side: own, 1 - side: other}
    vg = list(guard[start_k:stop]); aux = {side: vg, 1 - side: [False] * len(vg)}
    callback = None if decide is None else {(side, 0): lambda k, t, o: decide(k + start_k, t, o)}
    r = sim_joint(W, q, shf, None, aux, decide=callback, idle_only=1,
                  size=size, nr=1, mode='none', state=state,
                  rung_offsets=(rung,), active_sides=(side,), cooldown=cooldown)
    if any(r['stat']['viol'].values()):
        raise ValueError(f"reference accounting failed: {r['stat']['viol']}")
    return r['fills'][(side, 0)], r['orders'][(side, 0)], r['order_states'][(side, 0)], r


def portfolio_terminal(r, tf, mid):
    """Cash plus inventory after every executable cancellation has landed.

    All fills are included at their execution times. A post-terminal fill,
    still-executable order, missing mark or position mismatch fails closed.
    """
    terminal = float(r['terminal_time'])
    if any(t > terminal for _, _, t, _, _ in r['fl']):
        raise ValueError('post-terminal fill in terminal portfolio')
    for orders in r['order_states'].values():
        if any(not o['ref'] and o['rem'] > 1e-9 and o['tr'] > terminal for o in orders):
            raise ValueError('terminal reached with an executable/pending order')
    j = int(np.searchsorted(tf, terminal, side='right')) - 1
    if j < 0 or not np.isfinite(mid[j]):
        raise ValueError('no finite terminal mark')
    mark = float(mid[j])
    position = sum(q if s == 0 else -q for s, _, _, _, q in r['fl'])
    cash = sum((-q if s == 0 else q) * pt / 100.0 for s, _, _, pt, q in r['fl'])
    if abs(position - r['endpos']) > 1e-6:
        raise ValueError('terminal position differs from signed fills')
    pnl = cash + position * mark / 100.0
    return dict(time=terminal, mark=mark, cash=cash, position=position, pnl=pnl)


def stamp():
    return dict(contract=CONTRACT, decision_ms=DECISION_MS, repost_ms=REPOST_MS)


def require_contract(receipt):
    if any(receipt.get(k) != v for k, v in stamp().items()):
        raise ValueError('incompatible ORDER-HARM artifact: rebuild client-state / 50 ms pipeline')
