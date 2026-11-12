"""AGE, re-check of QR2 / GAP against B1a's young-order finding (2026-11-12, session ce68ebe0). USER: "The simulation has some
issues, profits are mainly from short-lived orders, but in practice it's hard for us to fill those orders, we have already checked
this previously, can you check that again".

B1a (20261110_b1a_live_orders.md) replayed live's own 11-08 orders through the fill model: of the replay's shares on orders under
100 ms old only 8.3% / 12.3% / 0.7% came from the taker orders that really filled live (M 0 / M 50 / M 250), against 54.8% / 28.6%
/ 7.7% at 100-500 ms and 46.8% / 47.7% / 22.0% at >= 500 ms ('precision' by age, reviews/b1a_20261110/b1a.json). This splits the
profit of QR2's arms (J-v3, Y, F30, Y+F30; QR2's code, pinned to QR2's committed day totals) by the order's age at the fill (replay
trade time - live time, B1a's bins) on the 9 consumed replay days, first 60 s, M 0 / M 50 / M 250. Post hoc diagnostic: the arms and
days are QR2's; nothing new is chosen.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule = >= 7 of 9 days):
 R310 YOUNG-SHARE (each arm, M 0 and M 50): the share of its $/day at 1 s from fills on orders < 100 ms old; YOUNG-DRIVEN if > 50%
      at both, else NOT-YOUNG-DRIVEN.
 R311 WITHOUT-YOUNG (each arm): $/day at 1 s from fills on orders >= 100 ms old > 0 by the day rule at M 0 and at M 50 -> SURVIVES;
      else FAILS. R311b the same with >= 500 ms.
 R312 GAIN (F30, Y and Y+F30, each vs J-v3): the difference in $/day at 1 s from fills on orders >= 100 ms old > 0 by the day rule at
      M 0 and at M 50 -> GAIN-SURVIVES; < 0 -> GAIN-REVERSES; else GAIN-IS-YOUNG.
 R313 B1a-WEIGHTED (each arm, M 0 and M 50): sum over the three age bins of the bin's $ x B1a's precision for that bin and config;
      pooled $/day printed; Y+F30 vs J-v3 on the weighted $/day by the day rule at both -> WEIGHTED-ABOVE / WEIGHTED-BELOW /
      NOT-SHOWN. (A scaling, not a model: B1a's rates are from another engine's orders on one day.)
 R314 vs TOP 5 (J-v3 and Y+F30): our c/sh at 1 s on fills on orders >= 100 ms old against the top 5's real at-best c/sh of the
      same day (GAP's day files) -> OURS-HIGHER / TOP-HIGHER by the day rule at M 0 and M 50, else MIXED.
Also printed: shares and c/sh per age bin, the cancel-pending part (fills after our cancel request; B1a found the replay credits
2.5x live's pending shares), and the finer split < 50 / 50-100 / 100-250 / 250-500 / 500-1000 / >= 1000 ms.
Controls (any failure -> no table): the arms reproduce QR2's committed day totals (fills, shares, value at 1 s); the age bins add up
to the totals; synthetic binning cases with a planted error (bin edges shifted by 1 ms) that must fail.
Usage: research/run_age.sh; python3 predictfun/research/pf_qr2_age_check.py selftest | day <yyyymmdd> | report
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as Q                              # noqa: E402
import pf_qr2_v3_early as Q2                              # noqa: E402

NAN = float('nan')
DAYS9 = K.DAYS9; OUTA = f'{K.IO}/age_20261112'; QR2D = f'{K.IO}/qr2_20261112'; GAPD = f'{K.IO}/gap_20261112'
CFG = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
ARMS = {'J-v3': dict(), 'Y': dict(y=1), 'F30': dict(f=1), 'Y+F30': dict(y=1, f=1)}
BINS = (('< 100 ms', 0.0, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, float('inf')))
FINE = (('< 50', -1e18, 50.0), ('50-100', 50.0, 100.0), ('100-250', 100.0, 250.0), ('250-500', 250.0, 500.0), ('500-1000', 500.0, 1000.0),
        ('>= 1000', 1000.0, float('inf')))
B1A = {'M|0': (0.08288286541654095, 0.5476125580415678, 0.46785435948727594), 'M|50': (0.12337820978184168, 0.2858571490112695, 0.47665862368157524),
       'M|250': (0.006996168670787272, 0.07718840565404715, 0.21964816762389805)}   # reviews/b1a_20261110/b1a.json configs[* P1].ages precision
MUT_EDGE = 0                                              # planted error (self-test only)


def agebin(a, bins=BINS):
    a = a + (1.0 if MUT_EDGE else 0.0)
    for nm, lo, hi in bins:
        if a < hi and (a >= lo or nm == bins[0][0]):
            return nm
    return bins[-1][0]


def selftest():
    bad = []
    for a, want in ((0.0, '< 100 ms'), (99.5, '< 100 ms'), (100.0, '100-500 ms'), (499.9, '100-500 ms'), (500.0, '>= 500 ms'), (-3.0, '< 100 ms')):
        if agebin(a) != want:
            bad.append(f'bin {a}')
    if agebin(49.5, FINE) != '< 50' or agebin(1000.0, FINE) != '>= 1000':
        bad.append('fine bins')
    return bad


def run_day(day):
    bad = selftest(); mut = {}
    globals()['MUT_EDGE'] = 1; mut['MUT_EDGE'] = len(selftest()) > 0; globals()['MUT_EDGE'] = 0
    if bad or not all(mut.values()) or Q2.selftest():
        sys.exit(f'SELFTEST FAILED -> no output: {bad} {mut}')
    os.makedirs(OUTA, exist_ok=True)
    D = K.load_day(day); st = D['stat']; T0 = time.time(); st.update(window_sides=0)
    q2 = json.load(open(f'{QR2D}/day_{day}.json'))['acc']
    A = {}
    for w in K.window_sides(D):
        st['window_sides'] += 1
        side = w['side']; tf = w['tf']; tfl = tf.tolist(); cache = w.setdefault('cache', {})
        b0, valid, rs, kind = Q.runs(tf, w['P5'], w['opp'], side)
        young = Q2.young_mask(w['tg'], w['jk'], tfl, rs, Q2.Y_MAX)
        bi = Q.fb_births(tf, b0, valid, w['ws'], w['te'], side)
        sch = [(float(tf[i] + Q2.D_F - Q2.L_PLACE), int(b0[i]), float(w['kf'][i])) for i in bi]
        for q_, shf in CFG:
            tt = w['anc'] - shf; key = f'{q_}|{shf:.0f}'
            a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stv3'], w['P90'], w['best'], w['bopp'], tt, w['ptl'], w['shl'], q_, w['te'])
            for arm, sw in ARMS.items():
                F, O = Q2.sim4(*a, allow=young if sw.get('y') else None, sched=sch if sw.get('f') else None)
                r = K.vals(w, F, tt)
                tot = A.setdefault(f'{arm}|{key}|ALL', dict(fills=0, q=0.0, v1=0.0))
                for f, qq, m1 in zip(F, r['q'], r['m1']):
                    vv = float(qq * m1) if np.isfinite(m1) else 0.0
                    labs = ['ALL', 'bin:' + agebin(f[4]), 'fine:' + agebin(f[4], FINE)] + (['pend', 'pend:' + agebin(f[4])] if f[5] == 'pend' else [])
                    for lab in labs:
                        e = A.setdefault(f'{arm}|{key}|{lab}', dict(fills=0, q=0.0, v1=0.0)); e['fills'] += 1; e['q'] += float(qq); e['v1'] += vv
                del tot
        w.pop('cache', None)
    bad = {}
    for arm in ARMS:
        for q_, shf in CFG:
            key = f'{q_}|{shf:.0f}'; e = A.get(f'{arm}|{key}|ALL', dict(fills=0, q=0.0, v1=0.0)); ref = q2[f'{arm}|{key}']
            if ref['fills'] != e['fills'] or abs(ref['q'] - e['q']) > 1e-6 or abs(ref['v1'] - e['v1']) > 1e-6:
                bad[f'qr2 {arm}|{key}'] = (ref['fills'], e['fills'], ref['v1'], e['v1'])
            sb = [A.get(f'{arm}|{key}|bin:{b[0]}', dict(fills=0, q=0.0, v1=0.0)) for b in BINS]
            if sum(x['fills'] for x in sb) != e['fills'] or abs(sum(x['v1'] for x in sb) - e['v1']) > 1e-6:
                bad[f'bins {arm}|{key}'] = 1
    if bad:
        sys.exit(f'AGE CONTROL FAILED on {day} -> no result: {bad}')
    json.dump(dict(day=day, days=len(D['W0']) / 288.0, stat=st, planted=mut, acc=A), open(f'{OUTA}/day_{day}.json', 'w'), default=float)
    print(f'{day}: done in {time.time() - T0:.0f} s; QR2 reproduced, bins add up', flush=True)


def report():
    DY = {d: json.load(open(f'{OUTA}/day_{d}.json')) for d in DAYS9}
    GP = {d: json.load(open(f'{GAPD}/day_{d}.json')) for d in DAYS9}
    nd = sum(DY[d]['days'] for d in DAYS9); L = []; P = L.append
    RES = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), days=DAYS9, b1a_precision=B1A)
    z = dict(fills=0, q=0.0, v1=0.0)
    g = lambda d, arm, key, lab: DY[d]['acc'].get(f'{arm}|{key}|{lab}', z)

    def pool(arm, key, lab):
        e = dict(fills=0, q=0.0, v1=0.0)
        for d in DAYS9:
            for k_, v_ in g(d, arm, key, lab).items():
                e[k_] += v_
        return e
    usd = lambda e: e['v1'] / 100.0 / nd
    cps = lambda e: e['v1'] / e['q'] if e['q'] > 0 else NAN
    dusd = lambda arm, key, labs: np.array([sum(g(d, arm, key, lb)['v1'] for lb in labs) / 100.0 / DY[d]['days'] for d in DAYS9])
    P(f'AGE re-check of QR2 / GAP against B1a (young orders) - 9 replay days, first 60 s; as of {RES["as_of"]}')
    P('controls passed per day (QR2 reproduced, bins add up); B1a precision by age (< 100 / 100-500 / >= 500 ms): '
      + '; '.join(f'{k_} ' + ' / '.join(f'{x:.1%}' for x in v_) for k_, v_ in B1A.items()))
    for q_, shf in CFG:
        key = f'{q_}|{shf:.0f}'
        P(''); P(f'{key}: $/day at 1 s (c/sh, shares/day) by order age at the fill; cancel-pending; B1a-weighted')
        for arm in ARMS:
            al = pool(arm, key, 'ALL'); bins = [pool(arm, key, f'bin:{b[0]}') for b in BINS]; pe = pool(arm, key, 'pend')
            wt = sum(usd(b) * p for b, p in zip(bins, B1A[key]))
            RES[f'{arm}|{key}'] = dict(all=al, bins={b[0]: x for b, x in zip(BINS, bins)}, pend=pe, weighted_usd=wt)
            P(f'  {arm:<6} all {usd(al):>7.2f} ({cps(al):+.2f}, {al["q"] / nd:.0f} sh) | ' + ' | '.join(
                f'{b[0]} {usd(x):>7.2f} ({cps(x):+.2f}, {x["q"] / nd:.0f})' for b, x in zip(BINS, bins))
              + f' | pending {usd(pe):.2f} ({pe["q"] / nd:.0f} sh) | B1a-weighted {wt:.2f}')
        if key == 'M|50':
            for arm in ARMS:
                P(f'    fine {arm}: ' + ', '.join(f'{f_[0]} {usd(pool(arm, key, "fine:" + f_[0])):.2f} ({cps(pool(arm, key, "fine:" + f_[0])):+.2f})' for f_ in FINE))
    P('')
    old1 = ['bin:100-500 ms', 'bin:>= 500 ms']; old5 = ['bin:>= 500 ms']
    RES['R310'] = {}; RES['R311'] = {}; RES['R311b'] = {}
    for arm in ARMS:
        sh = [usd(pool(arm, k_, 'bin:< 100 ms')) / usd(pool(arm, k_, 'ALL')) if usd(pool(arm, k_, 'ALL')) != 0 else NAN for k_ in ('M|0', 'M|50')]
        RES['R310'][arm] = dict(label='YOUNG-DRIVEN' if min(sh) > 0.5 else 'NOT-YOUNG-DRIVEN', share_M0_M50=sh)
        n1 = [int(np.sum(dusd(arm, k_, old1) > 0)) for k_ in ('M|0', 'M|50')]; n5 = [int(np.sum(dusd(arm, k_, old5) > 0)) for k_ in ('M|0', 'M|50')]
        RES['R311'][arm] = dict(label='SURVIVES' if min(n1) >= 7 else 'FAILS', days_pos=n1)
        RES['R311b'][arm] = dict(label='SURVIVES' if min(n5) >= 7 else 'FAILS', days_pos=n5)
        P(f'R310 {arm}: {RES["R310"][arm]["label"]} (share of $ from orders < 100 ms: M 0 {sh[0]:.0%}, M 50 {sh[1]:.0%}); '
          f'R311 without < 100 ms: {RES["R311"][arm]["label"]} (days > 0 {n1}); R311b only >= 500 ms: {RES["R311b"][arm]["label"]} (days > 0 {n5})')
    RES['R312'] = {}
    for arm in ('F30', 'Y', 'Y+F30'):
        up = [int(np.sum(dusd(arm, k_, old1) > dusd('J-v3', k_, old1))) for k_ in ('M|0', 'M|50')]
        dn = [int(np.sum(dusd(arm, k_, old1) < dusd('J-v3', k_, old1))) for k_ in ('M|0', 'M|50')]
        lab = 'GAIN-SURVIVES' if min(up) >= 7 else ('GAIN-REVERSES' if min(dn) >= 7 else 'GAIN-IS-YOUNG')
        RES['R312'][arm] = dict(label=lab, days_up=up, days_down=dn,
                                diff_usd_M0_M50=[float((dusd(arm, k_, old1) - dusd('J-v3', k_, old1)).mean()) for k_ in ('M|0', 'M|50')])
        P(f'R312 GAIN {arm} vs J-v3 without < 100 ms: {lab}; days above {up}, below {dn}; mean diff $/day M 0 / M 50 '
          + ' / '.join(f'{x:+.2f}' for x in RES['R312'][arm]['diff_usd_M0_M50']))
    wd = lambda arm, k_: np.array([sum(g(d, arm, k_, f'bin:{b[0]}')['v1'] / 100.0 / DY[d]['days'] * p for b, p in zip(BINS, B1A[k_])) for d in DAYS9])
    up = [int(np.sum(wd('Y+F30', k_) > wd('J-v3', k_))) for k_ in ('M|0', 'M|50')]; dn = [int(np.sum(wd('Y+F30', k_) < wd('J-v3', k_))) for k_ in ('M|0', 'M|50')]
    RES['R313'] = dict(label='WEIGHTED-ABOVE' if min(up) >= 7 else ('WEIGHTED-BELOW' if min(dn) >= 7 else 'NOT-SHOWN'), days_up=up, days_down=dn)
    P(f'R313 B1a-WEIGHTED Y+F30 vs J-v3: {RES["R313"]["label"]}; days above {up}, below {dn}')
    top = np.array([GP[d]['theirs']['top 5|ALL']['v1'] / GP[d]['theirs']['top 5|ALL']['q'] for d in DAYS9])
    RES['R314'] = {}
    for arm in ('J-v3', 'Y+F30'):
        oc = {k_: np.array([sum(g(d, arm, k_, lb)['v1'] for lb in old1) / max(sum(g(d, arm, k_, lb)['q'] for lb in old1), 1e-9) for d in DAYS9]) for k_ in ('M|0', 'M|50')}
        upo = [int(np.sum(oc[k_] > top)) for k_ in ('M|0', 'M|50')]; dno = [int(np.sum(oc[k_] < top)) for k_ in ('M|0', 'M|50')]
        lab = 'OURS-HIGHER' if min(upo) >= 7 else ('TOP-HIGHER' if min(dno) >= 7 else 'MIXED')
        RES['R314'][arm] = dict(label=lab, days_ours_higher=upo, days_top_higher=dno, ours_cps_M50_per_day=oc['M|50'].tolist(), top_cps_per_day=top.tolist())
        P(f'R314 {arm} (orders >= 100 ms) vs top 5: {lab}; days ours higher {upo}, top higher {dno}; ours M 50 per day '
          + ' '.join(f'{x:+.2f}' for x in oc['M|50']) + ' | top 5 ' + ' '.join(f'{x:+.2f}' for x in top))
    json.dump(RES, open(f'{OUTA}/age.json', 'w'), indent=1, default=float)
    open(f'{OUTA}/age.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        bad = selftest(); globals()['MUT_EDGE'] = 1; caught = len(selftest()) > 0; globals()['MUT_EDGE'] = 0
        print('failing:', bad, 'planted error caught:', caught); sys.exit(0 if not bad and caught else 1)
    elif cmd == 'day':
        run_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
