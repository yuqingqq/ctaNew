"""EXIT (2026-11-13, session ce68ebe0; USER: "how do we bridge the gap between our current strategy and to maker's" / "Yes" to the tape
check of the top makers' exits): do the top makers leave a price just before a trade takes the whole level, more than other makers?

Public tape only (real maker legs and real book frames; no replay fill model), 9 consumed replay days (10-23..27, 10-31, 11-01..03),
first 60 s of each window, WS-timed taker orders only (anchors exact; E2-timed legs only mark ambiguity).
  taker order     legs grouped by the taker order (legs_tk); its time t = its WS anchor; on one side, the prices it filled
  level-taking    a price p the taker order went through: it filled p and a worse price (bids: lower; asks: higher), so it took every
                  order resting at p (price-time priority, B12b R42). Event E = (taker order, p, t0)
  presence        maker x is the only maker filled at p by a taker order that STOPPED at p (p was its last price), and the first
                  book frame stamped after that trade still shows p on the side with size > 0: x's order had size left at p at t1
  observation     (E, x): x's last WS-timed fill at (p, side) before t0 is such a presence record, t0 - t1 <= 1000 ms, and x has no
                  E2-timed leg at (p, side) in [t1 - 250, t0 + 250] ms (else the status 'e2_ambiguous')
  outcome         STAYED if x has a leg at p in E (a cancel-pending order counts: it was live), else GONE (x's order left p in
                  (t1, t0): cancelled or moved)
  groups          makers ranked by shares in 0-240 s per day (B12's rank_groups): top 5 / ranks 6-20 / rest; field = ranks 6+
  P               FRONT-PULL's per-day model of 'the level goes in (t + 24, t + 274] ms' (fitted without the day), at the last 50 ms
                  grid step <= t0 - 24 ms (the last moment a pull could still land); undefined -> status
Readings DECLARED BEFORE THE FIRST RUN (evaluated in code):
 R390 EXIT: the GONE share, top 5 minus field, matched on the gap t0 - t1 (bins 0-100 / 100-250 / 250-1000 ms, weighted by the top 5's
      observations per bin): >= +5 points AND the top 5 higher on >= 7 of 9 days AND above the null p99 -> TOP-LEAVE-MORE; <= -5
      points AND lower on >= 7 days AND below the null p1 -> TOP-LEAVE-LESS; else NOT-SHOWN. Null: 500 permutations of the group label
      within day x gap bin.
 R391 TIMING (descriptive): the GONE share by gap bin and group.
 R392 SIGNAL: within each group, the GONE share in the top quintile of P minus the bottom quintile (quintiles over all observations):
      SIGNAL-USED if the top 5's difference is >= +10 points and larger than the field's; else NOT-SHOWN.
 R393 SIZE (descriptive): shares per maker at p in level-taking trades by group; each group's share of level-taken volume against its
      share of sole-maker stopping volume.
Controls (any failure -> no output): synthetic cases with known answers (clearing classification on both sides, a single-price taker
order, sole-maker detection, STAYED / GONE, the 1000 ms limit, no presence after t0, the level-present check), three planted errors
each caught (the stop price counted as taken, multi-maker stops accepted, presence after t0 accepted); on the data: every event has a
maker at p, no observation uses a record after t0 (asserted), the null's mean difference within 1 point of 0.
Limits: consumed days, first minute; a maker can hold several orders at one price (GONE needs all of them gone); presence requires a
sole-maker stop, so the population leans to small trades and front orders; a level refilled inside one frame interval can pass the
level-present check; the venue's leg order inside a taker order is not used.
Usage (run_light, PF_SLOT=claude): pf_exit_check.py selftest | day <yyyymmdd> | report; driver run_exit.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402  (day loader, window contexts)
import pf_gap_top_makers as GP                            # noqa: E402  (per-day maker ranks)

NAN = float('nan')
DAYS9 = K.DAYS9
OUTE = f'{K.IO}/exit_20261113'
FPDIR = f'{K.IO}/front_pull_20261113'
W_MAX = 1000.0; E2_TOL = 250.0; L_PULL = 24.0
GAPB = (('0-100 ms', 0.0, 100.0), ('100-250 ms', 100.0, 250.0), ('250-1000 ms', 250.0, 1000.0))
NULL_N = 500
MEM_CAP_MIB = 2048.0
MUT_STOP_TAKEN = MUT_MULTI_SOLE = MUT_FUTURE = 0          # planted errors (self-test only)


def classify(side, legs):
    """legs: [(signer, price, shares)] of one taker order on one side. Returns (taken prices, stop price, makers at the stop)."""
    ps = sorted({p for _, p, _ in legs}, reverse=(side == 0))  # bids are hit from the highest down, asks from the lowest up
    stop = ps[-1]
    taken = ps if MUT_STOP_TAKEN else ps[:-1]
    return set(taken), stop, {s for s, p, _ in legs if p == stop}


def observations(side, orders, present_after, e2_legs):
    """orders: [(t, legs)] time-sorted WS-timed taker orders of one window-side; present_after(p, t) -> bool (the first frame after t
    shows p with size); e2_legs: {(signer, p): [t]} E2-timed legs. Returns observations and event / record counts."""
    rec = {}                                              # (signer, p) -> [(t, is_sole_stop_presence)] of WS-timed fills
    obs = []; nev = 0; npres = 0; amb = 0
    for t, legs in orders:
        taken, stop, at_stop = classify(side, legs)
        for p in taken:                                   # level-taking events at p, before this order's own fills are recorded
            nev += 1; inE = {s for s, q, _ in legs if q == p}
            if not inE:
                raise ValueError('EXIT: a level-taking event without a maker at p')
            for (s, q), lst in rec.items():
                if q != p or not lst:
                    continue
                cand = lst if MUT_FUTURE else [r for r in lst if r[0] < t]
                if not cand:
                    continue
                t1, pres = cand[-1]
                if not pres or t - t1 > W_MAX or t1 > t:
                    continue
                if any(t1 - E2_TOL <= x <= t + E2_TOL for x in e2_legs.get((s, p), ())):
                    amb += 1; continue
                obs.append(dict(signer=s, p=p, t0=t, t1=t1, gap=t - t1, stayed=int(s in inE)))
        sole = len(at_stop) == 1 or (MUT_MULTI_SOLE and len(at_stop) >= 1)
        pres_ok = sole and present_after(stop, t)
        for s, p, _ in legs:
            rec.setdefault((s, p), []).append((t, bool(pres_ok and p == stop and s in at_stop)))
        npres += int(pres_ok)
    return obs, nev, npres, amb


def selftest():
    bad = []

    def chk(n, c):
        if not c:
            bad.append(n)
    chk('bid clear', classify(0, [(1, 50, 5), (2, 49, 3)])[:2] == ({50}, 49))
    chk('ask clear', classify(1, [(1, 51, 5), (2, 52, 3)])[:2] == ({51}, 52))
    chk('single price', classify(0, [(1, 50, 5), (2, 50, 3)])[:2] == (set(), 50))
    chk('stop makers', classify(0, [(1, 50, 5), (2, 49, 3), (3, 49, 1)])[2] == {2, 3})
    yes = lambda p, t: True
    # maker 7 proven at 50 (sole stop at t = 100), then a taker order through 50 at t = 300 with 7 in it -> STAYED
    o, nev, npres, _ = observations(0, [(100.0, [(7, 50, 2)]), (300.0, [(7, 50, 10), (8, 49, 1)])], yes, {})
    chk('stayed', len(o) == 1 and o[0]['stayed'] == 1 and o[0]['gap'] == 200.0 and nev == 1)
    o, _, _, _ = observations(0, [(100.0, [(7, 50, 2)]), (300.0, [(9, 50, 10), (8, 49, 1)])], yes, {})
    chk('gone', len(o) == 1 and o[0]['stayed'] == 0)
    o, _, _, _ = observations(0, [(100.0, [(7, 50, 2)]), (1200.0, [(9, 50, 10), (8, 49, 1)])], yes, {})
    chk('1000 ms limit', len(o) == 0)
    o, _, _, _ = observations(0, [(100.0, [(7, 50, 2), (6, 50, 2)]), (300.0, [(9, 50, 10), (8, 49, 1)])], yes, {})
    chk('multi-maker stop is not presence', len(o) == 0)
    o, _, _, _ = observations(0, [(100.0, [(7, 50, 2)]), (300.0, [(9, 50, 10), (8, 49, 1)])], lambda p, t: False, {})
    chk('level gone after the stop is not presence', len(o) == 0)
    o, _, _, a = observations(0, [(100.0, [(7, 50, 2)]), (300.0, [(9, 50, 10), (8, 49, 1)])], yes, {(7, 50): [200.0]})
    chk('E2 ambiguity', len(o) == 0 and a == 1)
    # a later fill of 7 at 50 in a multi-maker stop replaces the proof (its order may have been used up)
    o, _, _, _ = observations(0, [(100.0, [(7, 50, 2)]), (200.0, [(7, 50, 2), (6, 50, 2)]), (300.0, [(9, 50, 10), (8, 49, 1)])], yes, {})
    chk('last fill must be the proof', len(o) == 0)
    # equal times: a record at t0 itself is not before t0
    o, _, _, _ = observations(0, [(300.0, [(7, 50, 2)]), (300.0, [(9, 50, 10), (8, 49, 1)])], yes, {})
    chk('no presence at or after t0', len(o) == 0)
    return bad


def mutation_check():
    global MUT_STOP_TAKEN, MUT_MULTI_SOLE, MUT_FUTURE
    res = {}
    for name in ('MUT_STOP_TAKEN', 'MUT_MULTI_SOLE', 'MUT_FUTURE'):
        globals()[name] = 1
        try:
            res[name] = bool(selftest())
        except ValueError:
            res[name] = True
        finally:
            globals()[name] = 0
    return res


def run_day(day):
    import lightgbm as lgb
    bad = selftest(); mut = mutation_check()
    if bad or not all(mut.values()):
        sys.exit(f'EXIT SELFTEST FAILED: {bad} {mut}')
    os.makedirs(OUTE, exist_ok=True); T0 = time.time()
    D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tkf) != len(full):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    LT = D['LT']; ix = LT.index.values
    LT = LT.assign(signer=full.signer.values[ix].astype('int64'), tk=np.where(tkf[ix] >= 0, tkf[ix], -1 - ix))
    del full
    wsg = pd.Series(LT.how.values > 0).groupby(LT.tk.values).transform('any').values
    LT = LT.assign(wsg=wsg)
    mdl = lgb.Booster(model_file=f'{FPDIR}/models/av_{day}.txt')
    LTI = LT.groupby(['mid', 'side'], sort=False).indices
    st.update(window_sides=0, orders_ws=0, events=0, presence=0, e2_ambiguous=0, obs=0, p_undefined=0)
    rows = []; vol = []; nb = len(K.G32)
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side = w['m'], w['side']
        if (m, side) not in LTI:
            continue
        x = LT.iloc[LTI[(m, side)]]
        ws, te = w['ws'], w['te']
        e2 = {}
        for s_, p_, a_, g_ in zip(x.signer.values, x.ptick.values, x.anc.values, x.wsg.values):
            if not g_:
                e2.setdefault((int(s_), int(p_)), []).append(float(a_))
        y = x[x.wsg.values & (x.anc.values >= ws) & (x.anc.values < te)]
        orders = []
        for tk_, g in y.groupby('tk', sort=False):
            orders.append((float(g.anc.values.min()), [(int(s_), int(p_), float(q_)) for s_, p_, q_ in zip(g.signer.values, g.ptick.values, g.shares.values)]))
        orders.sort(key=lambda o: o[0]); st['orders_ws'] += len(orders)
        tfl = w['tf'].tolist(); P5, S5 = w['P5'], w['S5']

        def present_after(p, t):
            j = bisect.bisect_right(tfl, t)
            return j < len(tfl) and bool(np.any((P5[j] == p) & (S5[j] > 0)))
        obs, nev, npres, amb = observations(side, orders, present_after, e2)
        st['events'] += nev; st['presence'] += npres; st['e2_ambiguous'] += amb
        if obs:
            Xw = np.column_stack([w['X'], np.full(w['n'], float(side))]).astype(np.float32)
            P = mdl.predict(Xw); vk = w['X'][:, nb]
            und = ~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0)
            P = np.where(und, np.nan, P)
            for o in obs:
                if o['t1'] >= o['t0']:
                    sys.exit('EXIT K: an observation uses a record at or after t0')
                k = int(np.searchsorted(w['tg'], o['t0'] - L_PULL, side='right')) - 1
                pv = float(P[k]) if k >= 0 else NAN
                st['p_undefined'] += int(not np.isfinite(pv))
                rows.append((int(day), m, side, o['signer'], G.get(o['signer'], 'rest'), o['p'], o['t0'], o['gap'], o['stayed'], pv))
        for t, legs in orders:                            # R393: volume by group, level-taking vs sole-maker stops
            taken, stop, at_stop = classify(side, legs)
            for s_, p_, q_ in legs:
                kind = 'taken' if p_ in taken else ('sole_stop' if (p_ == stop and len(at_stop) == 1) else 'other_stop')
                vol.append((int(day), G.get(s_, 'rest'), kind, q_))
        if st['window_sides'] % 80 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {len(rows)} obs, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    st['obs'] = len(rows)
    pd.DataFrame(rows, columns=['day', 'mid', 'side', 'signer', 'grp', 'p', 't0', 'gap', 'stayed', 'P']).to_parquet(f'{OUTE}/obs_{day}.parquet', index=False)
    pd.DataFrame(vol, columns=['day', 'grp', 'kind', 'q']).groupby(['day', 'grp', 'kind']).q.agg(['sum', 'count']).reset_index().to_parquet(
        f'{OUTE}/vol_{day}.parquet', index=False)
    json.dump(st, open(f'{OUTE}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: done in {time.time() - T0:.0f} s; {st}', flush=True)


def gone_share(x):
    return float(1.0 - x.stayed.mean()) if len(x) else NAN


def matched_diff(A):
    """GONE share top 5 minus field, matched on gap bins, weighted by the top 5's observations per bin."""
    num = 0.0; den = 0
    for nm, lo, hi in GAPB:
        b = A[(A.gap > lo) & (A.gap <= hi)] if lo > 0 else A[A.gap <= hi]
        t5 = b[b.grp == 'top 5']; fd = b[b.grp != 'top 5']
        if len(t5) and len(fd):
            num += len(t5) * (gone_share(t5) - gone_share(fd)); den += len(t5)
    return num / den if den else NAN


def report():
    A = pd.concat([pd.read_parquet(f'{OUTE}/obs_{d}.parquet') for d in DAYS9], ignore_index=True)
    V = pd.concat([pd.read_parquet(f'{OUTE}/vol_{d}.parquet') for d in DAYS9], ignore_index=True)
    ST = {d: json.load(open(f'{OUTE}/day_{d}.json')) for d in DAYS9}
    A['gb'] = pd.cut(A.gap, [-1e-9, 100.0, 250.0, 1000.0], labels=[g[0] for g in GAPB]).astype(str)
    obs_d = matched_diff(A)
    per = {d: matched_diff(A[A.day == int(d)]) for d in DAYS9}
    hi = sum(per[d] > 0 for d in DAYS9 if per[d] == per[d]); lo = sum(per[d] < 0 for d in DAYS9 if per[d] == per[d])
    rng = np.random.default_rng(20261113); nul = []
    for _ in range(NULL_N):
        B = A.copy()
        B['grp'] = B.groupby(['day', 'gb'], sort=False).grp.transform(lambda s: s.values[rng.permutation(len(s))])
        nul.append(matched_diff(B))
    nul = np.array(nul); p1, p99 = np.quantile(nul, [0.01, 0.99])
    bad = {}
    if abs(nul.mean()) > 0.01:
        bad['null mean not ~0'] = float(nul.mean())
    if bad:
        sys.exit(f'EXIT REPORT CONTROL FAILED -> no table: {bad}')
    R = {}
    if obs_d >= 0.05 and hi >= 7 and obs_d > p99:
        R['R390'] = 'TOP-LEAVE-MORE'
    elif obs_d <= -0.05 and lo >= 7 and obs_d < p1:
        R['R390'] = 'TOP-LEAVE-LESS'
    else:
        R['R390'] = 'NOT-SHOWN'
    R['R390_detail'] = dict(diff=obs_d, days_higher=hi, days_lower=lo, null_p1=float(p1), null_p99=float(p99), null_mean=float(nul.mean()))
    grp3 = ['top 5', '6-20', 'rest']
    tim = {g: {b[0]: (gone_share(A[(A.grp == g) & (A.gb == b[0])]), int(((A.grp == g) & (A.gb == b[0])).sum())) for b in GAPB} for g in grp3}
    ok = A[np.isfinite(A.P)].copy(); q = np.quantile(ok.P, [0.2, 0.8])
    sig = {}
    for g, msk in (('top 5', ok.grp == 'top 5'), ('field', ok.grp != 'top 5')):
        lo_ = ok[msk & (ok.P <= q[0])]; hi_ = ok[msk & (ok.P >= q[1])]
        sig[g] = dict(low=gone_share(lo_), high=gone_share(hi_), n_low=int(len(lo_)), n_high=int(len(hi_)), diff=gone_share(hi_) - gone_share(lo_))
    R['R392'] = 'SIGNAL-USED' if (sig['top 5']['diff'] >= 0.10 and sig['top 5']['diff'] > sig['field']['diff']) else 'NOT-SHOWN'
    R['R392_detail'] = sig
    vv = V.groupby(['grp', 'kind'])[['sum', 'count']].sum()
    siz = {}
    for kind in ('taken', 'sole_stop', 'other_stop'):
        tot = sum(vv.loc[(g, kind), 'sum'] for g in grp3 if (g, kind) in vv.index)
        siz[kind] = {g: dict(share=float(vv.loc[(g, kind), 'sum'] / tot) if (g, kind) in vv.index and tot else NAN,
                             sh_per_leg=float(vv.loc[(g, kind), 'sum'] / vv.loc[(g, kind), 'count']) if (g, kind) in vv.index else NAN) for g in grp3}
    stat = {k_: sum(ST[d].get(k_, 0) for d in DAYS9) for k_ in ('window_sides', 'orders_ws', 'events', 'presence', 'e2_ambiguous', 'obs', 'p_undefined')}
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    res = dict(as_of=asof, days=DAYS9, readings=R, timing=tim, size=siz, per_day=per, stat=stat,
               gone={g: gone_share(A[A.grp == g]) for g in grp3}, n={g: int((A.grp == g).sum()) for g in grp3})
    json.dump(res, open(f'{OUTE}/exit.json', 'w'), indent=1, default=float)
    L = [f'EXIT public tape, 9 consumed replay days, first 60 s, WS-timed taker orders; as of {asof}; controls passed (synthetic cases, '
         f'3 planted errors caught, no record at or after t0, null mean {nul.mean():+.4f})', f'statuses {stat}', '',
         'Readings: ' + json.dumps({k: v for k, v in R.items() if not k.endswith('_detail')}),
         f'R390: GONE share top 5 minus field, gap-matched {obs_d:+.3f} (days higher {hi}, lower {lo}; null p1 / p99 {p1:+.3f} / {p99:+.3f}); '
         f'per day ' + ' '.join(f'{per[d]:+.3f}' for d in DAYS9),
         'GONE share by group (all gaps): ' + ', '.join(f'{g} {res["gone"][g]:.3f} (n {res["n"][g]})' for g in grp3), '',
         'R391 GONE share by gap t0 - t1 (n):']
    for g in grp3:
        L.append(f'  {g:<6} ' + ' | '.join(f'{b}: {tim[g][b][0]:.3f} ({tim[g][b][1]})' for b in tim[g]))
    L += ['', f'R392 GONE share in the bottom / top quintile of P (P <= {q[0]:.3f} / >= {q[1]:.3f}):']
    for g in sig:
        L.append(f'  {g:<6} {sig[g]["low"]:.3f} ({sig[g]["n_low"]}) -> {sig[g]["high"]:.3f} ({sig[g]["n_high"]}), diff {sig[g]["diff"]:+.3f}')
    L += ['', 'R393 volume share and shares per maker leg, by trade kind:']
    for kind in siz:
        L.append(f'  {kind:<10} ' + ' | '.join(f'{g}: {siz[kind][g]["share"]:.1%}, {siz[kind][g]["sh_per_leg"]:.1f} sh/leg' for g in grp3))
    open(f'{OUTE}/exit.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        b = selftest(); mu = mutation_check()
        print('self-test', 'PASS' if not b and all(mu.values()) else f'FAIL {b} {mu}')
        sys.exit(0 if not b and all(mu.values()) else 1)
    elif cmd == 'day' and len(sys.argv) > 2:
        run_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
