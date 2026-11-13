"""ORDERS (2026-11-13, session ce68ebe0; USER: "Start the information collection, we fill the gaps and figure out how they work"): how do
the top makers work, order by order? Public data on disk only, 9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s.

Build (extract <day>): the raw REST match pages of the day's own 24 hour files only (no padding hour: the sealed days 10-28..30 and the
forward days >= 11-04 are never opened; asserted) give, per maker leg, the maker's address and ORDER HASH, the taker's address and hash,
and the transaction hash (step 1 dropped them). They are joined to step 1's legs_timed2 rows on B1a's nine content keys (mid, exec_s,
first_recv_ns, side, ptick, shares, price_executed, taker_index, taker_quote); ties are resolved by the maker (step 1's per-day signer id
mapped to its address from the unique joins). Unmatched legs (first seen in a padding hour, or ambiguous) are a status.
Controls of the build (any failure -> no map): >= 98% of the first-minute legs of the cohort windows mapped; step 1's signer id <->
address one to one; every order hash has one maker, one side and one price.

Analysis (day <day>, report), on K.load_day's legs (WS / E2 anchors) of the first minute, groups by daily maker volume (B12's
rank_groups: top 5 / ranks 6-20 / rest; field = ranks 6+), WS-timed taker orders for the event-level measures:
  order       one maker order hash: its fills (distinct taker orders) in the first 61 s, shares filled (a lower bound of its size), life
              (first to last fill; censored at 61 s), sole-maker stops (it was the only maker filled by a taker order that stopped at its
              price, the next frame still showing the price: first in the queue with size left)
  creator     a level born as the side's best (the best moves to a better price between two frames) and later filled: the order of the first
              WS-timed fill at that price after the birth (first in that level's queue)
  ladder      per maker and window-side: distinct prices filled; and per fill, the other prices where the same maker has an order proven
              present at that time (an order hash whose first fill <= t <= last fill)
  takes       taker orders whose taker address is a maker of that day; the address's YES position in the window from all its fills
              (maker bid fill +, maker ask fill -, taker buy +, taker sell -) before the take; a take is inventory-reducing if it moves
              that position toward 0
  re-entry    per maker, side and price: the gap from one order's last fill to the next order's first fill (a new hash)
Readings DECLARED BEFORE THE FIRST ANALYSIS RUN (evaluated in code; day rule >= 7 of 9 days; top 5 against the field):
 R400 PERSIST: the share of orders filled by >= 3 taker orders, top 5 >= 1.5 x field pooled AND higher on >= 7 days -> PERSIST-MORE;
      <= 1 / 1.5 x and lower on >= 7 -> PERSIST-LESS; else NOT-SHOWN.
 R401 FRONT-KEEP: the share of orders with >= 2 sole-maker stops (first in queue at two or more trades), same rule -> FRONT-KEEP-MORE /
      FRONT-KEEP-LESS / NOT-SHOWN.
 R402 CREATOR: the top 5's share of creator orders against their share of first-minute fills at the best (legs): ratio >= 1.2 pooled and
      the share higher on >= 7 days -> CREATOR; <= 1 / 1.2 and lower on >= 7 -> FOLLOWER; else PROPORTIONAL.
 R403 LADDER (descriptive): distinct prices per maker and window-side; other prices held at a fill (median, p90).
 R404 TAKES: the share of the top 5's take shares that reduce their position: >= 60% -> REBALANCE; <= 40% -> DIRECTIONAL; else MIXED;
      and their take shares against their maker shares (descriptive).
 R405 RE-ENTRY (descriptive): the gap from an order's last fill to the same maker's next order at the same price (median, p25 / p75).
 Printed: per group orders, fills per order, shares per order, life, sole-maker stops, two-sided share (fills on both sides of the
 window), end-of-minute |position| / gross.
Controls of the analysis: synthetic cases for the position sign, the take classification, the creator and the ladder; a planted error
(inventory sign flipped) must fail them.
Limits: consumed days, first minute; an order's size is known only through its fills; orders that never fill are invisible; a maker can
split one quote into several orders; creator = first FILLED order at the new level (a creator cancelled before any fill is not seen).
Usage (run_light, PF_SLOT=claude): pf_orders.py extract <day> | day <day> | report; driver run_orders.sh
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import pf_kp_s4 as K                                      # noqa: E402  (day loader, window contexts, sealed / forward rules)
from predictfun.research import pf_io as io               # noqa: E402  (raw REST hour files)

DAYS9 = K.DAYS9
OUTO = f'{K.IO}/orders_20261113'
KEYS = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'taker_index', 'taker_quote']
MIN_MAPPED = 0.98
MEM_CAP_MIB = 2048.0


def tick(p):
    return int(round(p * 100))


def cohort(day):
    """The cohort windows of the day, as K.load_day: inside the day, not sealed / forward, with an outcome."""
    reg = pd.read_parquet(f'{K.IO}/step1/{day}/registry.parquet', columns=['mid', 'ws', 'outcome'])
    d0 = K._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}'); d1 = d0 + 86400000
    W0 = {}
    for m, ws, o in zip(reg.mid, reg.ws, reg.outcome):
        w = float(ws) * 1000.0
        if d0 <= w < d1 and not (K.SEAL_MS[0] <= w < K.SEAL_MS[1] or w >= K.FWD_MS) and o in ('Up', 'Down'):
            W0[int(m)] = w
    return W0


def extract(day):
    if day not in DAYS9:
        sys.exit(f'REFUSED: {day} is not a consumed replay day')
    hours = io.hours_of(day)                              # the day's own 24 hour files, no padding
    d0 = K._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}')
    if len(hours) != 24 or any(h[:8] != day for h in hours) or d0 >= K.FWD_MS or K.SEAL_MS[0] <= d0 < K.SEAL_MS[1]:
        sys.exit(f'REFUSED: hour files outside the consumed day {day}')
    os.makedirs(OUTO, exist_ok=True); T0 = time.time()
    W0 = cohort(day); rec = {}; st = dict(pages=0, matches=0, maker_records=0, outside_cohort=0, bad_time=0)
    for h in hours:
        for recv, params, path, pl in io.rest_rows(h, ('/v1/orders/matches',)):
            st['pages'] += 1
            for x in pl.get('data') or []:
                mid = (x.get('market') or {}).get('id')
                if mid not in W0:
                    st['outside_cohort'] += 1; continue
                try:
                    ts = int(io.iso_s(x['executedAt']))
                except (KeyError, TypeError, ValueError):
                    st['bad_time'] += 1; continue
                st['matches'] += 1; tx = x['transactionHash']; t = x['taker']
                for m in x['makers']:
                    st['maker_records'] += 1
                    k = tx[2:18] + m['hash'][2:18]
                    old = rec.get(k)
                    if old is not None:
                        if recv < old[2]:
                            rec[k] = old[:2] + (recv,) + old[3:]
                        continue
                    side, py = io.yes_side(m)
                    rec[k] = (mid, ts, recv, 0 if side == 'bid' else 1, tick(py), int(m['amount']) / io.E18, int(x['priceExecuted']) / io.E18,
                              t['outcome']['indexSet'], 0 if t['quoteType'] == 'Bid' else 1, m['signer'].lower(), m['hash'][2:18],
                              t['signer'].lower(), t['hash'][2:18], tx[2:18])
        if K.rss_mib() > MEM_CAP_MIB:
            sys.exit(f'REFUSED: resident memory {K.rss_mib():.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    R = pd.DataFrame(list(rec.values()), columns=KEYS + ['maker', 'ohash', 'taker', 'thash', 'tx']); del rec
    L = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet', columns=KEYS + ['signer', 't_rel'])
    L['row'] = np.arange(len(L)); L = L[L.mid.isin(list(W0))]
    J = L.merge(R, on=KEYS, how='inner')
    nL = J.groupby('row').row.transform('size'); nR = J.groupby(['ohash', 'tx']).row.transform('size')
    U = J[(nL == 1) & (nR == 1)]
    idmap = U.groupby('signer').maker.agg(lambda s: set(s))
    if any(len(v) != 1 for v in idmap.values):
        sys.exit(f'ORDERS BUILD CONTROL FAILED: a step-1 signer id maps to several addresses ({sum(len(v) != 1 for v in idmap.values)})')
    idm = {k: next(iter(v)) for k, v in idmap.items()}
    if len(set(idm.values())) != len(idm):
        sys.exit('ORDERS BUILD CONTROL FAILED: an address maps to several step-1 signer ids')
    A = J[~J.row.isin(U.row)]
    A = A[A.signer.map(idm).values == A.maker.values]
    nL2 = A.groupby('row').row.transform('size'); nR2 = A.groupby(['ohash', 'tx']).row.transform('size')
    M = pd.concat([U, A[(nL2 == 1) & (nR2 == 1)]], ignore_index=True)
    if M.row.duplicated().any() or M.duplicated(['ohash', 'tx']).any():
        sys.exit('ORDERS BUILD CONTROL FAILED: a leg or a record mapped twice')
    g = M.groupby('ohash')
    one = dict(makers=int((g.maker.nunique() > 1).sum()), sides=int((g.side.nunique() > 1).sum()), prices=int((g.ptick.nunique() > 1).sum()))
    f60 = (L.t_rel >= 0) & (L.t_rel < 60)
    share60 = float(L.row[f60].isin(M.row).mean()) if f60.any() else 0.0
    st.update(raw_legs=int(len(R)), cohort_legs=int(len(L)), joined_pairs=int(len(J)), unique=int(len(U)), resolved_by_maker=int(len(M) - len(U)),
              mapped=int(len(M)), mapped_share_all=float(L.row.isin(M.row).mean()), mapped_share_0_60=share60, hash_multi=one,
              windows=len(W0), seconds=round(time.time() - T0, 1))
    if share60 < MIN_MAPPED or any(one.values()):
        json.dump(st, open(f'{OUTO}/build_{day}.json', 'w'), indent=1)
        sys.exit(f'ORDERS BUILD CONTROL FAILED -> no map: {st}')
    M[['row', 'maker', 'ohash', 'taker', 'thash', 'tx']].to_parquet(f'{OUTO}/map_{day}.parquet', index=False)
    json.dump(st, open(f'{OUTO}/build_{day}.json', 'w'), indent=1)
    print(f'{day}: built {st}', flush=True)


# ---------------- analysis instruments ----------------
MUT_INV_SIGN = MUT_CREATOR_ANY = 0                        # planted errors (self-test only)
W1 = 61000.0                                              # legs anchored up to ws + 61 s (K.load_day's span)


def pos_delta(role, side, q):
    """YES position change of the maker (role 'm') or the taker (role 't') of one leg on the maker side `side` (0 = maker bid)."""
    d = q if side == 0 else -q
    if MUT_INV_SIGN:
        d = -d
    return d if role == 'm' else -d


def reducing(pos, d):
    return abs(pos + d) < abs(pos) - 1e-12


def births(tf, b1, side):
    """Frames where the side's best moves to a better price (a level born as the best): [(t, p)]."""
    out = []
    for j in range(1, len(tf)):
        a, b = b1[j - 1], b1[j]
        if a >= 0 and b >= 0 and ((b > a) if side == 0 else (b < a)):
            out.append((float(tf[j]), int(b)))
    return out


def first_fill(tb, p, tfl, present, legs):
    """The first leg at p after the birth at tb while the level lives (present(j) per frame index); legs [(t, p, key)] sorted."""
    j = int(np.searchsorted(tfl, tb, side='left'))
    td = float('inf')
    for jj in range(j, len(tfl)):
        if not present(jj):
            td = tfl[jj]; break
    for t, q, key in legs:
        if t > tb and q == p and (MUT_CREATOR_ANY or t <= td):
            return key
    return None


def others_held(iv, t, p):
    """iv: [(price, t_first, t_last)] of one maker's orders; distinct other prices with an order spanning t."""
    return len({q for q, a, b in iv if q != p and a <= t <= b})


def an_selftest():
    bad = []

    def chk(n, c):
        if not c:
            bad.append(n)
    chk('maker bid +', pos_delta('m', 0, 5.0) == 5.0)
    chk('maker ask -', pos_delta('m', 1, 5.0) == -5.0)
    chk('taker selling into bids -', pos_delta('t', 0, 5.0) == -5.0)
    chk('taker buying asks +', pos_delta('t', 1, 5.0) == 5.0)
    chk('reduce long', reducing(10.0, -5.0) and not reducing(0.0, 5.0) and reducing(-3.0, 5.0) and not reducing(2.0, 3.0))
    tf = [0.0, 50.0, 100.0, 150.0]
    chk('bid birth', births(tf, [50, 51, 51, 50], 0) == [(50.0, 51)])
    chk('ask birth', births(tf, [60, 59, 59, 60], 1) == [(50.0, 59)])
    pres = [True, True, True, False]
    legs = [(40.0, 51, 'x'), (60.0, 51, 'A'), (70.0, 51, 'B')]
    chk('creator', first_fill(50.0, 51, tf, lambda j: pres[j], legs) == 'A')
    chk('creator: level died first', first_fill(50.0, 51, tf, lambda j: pres[j], [(160.0, 51, 'C')]) is None)
    chk('ladder', others_held([(50, 0.0, 100.0), (49, 50.0, 150.0), (48, 120.0, 200.0)], 60.0, 50) == 1)
    return bad


def an_mutation_check():
    res = {}
    for name in ('MUT_INV_SIGN', 'MUT_CREATOR_ANY'):
        globals()[name] = 1
        try:
            res[name] = bool(an_selftest())
        finally:
            globals()[name] = 0
    return res


def day_measures(day):
    import pf_exit_check as E
    import pf_gap_top_makers as GP
    bad = an_selftest(); mut = an_mutation_check()
    if bad or not all(mut.values()):
        sys.exit(f'ORDERS SELFTEST FAILED: {bad} {mut}')
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tkf) != len(full):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    MP = pd.read_parquet(f'{OUTO}/map_{day}.parquet').set_index('row')
    LT = D['LT']; ix = LT.index.values
    LT = LT.assign(signer=full.signer.values[ix].astype('int64'), tk=np.where(tkf[ix] >= 0, tkf[ix], -1 - ix))
    del full
    LT = LT[(LT.anc.values >= LT.ws.values) & (LT.anc.values < LT.ws.values + W1)]
    mapped = LT.index.isin(MP.index); st['legs_0_61'] = int(len(LT)); st['legs_unmapped'] = int((~mapped).sum())
    LT = LT[mapped].copy()
    LT['maker'] = MP.maker.reindex(LT.index).values; LT['ohash'] = MP.ohash.reindex(LT.index).values; LT['taker'] = MP.taker.reindex(LT.index).values
    a2g = {}
    for s_, a_ in zip(LT.signer.values, LT.maker.values):
        a2g[a_] = G.get(int(s_), 'rest')
    LT['grp'] = [a2g[a_] for a_ in LT.maker.values]
    LT['wsg'] = pd.Series(LT.how.values > 0).groupby(LT.tk.values).transform('any').values
    LT['self'] = LT.maker.values == LT.taker.values
    st['self_legs'] = int(LT.self.sum())
    # sole-maker stops and creators per window-side (WS-timed taker orders, frames)
    sole = set(); BR = []; atb = []; nb = 0
    LI = LT.groupby(['mid', 'side'], sort=False).indices
    for w in K.window_sides(D):
        m, side = w['m'], w['side']
        if (m, side) not in LI:
            continue
        x = LT.iloc[LI[(m, side)]]; tfl = w['tf'].tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']
        y = x[x.wsg.values & (x.anc.values < w['ws'] + 60000.0)]
        for tk_, g in y.groupby('tk', sort=False):
            legs = [(int(s_), int(p_), float(q_)) for s_, p_, q_ in zip(g.signer.values, g.ptick.values, g.shares.values)]
            taken, stop, at_stop = E.classify(side, legs)
            t = float(g.anc.values.min()); j = int(np.searchsorted(tfl, t, side='right'))
            if len(at_stop) == 1 and j < len(tfl) and bool(np.any((P5[j] == stop) & (S5[j] > 0))):
                for i_, p_ in zip(g.index.values, g.ptick.values):
                    if p_ == stop:
                        sole.add(int(i_))
            for i_, p_, a_ in zip(g.index.values, g.ptick.values, g.anc.values):
                d_ = GP.depth_of(tfl, P5, opp, side, float(a_), int(p_))
                if d_ == 0:
                    atb.append(a2g[LT.maker.at[i_]])
        lw = sorted(zip(y.anc.values.tolist(), y.ptick.values.tolist(), y.index.values.tolist()))
        for tb, p in births(w['tf'], P5[:, 0], side):
            if not (w['ws'] <= tb < w['ws'] + 60000.0):
                continue
            nb += 1
            key = first_fill(tb, p, tfl, lambda jj: bool(np.any((P5[jj] == p) & (S5[jj] > 0))), lw)
            if key is not None:
                BR.append(dict(day=int(day), grp=LT.grp.at[key], tb=tb, gap=float(LT.anc.at[key] - tb)))
    st['births_0_60'] = nb; st['births_filled'] = len(BR)
    LT['sole'] = LT.index.isin(sorted(sole))
    f60 = LT[LT.anc.values < LT.ws.values + 60000.0]
    # orders whose first fill is in the first minute
    og = LT.groupby('ohash')
    O = pd.DataFrame(dict(grp=og.grp.first(), first=og.anc.min(), last=og.anc.max(), ws=og.ws.first(), fills=og.tk.nunique(),
                          shares=og.shares.sum(), sole=og.sole.sum(), side=og.side.first(), ptick=og.ptick.first(), maker=og.maker.first(),
                          mid=og.mid.first()))
    O = O[O['first'] < O.ws + 60000.0]; O['life'] = O['last'] - O['first']; O['day'] = int(day)
    # ladder: other prices held at each first-minute fill; distinct prices per maker and window-side
    lad = []; dpr = []; FG = f60.groupby(['maker', 'mid', 'side'], sort=False).indices
    for (mk, m, side), g in O.groupby(['maker', 'mid', 'side'], sort=False):
        iv = list(zip(g.ptick.values.tolist(), g['first'].values.tolist(), g['last'].values.tolist()))
        dpr.append((g.grp.values[0], len(set(g.ptick.values.tolist()))))
        xs = f60.iloc[FG[(mk, m, side)]] if (mk, m, side) in FG else f60.iloc[0:0]
        for t, p in zip(xs.anc.values.tolist(), xs.ptick.values.tolist()):
            lad.append((g.grp.values[0], others_held(iv, t, p)))
    # positions, takes, two-sided, end-of-minute
    ev = []
    for i_, r in zip(LT.index.values, LT.itertuples(index=False)):
        if r.self:
            continue
        ev.append((r.mid, r.maker, r.anc, pos_delta('m', r.side, r.shares), 'm', r.tk, r.shares, r.side))
        ev.append((r.mid, r.taker, r.anc, pos_delta('t', r.side, r.shares), 't', r.tk, r.shares, r.side))
    EV = pd.DataFrame(ev, columns=['mid', 'addr', 't', 'd', 'role', 'tk', 'q', 'side']).sort_values(['mid', 'addr', 't'], kind='stable')
    EV['grp'] = [a2g.get(a_, 'non-maker') for a_ in EV.addr.values]
    TK = []; MW = []
    for (m, a_), g in EV.groupby(['mid', 'addr'], sort=False):
        pos = 0.0; t_ = g.t.values; d_ = g.d.values; ro = g.role.values; tkk = g.tk.values; q_ = g.q.values
        k = 0; n = len(g)
        while k < n:                                      # one taker order of this address at one time: its legs together
            if ro[k] == 't':
                k2 = k
                while k2 + 1 < n and ro[k2 + 1] == 't' and tkk[k2 + 1] == tkk[k] and t_[k2 + 1] == t_[k]:
                    k2 += 1
                dd = float(d_[k:k2 + 1].sum())
                if g.grp.values[0] != 'non-maker':
                    TK.append(dict(grp=g.grp.values[0], q=float(q_[k:k2 + 1].sum()), red=bool(reducing(pos, dd)), flat=bool(abs(pos) < 1e-12)))
                pos += dd; k = k2 + 1
            else:
                pos += float(d_[k]); k += 1
        w0 = D['W0'][m]; mm = (g.t.values < w0 + 60000.0) & (g.role.values == 'm')
        if g.grp.values[0] != 'non-maker' and mm.sum() >= 2:
            gm = g[mm]; sides = set(gm.side.values.tolist())
            g60 = g[g.t.values < w0 + 60000.0]
            MW.append(dict(grp=g.grp.values[0], two_sided=len(sides) == 2, net=abs(float(g60.d.sum())), gross=float(g60.q.sum())))
    # re-entry gaps per maker, side, price
    RE = []
    for (mk, m, side, p), g in O.sort_values('first').groupby(['maker', 'mid', 'side', 'ptick'], sort=False):
        f_ = g['first'].values; l_ = g['last'].values
        for k in range(1, len(g)):
            if f_[k] > l_[k - 1]:
                RE.append((g.grp.values[0], float(f_[k] - l_[k - 1])))
    os.makedirs(OUTO, exist_ok=True)
    O.reset_index().to_parquet(f'{OUTO}/orders_{day}.parquet', index=False)
    pd.DataFrame(BR).to_parquet(f'{OUTO}/births_{day}.parquet', index=False)
    pd.DataFrame(dict(grp=atb)).groupby('grp').size().rename('n').reset_index().to_parquet(f'{OUTO}/atbest_{day}.parquet', index=False)
    pd.DataFrame(lad, columns=['grp', 'others']).to_parquet(f'{OUTO}/ladder_{day}.parquet', index=False)
    pd.DataFrame(dpr, columns=['grp', 'prices']).to_parquet(f'{OUTO}/prices_{day}.parquet', index=False)
    pd.DataFrame(TK).to_parquet(f'{OUTO}/takes_{day}.parquet', index=False)
    pd.DataFrame(MW).to_parquet(f'{OUTO}/mw_{day}.parquet', index=False)
    pd.DataFrame(RE, columns=['grp', 'gap']).to_parquet(f'{OUTO}/reentry_{day}.parquet', index=False)
    st.update(orders=int(len(O)), takes=len(TK), maker_windows=len(MW), reentries=len(RE), seconds=round(time.time() - T0, 1))
    json.dump(st, open(f'{OUTO}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st}', flush=True)


def report():
    rd = lambda nm: pd.concat([pd.read_parquet(f'{OUTO}/{nm}_{d}.parquet').assign(day=int(d)) for d in DAYS9], ignore_index=True)
    O = rd('orders'); BR = rd('births'); AB = rd('atbest'); LD = rd('ladder'); PR = rd('prices'); TK = rd('takes'); MW = rd('mw'); RE = rd('reentry')
    ST = {d: json.load(open(f'{OUTO}/day_{d}.json')) for d in DAYS9}
    grp3 = ['top 5', '6-20', 'rest']; fld = lambda x: x[x.grp != 'top 5']; top = lambda x: x[x.grp == 'top 5']
    R = {}

    def ratio_rule(name, f, labels):
        pt, pf = f(top(O)), f(fld(O))
        per = [(f(top(O[O.day == int(d)])), f(fld(O[O.day == int(d)]))) for d in DAYS9]
        hi = sum(a > b for a, b in per); lo = sum(a < b for a, b in per)
        r = pt / pf if pf else float('nan')
        R[name] = labels[0] if (r >= 1.5 and hi >= 7) else (labels[1] if (r <= 1 / 1.5 and lo >= 7) else 'NOT-SHOWN')
        R[f'{name}_detail'] = dict(top5=pt, field=pf, ratio=r, days_higher=hi, days_lower=lo)
    ratio_rule('R400', lambda x: float((x.fills >= 3).mean()) if len(x) else float('nan'), ('PERSIST-MORE', 'PERSIST-LESS'))
    ratio_rule('R401', lambda x: float((x.sole >= 2).mean()) if len(x) else float('nan'), ('FRONT-KEEP-MORE', 'FRONT-KEEP-LESS'))
    cs = lambda b, a: (float((b.grp == 'top 5').mean()) if len(b) else float('nan'), float(a.n[a.grp == 'top 5'].sum() / a.n.sum()) if a.n.sum() else float('nan'))
    c0, a0 = cs(BR, AB); per = [cs(BR[BR.day == int(d)], AB[AB.day == int(d)]) for d in DAYS9]
    hi = sum(c > a for c, a in per); lo = sum(c < a for c, a in per); r = c0 / a0 if a0 else float('nan')
    R['R402'] = 'CREATOR' if (r >= 1.2 and hi >= 7) else ('FOLLOWER' if (r <= 1 / 1.2 and lo >= 7) else 'PROPORTIONAL')
    R['R402_detail'] = dict(creator_share=c0, at_best_share=a0, ratio=r, days_higher=hi, days_lower=lo, births_filled=int(len(BR)))
    tt = top(TK); red = float(tt.q[tt.red].sum() / tt.q.sum()) if tt.q.sum() else float('nan')
    R['R404'] = 'REBALANCE' if red >= 0.6 else ('DIRECTIONAL' if red <= 0.4 else 'MIXED'); R['R404_detail'] = dict(reducing_share=red)
    q = lambda s, p: float(np.quantile(s, p)) if len(s) else float('nan')
    tab = {}
    for g in grp3:
        o = O[O.grp == g]; mw = MW[MW.grp == g]; tk = TK[TK.grp == g]; mk_sh = float(O.shares[O.grp == g].sum())
        tab[g] = dict(orders=int(len(o)), fills_med=q(o.fills, 0.5), fills_ge3=float((o.fills >= 3).mean()) if len(o) else float('nan'),
                      sh_med=q(o.shares, 0.5), sh_p90=q(o.shares, 0.9), life_med_ms=q(o.life[o.fills >= 2], 0.5), life_p90_ms=q(o.life[o.fills >= 2], 0.9),
                      sole_ge1=float((o.sole >= 1).mean()) if len(o) else float('nan'), sole_ge2=float((o.sole >= 2).mean()) if len(o) else float('nan'),
                      creator_share=float((BR.grp == g).mean()) if len(BR) else float('nan'),
                      atbest_share=float(AB.n[AB.grp == g].sum() / AB.n.sum()) if AB.n.sum() else float('nan'),
                      prices_med=q(PR.prices[PR.grp == g], 0.5), prices_p90=q(PR.prices[PR.grp == g], 0.9),
                      others_med=q(LD.others[LD.grp == g], 0.5), others_p90=q(LD.others[LD.grp == g], 0.9), others_ge1=float((LD.others[LD.grp == g] >= 1).mean()) if (LD.grp == g).any() else float('nan'),
                      take_sh=float(tk.q.sum()), maker_sh=mk_sh, take_over_maker=float(tk.q.sum() / mk_sh) if mk_sh else float('nan'),
                      take_reducing=float(tk.q[tk.red].sum() / tk.q.sum()) if tk.q.sum() else float('nan'),
                      take_from_flat=float(tk.q[tk.flat].sum() / tk.q.sum()) if tk.q.sum() else float('nan'),
                      two_sided=float(mw.two_sided.mean()) if len(mw) else float('nan'),
                      net_over_gross=float(mw.net.sum() / mw.gross.sum()) if mw.gross.sum() else float('nan'),
                      reentry_med_ms=q(RE.gap[RE.grp == g], 0.5), reentry_p25=q(RE.gap[RE.grp == g], 0.25), reentry_p75=q(RE.gap[RE.grp == g], 0.75),
                      creator_gap_med_ms=q(BR.gap[BR.grp == g], 0.5))
    stat = {k_: sum(ST[d].get(k_, 0) for d in DAYS9) for k_ in ('legs_0_61', 'legs_unmapped', 'self_legs', 'births_0_60', 'births_filled', 'orders', 'takes', 'maker_windows', 'reentries')}
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    res = dict(as_of=asof, days=DAYS9, readings=R, table=tab, stat=stat)
    json.dump(res, open(f'{OUTO}/orders.json', 'w'), indent=1, default=float)
    L = [f'ORDERS public tape + order hashes, 9 consumed replay days, first 60 s; as of {asof}; build and analysis controls passed', f'statuses {stat}', '',
         'Readings: ' + json.dumps({k: v for k, v in R.items() if not k.endswith('_detail')}),
         '  ' + json.dumps({k: v for k, v in R.items() if k.endswith('_detail')}, default=float), '']
    rows = [('orders (first fill in 0-60 s)', 'orders', '{:.0f}'), ('fills per order, median', 'fills_med', '{:.0f}'), ('orders filled >= 3 times', 'fills_ge3', '{:.1%}'),
            ('shares filled per order, median / p90', ('sh_med', 'sh_p90'), '{:.1f}'), ('life of orders filled >= 2 times, median / p90 ms', ('life_med_ms', 'life_p90_ms'), '{:.0f}'),
            ('orders first in queue >= 1 / >= 2 times', ('sole_ge1', 'sole_ge2'), '{:.1%}'), ('share of new-level creators / of at-best fills', ('creator_share', 'atbest_share'), '{:.1%}'),
            ('creator: birth to first fill, median ms', 'creator_gap_med_ms', '{:.0f}'),
            ('prices per maker and window-side, median / p90', ('prices_med', 'prices_p90'), '{:.0f}'), ('other prices held at a fill, median / p90', ('others_med', 'others_p90'), '{:.0f}'),
            ('fills with another price held', 'others_ge1', '{:.1%}'), ('take shares / maker shares', 'take_over_maker', '{:.1%}'),
            ('take shares reducing the position / from flat', ('take_reducing', 'take_from_flat'), '{:.1%}'), ('maker-windows two-sided', 'two_sided', '{:.1%}'),
            ('|net| / gross at the end of the minute', 'net_over_gross', '{:.1%}'), ('re-entry gap at the same price, median (p25 / p75) ms', ('reentry_med_ms', 'reentry_p25', 'reentry_p75'), '{:.0f}')]
    L.append(f'{"measure":<58} | ' + ' | '.join(f'{g:>22}' for g in grp3))
    for lab, key, fmt in rows:
        ks = key if isinstance(key, tuple) else (key,)
        L.append(f'{lab:<58} | ' + ' | '.join(f'{" / ".join(fmt.format(tab[g][k_]) for k_ in ks):>22}' for g in grp3))
    open(f'{OUTO}/orders.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'extract' and len(sys.argv) > 2:
        extract(sys.argv[2])
    elif cmd == 'selftest':
        b = an_selftest(); mu = an_mutation_check()
        print('self-test', 'PASS' if not b and all(mu.values()) else f'FAIL {b} {mu}')
        sys.exit(0 if not b and all(mu.values()) else 1)
    elif cmd == 'day' and len(sys.argv) > 2:
        day_measures(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
