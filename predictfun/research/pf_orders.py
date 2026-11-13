"""ORDERS (2026-11-13, session ce68ebe0; USER: "Start the information collection, we fill the gaps and figure out how they work"): how do
the top makers work, order by order? Public data on disk only, 9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s.

Build (extract <day>), v2 (2026-11-13): v1 (8fd16d97) read only the day's own hours and joined on content without rank; its control
stopped 10-23 at 86% mapped and wrote no map. v2 follows pf_txmap (legs_tk's builder): pf_step1's hours (day - 1 h .. + 2 h) minus the
sealed days and the forward days (asserted), and a join to legs_timed2 on B1a's nine content keys + rank within equal content, refined by
the maker (step 1's signer id -> address). Per row: maker / order hash / taker order (transaction + taker hash) each certain only if its
equal-content group has one candidate; else unknown (status). Controls: signer id <-> address one to one; one maker, side and price per
certain order hash; our taker orders agree with legs_tk's where both are certain (>= 99.9%); coverage of the first-minute cohort legs:
maker >= 90%, order hash >= 80% (thresholds set after v1's failure, before any analysis output; pf_txmap identifies 84-94% of legs).
Order-level measures use certain hashes; event-level ones (first in queue, at-best, takes, EXIT-v2) only COMPLETE trades (every leg of
the trade's content key has a certain taker order); unknown fills still count as a maker's last fill.
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
Controls of the analysis: synthetic cases for the position sign, the take classification, the creator, the ladder and EXIT-v2's handling
of incomplete trades; planted errors (inventory sign flipped, creator after the level died) must fail them.
POST HOC (added 2026-11-13 before any analysis output, after finding that legs without a certain taker order become one-leg 'taker orders'
in K.load_day, which EXIT used): EXIT-v2 = EXIT's GONE / STAYED observations rebuilt on exact, complete taker orders (no E2 tolerance rule;
unknown fills replace the proof); printed beside EXIT's numbers, no reading.
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


SEALED = ('20261028', '20261029', '20261030'); FORWARD = '20261104'
OLDKEY = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']   # a trade's content key (pf_txmap)
MIN_MAKER = 0.90; MIN_HASH = 0.80


def extract(day):
    """Build v2 (2026-11-13; v1 = 8fd16d97 read only the day's own hours and joined on content without rank: its control stopped 10-23 at 86%
    mapped, and no map or analysis output was written). As pf_txmap: pf_step1's hours (day - 1 h .. + 2 h) minus the sealed days and the
    forward days, and a join on content + rank within equal content. Per legs_timed2 row: maker certain, order hash certain and taker order
    (transaction + taker hash) certain only if its equal-content group (refined by the maker where needed) has one candidate; status 1 absent,
    2 counts differ. Controls: step 1's signer id <-> address one to one; one maker / side / price per certain order hash; coverage of the
    cohort's first-minute legs: maker >= 90%, order hash >= 80% (thresholds set after v1's failure, before any analysis output)."""
    if day not in DAYS9:
        sys.exit(f'REFUSED: {day} is not a consumed replay day')
    HALL = io.hours_of(day, extra_after=2, extra_before=1)
    hours = [h for h in HALL if h[:8] not in SEALED and h[:8] < FORWARD]
    if any(h[:8] in SEALED or h[:8] >= FORWARD for h in hours):
        sys.exit('REFUSED: a sealed or forward hour file in the read list')
    os.makedirs(OUTO, exist_ok=True); T0 = time.time()
    W0 = cohort(day)
    L = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet', columns=KEYS + ['signer', 't_rel'])
    L['_row'] = np.arange(len(L)); mids = set(L.mid.unique().tolist())
    rec = {}; st = dict(hours_read=hours, hours_excluded=[h for h in HALL if h not in hours], pages=0, maker_records=0)
    for h in hours:
        for recv, params, path, pl in io.rest_rows(h, ('/v1/orders/matches',)):
            st['pages'] += 1
            for x in pl.get('data') or []:
                mid = (x.get('market') or {}).get('id')
                if mid not in mids:
                    continue
                try:
                    ts = int(io.iso_s(x['executedAt']))
                except (KeyError, TypeError, ValueError):
                    continue
                tx = x['transactionHash']; t = x['taker']
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
                              t['signer'].lower(), tx[2:18] + '|' + (t.get('hash') or '')[2:18])
        if K.rss_mib() > MEM_CAP_MIB:
            sys.exit(f'REFUSED: resident memory {K.rss_mib():.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    R = pd.DataFrame(list(rec.values()), columns=KEYS + ['maker', 'ohash', 'taker', 'tko']); del rec
    R['mo'] = R.maker + ':' + R.ohash

    def rank_join(Lx, Rx, keys):
        gl = Lx.groupby(keys, sort=False, dropna=False); Lx = Lx.assign(_n=gl.cumcount().values, _cnt=gl[keys[0]].transform('size').values)
        gr = Rx.groupby(keys, sort=False, dropna=False)
        Rx = Rx.assign(_n=gr.cumcount().values, _cnt=gr[keys[0]].transform('size').values, _nid=gr.mo.transform('nunique').values,
                       _ntk=gr.tko.transform('nunique').values, _nmk=gr.maker.transform('nunique').values)
        J = Lx.merge(Rx[keys + ['_n', '_cnt', '_nid', '_ntk', '_nmk', 'maker', 'ohash', 'taker', 'tko']], on=keys + ['_n'], how='left', suffixes=('', '_r'))
        if len(J) != len(Lx) or not (J._row.values == Lx._row.values).all():
            sys.exit('ORDERS BUILD: the join changed rows')
        return J
    J = rank_join(L, R, KEYS)
    st0 = np.zeros(len(J), np.int8); st0[J.maker.isna().values] = 1; st0[(st0 == 0) & (J._cnt.values != J._cnt_r.values)] = 2
    clean = (st0 == 0) & (J._nid.values == 1) & (J._ntk.values == 1)
    idmap = J[clean].groupby('signer').maker.agg(lambda s_: set(s_))
    if any(len(v) != 1 for v in idmap.values):
        sys.exit('ORDERS BUILD CONTROL FAILED: a step-1 signer id maps to several addresses')
    idm = {k_: next(iter(v)) for k_, v in idmap.items()}
    if len(set(idm.values())) != len(idm):
        sys.exit('ORDERS BUILD CONTROL FAILED: an address maps to several step-1 signer ids')
    mk1 = (st0 == 0) & (J._nmk.values == 1); hk1 = (st0 == 0) & (J._nid.values == 1); tk1 = (st0 == 0) & (J._ntk.values == 1)
    OUT = pd.DataFrame(dict(row=J._row.values, status=st0, maker=np.where(mk1, J.maker.values, None), ohash=np.where(hk1, J.ohash.values, None),
                            taker=np.where(tk1, J.taker.values, None), tko=np.where(tk1, J.tko.values, None)))
    amb = (st0 == 0) & ~clean                             # refine by the maker (address of step 1's signer id)
    if amb.any():
        La = L[amb].assign(maker=L.signer[amb].map(idm).values)
        La = La[La.maker.notna()]
        keys2 = KEYS + ['maker']
        Ra = R[R.set_index(KEYS).index.isin(La.set_index(KEYS).index)]
        gl = La.groupby(keys2, sort=False); La = La.assign(_n=gl.cumcount().values, _cnt=gl.mid.transform('size').values)
        gr = Ra.groupby(keys2, sort=False)
        Ra = Ra.assign(_n=gr.cumcount().values, _cnt=gr.mid.transform('size').values, _nid=gr.ohash.transform('nunique').values, _ntk=gr.tko.transform('nunique').values)
        Ja = La.merge(Ra[keys2 + ['_n', '_cnt', '_nid', '_ntk', 'ohash', 'taker', 'tko']], on=keys2 + ['_n'], how='left', suffixes=('', '_r'))
        okc = Ja.ohash.notna().values & (Ja._cnt.values == Ja._cnt_r.values)
        hk = okc & (Ja._nid.values == 1); tk_ = okc & (Ja._ntk.values == 1)
        ix = Ja._row.values
        for col, ok_, src in (('maker', okc, Ja.maker.values), ('ohash', hk, Ja.ohash.values), ('tko', tk_, Ja.tko.values), ('taker', tk_, Ja.taker.values)):
            OUT.loc[ix, col] = np.where(ok_, src, OUT[col].values[ix])  # refinement only fills; a group-level certain value stays
    OUT['hash_ok'] = OUT.ohash.notna(); OUT['maker_ok'] = OUT.maker.notna(); OUT['tko_ok'] = OUT.tko.notna()
    c = OUT[OUT.hash_ok].merge(L[['_row', 'side', 'ptick']], left_on='row', right_on='_row')
    g = c.groupby('ohash')
    one = dict(makers=int((g.maker.nunique() > 1).sum()), sides=int((g.side.nunique() > 1).sum()), prices=int((g.ptick.nunique() > 1).sum()))
    coh = L.mid.isin(list(W0)).values; f60 = coh & (L.t_rel.values >= 0) & (L.t_rel.values < 60)
    cov = dict(maker=float(OUT.maker_ok.values[f60].mean()), hash=float(OUT.hash_ok.values[f60].mean()), taker_order=float(OUT.tko_ok.values[f60].mean()))
    tkr = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet')
    both = OUT.tko_ok.values & (tkr.tk_status.isin([0, 5]).values)
    pr = pd.DataFrame(dict(a=tkr.tk.values[both], b=OUT.tko.values[both]))
    agree = float(min((pr.groupby('a').b.nunique() == 1).mean(), (pr.groupby('b').a.nunique() == 1).mean())) if len(pr) else float('nan')
    st.update(raw_legs=int(len(R)), legs=int(len(L)), status={int(k_): int(v) for k_, v in zip(*np.unique(OUT.status.values, return_counts=True))},
              coverage_0_60=cov, hash_multi=one, taker_orders_agree_with_legs_tk=agree, windows=len(W0), seconds=round(time.time() - T0, 1))
    if cov['maker'] < MIN_MAKER or cov['hash'] < MIN_HASH or any(one.values()) or not (agree >= 0.999):
        json.dump(st, open(f'{OUTO}/build_{day}.json', 'w'), indent=1)
        sys.exit(f'ORDERS BUILD CONTROL FAILED -> no map: {st}')
    OUT[coh].to_parquet(f'{OUTO}/map_{day}.parquet', index=False)
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
    yes = lambda p, t: True
    o = exit_obs(0, [(100.0, [(7, 50, 2.0)], True), (300.0, [(9, 50, 10.0), (8, 49, 1.0)], True)], yes)
    chk('exit2 gone', len(o) == 1 and o[0]['stayed'] == 0)
    o = exit_obs(0, [(100.0, [(7, 50, 2.0)], True), (200.0, [(7, 50, 1.0)], False), (300.0, [(9, 50, 10.0), (8, 49, 1.0)], True)], yes)
    chk('exit2 an incomplete fill replaces the proof', len(o) == 0)
    o = exit_obs(0, [(100.0, [(7, 50, 2.0)], True), (300.0, [(9, 50, 10.0), (8, 49, 1.0)], False)], yes)
    chk('exit2 an incomplete trade is no event', len(o) == 0)
    o = exit_obs(0, [(100.0, [(7, 50, 2.0)], False), (300.0, [(9, 50, 10.0), (8, 49, 1.0)], True)], yes)
    chk('exit2 an incomplete trade is no proof', len(o) == 0)
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


def exit_obs(side, orders, present_after):
    """EXIT's observations (pf_exit_check.observations, 11fb2adc) on exact taker orders. orders: [(t, legs, complete)] time-sorted, legs
    [(maker, price, shares)]. An incomplete taker order (a leg of the trade has no certain taker order) gives no event and no proof; its
    known fills still replace a maker's last fill (they may have used the order up)."""
    import pf_exit_check as E
    rec = {}; obs = []
    for t, legs, comp in orders:
        if comp:
            taken, stop, at_stop = E.classify(side, legs)
            for p in taken:
                inE = {s_ for s_, q, _ in legs if q == p}
                for (s_, q), lst in rec.items():
                    if q != p or not lst:
                        continue
                    cand = [r for r in lst if r[0] < t]
                    if not cand:
                        continue
                    t1, pres = cand[-1]
                    if pres and t - t1 <= E.W_MAX:
                        obs.append(dict(maker=s_, p=p, t0=t, t1=t1, gap=t - t1, stayed=int(s_ in inE)))
            pres_ok = len(at_stop) == 1 and present_after(stop, t)
        else:
            stop = None; at_stop = set(); pres_ok = False
        for s_, p, _ in legs:
            rec.setdefault((s_, p), []).append((t, bool(pres_ok and p == stop and s_ in at_stop)))
    return obs


def day_measures(day):
    import lightgbm as lgb
    import pf_exit_check as E
    import pf_gap_top_makers as GP
    bad = an_selftest(); mut = an_mutation_check(); be = E.selftest(); bm = E.mutation_check()
    if bad or not all(mut.values()) or be or not all(bm.values()):
        sys.exit(f'ORDERS SELFTEST FAILED: {bad} {mut} {be} {bm}')
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'] + OLDKEY[1:] + ['how', 't_ms'])
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    MP = pd.read_parquet(f'{OUTO}/map_{day}.parquet').set_index('row')
    F = full[inw].join(MP, how='left')
    if F.status.isna().any():
        sys.exit('ORDERS: a cohort leg without a map row')
    F['comp'] = F.groupby(OLDKEY, sort=False).tko_ok.transform('all').values   # every leg of the trade has a certain taker order
    wsT = np.where((F.how.values > 0) & F.tko_ok.values, F.t_ms.values.astype(float), np.nan)
    F['tws'] = pd.Series(wsT, index=F.index).groupby(F.tko.values).transform('min').values if F.tko_ok.any() else np.nan
    idm = F[F.maker_ok].groupby('signer').maker.first().to_dict(); a2g = {a_: G.get(int(k_), 'rest') for k_, a_ in idm.items()}
    LT = D['LT']
    LT = LT[(LT.anc.values >= LT.ws.values) & (LT.anc.values < LT.ws.values + W1)].copy()
    for c in ('signer', 'maker', 'ohash', 'taker', 'tko', 'hash_ok', 'maker_ok', 'tko_ok', 'comp', 'tws'):
        LT[c] = F[c].reindex(LT.index).values
    LT['grp'] = [G.get(int(s_), 'rest') for s_ in LT.signer.values]
    LT['self'] = LT.maker_ok.values & LT.tko_ok.values & (LT.maker.values == LT.taker.values)
    st.update(legs_0_61=int(len(LT)), hash_ok=int(LT.hash_ok.sum()), tko_ok=int(LT.tko_ok.sum()), complete=int(LT.comp.sum()), self_legs=int(LT.self.sum()))
    mdl = lgb.Booster(model_file=f'{K.IO}/front_pull_20261113/models/av_{day}.txt')
    sole = set(); BR = []; atb = []; nb = 0; XO = []; nbB = len(K.G32)
    LI = LT.groupby(['mid', 'side'], sort=False).indices
    for w in K.window_sides(D):
        m, side = w['m'], w['side']
        if (m, side) not in LI:
            continue
        x = LT.iloc[LI[(m, side)]]; tfl = w['tf'].tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']
        x60 = x[x.anc.values < w['ws'] + 60000.0]
        present_after = lambda p, t: (lambda j: j < len(tfl) and bool(np.any((P5[j] == p) & (S5[j] > 0))))(int(np.searchsorted(tfl, t, side='right')))
        orders = []                                       # exact taker orders (WS-timed, complete) and the rest as single known fills
        y = x60[x60.comp.values & np.isfinite(x60.tws.values)]
        for tko, g in y.groupby('tko', sort=False):
            legs = [(int(s_), int(p_), float(q_)) for s_, p_, q_ in zip(g.signer.values, g.ptick.values, g.shares.values)]
            t = float(g.tws.values[0]); orders.append((t, legs, True))
            taken, stop, at_stop = E.classify(side, legs)
            if len(at_stop) == 1 and present_after(stop, t):
                for i_, p_ in zip(g.index.values, g.ptick.values):
                    if p_ == stop:
                        sole.add(int(i_))
            for i_, p_ in zip(g.index.values, g.ptick.values):
                if GP.depth_of(tfl, P5, opp, side, t, int(p_)) == 0:
                    atb.append(G.get(int(LT.signer.at[i_]), 'rest'))
        z = x60[~(x60.comp.values & np.isfinite(x60.tws.values))]
        for s_, p_, q_, a_ in zip(z.signer.values, z.ptick.values, z.shares.values, z.anc.values):
            orders.append((float(a_), [(int(s_), int(p_), float(q_))], False))
        orders.sort(key=lambda o: (o[0], not o[2]))
        ob = exit_obs(side, orders, present_after)
        if ob:
            Xw = np.column_stack([w['X'], np.full(w['n'], float(side))]).astype(np.float32)
            P = mdl.predict(Xw); vk = w['X'][:, nbB]
            P = np.where(~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0), np.nan, P)
            for o in ob:
                k = int(np.searchsorted(w['tg'], o['t0'] - E.L_PULL, side='right')) - 1
                XO.append((int(day), m, side, G.get(o['maker'], 'rest'), o['gap'], o['stayed'], float(P[k]) if k >= 0 else np.nan))
        lw = sorted((float(a_), int(p_), int(i_)) for a_, p_, i_ in zip(x60.anc.values, x60.ptick.values, x60.index.values) if np.isfinite(a_))
        for tb, p in births(w['tf'], P5[:, 0], side):
            if not (w['ws'] <= tb < w['ws'] + 60000.0):
                continue
            nb += 1
            key = first_fill(tb, p, tfl, lambda jj: bool(np.any((P5[jj] == p) & (S5[jj] > 0))), lw)
            if key is not None:
                BR.append(dict(grp=LT.grp.at[key], gap=float(LT.anc.at[key] - tb), hash_ok=bool(LT.hash_ok.at[key])))
    st.update(births_0_60=nb, births_filled=len(BR), exit2_obs=len(XO))
    LT['sole'] = LT.index.isin(sorted(sole))
    H = LT[LT.hash_ok.values].copy()
    H['fid'] = np.where(H.tko_ok.values, H.tko.values.astype(object), 'a' + H.anc.astype(str).values)   # one fill = one taker order
    og = H.groupby('ohash')
    O = pd.DataFrame(dict(grp=og.grp.first(), first=og.anc.min(), last=og.anc.max(), ws=og.ws.first(), fills=og.fid.nunique(),
                          shares=og.shares.sum(), sole=og.sole.sum(), side=og.side.first(), ptick=og.ptick.first(), maker=og.maker.first(), mid=og.mid.first()))
    O = O[O['first'] < O.ws + 60000.0]; O['life'] = O['last'] - O['first']
    f60 = H[H.anc.values < H.ws.values + 60000.0]
    lad = []; dpr = []; FG = f60.groupby(['maker', 'mid', 'side'], sort=False).indices
    for (mk, m, side), g in O.groupby(['maker', 'mid', 'side'], sort=False):
        iv = list(zip(g.ptick.values.tolist(), g['first'].values.tolist(), g['last'].values.tolist()))
        dpr.append((g.grp.values[0], len(set(g.ptick.values.tolist()))))
        xs = f60.iloc[FG[(mk, m, side)]] if (mk, m, side) in FG else f60.iloc[0:0]
        for t, p in zip(xs.anc.values.tolist(), xs.ptick.values.tolist()):
            lad.append((g.grp.values[0], others_held(iv, t, p)))
    ev = []                                               # positions: maker legs by step 1's signer id, taker legs by the certain taker address
    for r in LT.itertuples(index=False):
        if r.self:
            continue
        ev.append((r.mid, f's{int(r.signer)}', r.anc, pos_delta('m', r.side, r.shares), 'm', None, r.shares, r.side))
        if r.tko_ok:
            ev.append((r.mid, r.taker, r.anc, pos_delta('t', r.side, r.shares), 't', r.tko, r.shares, r.side))
    a2s = {a_: f's{int(k_)}' for k_, a_ in idm.items()}
    EV = pd.DataFrame(ev, columns=['mid', 'addr', 't', 'd', 'role', 'tko', 'q', 'side'])
    EV['addr'] = [a2s.get(a_, a_) for a_ in EV.addr.values]          # a taker that is a maker of the day: one identity
    s2g = {f's{int(k_)}': G.get(int(k_), 'rest') for k_ in set(LT.signer.values.tolist())}
    EV['grp'] = [s2g.get(a_, 'non-maker') for a_ in EV.addr.values]
    EV['comp'] = EV.tko.map(LT.drop_duplicates('tko').set_index('tko').comp) if EV.tko.notna().any() else False
    EV = EV.sort_values(['mid', 'addr', 't'], kind='stable')
    TK = []; MW = []
    for (m, a_), g in EV.groupby(['mid', 'addr'], sort=False):
        pos = 0.0; ro = g.role.values; d_ = g.d.values; tk_ = g.tko.values; q_ = g.q.values; cp = g.comp.values; gr = g.grp.values[0]
        k = 0; n = len(g)
        while k < n:
            if ro[k] == 't':
                k2 = k
                while k2 + 1 < n and ro[k2 + 1] == 't' and tk_[k2 + 1] == tk_[k]:
                    k2 += 1
                dd = float(d_[k:k2 + 1].sum())
                if gr != 'non-maker' and bool(cp[k]):
                    TK.append(dict(grp=gr, q=float(q_[k:k2 + 1].sum()), red=bool(reducing(pos, dd)), flat=bool(abs(pos) < 1e-12)))
                pos += dd; k = k2 + 1
            else:
                pos += float(d_[k]); k += 1
        w0 = D['W0'][m]; mm = (g.t.values < w0 + 60000.0) & (ro == 'm')
        if gr != 'non-maker' and mm.sum() >= 2:
            g60 = g[g.t.values < w0 + 60000.0]
            MW.append(dict(grp=gr, two_sided=len(set(g[mm].side.values.tolist())) == 2, net=abs(float(g60.d.sum())), gross=float(g60.q.sum())))
    RE = []
    for (mk, m, side, p), g in O.sort_values('first').groupby(['maker', 'mid', 'side', 'ptick'], sort=False):
        f_ = g['first'].values; l_ = g['last'].values
        for k in range(1, len(g)):
            if f_[k] > l_[k - 1]:
                RE.append((g.grp.values[0], float(f_[k] - l_[k - 1])))
    os.makedirs(OUTO, exist_ok=True)
    O.reset_index().to_parquet(f'{OUTO}/orders_{day}.parquet', index=False)
    pd.DataFrame(BR, columns=['grp', 'gap', 'hash_ok']).to_parquet(f'{OUTO}/births_{day}.parquet', index=False)
    pd.DataFrame(dict(grp=atb)).groupby('grp').size().rename('n').reset_index().to_parquet(f'{OUTO}/atbest_{day}.parquet', index=False)
    pd.DataFrame(lad, columns=['grp', 'others']).to_parquet(f'{OUTO}/ladder_{day}.parquet', index=False)
    pd.DataFrame(dpr, columns=['grp', 'prices']).to_parquet(f'{OUTO}/prices_{day}.parquet', index=False)
    pd.DataFrame(TK, columns=['grp', 'q', 'red', 'flat']).to_parquet(f'{OUTO}/takes_{day}.parquet', index=False)
    pd.DataFrame(MW, columns=['grp', 'two_sided', 'net', 'gross']).to_parquet(f'{OUTO}/mw_{day}.parquet', index=False)
    pd.DataFrame(RE, columns=['grp', 'gap']).to_parquet(f'{OUTO}/reentry_{day}.parquet', index=False)
    pd.DataFrame(XO, columns=['day', 'mid', 'side', 'grp', 'gap', 'stayed', 'P']).to_parquet(f'{OUTO}/exit2_{day}.parquet', index=False)
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
    import pf_exit_check as E                             # POST HOC (2026-11-13): EXIT on exact, complete taker orders (EXIT-v2)
    X2 = rd('exit2'); X2 = X2.assign(gb=pd.cut(X2.gap, [-1e-9, 100.0, 250.0, 1000.0], labels=[g[0] for g in E.GAPB]).astype(str))
    d2 = E.matched_diff(X2); per2 = [E.matched_diff(X2[X2.day == int(d)]) for d in DAYS9]
    ok2 = X2[np.isfinite(X2.P)]; q2 = np.quantile(ok2.P, [0.2, 0.8]) if len(ok2) else [np.nan, np.nan]
    sig2 = {}
    for g, msk in (('top 5', ok2.grp == 'top 5'), ('field', ok2.grp != 'top 5')):
        lo_ = ok2[msk & (ok2.P <= q2[0])]; hi_ = ok2[msk & (ok2.P >= q2[1])]
        sig2[g] = dict(low=E.gone_share(lo_), high=E.gone_share(hi_), n_low=int(len(lo_)), n_high=int(len(hi_)))
    up2 = sum(1 for d in DAYS9 if (lambda z: (lambda t_, f_: (E.gone_share(t_[t_.P >= q2[1]]) - E.gone_share(t_[t_.P <= q2[0]])) > (E.gone_share(f_[f_.P >= q2[1]]) - E.gone_share(f_[f_.P <= q2[0]])))(z[z.grp == 'top 5'], z[z.grp != 'top 5']))(ok2[ok2.day == int(d)]))
    res['exit_v2'] = dict(obs=int(len(X2)), gone={g: E.gone_share(X2[X2.grp == g]) for g in grp3}, matched_diff=d2, per_day=per2, signal=sig2, days_top_effect_larger=up2)
    json.dump(res, open(f'{OUTO}/orders.json', 'w'), indent=1, default=float)
    L += ['', f'EXIT-v2 (POST HOC: EXIT on exact, complete taker orders): observations {len(X2)}; GONE share ' + ', '.join(f'{g} {res["exit_v2"]["gone"][g]:.3f}' for g in grp3)
          + f'; gap-matched top 5 minus field {d2:+.3f} (per day ' + ' '.join(f'{x:+.3f}' for x in per2) + ')',
          '  GONE share bottom -> top P quintile: ' + '; '.join(f'{g} {v["low"]:.3f} ({v["n_low"]}) -> {v["high"]:.3f} ({v["n_high"]})' for g, v in sig2.items())
          + f'; days the top-5 P effect exceeds the field\'s: {up2} of 9']
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
