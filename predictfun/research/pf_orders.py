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


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'extract' and len(sys.argv) > 2:
        extract(sys.argv[2])
    else:
        sys.exit(__doc__)
