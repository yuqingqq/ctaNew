"""Do the top takers that fire together take the same makers' quotes? (USER 2026-11-26: "You mean the top takers they taken the same signal,
their orders match almost at the same time? Do they target same maker?"). Readout on the four consumed days; no new data read.
Legs: step1 maker legs (maker = (day, code): step1 numbers signers per day) joined to their taker order on winners_legs' key (duplicate keys
dropped, as in info_batch2); orders and ranks: fire_times_orders (complete markets; settlement time per order).
Declared before the run: pairs = a top-taker order (rank 1) and an order of ranks 2-10 in the same market and direction whose settlement
times differ by <= 50 ms (CO-TIMED) or by 2-30 s (CONTROL: the same takers, not acting together). Per pair: they share a maker; they share a
maker at the same price; the share of the top-taker order's shares taken from a maker the other order also took from. Per class and per
rank; share of the top taker's co-timed orders that come first. Counterparty similarity per rank: sum over makers of min(share of the top
taker's volume, share of that group's volume), against other takers'.
Output (write-once): same_maker.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/same_maker.py [<rehearsal dir> <day>]
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import info_batch2 as I2  # noqa: E402

FI, C, TR, WT, MM = I2.FI, I2.C, I2.TR, I2.WT, I2.MM
KEY, DAYS = I2.KEY, I2.DAYS
CO_MS, CTL_LO, CTL_HI = 50., 2000., 30000.


def legs(days):
    parts = []
    for d_ in days:
        reg = set(MM.registry(d_).reset_index().mid.astype(int))
        S = pd.read_parquet(FI.S1 / d_ / 'legs.parquet', columns=KEY + ['signer'])
        Wl = pd.read_parquet(WT.OUT / f'winners_legs_{d_}.parquet', columns=KEY + ['taker', 'taker_hash', 'ambiguous'])
        S = S[S.mid.isin(reg)].drop_duplicates(KEY, keep=False)
        Wl = Wl[Wl.mid.isin(reg) & ~Wl.ambiguous & Wl.taker.notna()].drop_duplicates(KEY, keep=False)
        J = S.merge(Wl, on=KEY, how='inner')
        J['maker'] = d_ + ':' + J.signer.astype(str)
        J['taker'] = J.taker.str.lower()
        parts.append(J[['mid', 'maker', 'ptick', 'shares', 'taker', 'taker_hash']])
    return pd.concat(parts, ignore_index=True)


def main():
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else H
    days = (sys.argv[2],) if len(sys.argv) > 2 else DAYS
    fn = od / 'same_maker.json'
    C.require(not fn.exists(), 'same_maker.json exists (write-once)')
    L = legs(days)
    by_order = {h: g for h, g in L.groupby('taker_hash')}
    O = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['taker_hash', 'taker', 'mid', 'd', 't_set', 'rank', 'day'])
    O = O[O.t_set.notna() & O.day.isin(days) & O.taker_hash.isin(by_order)]
    T1, Rk = O[O['rank'] == 1], O[O['rank'] > 1]
    rk_by = {k: g.sort_values('t_set') for k, g in Rk.groupby(['mid', 'd'])}
    res = {c: collections.Counter() for c in ('CO-TIMED', 'CONTROL')}
    share_num = {c: 0. for c in res}; share_den = {c: 0. for c in res}
    by_rank = collections.defaultdict(collections.Counter)
    first = collections.Counter()
    for r in T1.itertuples():
        g = rk_by.get((r.mid, r.d))
        if g is None:
            continue
        lag = g.t_set.to_numpy(float) - float(r.t_set)
        A = by_order[r.taker_hash]
        ma, mpa = set(A.maker), set(zip(A.maker, A.ptick))
        for cls, sel in (('CO-TIMED', np.abs(lag) <= CO_MS), ('CONTROL', (np.abs(lag) >= CTL_LO) & (np.abs(lag) <= CTL_HI))):
            for j in np.flatnonzero(sel):
                o = g.iloc[j]
                B = by_order[o.taker_hash]
                mb, mpb = set(B.maker), set(zip(B.maker, B.ptick))
                cm = bool(ma & mb)
                res[cls]['pairs'] += 1; res[cls]['common_maker'] += int(cm); res[cls]['common_maker_price'] += int(bool(mpa & mpb))
                share_num[cls] += float(A.shares[A.maker.isin(mb)].sum()); share_den[cls] += float(A.shares.sum())
                if cls == 'CO-TIMED':
                    by_rank[int(o['rank'])]['pairs'] += 1; by_rank[int(o['rank'])]['common_maker'] += int(cm)
                    first['top_first' if lag[j] > 0 else ('other_first' if lag[j] < 0 else 'same_ms')] += 1
    out = dict(asof='2026-11-26', days=list(days), co_ms=CO_MS, control_ms=[CTL_LO, CTL_HI], legs_joined=int(len(L)),
               top_orders_with_legs=int(len(T1)), classes={}, by_rank={}, cotimed_order=dict(first))
    for cls, c in res.items():
        out['classes'][cls] = dict(pairs=int(c['pairs']), common_maker=c['common_maker'] / c['pairs'] if c['pairs'] else None,
                                   common_maker_price=c['common_maker_price'] / c['pairs'] if c['pairs'] else None,
                                   top_shares_on_common_makers=share_num[cls] / share_den[cls] if share_den[cls] else None)
    out['by_rank'] = {r_: dict(pairs=int(c['pairs']), common_maker=c['common_maker'] / c['pairs']) for r_, c in sorted(by_rank.items()) if c['pairs']}
    top = O.groupby('rank').taker.first().to_dict()
    vol = L.groupby(['taker', 'maker']).shares.sum()
    tt = top.get(1, '').lower()
    pt = vol.loc[tt] / vol.loc[tt].sum() if tt in vol.index.get_level_values(0) else pd.Series(dtype=float)
    sim = {}
    for r_, a in sorted(top.items()):
        if r_ == 1:
            continue
        a = a.lower()
        if a in vol.index.get_level_values(0):
            pr = vol.loc[a] / vol.loc[a].sum()
            sim[f'rank{r_}'] = float(np.minimum(pt, pr.reindex(pt.index).fillna(0.)).sum())
    oth = L[~L.taker.isin([x.lower() for x in top.values()])].groupby('maker').shares.sum()
    oth = oth / oth.sum()
    sim['other_takers'] = float(np.minimum(pt, oth.reindex(pt.index).fillna(0.)).sum())
    out['similarity_to_top'] = sim
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
