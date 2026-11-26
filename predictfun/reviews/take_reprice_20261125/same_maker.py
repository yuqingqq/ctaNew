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
    O_all = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['taker_hash', 'taker', 'mid', 'd', 't_set', 'rank', 'day'])
    O_all = O_all[O_all.t_set.notna() & O_all.day.isin(days)]
    addr = {int(r_): a.lower() for r_, a in O_all.groupby('rank').taker.first().items()}
    res = {c: collections.Counter() for c in ('CO-TIMED', 'CONTROL')}
    share_num = {c: 0. for c in res}; share_den = {c: 0. for c in res}
    by_rank = collections.defaultdict(collections.Counter)
    first = collections.Counter()
    vol = collections.defaultdict(float)                     # (group, maker) -> shares
    n_legs, n_top = 0, 0
    for d_ in days:                                          # pairs never cross days: one day in memory at a time
        L = legs([d_])
        n_legs += len(L)
        grp = L.taker.map({a: f'rank{r_}' for r_, a in addr.items()}).fillna('other_takers')
        for (g_, m_), v in L.groupby([grp, L.maker]).shares.sum().items():
            vol[(g_, m_)] += float(v)
        Od = O_all[O_all.day == d_]
        Ld = L[L.taker_hash.isin(set(Od.taker_hash))].copy()
        del L
        Ld['mp'] = Ld.maker + '@' + Ld.ptick.astype(str)
        mk = Ld.groupby('taker_hash').maker.agg(frozenset).to_dict()
        mkp = Ld.groupby('taker_hash').mp.agg(frozenset).to_dict()
        T1 = Od[(Od['rank'] == 1) & Od.taker_hash.isin(mk)]
        sh1 = {h: g.groupby('maker').shares.sum().to_dict() for h, g in Ld[Ld.taker_hash.isin(set(T1.taker_hash))].groupby('taker_hash')}
        Rk = Od[(Od['rank'] > 1) & Od.taker_hash.isin(mk)]
        rk_by = {k: (g.t_set.to_numpy(float), g.taker_hash.to_numpy(), g['rank'].to_numpy(int)) for k, g in Rk.groupby(['mid', 'd'])}
        n_top += len(T1)
        for r in T1.itertuples():
            if (r.mid, r.d) not in rk_by:
                continue
            ts, hs, rs = rk_by[(r.mid, r.d)]
            lag = ts - float(r.t_set)
            ma, mpa, sa = mk[r.taker_hash], mkp[r.taker_hash], sh1[r.taker_hash]
            tot = sum(sa.values())
            for cls, sel in (('CO-TIMED', np.abs(lag) <= CO_MS), ('CONTROL', (np.abs(lag) >= CTL_LO) & (np.abs(lag) <= CTL_HI))):
                for j in np.flatnonzero(sel):
                    mb, mpb = mk[hs[j]], mkp[hs[j]]
                    cm = bool(ma & mb)
                    res[cls]['pairs'] += 1; res[cls]['common_maker'] += int(cm); res[cls]['common_maker_price'] += int(bool(mpa & mpb))
                    share_num[cls] += sum(v for m_, v in sa.items() if m_ in mb); share_den[cls] += tot
                    if cls == 'CO-TIMED':
                        by_rank[int(rs[j])]['pairs'] += 1; by_rank[int(rs[j])]['common_maker'] += int(cm)
                        first['top_first' if lag[j] > 0 else ('other_first' if lag[j] < 0 else 'same_ms')] += 1
        del Ld, mk, mkp, sh1
        C.require(FI.F.rss_mib() < FI.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = dict(asof='2026-11-26', days=list(days), co_ms=CO_MS, control_ms=[CTL_LO, CTL_HI], legs_joined=int(n_legs),
               top_orders_with_legs=int(n_top), classes={}, by_rank={}, cotimed_order=dict(first))
    for cls, c in res.items():
        out['classes'][cls] = dict(pairs=int(c['pairs']), common_maker=c['common_maker'] / c['pairs'] if c['pairs'] else None,
                                   common_maker_price=c['common_maker_price'] / c['pairs'] if c['pairs'] else None,
                                   top_shares_on_common_makers=share_num[cls] / share_den[cls] if share_den[cls] else None)
    out['by_rank'] = {r_: dict(pairs=int(c['pairs']), common_maker=c['common_maker'] / c['pairs']) for r_, c in sorted(by_rank.items()) if c['pairs']}
    V = pd.Series(vol)
    V.index = pd.MultiIndex.from_tuples(V.index, names=['grp', 'maker'])
    pt = V.loc['rank1'] / V.loc['rank1'].sum()
    sim = {}
    for g_ in sorted(set(V.index.get_level_values(0)) - {'rank1'}):
        pr = V.loc[g_] / V.loc[g_].sum()
        sim[g_] = float(np.minimum(pt, pr.reindex(pt.index).fillna(0.)).sum())
    out['similarity_to_top'] = sim
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
