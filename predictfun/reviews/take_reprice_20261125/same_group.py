"""Is another top taker the same group as the top taker? (USER 2026-11-26: "they take almost in the same ms, so they probably run by the
same group, other takers actually do not fire at the exact same time"). Readout on the four consumed days; no new data read.
Declared before the run (orders: fire_times_orders, complete markets, settlement ms per order; legs: winners_legs; the top taker's maker legs:
exit_times' scan):
A exact timing: per rank 2-10, settlement-time differences to the top taker's orders in the same market, -10..+10 ms by 1 ms, same direction
  and opposite direction; the share at exactly 0 ms against the mean of the 1-5 ms bins (a same-batch / same-engine spike).
B sizing fingerprint: per rank 1-10, order size (shares) and notional (sum of shares x price paid): median, the most common size and its
  share, the share at exactly 15 shares.
C opposite bets: per rank, pairs within 50 ms in the same market in the opposite direction against the same direction.
D trading with each other: the top taker's resting (maker) fills, by the rank of the taker order that hit them.
Output (write-once): same_group.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/same_group.py [<rehearsal dir>]
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

FI, C, TR, WT = I2.FI, I2.C, I2.TR, I2.WT
DAYS = I2.DAYS


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'same_group.json'
    C.require(not fn.exists(), 'same_group.json exists (write-once)')
    O = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['taker_hash', 'taker', 'mid', 'd', 'shares', 't_set', 'rank', 'day'])
    O = O[O.t_set.notna() & O.day.isin(DAYS)].copy()
    O['t_set'] = O.t_set.astype(np.int64)
    out = dict(asof='2026-11-26', exact_timing={}, sizing={}, opposite={}, maker_fills_by_taker_rank={})
    T1 = O[O['rank'] == 1]
    t1_by = {m: (g.t_set.to_numpy(), g.d.to_numpy()) for m, g in T1.groupby('mid')}
    for r_, G in O[O['rank'] > 1].groupby('rank'):
        same, opp = collections.Counter(), collections.Counter()
        n_same50 = n_opp50 = 0
        for m, g in G.groupby('mid'):
            if m not in t1_by:
                continue
            ta, da = t1_by[m]
            for t, d in zip(g.t_set.to_numpy(), g.d.to_numpy()):
                lo, hi = np.searchsorted(ta, t - 50), np.searchsorted(ta, t + 50, 'right')
                for k in range(lo, hi):
                    diff = int(t - ta[k])
                    if da[k] == d:
                        n_same50 += 1
                        if abs(diff) <= 10:
                            same[diff] += 1
                    else:
                        n_opp50 += 1
                        if abs(diff) <= 10:
                            opp[diff] += 1
        base = np.mean([same[k] for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)])
        out['exact_timing'][int(r_)] = dict(same_dir_hist=[int(same[k]) for k in range(-10, 11)], opp_dir_hist=[int(opp[k]) for k in range(-10, 11)],
                                            at_0ms=int(same[0]), mean_1_5ms=float(base), spike_ratio=float(same[0] / base) if base > 0 else None)
        out['opposite'][int(r_)] = dict(same_direction_within_50ms=int(n_same50), opposite_direction_within_50ms=int(n_opp50))
    TO = T1.sort_values('t_set')
    W = pd.concat([pd.read_parquet(WT.OUT / f'winners_legs_{d_}.parquet', columns=['taker_hash', 'shares', 'price_executed']) for d_ in DAYS], ignore_index=True)
    notional = (W.shares * W.price_executed).groupby(W.taker_hash).sum()
    O['notional'] = O.taker_hash.map(notional)
    for r_, G in O.groupby('rank'):
        sz = G.shares.round(6)
        vc = sz.value_counts()
        out['sizing'][int(r_)] = dict(orders=int(len(G)), median_shares=float(sz.median()), mode_shares=float(vc.index[0]), mode_share=float(vc.iloc[0] / len(G)),
                                      at_15_shares=float((np.abs(sz - 15.) < 1e-6).mean()), median_notional_usd=float(G.notional.median()),
                                      notional_mode=float(G.notional.round(2).value_counts().index[0]), notional_mode_share=float(G.notional.round(2).value_counts().iloc[0] / len(G)))
    X = pd.concat([pd.read_parquet(WT.OUT / f'exit_legs_{d_}.parquet') for d_ in DAYS], ignore_index=True)
    X = X.sort_values('recv_ns', kind='stable').drop_duplicates(['tx', 'maker_hash'], keep='first')
    Mk = X[X.role == 'maker']
    rank_of = O.drop_duplicates('taker_hash').set_index('taker_hash')['rank']
    Wt = pd.concat([pd.read_parquet(WT.OUT / f'winners_legs_{d_}.parquet', columns=['taker_hash', 'taker']) for d_ in DAYS]).drop_duplicates('taker_hash').set_index('taker_hash').taker
    top10 = {a.lower(): i + 1 for i, a in enumerate(I2.FT.top10())}
    rk = Mk.taker_hash.map(lambda h: rank_of.get(h, top10.get(str(Wt.get(h, '')).lower(), 'other')))
    out['maker_fills_by_taker_rank'] = {str(k): dict(legs=int(v), shares=float(Mk.shares[rk == k].sum())) for k, v in rk.value_counts().items()}
    out['maker_legs'] = int(len(Mk))
    TR.dump(fn, out)
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
