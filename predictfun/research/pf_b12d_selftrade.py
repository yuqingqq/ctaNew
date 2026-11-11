"""B12d POST HOC 2 (2026-11-11, written after the user asked whether the top makers must pull their own quotes to
take; descriptive, no reading). Venue allows self-matches (56 cohort match rows have taker == maker), so avoiding them
is up to the trader. For each top-5 maker (address from the same join as pf_b12d_takers.py): its taker shares in the
cohort's first minute; its self-match shares; and, for its taker orders at the best (first minute), whether the same
address had a maker fill at the same price and side in the 2 s before (it was quoting there) and in the 2 s after (it
quoted there again). Uses the committed pf_b12b_reason build unchanged (c882da77).
Usage: predictfun/research/run_light.sh b12d_selftrade -- python3 -m predictfun.research.pf_b12d_selftrade <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12b_reason as BB

MATCH = f'{BB.LV}/matches_20261108.parquet'


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = BB.controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    X = BB.build(); R = X['R']; W0 = X['W0']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    M = pd.read_parquet(MATCH, columns=['mid', 'exec_s', 'taker', 'maker', 'ptick', 'shares'])
    M = M[M.mid.isin(W0)].copy(); M['exec_s'] = M.exec_s.astype('int64'); M['sh6'] = M.shares.round(6)
    M['taker'] = M.taker.str.lower(); M['maker'] = M.maker.str.lower()
    key = ['mid', 'exec_s', 'ptick', 'sh6']
    Mu = M.merge(M.groupby(key).size().rename('n').reset_index().query('n == 1')[key], on=key)
    L = R[['mid', 'exec_s', 'ptick', 'shares', 'signer', 'grp', 'side', 'anc', 'g']].copy(); L['exec_s'] = L.exec_s.astype('int64'); L['sh6'] = L.shares.round(6)
    Lu = L.merge(L.groupby(key).size().rename('nl').reset_index().query('nl == 1')[key], on=key)
    J = Lu.merge(Mu[key + ['taker', 'maker']], on=key)
    addr = J.groupby('signer').maker.agg(lambda s: s.value_counts().index[0])
    tk = J.groupby('grp').taker.agg(lambda s: s.value_counts().index[0])
    R['taker'] = tk.reindex(R.grp.values).values; R['maker_addr'] = addr.reindex(R.signer.values).values
    top5 = sorted(int(s) for s in R.signer[R.g == 'top 5'].unique())
    res = dict(note='POST HOC, descriptive', self_matches_all=dict(rows=int((M.taker == M.maker).sum()), shares=float(M.shares[M.taker == M.maker].sum())), makers={})
    A = R[(R.status == 'ok') & (R.depth == 0)]
    for s in top5:
        a = str(addr.get(s, '')).lower()
        tk_legs = R[R.taker == a]; self_sh = float(M.shares[(M.taker == a) & (M.maker == a)].sum())
        mf = R[R.signer == s][['mid', 'side', 'ptick', 'anc']]
        tos = A[A.taker == a].groupby('grp').agg(mid=('mid', 'first'), side=('side', 'first'), ptick=('ptick', 'first'), anc=('anc', 'first')).reset_index()
        before = after = 0
        if len(tos):
            mm = mf.groupby(['mid', 'side', 'ptick']).anc.apply(lambda x: np.sort(x.values)).to_dict()
            for r in tos.itertuples():
                ts = mm.get((r.mid, r.side, r.ptick))
                if ts is None:
                    continue
                before += int(((ts >= r.anc - 2000.0) & (ts < r.anc)).any()); after += int(((ts > r.anc) & (ts <= r.anc + 2000.0)).any())
        res['makers'][str(s)] = dict(taker_legs_hit=int(len(tk_legs)), taker_shares=float(tk_legs.shares.sum()), self_match_shares=self_sh,
                                     taker_orders_at_best=int(len(tos)), quoted_same_price_2s_before=before, quoted_same_price_2s_after=after,
                                     own_maker_fills=int(len(mf)))
    json.dump(res, open(f'{out}/b12d_selftrade.json', 'w'), indent=1, default=float)
    Lt = ['B12d POST HOC 2: do the top makers hit their own quotes when they take? (descriptive)',
          f"All cohort self-matches: {res['self_matches_all']}"]
    for s, v in res['makers'].items():
        Lt.append(f"maker {s}: taker shares (first minute) {v['taker_shares']:,.1f} on {v['taker_legs_hit']} legs; self-match shares (all minutes) {v['self_match_shares']:.1f}; "
                  f"taker orders at the best {v['taker_orders_at_best']}: its own maker fill at that price and side in the 2 s before {v['quoted_same_price_2s_before']}, "
                  f"in the 2 s after {v['quoted_same_price_2s_after']}; own maker fills {v['own_maker_fills']}")
    open(f'{out}/b12d_selftrade.txt', 'w').write('\n'.join(Lt) + '\n'); print('\n'.join(Lt))


if __name__ == '__main__':
    main()
